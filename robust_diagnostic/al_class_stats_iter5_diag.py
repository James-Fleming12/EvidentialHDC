"""al_class_stats_iter5_diag.py: Iteration 5 -- the COEFFICIENT-ESTIMATION
diagnostic (the single remaining bottleneck of the class-statistics
reformulation, class_stats_iters.md).

Iteration 4 localized the failure completely: the pool direction v_c =
M_pseudo_c - M0_c points at the true shift (align +0.83-0.92), the direction is
robust to ~50% corruption (Iteration 3), and the confusion-correction saturates
at +0.05 -- but the label-estimated SCALAR gamma_c from 2-8 labels is wrong and
the whitening amplifies the mis-scaled step. This iteration asks THE remaining
question:

  Given the known-good direction v_c, can a practical estimator of the single
  scalar gamma_c land in a useful operating window?

Measured pieces:

A. THE WINDOW -- gc(gamma) for a GLOBAL scalar applied to the top-K suspicious
   classes, M_corr_c = M0_c + gamma * v_c, gamma in a sweep. Establishes:
   - is there a positive plateau (a real operating window)?
   - how wide is it (required estimation precision)?
   Also the PER-CLASS ORACLE gamma*_c = <M*_c - M0_c, v_c>/||v_c||^2 and its gc.
   THE DECISIVE NUMBER: if gc(per-class gamma*) ~ W_mean_oracle gc, then scalar
   estimation is THE WHOLE problem (the direction family is sufficient). If
   gc(gamma*) << W_mean_oracle, the direction family itself caps out.

B. THE ESTIMATORS -- practical gamma_hat from b labels, applied per class on
   v_c, compared to gamma*:
   raw       gamma = <M_lab_c - M0_c, v_c>/||v_c||^2   (the Iteration-4 failure)
   shrink1   (raw + alpha * 1)/(1 + alpha)  alpha in {1, 4}  -- pool-regularize
             the SCALAR toward 1 (gamma=1 is the label-free pseudo-mean default,
             the +0.05-0.12 positive)
   gamma1    gamma = 1.0 (the label-free default, reference)
   normscale GLOBAL: choose gamma so the whitened update norm ~ ||R|| (the
             corruption-control operating point) -- step-size calibration
   oracle    gamma = gamma* (upper bound, diagnostic only)
   Report gc per estimator AND the gamma_hat values vs gamma* (bias: are they
   systematically too small/large?).

Usage:
  uv run python robust_diagnostic/al_class_stats_iter5_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_class_stats_iter5_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                  labels=data["labels"], color_map=data["color_map"], learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"], sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_clean(model, parser, device, num_frames=100):
    feats, lbls = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device); labels = batch[2].to(device).view(-1); mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol); z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(zf.cpu()); lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(lbls)


def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)


def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc); y[torch.arange(len(lbls)), lbls.long()] = 1; return y


def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    p = []
    for s in range(0, len(codes), chunk):
        p.append((codes[s:s+chunk].float() @ W).argmax(1))
    return torch.cat(p)


def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)


def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device); torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1], max(1, X.shape[0] - 1))
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P; Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device); That = XP.t() @ Yd
    x0 = P @ torch.linalg.solve(Shat, That)
    if X.shape[0] <= 8:
        return x0.float()
    x = x0; b = X.t() @ Yd
    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); d = (p * Ap).sum(0)
        if not torch.isfinite(d).all() or d.abs().max().item() < 1e-20:
            break
        a = rs / (d + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0); be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p; rs = rsn
        if not torch.isfinite(x).all():
            return x0.float()
    return x.float()


def solve_whitened(X, B, lam, iters, m, device):
    X = X.to(device); B = B.float().to(device)
    torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1], max(1, X.shape[0] - 1))
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    w0 = P @ torch.linalg.solve(Shat, P.t() @ B)
    if B.shape[0] <= 8:
        return w0.float()
    x = w0; b = B
    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); d = (p * Ap).sum(0)
        if not torch.isfinite(d).all() or d.abs().max().item() < 1e-20:
            break
        a = rs / (d + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0); be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p; rs = rsn
        if not torch.isfinite(x).all():
            return w0.float()
    return x.float()


def class_means(X, y, nc):
    M = torch.zeros(nc, X.shape[1]); C = torch.zeros(nc)
    for c in range(nc):
        m = (y == c)
        if int(m.sum().item()) > 0:
            M[c] = X[m].mean(dim=0)
            C[c] = float(int(m.sum().item()))
    return M, C


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync(); return time.time()


def toc(t0):
    sync(); return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--pool_size", type=int, default=20000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=30000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--b_per_class", type=str, default="2,4,8")
    ap.add_argument("--k", type=int, default=5, help="top-K suspicious classes")
    ap.add_argument("--gamma_sweep", type=str, default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--alpha_sweep", type=str, default="1,4")
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    b_sweep = [int(x) for x in args.b_per_class.split(',')]
    gamma_sweep = [float(x) for x in args.gamma_sweep.split(',')]
    alpha_sweep = [float(x) for x in args.alpha_sweep.split(',')]
    K = args.k

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_per_class': b_sweep,
               'k': K, 'gamma_sweep': gamma_sweep, 'alpha_sweep': alpha_sweep, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    lac = la[ci]
    W0 = ridge_fit_soft(Xc, onehot(lac, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
    M0, C0 = class_means(Xc, lac, NUM_CLASSES)
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        fd, ld = extract_clean(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(fd))
        pool_f, pl = fd[perm[:args.pool_size]], ld[perm[:args.pool_size]]
        val_f, vl = fd[perm[-args.val_size:]], ld[perm[-args.val_size:]]
        Xp = hdc_codes(pool_f, proj, device).float()
        Xv = hdc_codes(val_f, proj, device).float()
        del val_f, fd, ld
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        def gc(mi):
            return (mi - refs['frozen']) / gap if gap > 1e-9 else None

        M_star, C_star = class_means(Xp, pl, NUM_CLASSES)
        R = (Ws - W0).detach().cpu().float()
        R_norm = R.norm().item()

        B_mean_oracle = (M_star * C0.unsqueeze(1)).t().contiguous()
        W_mean_oracle = solve_whitened(Xp, B_mean_oracle, args.lam, args.cg_iters, args.nystrom_m, device)

        # frozen probe on the pool -> pseudo-mean direction v
        Lp = Xp.float() @ W0c
        pseudo = Lp.argmax(1)
        M_hard, C_ph = class_means(Xp, pseudo, NUM_CLASSES)
        V = M_hard - M0                       # the known-good direction (align 0.9)
        shift_norm = torch.norm(V, p=2, dim=1)
        suspicious = [int(c) for c in torch.argsort(shift_norm, descending=True) if c != 0][:K]

        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        Xp_cpu = Xp

        # per-class oracle gamma* = projection of the TRUE shift onto v_c
        gamma_star = {}
        for c in suspicious:
            v_c = V[c]
            vn = v_c.norm().item() ** 2 + 1e-12
            gamma_star[c] = float(((M_star[c] - M0[c]) * v_c).sum().item() / vn)

        def build_W(gammas):
            M_c = M0.clone()
            for c in suspicious:
                M_c[c] = M0[c] + gammas[c] * V[c]
            return solve_whitened(Xp, (M_c * C0.unsqueeze(1)).t().contiguous(),
                                  args.lam, args.cg_iters, args.nystrom_m, device)

        # ---- A. THE WINDOW ----
        curve = {}
        for g in gamma_sweep:
            W_g = build_W({c: g for c in suspicious})
            curve[str(g)] = gc(mw(W_g, Xv, vl))
        W_star = build_W(gamma_star)
        gc_gamma_star = gc(mw(W_star, Xv, vl))
        # update norm of the gamma=1 step vs R (the corruption-control operating pt)
        W_one = build_W({c: 1.0 for c in suspicious})
        upd_norm_one = float((W_one - W0).detach().cpu().norm().item() / (R_norm + 1e-12))

        cond_res = {'refs': refs, 'gap': float(gap),
                    'ladder': {'W0': 0.0,
                               'W_mean_oracle': gc(mw(W_mean_oracle, Xv, vl)),
                               'gamma_star_perclass': gc_gamma_star,
                               'W*': 1.0},
                    'gamma_star': {str(c): round(v, 2) for c, v in gamma_star.items()},
                    'curve': {k: v for k, v in curve.items()},
                    'upd_norm_gamma1': upd_norm_one,
                    'estimators': {}}

        # ---- B. THE ESTIMATORS ----
        for b in b_sweep:
            obs_means = M0.clone(); obs_counts = torch.zeros(NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                idx = class_idx[c]
                if len(idx) == 0:
                    continue
                torch.manual_seed(7 + c)
                sub = idx[torch.randperm(len(idx))[:min(b, len(idx))]]
                obs_means[c] = Xp_cpu[sub].mean(dim=0)
                obs_counts[c] = float(len(sub))
            gamma_raw = {}
            for c in suspicious:
                v_c = V[c]
                vn = v_c.norm().item() ** 2 + 1e-12
                gamma_raw[c] = float(((obs_means[c] - M0[c]) * v_c).sum().item() / vn) \
                    if obs_counts[c] > 0 else 0.0
            est = {}
            W_r = build_W(gamma_raw)
            est['raw'] = {'gc': gc(mw(W_r, Xv, vl)),
                          'gamma': {str(c): round(gamma_raw[c], 2) for c in suspicious}}
            for alpha in alpha_sweep:
                gamma_sh = {c: (gamma_raw[c] + alpha * 1.0) / (1.0 + alpha) for c in suspicious}
                W_s = build_W(gamma_sh)
                est[f'shrink1_a{alpha}'] = {'gc': gc(mw(W_s, Xv, vl)),
                                            'gamma': {str(c): round(gamma_sh[c], 2) for c in suspicious}}
            W_1 = build_W({c: 1.0 for c in suspicious})
            est['gamma1'] = {'gc': gc(mw(W_1, Xv, vl))}
            # normscale: global scalar so the update norm ~ ||R||
            W_one = build_W({c: 1.0 for c in suspicious})
            step1 = (W_one - W0).detach().cpu().norm().item()
            g_norm = R_norm / (step1 + 1e-12)
            W_n = build_W({c: g_norm for c in suspicious})
            est['normscale'] = {'gc': gc(mw(W_n, Xv, vl)), 'gamma': round(g_norm, 3)}
            W_o = build_W(gamma_star)
            est['oracle'] = {'gc': gc(mw(W_o, Xv, vl))}
            cond_res['estimators'][str(b)] = est

        results['conds'][cond] = cond_res
        del Xv, Ws, R, M_star, Xp_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        l = cond_res['ladder']
        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    ladder W0 {l['W0']:+.2f} W_mean_oracle {l['W_mean_oracle']:+.2f} "
              f"gamma*_perclass {l['gamma_star_perclass']:+.2f} W* {l['W*']:+.2f}")
        print("    gamma*: " + " ".join(f"c{c}:{v:.2f}" for c, v in gamma_star.items()))
        print("    curve: " + " ".join(f"g{k}:{v:+.2f}" for k, v in curve.items()))
        print(f"    upd_norm@gamma1 {upd_norm_one:.2f}xR")
        for b in b_sweep:
            e = cond_res['estimators'][str(b)]
            line = " ".join(f"{k}:{v['gc']:+.2f}" for k, v in e.items())
            print(f"    b{b}: {line}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. THE WINDOW (curve): is there a positive plateau of gc(gamma)? How")
    print("   wide? The decisive number: gc(gamma*_perclass) ~ W_mean_oracle ->")
    print("   scalar estimation is THE WHOLE problem; the direction is sufficient.")
    print("   upd_norm@gamma1 ~ 35x means the gamma=1 step still oversteps.")
    print("B. THE ESTIMATORS: raw (Iteration-4 failure), shrink1 (pool-regularize")
    print("   the scalar toward 1), gamma1 (label-free default), normscale (step-")
    print("   size calibration), oracle (gamma*). gc near the curve's peak -> the")
    print("   estimator works; gamma values vs gamma* show the bias.")


if __name__ == "__main__":
    main()
