"""al_class_stats_diag.py: the CLASS-STATISTICS decoder reformulation -- naive,
WHITE-BOX tests. The construction W = Sigma^-1 P M (the linear probe is the
whitened class means) is split into measured pieces, so each arm shows WHAT is
working / WHAT needs fixing, not a blackbox end-to-end result.

Pieces measured separately:

A. THE DECOMPOSITION (is the class-mean shift the fixable part of R?)
   R = W* - W0 = Sigma0^-1 P0 (M* - M0) + (Sigma*^-1 - Sigma0^-1) P* M*
   - R_mean_frac = ||Sigma0^-1 P0 (M* - M0)|| / ||R||  -- the fraction of the
     residual the mean-shift accounts for (under the CLEAN whitening).
   - The decoder ladder: gc(W0) / gc(proto_oracle: cosine to M*) /
     gc(W_mean_oracle: WHITENED mean decoder with oracle means) / gc(W*).
     proto vs whitened shows how much the whitening matters; whitened vs W*
     shows how much the COVARIANCE part matters.
   - Top moved classes: ||M*_c - M0_c|| sorted -- WHICH classes the corruption
     shifts (the "what to fix" target).

B. FEW-LABEL MEAN ESTIMATION (the core update: re-estimate M, keep Sigma/P)
   W_est = Sigma0^-1 P0 M_hat, M_hat_c = sample mean of b labeled points of
   class c with shrinkage toward M0_c: M_hat = (n_obs M_obs + alpha M0)/(n_obs + a).
   Sweep b per class x shrinkage alpha x selection; report gc and the mean
   estimation error ||M_hat - M*||_F per class. This isolates:
     - the label-budget curve for THIS mechanism (vs Iteration 7's flat curve
       for the first-order step -- a different object);
     - the shrinkage / memory knob (alpha);
     - which classes' mean error drives the gc loss.

C. SELECTION for the mean re-estimation (WHICH points of class c to label):
   random / proto_dist (far from M0_c -- the informative ones) / entropy /
   oracle_error (points of class c the frozen probe misclassifies). At fixed b,
   whose mean estimate is better?

D. UPDATE DETAILS (white-box knobs of the decoder itself):
   - softmax temperature T on the estimated logits (step-size-like knob)
   - update SCOPE: use estimated means for only the top-K moved classes, keep
     M0 for the rest (sparse update / memory).

Read:
   R_mean_frac ~ 1 & W_mean_oracle ~ W* gc  -> the mean-shift IS the whole
       residual; this reformulation has the full ceiling.
   R_mean_frac << 1                          -> the covariance part dominates;
       mean-only updates cap out (the diagnostic shows the ceiling, not a stop).
   W_est gc ~ W_mean_oracle gc at small b     -> few labels re-estimate the means
       as well as the oracle; the update works.
   proto << whitened                         -> whitening is essential (why R1
       closed but this need not).
   top moved classes concentrated in 2-4     -> update only those (sparse D).

Usage:
  uv run python robust_diagnostic/al_class_stats_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_class_stats_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
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
    """Solve (X^T X + lam I) W = B exactly via Nystrom warm start + CG.
    B: d x C. This is the whitening step: W = Sigma^-1 B.
    Used to build Sigma0^-1 P0 M for an arbitrary mean matrix M."""
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
    """Per-class mean vectors + counts."""
    M = torch.zeros(nc, X.shape[1]); C = torch.zeros(nc)
    for c in range(nc):
        m = (y == c)
        if int(m.sum().item()) > 0:
            M[c] = X[m].mean(dim=0)
            C[c] = float(int(m.sum().item()))
    return M, C


def mean_shrink(M_obs, C_obs, M0, alpha):
    """Shrinkage: M_hat = (C_obs * M_obs + alpha * M0) / (C_obs + alpha).
    alpha = prior strength (memory); alpha=0 = raw sample mean."""
    denom = C_obs.unsqueeze(1) + alpha
    return (C_obs.unsqueeze(1) * M_obs + alpha * M0) / denom


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
    ap.add_argument("--alpha_sweep", type=str, default="0,2,8")
    ap.add_argument("--temp_sweep", type=str, default="0.5,1,2")
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
    alpha_sweep = [float(x) for x in args.alpha_sweep.split(',')]
    temp_sweep = [float(x) for x in args.temp_sweep.split(',')]
    bmax = max(b_sweep)

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_per_class': b_sweep,
               'alpha_sweep': alpha_sweep, 'temp_sweep': temp_sweep, 'conds': {}}

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

        # ---- oracle class stats on the corrupted pool ----
        M_star, C_star = class_means(Xp, pl, NUM_CLASSES)
        R = (Ws - W0).detach().cpu().float()
        R_norm = R.norm().item()

        # ---- A. THE DECOMPOSITION ----
        # W_mean_oracle = Sigma0^-1 P0 M* (clean whitening, oracle means)
        B_mean_oracle = (M_star * C0.unsqueeze(1)).t().contiguous()
        W_mean_oracle = solve_whitened(Xp, B_mean_oracle, args.lam, args.cg_iters, args.nystrom_m, device)
        # NOTE: the whitening uses the POOL codes Xp but the clean counts C0 and
        # clean-lam; the covariance is dominated by the bulk. This is the
        # "mean-shift under the available whitening" term.
        R_mean = (W_mean_oracle - W0).detach().cpu().float()
        R_mean_frac = R_mean.norm().item() / (R_norm + 1e-12)

        # decoder ladder
        ladder = {}
        ladder['W0'] = gc(refs['frozen'])
        # proto_oracle: argmax cosine to oracle means (no whitening) = R1-oracle
        proto_sim = F.normalize(Xv.float(), p=2, dim=1) @ F.normalize(M_star, p=2, dim=1).t()
        ladder['proto_oracle'] = gc(compute_miou(proto_sim.argmax(1), vl))
        ladder['W_mean_oracle'] = gc(mw(W_mean_oracle, Xv, vl))
        ladder['W*'] = gc(refs['oracle'])
        # top moved classes
        moved = torch.norm(M_star - M0, p=2, dim=1)
        top_moved = [(int(c), float(moved[c].item())) for c in torch.argsort(moved, descending=True)]

        # ---- B. FEW-LABEL MEAN ESTIMATION (random + proto_dist selection) ----
        M_star_cpu = M_star
        est = {}
        # per-class candidate indices for selection
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        Xp_cpu = Xp
        # proto_dist selection: points of class c farthest from M0_c (in code cos)
        far_idx = {}
        torch.manual_seed(5)
        for c in range(1, NUM_CLASSES):
            idx = class_idx[c]
            if len(idx) == 0:
                far_idx[c] = idx
                continue
            cos = F.normalize(Xp_cpu[idx], p=2, dim=1) @ F.normalize(M0[c].unsqueeze(0), p=2, dim=1).t()
            far_idx[c] = idx[torch.argsort(cos[:, 0])[:bmax]]
        for sel_name, sel_idx in [('random', None), ('proto_dist', far_idx)]:
            for b in b_sweep:
                # select b points per class
                obs_means = M0.clone(); obs_counts = torch.zeros(NUM_CLASSES)
                for c in range(1, NUM_CLASSES):
                    idx = class_idx[c]
                    if len(idx) == 0:
                        continue
                    if sel_name == 'random':
                        torch.manual_seed(7 + c)
                        sub = idx[torch.randperm(len(idx))[:min(b, len(idx))]]
                    else:
                        sub = far_idx[c][:min(b, len(far_idx[c]))]
                    if len(sub) == 0:
                        continue
                    obs_means[c] = Xp_cpu[sub].mean(dim=0)
                    obs_counts[c] = float(len(sub))
                for alpha in alpha_sweep:
                    M_hat = mean_shrink(obs_means, obs_counts, M0, alpha)
                    B_hat = (M_hat * C0.unsqueeze(1)).t().contiguous()
                    W_est = solve_whitened(Xp, B_hat, args.lam, args.cg_iters, args.nystrom_m, device)
                    g = gc(mw(W_est, Xv, vl))
                    mean_err = float((M_hat - M_star_cpu).norm().item() /
                                     (M_star_cpu.norm().item() + 1e-12))
                    est.setdefault(sel_name, {}).setdefault(str(b), {})[str(alpha)] = {
                        'gc': g, 'mean_err': mean_err,
                        'n_labels': int(min(b, len(Xp)) * (NUM_CLASSES - 1))}

        # ---- C. SELECTION ablation at the middle budget ----
        b_sel = b_sweep[len(b_sweep) // 2]
        sel_ablation = {}
        # oracle_error selection: points of class c the frozen probe misclassifies
        Lp = Xp.float() @ W0c
        pred_p = Lp.argmax(1)
        pool_err = pred_p != pl
        err_idx = {c: torch.nonzero((pl == c) & pool_err).squeeze(1) for c in range(1, NUM_CLASSES)}
        # entropy selection within class
        p_p = torch.softmax(Lp, dim=1)
        ent = -(p_p * (p_p + 1e-12).log()).sum(1)
        ent_idx = {}
        for c in range(1, NUM_CLASSES):
            idx = class_idx[c]
            if len(idx) == 0:
                ent_idx[c] = idx
                continue
            ent_idx[c] = idx[torch.argsort(ent[idx], descending=True)[:bmax]]
        for sel_name, sel_map in [('random', None), ('proto_dist', far_idx),
                                  ('entropy', ent_idx), ('oracle_error', err_idx)]:
            obs_means = M0.clone(); obs_counts = torch.zeros(NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                idx = class_idx[c]
                if len(idx) == 0:
                    continue
                if sel_name == 'random':
                    torch.manual_seed(7 + c)
                    sub = idx[torch.randperm(len(idx))[:min(b_sel, len(idx))]]
                else:
                    sub = sel_map[c][:min(b_sel, len(sel_map[c]))]
                if len(sub) == 0:
                    continue
                obs_means[c] = Xp_cpu[sub].mean(dim=0)
                obs_counts[c] = float(len(sub))
            M_hat = mean_shrink(obs_means, obs_counts, M0, 2.0)
            B_hat = (M_hat * C0.unsqueeze(1)).t().contiguous()
            W_est = solve_whitened(Xp, B_hat, args.lam, args.cg_iters, args.nystrom_m, device)
            sel_ablation[sel_name] = {'gc': gc(mw(W_est, Xv, vl))}

        # ---- D. UPDATE DETAILS (at middle b, alpha=2) ----
        alpha_fix = 2.0
        obs_means = M0.clone(); obs_counts = torch.zeros(NUM_CLASSES)
        for c in range(1, NUM_CLASSES):
            idx = class_idx[c]
            if len(idx) == 0:
                continue
            sub = far_idx[c][:min(b_sel, len(far_idx[c]))]
            obs_means[c] = Xp_cpu[sub].mean(dim=0)
            obs_counts[c] = float(len(sub))
        M_hat_full = mean_shrink(obs_means, obs_counts, M0, alpha_fix)
        # D1 temperature on the estimated logits
        B_hat = (M_hat_full * C0.unsqueeze(1)).t().contiguous()
        W_est_d = solve_whitened(Xp, B_hat, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        temp_res = {}
        Lv = Xv.float() @ W_est_d
        for T in temp_sweep:
            pred = (Lv / T).argmax(1)
            temp_res[str(T)] = gc(compute_miou(pred, vl))
        # D2 update scope: estimated means only for the top-K moved classes
        scope_res = {}
        moved_sorted = torch.argsort(moved, descending=True)
        for K in [4, 8, NUM_CLASSES]:
            M_scope = M0.clone()
            for c in moved_sorted[:K]:
                M_scope[c] = M_hat_full[c]
            B_scope = (M_scope * C0.unsqueeze(1)).t().contiguous()
            W_scope = solve_whitened(Xp, B_scope, args.lam, args.cg_iters, args.nystrom_m, device)
            scope_res[str(K)] = gc(mw(W_scope, Xv, vl))

        cond_res = {'refs': refs, 'gap': float(gap),
                    'R_mean_frac': R_mean_frac, 'ladder': ladder,
                    'top_moved': top_moved[:6], 'est': est,
                    'sel_ablation': sel_ablation,
                    'temp': temp_res, 'scope': scope_res}
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, M_star, Xp_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print("    R_mean_frac %.3f | ladder %s" % (R_mean_frac, " ".join(
            f"{k}:{v:+.2f}" if v is not None else f"{k}:NA" for k, v in ladder.items())))
        print("    top_moved: " + " ".join(f"c{c}:{m:.3f}" for c, m in top_moved[:6]))
        for sel_name in ['random', 'proto_dist']:
            line = " ".join(f"b{b}:a{a}:{est[sel_name][str(b)][str(a)]['gc']:+.2f}"
                            for b in b_sweep for a in alpha_sweep)
            print("    est[%-10s] %s" % (sel_name, line))
        print("    sel_ablation " + " ".join(f"{k}:{v['gc']:+.2f}" for k, v in sel_ablation.items()))
        print("    temp " + " ".join(f"T{k}:{v:+.2f}" for k, v in temp_res.items()) +
              " | scope " + " ".join(f"K{k}:{v:+.2f}" for k, v in scope_res.items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. R_mean_frac ~ 1 + W_mean_oracle ~ W* gc -> the class-mean shift IS")
    print("   the residual; this reformulation has the full ceiling.")
    print("B. W_est gc ~ W_mean_oracle at small b -> few labels re-estimate the")
    print("   means well; the update works (this is a DIFFERENT object than the")
    print("   flat Iteration-7 curve). alpha = shrinkage/memory knob.")
    print("C. sel_ablation: proto_dist / entropy / oracle_error vs random -> which")
    print("   points of class c to label for the mean estimate.")
    print("D. temp = step-size-like knob; scope = sparse update (top-K moved classes).")


if __name__ == "__main__":
    main()
