"""al_class_stats_iter3_diag.py: Iteration 3 -- the three-arm DECISIVE test for
the class-statistics reformulation (class_stats_iters.md).

Iteration 2 closed the raw few-label mean estimator (35x whitening overstep) but
NOT the class-statistics program. The decomposition is validated (W_mean_oracle
+0.72 to +1.15 gc). The open question: can the labels estimate a LOW-DIMENSIONAL
correction on top of pool-derived class statistics, instead of a 10000-d mean?

Three arms answer three hypotheses:

ARM A -- pure pool baseline (the label-free ceiling of the construction):
  W_pseudo = Sigma0^-1 M_pseudo^T P0 with three pseudo-label estimators:
    hard   M_c = mean of pool codes with argmax(W0 x) = c
    soft   M_c = sum_i p_i(c) x_i / sum_i p_i(c)
    tta    soft mean with p averaged over bit-flip augmentations
  If soft/TTA pushes +0.05-0.11 toward +0.72-1.15, the label-free route is the
  strong one and labels may be unnecessary.

ARM B -- pool basis + scalar/class calibration (labels estimate alpha in R^K):
  Identify K suspicious classes (pool evidence: ||M_pseudo_c - M0_c||). For each,
  v_c = M_pseudo_c - M0_c is a pool-derived shift DIRECTION. The b labels of
  class c estimate only the scalar gamma_c = <M_lab_c - M0_c, v_c>/||v_c||^2
  ("how much should class c move along v_c"). Then
      M_corr_c = M0_c + gamma_c v_c
  No 10000-d labeled mean. Report gamma_c per class and gc.

ARM C -- pseudo-label confusion correction (labels estimate a small Q matrix):
  M_tilde ~ Q M* with Q_cj = P(pseudo = j | true = c). The b labels of true
  class c directly estimate row c of Q (fraction of their pseudo labels). Then
      M_corr_c = sum_j Q_cj M_tilde_j
  restricted to the K x K suspicious-class block. Labels estimate C x C (or K x K)
  scalars, not a 10000-d vector.

CORRUPTION CONTROL D_rho -- how precisely must the mean direction be estimated?
  D_oracle = W_mean_oracle - W0. Corrupt: D_rho = sqrt(1-rho^2) D + rho N with N
  random noise. If gc survives large rho, the estimator problem is solvable; if
  it collapses at small rho, the route is brittle. Translates the 35x-noise
  observation into a required estimation precision.

DIAGNOSTICS -- the two missing measurements from Iteration 2:
  whitened mean error   ||Sigma^-1 (M_hat - M*)|| / ||Sigma^-1 M*||
  residual-relevant err <Sigma^-1 (M_hat - M0), R> / ||R||^2
  per class, for each estimator -- WHICH classes account for the +0.05-0.11 to
  +0.72-1.15 gap (probably not all 17).

Usage:
  uv run python robust_diagnostic/al_class_stats_iter3_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_class_stats_iter3_<label>.json
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


def soft_means(X, Pmat, nc):
    """Soft pseudo-means: M_c = sum_i p_i(c) x_i / sum_i p_i(c)."""
    M = torch.zeros(nc, X.shape[1])
    for c in range(nc):
        w = Pmat[:, c]
        s = w.sum().item()
        if s > 1e-9:
            M[c] = (X * w.unsqueeze(1)).sum(dim=0) / s
    return M


def whitened_mean_err(S, M_hat, M_star, lam):
    """||Sigma^-1 (M_hat - M*)|| / ||Sigma^-1 M*|| with Sigma = S^T S + lam I
    approximated via the pool codes S. Uses the truncated-ridge inverse."""
    # (S^T S + lam I)^-1 applied to a d x C difference: solve in the row space
    D = (M_hat - M_star).t().contiguous()          # d x C
    # use CG via solve_whitened's inner solve on S directly
    Wd = solve_whitened(S, D, lam, 8, 1000, 'cuda' if torch.cuda.is_available() else 'cpu')
    Ws = solve_whitened(S, M_star.t().contiguous(), lam, 8, 1000,
                        'cuda' if torch.cuda.is_available() else 'cpu')
    return float(Wd.norm().item() / (Ws.norm().item() + 1e-12))


def residual_relevant_err(S, M_hat, M0, R, lam, device):
    """<Sigma^-1 (M_hat - M0), R> / ||R||^2. The fraction of the update along R."""
    D = (M_hat - M0).t().contiguous()
    Wd = solve_whitened(S, D, lam, 8, 1000, device).cpu()
    return float((Wd * R).sum().item() / (R.norm().item() ** 2 + 1e-12))


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
    ap.add_argument("--k_sweep", type=str, default="3,5,8")
    ap.add_argument("--rho_sweep", type=str, default="0,0.1,0.3,0.5,0.8,1.0")
    ap.add_argument("--tta_augs", type=int, default=5)
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
    k_sweep = [int(x) for x in args.k_sweep.split(',')]
    rho_sweep = [float(x) for x in args.rho_sweep.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_per_class': b_sweep,
               'k_sweep': k_sweep, 'rho_sweep': rho_sweep, 'conds': {}}

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

        # references
        B_mean_oracle = (M_star * C0.unsqueeze(1)).t().contiguous()
        W_mean_oracle = solve_whitened(Xp, B_mean_oracle, args.lam, args.cg_iters, args.nystrom_m, device)

        # frozen probe softmax on the pool
        Lp = Xp.float() @ W0c
        p0 = torch.softmax(Lp, dim=1)
        pseudo_hard = Lp.argmax(1)

        # ---- ARM A: label-free pseudo-means (hard / soft / TTA) ----
        M_hard, C_ph = class_means(Xp, pseudo_hard, NUM_CLASSES)
        M_soft = soft_means(Xp, p0, NUM_CLASSES)
        # TTA-averaged soft means
        draws = []
        Xe = Xp
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xe) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xe, Xe) @ W0c, dim=1))
        p_tta = torch.stack(draws).mean(dim=0)
        M_tta = soft_means(Xp, p_tta, NUM_CLASSES)

        arm_a = {}
        for name, Mh in [('hard', M_hard), ('soft', M_soft), ('tta', M_tta)]:
            B_h = (Mh * C0.unsqueeze(1)).t().contiguous()
            W_h = solve_whitened(Xp, B_h, args.lam, args.cg_iters, args.nystrom_m, device)
            arm_a[name] = {'gc': gc(mw(W_h, Xv, vl)),
                           'whitened_err': whitened_mean_err(Xp, Mh, M_star, args.lam),
                           'resid_rel': residual_relevant_err(Xp, Mh, M0, R, args.lam, device)}

        # ---- suspicious classes (pool evidence: pseudo-vs-clean shift) ----
        shift_norm = torch.norm(M_hard - M0, p=2, dim=1)
        suspicious = [int(c) for c in torch.argsort(shift_norm, descending=True) if c != 0]

        # per-class labeled indices
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        Xp_cpu = Xp

        # ---- ARM B: pool basis + scalar gamma_c ----
        arm_b = {}
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
            gammas = {}
            for K in k_sweep:
                M_corr = M0.clone()
                for c in suspicious[:K]:          # top-K shifted real classes
                    v_c = M_hard[c] - M0[c]
                    vn = v_c.norm().item() ** 2 + 1e-12
                    # labels estimate the SCALAR gamma_c = how much to move along v_c
                    gam = float(((obs_means[c] - M0[c]) * v_c).sum().item() / vn) \
                        if obs_counts[c] > 0 else 0.0
                    gammas[c] = gam
                    M_corr[c] = M0[c] + gam * v_c
                B_c = (M_corr * C0.unsqueeze(1)).t().contiguous()
                W_c = solve_whitened(Xp, B_c, args.lam, args.cg_iters, args.nystrom_m, device)
                arm_b.setdefault(str(b), {})[str(K)] = {'gc': gc(mw(W_c, Xv, vl)),
                                                        'gamma': {k: round(v, 2) for k, v in
                                                                  list(gammas.items())[:K]}}

        # ---- ARM C: pseudo-label confusion correction (few labels estimate Q) ----
        arm_c = {}
        for b in b_sweep:
            # Q_cj = P(pseudo = j | true = c), estimated from the b labeled points
            # of true class c. Restrict to the K x K suspicious block.
            for K in k_sweep:
                sus = suspicious[:K]
                M_corr = M_hard.clone()
                for c in sus:
                    idx = class_idx[c]
                    if len(idx) == 0:
                        continue
                    torch.manual_seed(7 + c)
                    sub = idx[torch.randperm(len(idx))[:min(b, len(idx))]]
                    plab = pseudo_hard[sub]             # pseudo labels of true-c points
                    # row c of Q: fraction of these points with each pseudo class
                    row = torch.zeros(NUM_CLASSES)
                    for j in range(NUM_CLASSES):
                        row[j] = float((plab == j).float().mean().item())
                    # correct: M*_c ~= sum_j Q_cj M_tilde_j (full row, sums to 1)
                    M_corr[c] = sum(row[j] * M_hard[j] for j in range(NUM_CLASSES) if j != 0)
                B_c = (M_corr * C0.unsqueeze(1)).t().contiguous()
                W_c = solve_whitened(Xp, B_c, args.lam, args.cg_iters, args.nystrom_m, device)
                arm_c.setdefault(str(b), {})[str(K)] = {'gc': gc(mw(W_c, Xv, vl))}

        # ---- CORRUPTION CONTROL: how precise must the mean direction be? ----
        D_oracle = (W_mean_oracle - W0).detach().cpu().float()
        W0_cpu = W0.detach().cpu()
        dn = D_oracle.norm().item() + 1e-12
        torch.manual_seed(21)
        N = torch.randn_like(D_oracle)
        N = N / N.norm().item() * dn
        corr = {}
        for rho in rho_sweep:
            D_r = (1 - rho ** 2) ** 0.5 * D_oracle + rho * N
            W_r = W0_cpu + D_r
            corr[str(rho)] = gc(mw(W_r, Xv, vl))

        # ---- per-class diagnostics: which classes drive the gap? ----
        per_class = {}
        for c in range(1, NUM_CLASSES):
            if C_star[c] < 10:
                continue
            per_class[str(c)] = {
                'shift': float(shift_norm[c].item()),
                'hard_err': float((M_hard[c] - M_star[c]).norm().item() /
                                  (M_star[c].norm().item() + 1e-12))}

        cond_res = {'refs': refs, 'gap': float(gap),
                    'ladder': {'W0': 0.0,
                               'W_mean_oracle': gc(mw(W_mean_oracle, Xv, vl)),
                               'W*': 1.0},
                    'arm_a': arm_a, 'arm_b': arm_b, 'arm_c': arm_c,
                    'corruption': corr, 'per_class': per_class,
                    'suspicious': suspicious[:8]}
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, M_star, Xp_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        l = cond_res['ladder']
        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    ladder W0 {l['W0']:+.2f} W_mean_oracle {l['W_mean_oracle']:+.2f} W* {l['W*']:+.2f}")
        print("    ARM A: " + " ".join(
            f"{k}:{v['gc']:+.2f}(we {v['whitened_err']:.2f},rr {v['resid_rel']:+.2f})" for k, v in arm_a.items()))
        for b in b_sweep:
            bb = " ".join(f"K{k}:{arm_b[str(b)][str(k)]['gc']:+.2f}" for k in k_sweep)
            cc = " ".join(f"K{k}:{arm_c[str(b)][str(k)]['gc']:+.2f}" for k in k_sweep)
            print(f"    b{b}: ARM B [{bb}] ARM C [{cc}]")
        print("    corruption: " + " ".join(f"r{k}:{v:+.2f}" for k, v in corr.items()))
        print(f"    suspicious: {cond_res['suspicious']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Arm A > B,C         -> the label-free route dominates (labels not needed).")
    print("B or C > A          -> labels ARE useful for a LOW-DIMENSIONAL correction")
    print("                       (the reformulation's actual claim).")
    print("Arm B gamma values  -> 'how much class c should move along v_c', from labels.")
    print("corruption rho      -> if gc survives large rho, the estimator problem is")
    print("                       solvable; if it collapses at small rho, brittle.")
    print("whitened_err/resid_rel -> is the mean error in suppressed or in-R directions?")


if __name__ == "__main__":
    main()
