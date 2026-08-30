"""al_class_stats_fix_diag.py: validate the THREE fix paths from Iteration 1
(class_stats_iters.md), all white-box, before exploring any other method.

Iteration 1 showed the class-statistics decoder decomposition is CORRECT (whitened
class means capture the full residual: W_mean_oracle +0.72 to +1.15 gc) but the
raw few-label mean estimator fails (8 labels can't estimate a 10000-d mean; the
whitening amplifies the noise). This diagnostic validates the three concrete
fix paths independently, each against the same references (W0 frozen, W_mean_
oracle ceiling, W* full oracle):

FIX 1. POOL-PSEUDO-LABEL MEANS + FEW-LABEL BIAS CORRECTION.
  M_pseudo_c = mean of the pool codes the frozen probe pseudo-labels as class c.
  This is LOW-VARIANCE (20k points) but BIASED (pseudo-labels are wrong near
  boundaries). The few labels correct the BIAS -- the mean SHIFT -- which is a
  concentrated low-rank object:
    M_corr_c = M_pseudo_c + delta_c,  delta_c = (M_lab_c - M_pseudo_of_lab_c)
  where M_lab_c = mean of the b labeled points of class c and M_pseudo_of_lab_c
  = the pseudo-mean of those same points (the bias the labels reveal). Sweep
  b x alpha (shrinkage of delta). Report gc and the corrected-mean error.

FIX 2. ESTIMATE ONLY THE MEAN SHIFT (not the absolute mean).
  Instead of M_hat_c -> M*_c, estimate Delta_mu_c = M*_c - M0_c directly from the
  b labels: Delta_hat_c = (M_lab_c - M0_c), then apply the CLEAN whitening with
  STRONG SHRINKAGE on the shift (the shift is concentrated in a few classes, so
  alpha large): M_shift_c = M0_c + alpha_shift * Delta_hat_c. Report gc as a
  function of alpha_shift -- the "step size" on the shift. If a positive alpha
  exists, the shift formulation is the right object (vs the absolute-mean
  failure of Iteration 1).

FIX 3. REGULARIZE THE WHITENING (don't amplify the noise floor).
  The Iteration-1 failure was Sigma^-1 amplifying high-dim noise. Two
  regularizations of the whitening solve, both applied to the SAME estimated
  means (so they isolate the whitening, not the estimator):
    (a) lambda_whitening: increase the ridge on the whitening solve (damp the
        smallest eigenvalues of Sigma).
    (b) spectral truncation: keep only the top-k eigen-directions of the pool
        covariance (project the update onto the high-variance subspace).
  Sweep lambda_whitening x rank-k; report gc and the post-whitening update norm
  (||W_est - W0|| / ||W* - W0|| -- the step size the whitening produces).

Each fix is reported at the same budgets b in {2,4,8} per class where applicable,
plus the mean-estimation error and update-norm diagnostics so we can SEE why a
fix works or fails, not just the final gc.

Usage:
  uv run python robust_diagnostic/al_class_stats_fix_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_class_stats_fix_<label>.json
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


def solve_whitened_ridge(X, B, lam, lam_w, iters, m, device):
    """FIX 3(a): whitening with an EXTRA ridge (regularized Sigma): solve
    (X^T X + (lam + lam_w) I) W = B. Larger lam_w damps the small eigen-directions
    that amplify noise."""
    return solve_whitened(X, B, lam + lam_w, iters, m, device)


def solve_whitened_trunc(X, B, lam, rank_k, iters, m, device):
    """FIX 3(b): spectral truncation -- project B onto the top-rank_k eigen-
    directions of X^T X, whiten only there, and drop the rest. Uses a randomized
    SVD of X to get the top-k right singular vectors (the top-k eigen-directions
    of the covariance)."""
    X = X.to(device); B = B.float().to(device)
    torch.manual_seed(SKETCH_SEED)
    n, d = X.shape
    k = min(rank_k, min(n, d) - 1)
    # randomized SVD: top-k right singular vectors of X (d x k)
    Omega = torch.randn(d, k + 8, device=device)
    Y = X @ Omega
    Q, _ = torch.linalg.qr(Y)
    Bm = Q.t() @ X                        # (k+8) x d
    U, S, Vh = torch.linalg.svd(Bm, full_matrices=False)
    V = Vh[:k].t()                        # d x k right singular vectors
    sig = S[:k]
    # whiten only in the top-k subspace: W = V diag(1/(sig+lam)) V^T B
    W = V @ ((V.t() @ B) / (sig.unsqueeze(1) + lam))
    return W.float()


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
    ap.add_argument("--alpha_sweep", type=str, default="0,2,8,32")
    ap.add_argument("--shift_sweep", type=str, default="0.1,0.3,1.0")
    ap.add_argument("--lam_w_sweep", type=str, default="0.001,0.01,0.1,1.0")
    ap.add_argument("--rank_sweep", type=str, default="128,512,2048")
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
    shift_sweep = [float(x) for x in args.shift_sweep.split(',')]
    lam_w_sweep = [float(x) for x in args.lam_w_sweep.split(',')]
    rank_sweep = [int(x) for x in args.rank_sweep.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_per_class': b_sweep,
               'alpha_sweep': alpha_sweep, 'shift_sweep': shift_sweep,
               'lam_w_sweep': lam_w_sweep, 'rank_sweep': rank_sweep, 'conds': {}}

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

        # ---- references: W0, W_mean_oracle, W* ----
        B_mean_oracle = (M_star * C0.unsqueeze(1)).t().contiguous()
        W_mean_oracle = solve_whitened(Xp, B_mean_oracle, args.lam, args.cg_iters, args.nystrom_m, device)

        # ---- pool pseudo-labels (frozen probe on the pool) ----
        pseudo = (Xp.float() @ W0c).argmax(1)
        M_pseudo, C_pseudo = class_means(Xp, pseudo, NUM_CLASSES)

        # ---- per-class labeled point indices (for the few-label estimators) ----
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        Xp_cpu = Xp

        cond_res = {'refs': refs, 'gap': float(gap),
                    'ladder': {'W0': 0.0,
                               'W_mean_oracle': gc(mw(W_mean_oracle, Xv, vl)),
                               'W*': 1.0},
                    'fix1': {}, 'fix2': {}, 'fix3': {}}

        # ================= FIX 1: pool pseudo-means + few-label bias correction
        # M_corr_c = M_pseudo_c + delta_c; delta_c from the b labeled points:
        #   M_lab_c = mean of labeled points of class c
        #   the bias the labels reveal = M_lab_c - (pseudo-mean at those points'
        #   pseudo-class). We shrink delta_c by alpha_delta.
        for b in b_sweep:
            # select b points per class (random)
            obs_means = M0.clone(); obs_counts = torch.zeros(NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                idx = class_idx[c]
                if len(idx) == 0:
                    continue
                torch.manual_seed(7 + c)
                sub = idx[torch.randperm(len(idx))[:min(b, len(idx))]]
                obs_means[c] = Xp_cpu[sub].mean(dim=0)
                obs_counts[c] = float(len(sub))
            # pseudo-means of the labeled points (their pseudo-class assignment)
            # (the raw bias delta_c is estimated directly from obs_means vs M_pseudo)
            for alpha_d in alpha_sweep:
                M_corr = M_pseudo.clone()
                for c in range(1, NUM_CLASSES):
                    if obs_counts[c] == 0:
                        continue
                    delta_c = obs_means[c] - M_pseudo[c]
                    M_corr[c] = M_pseudo[c] + (alpha_d / (alpha_d + 1.0)) * delta_c
                B_corr = (M_corr * C0.unsqueeze(1)).t().contiguous()
                W_c = solve_whitened(Xp, B_corr, args.lam, args.cg_iters, args.nystrom_m, device)
                g = gc(mw(W_c, Xv, vl))
                mean_err = float((M_corr - M_star).norm().item() / (M_star.norm().item() + 1e-12))
                cond_res['fix1'].setdefault(str(b), {})[str(alpha_d)] = {
                    'gc': g, 'mean_err': mean_err}

        # ================= FIX 2: mean-shift-only estimation with shrinkage
        # M_shift_c = M0_c + s * (M_lab_c - M0_c), s in shift_sweep. The shift is
        # concentrated in a few classes; s < 1 is strong shrinkage on the step.
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
            for s in shift_sweep:
                M_shift = M0.clone()
                for c in range(1, NUM_CLASSES):
                    if obs_counts[c] == 0:
                        continue
                    M_shift[c] = M0[c] + s * (obs_means[c] - M0[c])
                B_shift = (M_shift * C0.unsqueeze(1)).t().contiguous()
                W_s = solve_whitened(Xp, B_shift, args.lam, args.cg_iters, args.nystrom_m, device)
                g = gc(mw(W_s, Xv, vl))
                cond_res['fix2'].setdefault(str(b), {})[str(s)] = {'gc': g}

        # ================= FIX 3: regularized whitening (fix the estimator's
        # noise amplification). Use the SAME estimated absolute means (b=4,
        # alpha=2 -- the Iteration-1 operating point) and vary ONLY the
        # whitening. This isolates the whitening from the estimator.
        b3 = b_sweep[len(b_sweep) // 2]
        obs_means = M0.clone(); obs_counts = torch.zeros(NUM_CLASSES)
        for c in range(1, NUM_CLASSES):
            idx = class_idx[c]
            if len(idx) == 0:
                continue
            torch.manual_seed(7 + c)
            sub = idx[torch.randperm(len(idx))[:min(b3, len(idx))]]
            obs_means[c] = Xp_cpu[sub].mean(dim=0)
            obs_counts[c] = float(len(sub))
        denom = obs_counts.unsqueeze(1) + 2.0
        M_est = (obs_counts.unsqueeze(1) * obs_means + 2.0 * M0) / denom
        B_est = (M_est * C0.unsqueeze(1)).t().contiguous()

        # (a) lambda_whitening sweep
        for lam_w in lam_w_sweep:
            W_r = solve_whitened_ridge(Xp, B_est, args.lam, lam_w, args.cg_iters, args.nystrom_m, device)
            g = gc(mw(W_r, Xv, vl))
            upd = float((W_r - W0).detach().cpu().norm().item() / (R_norm + 1e-12))
            cond_res['fix3'].setdefault('lam_w', {})[str(lam_w)] = {'gc': g, 'update_norm': upd}
        # (b) rank truncation sweep
        for rk in rank_sweep:
            W_t = solve_whitened_trunc(Xp, B_est, args.lam, rk, args.cg_iters, args.nystrom_m, device)
            g = gc(mw(W_t, Xv, vl))
            upd = float((W_t - W0).detach().cpu().norm().item() / (R_norm + 1e-12))
            cond_res['fix3'].setdefault('rank', {})[str(rk)] = {'gc': g, 'update_norm': upd}

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, M_star, Xp_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        l = cond_res['ladder']
        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    ladder W0 {l['W0']:+.2f} W_mean_oracle {l['W_mean_oracle']:+.2f} W* {l['W*']:+.2f}")
        for b in b_sweep:
            f1 = " ".join(f"a{a}:{cond_res['fix1'][str(b)][str(a)]['gc']:+.2f}" for a in alpha_sweep)
            f2 = " ".join(f"s{s}:{cond_res['fix2'][str(b)][str(s)]['gc']:+.2f}" for s in shift_sweep)
            print(f"    b{b}: fix1({f1}) fix2({f2})")
        lw = " ".join(f"lw{l}:{cond_res['fix3']['lam_w'][str(l)]['gc']:+.2f}" for l in lam_w_sweep)
        rk = " ".join(f"rk{r}:{cond_res['fix3']['rank'][str(r)]['gc']:+.2f}" for r in rank_sweep)
        print(f"    fix3: {lw} | {rk}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("All three fixes are compared to the SAME references: W0 (0),")
    print("W_mean_oracle (the Iteration-1 ceiling, oracle means), W* (1).")
    print("FIX 1 (pool pseudo-means + bias correction): gc ~ W_mean_oracle at")
    print("  small b -> the pool gives low-variance means and labels fix the bias.")
    print("  If fix1 ~ W_mean_oracle but W_pseudo (b=0 implied) is far below, the")
    print("  bias correction is the active ingredient.")
    print("FIX 2 (shift-only): positive gc for some s -> the shift formulation is")
    print("  the right object (vs the absolute-mean failure). s = step size on the")
    print("  shift; a peak at s < 1 confirms strong shrinkage is needed.")
    print("FIX 3 (regularized whitening): does a lam_w / rank exist where the same")
    print("  noisy means become positive? If yes, the whitening was the amplifier.")
    print("  update_norm shows the step size the whitening produces -- if huge at")
    print("  small lam_w/rank, the regularization is doing its job.")


if __name__ == "__main__":
    main()
