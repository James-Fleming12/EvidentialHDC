"""al_class_stats_iter7_diag.py: Iteration 7 -- the DECISION-BOUNDARY diagnostic
matrix (revised per the Iteration-6 finding: recoverable statistic !=
decision-relevant statistic).

Iteration 6 established: per-class mean directions survive decoder geometry
(align ~1.0), but the per-class mean shift is 2-117x too large and nearly
orthogonal to the class's OWN residual column. The correct object is the
DECISION object: Delta(w_a - w_b) -- and whether the mean/cov/prior components
of it explain the oracle pairwise correction. This iteration is a diagnostic
matrix, NOT a method.

A. EXACT DECISION DECOMPOSITION (per important pair)
   Delta w*_ab = (w*_a - w*_b) - (w0_a - w0_b)
   = Delta w_mean,ab + Delta w_prior,ab + Delta w_cov,ab
     mean   = Sigma^-1 (p_a Delta_mu_a - p_b Delta_mu_b)
     prior  = Sigma^-1 ((p*_a-p0_a) mu*_a - (p*_b-p0_b) mu*_b)
     cov    = Delta w*_ab - mean - prior   (residual)
   Report per pair: norm of each term, cos(mean,total), cos(cov,total),
   AND the decision-flip count between the mean-only decoder and the oracle.

B. ORACLE PAIRWISE SCALAR test (BEFORE any direction estimator)
   Give the pool-derived pair direction d_ab = Sigma^-1 (v_a - v_b) its best
   oracle scalar: m_ab(x) -> m_ab(x) + gamma_ab d_ab^T x, gamma* from oracle.
   Three outcomes: gamma* helps (direction usable), gamma* ~ 0 (irrelevant),
   helps only some pairs (sparse pairwise AL).

C. BOUNDARY-LOCAL pair directions
   The mean summarizes P(x|y=c) globally, but decisions are set near the a-b
   boundary. Build v_ab,boundary from pool points with |m_ab(x)| < tau, test
   decoder-space alignment with Delta w*_ab. Compare vs the global mean
   direction (the untested, potentially much better pivot). CAVEAT: Iteration 8
   found margin anti-predicts ERRORS (detection), but boundary-local for
   CORRECTION DIRECTION is a different, untested object.

D. COVARIANCE DIAGNOSIS
   R_cov = R - R_mean - R_prior (global); report ||R_cov||/||R|| and
   cos(R_cov, R) -- the direct test of "covariance-dominated" (Iteration 6's
   inference was indirect). Plus the effective rank of Delta_Sigma =
   Sigma* - Sigma0 (is the covariance change low-dimensional?).

E. PRIOR-ONLY ceiling -- ALREADY MEASURED (Iteration 4 Part 3: P_oracle ~ -0.01
   to -0.03 on dglsspp fog, negligible since same-scan P*~P0). Reported here for
   completeness from the stored result, NOT re-run.

F. TTA DECISION-SPACE direction
   Per point, Delta z = W0^T f(aug(x)) - W0^T f(x) (logit displacement under
   bit-flip TTA). Test whether E[Delta z_a - Delta z_b] over the a-b boundary
   predicts Delta(w_a - w_b)*: cos(E[Delta z_a - Delta z_b], Delta w*_ab).
   This is a NEW use of TTA (not variance -- displacement direction).

G. TINY PAIRWISE DECISION CORRECTIONS (logit space, not W)
   z'_a - z'_b = alpha_ab (z_a - z_b) + b_ab. Fit alpha,b on a few labels, and
   the oracle (ceiling). This is the tiny logit-space correction that Iteration
   6's "recoverable != decision-relevant" points to -- before any high-dim W.

The bigger picture: W -> class means -> mean directions -> decoder geometry ->
decision boundaries. The evidence says the useful object is Delta(w_a - w_b)
near the actual decision boundary. A and B tell us whether that is a
pairwise-mean, covariance, prior, or boundary-local problem -- or whether to
abandon parameter reconstruction for tiny logit-space / TTA corrections (G, F).

Usage:
  uv run python robust_diagnostic/al_class_stats_iter7_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_class_stats_iter7_<label>.json
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


def cos_(a, b):
    return float((a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12))


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
    ap.add_argument("--k_classes", type=int, default=5)
    ap.add_argument("--n_pairs", type=int, default=5)
    ap.add_argument("--tau", type=float, default=1.0, help="boundary margin threshold for C")
    ap.add_argument("--gamma_sweep", type=str, default="-0.5,0,0.5,1.0,1.5,2.0")
    ap.add_argument("--b_per_class", type=str, default="4")
    ap.add_argument("--k_eig", type=int, default=512)
    ap.add_argument("--tta_augs", type=int, default=8)
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
    gamma_sweep = [float(x) for x in args.gamma_sweep.split(',')]
    b_sweep = [int(x) for x in args.b_per_class.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'conds': {}}

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
        Ws_c = Ws.detach().cpu()
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        def gc(mi):
            return (mi - refs['frozen']) / gap if gap > 1e-9 else None

        M_star, C_star = class_means(Xp, pl, NUM_CLASSES)
        R = (Ws_c - W0c)
        R_norm = R.norm().item()
        p0c = C0 / (C0.sum() + 1e-12)
        p_sc = C_star / (C_star.sum() + 1e-12)

        Lp = Xp.float() @ W0c
        pseudo = Lp.argmax(1)
        M_hard, C_ph = class_means(Xp, pseudo, NUM_CLASSES)
        V = M_hard - M0
        Delta = M_star - M0

        # ---- important pairs: top pairs by oracle pairwise residual norm ----
        pair_cands = []
        for a in range(1, NUM_CLASSES):
            for b in range(a + 1, NUM_CLASSES):
                if C_star[a] < 20 or C_star[b] < 20:
                    continue
                d = (Ws_c[:, a] - Ws_c[:, b]) - (W0c[:, a] - W0c[:, b])
                pair_cands.append((a, b, float(d.norm().item())))
        pairs = [(a, b) for a, b, _ in sorted(pair_cands, key=lambda x: -x[2])[:args.n_pairs]]

        # ---- A. EXACT DECISION DECOMPOSITION per pair ----
        A = {}
        W_mean_oracle = None
        for (a, b) in pairs:
            dw_ab = (Ws_c[:, a] - Ws_c[:, b]) - (W0c[:, a] - W0c[:, b])
            mean_ab = solve_whitened(Xp, (p0c[a] * Delta[a] - p0c[b] * Delta[b]).unsqueeze(1),
                                     args.lam, args.cg_iters, args.nystrom_m, device).cpu()[:, 0]
            prior_ab = solve_whitened(Xp, ((p_sc[a] - p0c[a]) * M_star[a] -
                                           (p_sc[b] - p0c[b]) * M_star[b]).unsqueeze(1),
                                      args.lam, args.cg_iters, args.nystrom_m, device).cpu()[:, 0]
            cov_ab = dw_ab - mean_ab - prior_ab
            A[f"{a}-{b}"] = {
                'n_mean': float(mean_ab.norm().item()), 'n_prior': float(prior_ab.norm().item()),
                'n_cov': float(cov_ab.norm().item()), 'n_total': float(dw_ab.norm().item()),
                'cos_mean': cos_(mean_ab, dw_ab), 'cos_cov': cos_(cov_ab, dw_ab),
            }

        # mean-only decoder decision agreement vs oracle (the key sanity check)
        M_mean_dec = M_star.clone()
        W_mean_oracle = solve_whitened(Xp, (M_mean_dec * C0.unsqueeze(1)).t().contiguous(),
                                       args.lam, args.cg_iters, args.nystrom_m, device)
        pred_mean = decode(W_mean_oracle, Xv)
        pred_or = decode(Ws, Xv)
        pred_0 = decode(W0, Xv)
        err_mask = (pred_0 != vl)
        dec_agree_all = float((pred_mean == pred_or).float().mean().item())
        dec_agree_err = float((pred_mean[err_mask] == pred_or[err_mask]).float().mean().item()) \
            if int(err_mask.sum().item()) > 0 else None

        # ---- B. ORACLE PAIRWISE SCALAR test ----
        B = {}
        for (a, b) in pairs:
            d_ab = solve_whitened(Xp, (V[a] - V[b]).unsqueeze(1),
                                  args.lam, args.cg_iters, args.nystrom_m, device).cpu()[:, 0]
            # apply gamma to the a-b margin via logits
            out = {}
            for g in gamma_sweep:
                L = (Xv.float() @ W0c).clone()
                L[:, a] += g * (Xv.float() @ d_ab)
                L[:, b] -= g * (Xv.float() @ d_ab)
                out[str(g)] = gc(compute_miou(L.argmax(1), vl))
            B[f"{a}-{b}"] = out

        # ---- C. BOUNDARY-LOCAL pair direction ----
        Lp_full = Xp.float() @ W0c
        C_ = {}
        for (a, b) in pairs:
            m_ab = Lp_full[:, a] - Lp_full[:, b]
            bnd = (m_ab.abs() < args.tau)
            if int(bnd.sum().item()) < 20:
                C_[f"{a}-{b}"] = None
                continue
            # boundary-local pseudo displacement: mean of (x - M0) for points
            # near the boundary, split by which side (a-side vs b-side)
            a_side = bnd & (m_ab > 0)
            b_side = bnd & (m_ab <= 0)
            mu_a_b = Xp[a_side].mean(dim=0) if int(a_side.sum().item()) > 5 else M_hard[a]
            mu_b_b = Xp[b_side].mean(dim=0) if int(b_side.sum().item()) > 5 else M_hard[b]
            v_bnd = (mu_a_b - mu_b_b) - (M0[a] - M0[b])
            d_bnd = solve_whitened(Xp, v_bnd.unsqueeze(1), args.lam, args.cg_iters,
                                   args.nystrom_m, device).cpu()[:, 0]
            dw_ab = (Ws_c[:, a] - Ws_c[:, b]) - (W0c[:, a] - W0c[:, b])
            C_[f"{a}-{b}"] = {'align': cos_(d_bnd, dw_ab), 'n_bnd_pts': int(bnd.sum().item())}

        # ---- D. COVARIANCE DIAGNOSIS (global) ----
        R_mean = solve_whitened(Xp, (Delta * p0c.unsqueeze(1)).t().contiguous(),
                                args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        R_prior = solve_whitened(Xp, ((p_sc - p0c).unsqueeze(1) * M_star).t().contiguous(),
                                 args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        R_cov = R - R_mean - R_prior
        D = {
            'frac_mean': float(R_mean.norm().item() / (R_norm + 1e-12)),
            'frac_prior': float(R_prior.norm().item() / (R_norm + 1e-12)),
            'frac_cov': float(R_cov.norm().item() / (R_norm + 1e-12)),
            'cos_cov_R': cos_(R_cov, R),
            'cos_mean_R': cos_(R_mean, R),
        }

        # ---- F. TTA DECISION-SPACE direction ----
        F = {}
        for (a, b) in pairs:
            # average logit displacement over boundary points
            m_ab = Lp_full[:, a] - Lp_full[:, b]
            bnd = torch.nonzero(m_ab.abs() < args.tau).squeeze(1)
            if len(bnd) < 20:
                F[f"{a}-{b}"] = None
                continue
            draws = []
            for _ in range(args.tta_augs):
                torch.manual_seed(100 + _)
                flip = torch.rand(len(bnd), Xp.shape[1]) < 0.02
                Xa = torch.where(flip, -Xp[bnd], Xp[bnd])
                draws.append(Xa.float() @ W0c)
            avg_shift = (torch.stack(draws).mean(0) - Xp[bnd].float() @ W0c).mean(0)  # C-dim
            dz_ab = avg_shift[a] - avg_shift[b]
            dw_ab = (Ws_c[:, a] - Ws_c[:, b]) - (W0c[:, a] - W0c[:, b])
            F[f"{a}-{b}"] = {'align': cos_(dz_ab, dw_ab)}

        # ---- G. TINY PAIRWISE LOGIT CORRECTION ----
        # z'_a - z'_b = alpha (z_a - z_b) + beta. Fit on few labels (b per class)
        # and on oracle (ceiling).
        b = b_sweep[0]
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        lab_idx = []
        for c in range(1, NUM_CLASSES):
            idx = class_idx[c]
            if len(idx) == 0:
                continue
            torch.manual_seed(7 + c)
            lab_idx.append(idx[torch.randperm(len(idx))[:min(b, len(idx))]])
        lab_idx = torch.cat(lab_idx)
        Xl = Xp[lab_idx].float(); yl = pl[lab_idx]
        L_lab = Xl @ W0c
        G = {}
        for (a, b) in pairs:
            m = L_lab[:, a] - L_lab[:, b]
            tgt = (yl == a).float() - (yl == b).float()
            # oracle alpha/beta: fit on ALL pool points of the pair
            m_all = (Lp_full[:, a] - Lp_full[:, b])
            t_all = ((pl == a).float() - (pl == b).float())
            A_mat = torch.stack([m_all, torch.ones_like(m_all)], dim=1)
            sol = torch.linalg.lstsq(A_mat.double(), t_all.double().unsqueeze(1)).solution
            alpha_o = float(sol[0].item()); beta_o = float(sol[1].item())
            # apply on val
            mv = (Xv.float() @ W0c)[:, a] - (Xv.float() @ W0c)[:, b]
            corr = alpha_o * mv + beta_o
            pred = (Xv.float() @ W0c).clone()
            pred[:, a] += corr
            pred[:, b] -= corr
            gc_o = gc(compute_miou(pred.argmax(1), vl))
            # few-label fit
            A_lab = torch.stack([m, torch.ones_like(m)], dim=1)
            if A_lab.shape[0] >= 2:
                sol_l = torch.linalg.lstsq(A_lab.double(), tgt.double().unsqueeze(1)).solution
                alpha_l = float(sol_l[0].item()); beta_l = float(sol_l[1].item())
                corr_l = alpha_l * mv + beta_l
                pred_l = (Xv.float() @ W0c).clone()
                pred_l[:, a] += corr_l
                pred_l[:, b] -= corr_l
                gc_l = gc(compute_miou(pred_l.argmax(1), vl))
            else:
                gc_l = None
            G[f"{a}-{b}"] = {'oracle': gc_o, 'fewlabel': gc_l,
                             'alpha_oracle': alpha_o, 'beta_oracle': beta_o}

        cond_res = {'refs': refs, 'gap': float(gap),
                    'pairs': [f"{a}-{b}" for a, b in pairs],
                    'A_decomp': A,
                    'dec_agree_mean_oracle': dec_agree_all,
                    'dec_agree_mean_oracle_errs': dec_agree_err,
                    'B_pairwise_scalar': B,
                    'C_boundary': C_,
                    'D_cov': D,
                    'F_tta_dir': F,
                    'G_pairwise_logit': G}
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, M_star
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    pairs: {[f'{a}-{b}' for a,b in pairs]}")
        print("    A decomp: " + " ".join(
            f"{k}:mn{v['n_mean']:.1f}pr{v['n_prior']:.1f}cv{v['n_cov']:.1f}tot{v['n_total']:.1f}"
            f"cm{v['cos_mean']:+.2f}cc{v['cos_cov']:+.2f}" for k, v in A.items()))
        print(f"    dec_agree mean-oracle: all {dec_agree_all:.3f} errs {dec_agree_err}")
        for k, v in B.items():
            best = max(v.items(), key=lambda kv: kv[1])
            print("    B scalar %s: best g%s %+.2f | " % (k, best[0], best[1]) +
                  " ".join(f"g{g}:{x:+.2f}" for g, x in list(v.items())[:3]))
        print("    C boundary: " + " ".join(f"{k}:{v['align']:+.2f}" if v else f"{k}:NA"
                                            for k, v in C_.items()))
        print("    D cov: " + " ".join(f"{k}:{v:.2f}" for k, v in D.items()))
        print("    F tta-dir: " + " ".join(f"{k}:{v['align']:+.2f}" if v else f"{k}:NA"
                                           for k, v in F.items()))
        print("    G pairwise-logit: " + " ".join(
            f"{k}:or{v['oracle']:+.2f}fl{v['fewlabel']:+.2f}" if v['fewlabel'] is not None
            else f"{k}:or{v['oracle']:+.2f}" for k, v in G.items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. Which term (mean/prior/cov) dominates the PAIRWISE decision residual?")
    print("   cos_mean / cos_cov tell which component aligns with the true pair shift.")
    print("   dec_agree mean-oracle: if ~1.0 (esp. on errors), the mean-only decoder")
    print("   ALREADY matches oracle decisions -> covariance unnecessary.")
    print("B. Oracle pairwise scalar: does gamma* on the pool pair direction help?")
    print("   If yes, the pairwise direction is usable; estimate gamma from labels.")
    print("C. Boundary-local direction alignment vs the global mean direction.")
    print("D. Direct ||R_cov||/||R|| and cos(R_cov,R) -- the Iteration-6 'covariance")
    print("   dominated' claim, now measured directly.")
    print("F. Does TTA logit displacement predict the oracle decision movement?")
    print("G. Tiny pairwise logit correction (alpha,beta): oracle vs few-label.")


if __name__ == "__main__":
    main()
