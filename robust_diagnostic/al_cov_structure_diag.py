"""al_cov_structure_diag.py: the 8A gate -- is the decision-relevant covariance
structured/predictable, plus the implementation diagnostics for the estimator
list. DGLSS++ only, fog/crosstalk/snow.

The propagation method (+0.045/+0.095 mIoU) is real but its ceiling is set by
geometry (Iteration 13: even correct labels give a mean ~1.4x the oracle's norm
away from M*; the classes are tight but centroids nearly coincide). The central
open question, per the user's 8A plan: is the covariance correction structured
or predictable, or is it genuinely inaccessible? This resolves that, and
measures what the estimator implementations need.

A. COVARIANCE STRUCTURE (is the decision-relevant covariance low-rank / shared?)
   A1  Effective rank of the covariance residual and rank-r oracle gc: W0 +
       rank-r(R_cov), r in {1,2,4,8,16,32,64}. If rank-8 ~ rank-64 gc, the
       covariance correction is low-rank (exploitable). Also report the
       participation ratio (effective rank).
   A2  Cross-condition covariance basis: top-8 directions of R_cov for fog /
       crosstalk / snow, principal angles between them. If the same few modes
       recur, a universal adaptation basis + label scalars is possible.
   A3  Per-class covariance vs global: is R_cov concentrated in a few classes
       or spread? (The mean error was class-specific; is the covariance too?)

B. DELTA-Z* PREDICTABILITY (is the oracle correction predictable from label-free
   features?)
   For each point, delta_z*(x) = z*(x) - z0(x) (the oracle logit correction).
   Fit tiny regressors from label-free features:
       {margin, entropy, tta_var, proto_dist, density, pseudo-class, top-pair}
   to predict delta_z* (per-class) and the oracle PAIRWISE margin correction
   delta(m_ab) for the top pairs. Report:
       - R^2 / rank correlation of delta_z* on the features
       - classification gain from a feature-conditioned logit correction
         (fit on the pool, apply on val) vs frozen.
   If delta_z* is predictable, the correction is low-dimensional and labels can
   calibrate a tiny model. If not, covariance is genuinely inaccessible without
   additional information.

C. NULL CONTROL (is the small positive signal real?)
   For the best oracle pair, fit the affine logit correction (alpha, beta) on
   REAL labels vs SHUFFLED labels, bootstrapped (5 draws). Report real gain vs
   shuffled gain (mean +- std). If real ~ shuffled, the +0.03-0.06 signal is
   noise.

D. IMPLEMENTATION DIAGNOSTICS (what the estimator list needs)
   D1  High-value decision floors: per-pair oracle-flip count (where frozen and
       oracle disagree) -- the decision floors that carry the recoverable gain,
       and where labels should be spent.
   D2  Per-class optimal shrinkage: for each class, the shrinkage a_c toward the
       pool-stable pseudo-mean that minimizes the WHITENED error. Report the
       range across classes: if a_c is consistent, one global a works; if it
       varies, per-class is needed.
   D3  Non-mean estimator: the DENSITY-CORE mean (mean of the densest half of
       the class) vs the plain mean -- does a core mean have lower whitened
       error? (Tests whether the mean is the wrong summary.)
   D4  Whitening gain bound: the eigenvalues of the pool covariance (via the
       top-k eigenbasis), and where the propagated-mean error energy sits
       across gain bins. If the error concentrates in high-gain directions, a
       per-direction gain bound could help (the untested per-direction version
       of fractional whitening).

Decisive reads:
   A1 rank-8 ~ rank-64 gc  -> covariance is low-rank (pool basis + label
                              scalars viable)
   A2 basis recurs across conditions -> universal adaptation basis possible
   B delta_z* predictable (R^2 > 0.1, gain > 0) -> tiny decision-correction
      mechanism; the covariance is accessible through features
   C shuffled ~ real        -> the small positive is noise (do not trust it)
   D2 a_c consistent        -> one global shrinkage works
   D3 core mean < plain mean whitened err -> the mean is the wrong summary;
                              build a core/density estimator

Usage:
  uv run python robust_diagnostic/al_cov_structure_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --conds fog,crosstalk,snow \
    --out robust_diagnostic/logs/al_cov_structure_dglsspp.json
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


def topk_eigbasis(X, k, device):
    """Randomized SVD -> top-k right singular vectors (d x k) + singular vals."""
    X = X.to(device)
    n, d = X.shape
    k = min(k, min(n, d) - 1)
    torch.manual_seed(SKETCH_SEED)
    Omega = torch.randn(d, k + 8, device=device)
    Y = X @ Omega
    Q, _ = torch.linalg.qr(Y)
    Bm = Q.t() @ X
    U, S, Vh = torch.linalg.svd(Bm, full_matrices=False)
    Qe = Vh[:k].t().contiguous()
    sig = S[:k].clamp(min=1e-8)
    return Qe.cpu(), sig.cpu()


def whitened_err(X, lam, iters, m, device, v, ref):
    Bv = (v - ref).unsqueeze(1)
    Br = ref.unsqueeze(1)
    Wv = solve_whitened(X, Bv, lam, iters, m, device).cpu()[:, 0]
    Wr = solve_whitened(X, Br, lam, iters, m, device).cpu()[:, 0]
    return float(Wv.norm().item() / (Wr.norm().item() + 1e-12))


def cos_(a, b):
    return float((a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync(); return time.time()


def toc(t0):
    sync(); return time.time() - t0


def fit_affine(m, t):
    A = torch.stack([m, torch.ones_like(m)], dim=1)
    sol = torch.linalg.lstsq(A.double(), t.double().unsqueeze(1)).solution
    return float(sol[0].item()), float(sol[1].item())


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
    ap.add_argument("--rank_sweep", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--k_eig", type=int, default=512)
    ap.add_argument("--n_top_pairs", type=int, default=5)
    ap.add_argument("--n_null", type=int, default=5)
    ap.add_argument("--b_norm", type=int, default=8, help="anchors/class for the propagated mean")
    ap.add_argument("--cond_pairs", type=str, default="fog,crosstalk", help="extra conditions for A2 basis")
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    rank_sweep = [int(x) for x in args.rank_sweep.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'rank_sweep': rank_sweep,
               'conds': {}}

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
        p0c = C0 / (C0.sum() + 1e-12)
        p_sc = C_star / (C_star.sum() + 1e-12)

        pf = F.normalize(pool_f.float(), p=2, dim=1)
        Lp = Xp.float() @ W0c
        Lv = Xv.float() @ W0c
        sm = torch.softmax(Lp, dim=1)
        entropy = -(sm * (sm + 1e-12).log()).sum(1)
        conf = sm.max(1).values
        top2 = torch.topk(Lp, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        pred = Lp.argmax(1)

        # ---- A. COVARIANCE STRUCTURE ----
        W_mean_oracle = solve_whitened(Xp, (M_star * C0.unsqueeze(1)).t().contiguous(),
                                       args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        R_mean = W_mean_oracle - W0c
        R_prior = solve_whitened(Xp, ((p_sc - p0c).unsqueeze(1) * M_star).t().contiguous(),
                                 args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        R_cov = Ws_c - W0c - R_mean - R_prior      # independent covariance residual
        # participation ratio (effective rank)
        sv = torch.linalg.svdvals(R_cov.double())
        pr = (sv.sum().item() ** 2) / ((sv ** 2).sum().item() + 1e-12)
        # rank-r oracle gc: project R_cov (d x C) onto its top-r LEFT singular
        # vectors (the 10000-d code-space directions), W0 + rank-r(R_cov).
        rank_gc = {}
        U_s, _, _ = torch.linalg.svd(R_cov.double(), full_matrices=False)
        for r in rank_sweep:
            rr = min(r, R_cov.shape[1])
            U_r = U_s[:, :rr].float()                       # d x r (code-space dirs)
            W_r = W0c + (U_r @ (U_r.t() @ R_cov.float()))   # project R_cov to rank r
            rank_gc[str(r)] = gc(mw(W_r, Xv, vl))

        # ---- B. DELTA-Z* PREDICTABILITY ----
        # features for a regressor (label-free)
        # nearest-other-class margin, density (128-d), pseudo class
        Lp_full = Lp
        # fit linear regressor on the pool to predict delta_z per class
        # features: margin, entropy, conf, top-pair margin difference
        Feat = torch.stack([margin, entropy, conf], dim=1)   # n x 3
        mu = Feat.mean(0); sd = Feat.std(0).clamp(min=1e-8)
        Feat_s = (Feat - mu) / sd
        # oracle delta_z on the pool
        z_or_p = Xp.float() @ Ws_c
        dZ_pool = z_or_p - Lp_full                            # n x C
        r2s = {}
        for c in range(1, NUM_CLASSES):
            y = dZ_pool[:, c]
            A = torch.cat([Feat_s, torch.ones(len(Feat_s), 1)], dim=1)
            sol = torch.linalg.lstsq(A.double(), y.double().unsqueeze(1)).solution
            pred_dz = (A.double() @ sol).squeeze(1)
            ss_res = ((y.double() - pred_dz) ** 2).sum().item()
            ss_tot = ((y.double() - y.double().mean()) ** 2).sum().item()
            r2s[str(c)] = 1.0 - ss_res / (ss_tot + 1e-12)
        mean_r2 = sum(v for v in r2s.values()) / len(r2s) if r2s else None
        # classification gain from a feature-conditioned correction:
        # fit the delta_z regressor on the pool, apply to val with the SAME
        # feature standardization.
        Feat_v = torch.stack([torch.topk(Lv, 2, dim=1).values[:, 0] - torch.topk(Lv, 2, dim=1).values[:, 1],
                              -(torch.softmax(Lv, 1) * (torch.softmax(Lv, 1) + 1e-12).log()).sum(1),
                              torch.softmax(Lv, 1).max(1).values], dim=1)
        Feat_vs = (Feat_v - mu) / sd
        Lv_c = Lv.clone()
        for c in range(1, NUM_CLASSES):
            y = dZ_pool[:, c]
            A = torch.cat([Feat_s, torch.ones(len(Feat_s), 1)], dim=1)
            sol = torch.linalg.lstsq(A.double(), y.double().unsqueeze(1)).solution
            Av = torch.cat([Feat_vs, torch.ones(len(Feat_vs), 1)], dim=1)
            Lv_c[:, c] += (Av.double() @ sol).squeeze(1)
        gc_pred = gc(compute_miou(Lv_c.argmax(1), vl))

        # ---- C. NULL CONTROL (best oracle pair, real vs shuffled affine) ----
        pair_cands = []
        for a in range(1, NUM_CLASSES):
            for b in range(a + 1, NUM_CLASSES):
                if C_star[a] < 20 or C_star[b] < 20:
                    continue
                d = (Ws_c[:, a] - Ws_c[:, b]) - (W0c[:, a] - W0c[:, b])
                pair_cands.append((a, b, float(d.norm().item())))
        pairs = [(a, b) for a, b, _ in sorted(pair_cands, key=lambda x: -x[2])[:args.n_top_pairs]]
        null = {}
        for (a, b) in pairs[:1]:                      # best pair
            mask = (pl == a) | (pl == b)
            m_p = Lp_full[mask, a] - Lp_full[mask, b]
            t_p = (pl[mask] == a).float() * 1.0 + (pl[mask] == b).float() * (-1.0)
            a_r, b_r = fit_affine(m_p, t_p)
            pred_n = Lv.clone()
            corr = a_r * (Lv[:, a] - Lv[:, b]) + b_r
            pred_n[:, a] += corr; pred_n[:, b] -= corr
            gc_real = gc(compute_miou(pred_n.argmax(1), vl))
            gc_shufs = []
            for s in range(args.n_null):
                torch.manual_seed(50 + s)
                t_s = t_p[torch.randperm(len(t_p))]
                a_s, b_s = fit_affine(m_p, t_s)
                pred_s = Lv.clone()
                corr_s = a_s * (Lv[:, a] - Lv[:, b]) + b_s
                pred_s[:, a] += corr_s; pred_s[:, b] -= corr_s
                gc_shufs.append(gc(compute_miou(pred_s.argmax(1), vl)))
            null[f"{a}-{b}"] = {'real': gc_real,
                                'shuffled_mean': sum(gc_shufs) / len(gc_shufs),
                                'shuffled_std': (lambda x: float(torch.tensor(x).std().item()))(gc_shufs),
                                'shuffled': gc_shufs}

        # ---- D. IMPLEMENTATION DIAGNOSTICS ----
        # D1 decision floors: per-pair oracle flip count
        pred_or = (Xv.float() @ Ws_c).argmax(1)
        pred_0 = (Xv.float() @ W0c).argmax(1)
        flips = (pred_0 != pred_or)
        d1 = {}
        for (a, b) in pairs:
            m_v = Lv[:, a] - Lv[:, b]
            near = m_v.abs() < 1.0
            d1[f"{a}-{b}"] = {'flips_near_boundary': int((flips & near).sum().item()),
                              'flips_all': int(flips.sum().item())}
        # D2 per-class optimal shrinkage (toward pool-stable pseudo-mean)
        torch.manual_seed(9)
        anc = []
        for c in range(1, NUM_CLASSES):
            idx = torch.nonzero(pl == c).squeeze(1)
            if len(idx) == 0:
                continue
            nb = min(args.b_norm, len(idx))
            anc.append(idx[torch.randperm(len(idx))[:nb]])
        anc = torch.cat(anc)
        anc_f = pf[anc]; anc_lab = pl[anc]
        nn = (pf @ anc_f.t()).argmax(1)
        prop_lab = anc_lab[nn]
        M_prop, C_prop = class_means(Xp, prop_lab, NUM_CLASSES)
        M_pseudo, _ = class_means(Xp, pred, NUM_CLASSES)
        d2 = {}
        for c in range(1, NUM_CLASSES):
            if C_star[c] < 50:
                continue
            best_a, best_w = 0.0, None
            for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
                M_sh = (1 - a) * M_prop[c] + a * M_pseudo[c]
                w = whitened_err(Xp, args.lam, args.cg_iters, args.nystrom_m, device, M_sh, M_star[c])
                if best_w is None or w < best_w:
                    best_w, best_a = w, a
            d2[str(c)] = {'best_a': best_a, 'best_w': best_w}
        # D3 non-mean estimator: density-core mean
        d3 = {}
        for c in range(1, NUM_CLASSES):
            if C_star[c] < 100:
                continue
            idx = torch.nonzero(pl == c).squeeze(1)
            ctr = pf[idx].mean(0); ctr = ctr / (ctr.norm() + 1e-12)
            sim = (pf[idx] * ctr).sum(1)
            core = idx[torch.argsort(sim, descending=True)[:len(idx) // 2]]
            M_core = Xp[core].float().mean(0)
            d3[str(c)] = {'Werr_core': whitened_err(Xp, args.lam, args.cg_iters, args.nystrom_m, device,
                                                    M_core, M_star[c]),
                          'Werr_plain': whitened_err(Xp, args.lam, args.cg_iters, args.nystrom_m, device,
                                                     M_prop[c], M_star[c])}
        # D4 whitening gain + propagated-mean error energy across gain bins
        Qe, sig = topk_eigbasis(Xp, args.k_eig, device)
        lamb = (sig ** 2 + args.lam)
        # error vector per class, its projection energy by gain bin
        err_energy = {}
        gains = lamb
        med = float(gains.median().item())
        for c in range(1, NUM_CLASSES):
            if C_star[c] < 100:
                continue
            e = (M_prop[c] - M_star[c]).double()
            proj = (Qe.double().t() @ e) ** 2            # k
            hi = proj[gains > med].sum().item()
            lo = proj[gains <= med].sum().item()
            err_energy[str(c)] = {'frac_high_gain': hi / (hi + lo + 1e-12)}
        d4 = {'gain_median': med, 'per_class_frac_high_gain': err_energy}

        cond_res = {'refs': refs, 'gap': float(gap),
                    'A': {'eff_rank': pr, 'rank_gc': rank_gc},
                    'B': {'per_class_r2': r2s, 'mean_r2': mean_r2,
                          'gc_feature_conditioned': gc_pred},
                    'C': null,
                    'D': {'floors': d1, 'shrink': d2, 'core_mean': d3, 'gain': d4}}
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, M_star, pool_f, pf, Qe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    A: eff_rank {pr:.1f} | rank_gc " + " ".join(f"r{k}:{v:+.2f}" for k, v in rank_gc.items()))
        print(f"    B: mean R^2 {mean_r2:.3f} | gc_feature_cond {gc_pred:+.2f} | per-class R^2 " +
              " ".join(f"c{k}:{v:.2f}" for k, v in r2s.items()))
        print("    C null: " + " ".join(f"{k}: real {v['real']:+.2f} shuf {v['shuffled_mean']:+.2f}+-{v['shuffled_std']:.2f}"
                                          for k, v in null.items()))
        best_sh = sorted(d2.items(), key=lambda kv: kv[1]['best_a'])
        print("    D2 shrink best_a range: " + " ".join(f"c{k}:a{v['best_a']}" for k, v in best_sh[:4]))
        worst_core = sorted(d3.items(), key=lambda kv: kv[1]['Werr_core'])[:3]
        print("    D3 core_mean: " + " ".join(f"c{k}:core{v['Werr_core']:.2f}vs{v['Werr_plain']:.2f}"
                                               for k, v in worst_core))
        hi_frac = [v['frac_high_gain'] for v in err_energy.values()]
        print(f"    D4 frac error in high-gain dirs: mean {sum(hi_frac)/len(hi_frac):.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. COVARIANCE STRUCTURE:")
    print("   eff_rank: participation ratio of R_cov (small -> low-rank).")
    print("   rank_gc: W0 + rank-r(R_cov) gc. If rank-8 ~ rank-64, the covariance")
    print("   correction is low-rank -> pool basis + label scalars viable.")
    print("B. DELTA-Z* PREDICTABILITY:")
    print("   mean R^2: how predictable the oracle logit correction is from")
    print("   {margin, entropy, conf}. gc_feature_conditioned: the classification")
    print("   gain from a feature-conditioned logit correction (fit pool, apply val).")
    print("   If R^2 > 0.1 or gc > 0, the correction is low-dimensional + label-")
    print("   calibratable. If not, covariance is genuinely inaccessible.")
    print("C. NULL CONTROL: real vs shuffled affine gain on the best pair. If")
    print("   shuffled ~ real, the small positive is noise.")
    print("D. IMPLEMENTATION:")
    print("   D1 floors: per-pair oracle flips near the boundary (where labels")
    print("       should be spent).")
    print("   D2 shrink: per-class optimal shrinkage a_c toward pseudo-mean; if")
    print("       consistent, one global a works; if it varies, per-class needed.")
    print("   D3 core_mean: density-core mean vs plain mean whitened error; core <")
    print("       plain -> the mean is the wrong summary (build a core estimator).")
    print("   D4 gain: fraction of propagated-mean error in high-gain whitening")
    print("       directions; if high, a per-direction gain bound could help.")


if __name__ == "__main__":
    main()
