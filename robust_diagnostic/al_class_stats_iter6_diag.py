"""al_class_stats_iter6_diag.py: Iteration 6 -- COVARIANCE-SPACE LOCALIZATION.

The Iteration-5 negative (gamma*_perclass < 0, resid_rel ~ 0.001) compares a
CLASS-SPECIFIC direction against the GLOBAL flattened residual R = sum_c R_c.
That is a metric mismatch (the same trap as Iteration 3's flattened cosine): a
direction that perfectly reconstructs class c can still have near-zero cosine
with the aggregate R if other classes dominate ||R||. This diagnostic tests the
per-class / per-pair / per-eigenspace questions directly, resolving three
hypotheses:

  H1  whitening destroys each useful class direction
  H2  individual directions are useful, but global residual alignment is
      misleading (the per-class test would show alignment even if global is ~0)
  H3  whitening amplifies irrelevant/noisy covariance eigendirections

Decisive per-class measurements (suspicious classes, K = 5):

A. PER-CLASS DECODER ALIGNMENT (the decisive test)
   G_cc = cos( Sigma^-1 v_c, Sigma^-1 Delta_mu_c )
   plus recoverability <Sigma^-1 v_c, Sigma^-1 Delta_mu_c>/||Sigma^-1 Delta_mu_c||^2.
   If G_cc >> 0 for the important classes, the direction SURVIVES decoder
   geometry and the Iteration-5 negative was H2 (global residual mismatch).

B. PAIRWISE DECODER ALIGNMENT
   cos( Sigma^-1 (v_a - v_b), (w*_a - w*_b) - (w0_a - w0_b) ) for the top pairs.
   The decision-relevant object is w_a - w_b (class competition), not w_c alone.

C. COVARIANCE EIGEN-SPECTRUM
   Decompose v_c = sum a_j q_j and Delta_mu_c = sum b_j q_j in the pool
   covariance eigenbasis (q_j, lambda_j = s_j^2 + lam). Report where a_j and b_j
   live across the spectrum, and the whitened coefficients a_j/lambda_j vs
   b_j/lambda_j. If v_c has too much energy in tiny-lambda directions, the
   whitening amplifies noise (H3); if the wrong eigenspace structure, the pool
   direction is covariance-unaware.

D. FRACTIONAL WHITENING  Sigma^-beta, beta in {0, .25, .5, .75, 1}
   Per-class alignment cos(Sigma^-beta v_c, Sigma^-beta Delta_mu_c). If -1/2 is
   excellent but -1 is terrible, over-whitening is the culprit.

E. RANK SWEEP  r in {8, 16, 32, 64, 128, 256, 512}
   Per-class alignment with truncated Sigma_r^-1, PLUS the actual decoder gc(r):
   W_r = W0 + Sigma_r^-1 (V masked to suspicious classes). If alignment peaks at
   r << d, the full covariance inverse amplifies irrelevant directions.

F. PER-CLASS SCALAR CORRECTION (the low-dimensional formulation)
   W_c = W0 + alpha_c (Sigma^-1 v_c) on class c only; sweep alpha, measure gc.
   If fixing class c alone improves gc, the direction is decision-useful.

G. PER-CLASS RESIDUAL DECOMPOSITION
   R_c = (W* - W0)[:, c]; R_mean,c = Sigma0^-1 (p_c Delta_mu_c) column.
   Report ||R_mean,c||/||R_c|| and their alignment per class -- is the mean-shift
   the dominant part of class c's OWN residual column?

H. CORRUPTION CONTROL IN MEAN SPACE (revisited)
   Corrupt the RAW mean shift v_rho = sqrt(1-rho^2) Delta_mu + rho N in MEAN
   space, THEN decode with Sigma^-1, measure gc(rho). This is the correct test of
   decoder tolerance to mean-space estimation error (the Iteration-3 control
   corrupted the already-whitened difference).

I. ALTERNATIVE PSEUDO-MEANS (decoder-space alignment only)
   soft / tta / highconf / core means -> per-class alignment
   cos(Sigma^-1 (M_scheme_c - M0_c), Sigma^-1 Delta_mu_c). If none gets positive
   diagonal alignment, close it.

Decision (clean):
  G_cc >> 0 for important classes -> direction alive; move to pool-derived
      directions + few-label scalar coefficients (per class/pair), not mean
      estimation.
  full whitening bad, fractional/rank good -> covariance regularization /
      subspace selection is the missing ingredient.
  G_cc ~ 0 even per class -> the +0.92 raw direction does not survive decoder
      geometry; close the line.

Usage:
  uv run python robust_diagnostic/al_class_stats_iter6_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_class_stats_iter6_<label>.json
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
    M = torch.zeros(nc, X.shape[1])
    for c in range(nc):
        w = Pmat[:, c]
        s = w.sum().item()
        if s > 1e-9:
            M[c] = (X * w.unsqueeze(1)).sum(dim=0) / s
    return M


def topk_eigbasis(X, k, device):
    """Randomized SVD of X -> top-k right singular vectors Q (d x k) and
    covariance eigenvalues lambda_j = s_j^2 (ridge added by the caller)."""
    X = X.to(device)
    n, d = X.shape
    k = min(k, min(n, d) - 1)
    torch.manual_seed(SKETCH_SEED)
    Omega = torch.randn(d, k + 8, device=device)
    Y = X @ Omega
    Q, _ = torch.linalg.qr(Y)
    Bm = Q.t() @ X
    U, S, Vh = torch.linalg.svd(Bm, full_matrices=False)
    Qe = Vh[:k].t().contiguous()       # d x k right singular vectors
    sig = S[:k].clamp(min=1e-8)        # singular values
    return Qe.cpu(), sig.cpu()


def cos_(a, b):
    return float((a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12))


def frac_whiten(Qe, sig, lam, v, beta):
    """Sigma^-beta v using the top-k eigenbasis (Qe: d x k, sig: k).
    lambda_j = sig_j^2 + lam. Returns (k-d) whitened vector on CPU."""
    lamb = (sig ** 2 + lam)
    proj = Qe.t() @ v                      # k
    return Qe @ (proj / (lamb ** beta))


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
    ap.add_argument("--k_eig", type=int, default=1024)
    ap.add_argument("--rank_sweep", type=str, default="8,16,32,64,128,256,512")
    ap.add_argument("--beta_sweep", type=str, default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--alpha_sweep", type=str, default="-0.5,0,0.5,1.0,1.5,2.0")
    ap.add_argument("--rho_sweep", type=str, default="0,0.3,0.5,0.8")
    ap.add_argument("--tta_augs", type=int, default=5)
    ap.add_argument("--conf_thresh", type=float, default=0.5)
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
    rank_sweep = [int(x) for x in args.rank_sweep.split(',')]
    beta_sweep = [float(x) for x in args.beta_sweep.split(',')]
    alpha_sweep = [float(x) for x in args.alpha_sweep.split(',')]
    rho_sweep = [float(x) for x in args.rho_sweep.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'k_classes': args.k_classes,
               'rank_sweep': rank_sweep, 'beta_sweep': beta_sweep, 'conds': {}}

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

        Lp = Xp.float() @ W0c
        p0 = torch.softmax(Lp, dim=1)
        pseudo = Lp.argmax(1)
        pseudo_conf = p0.gather(1, pseudo.unsqueeze(1)).squeeze(1)
        M_hard, C_ph = class_means(Xp, pseudo, NUM_CLASSES)

        V = M_hard - M0
        Delta = M_star - M0
        shift_norm = torch.norm(V, p=2, dim=1)
        suspicious = [int(c) for c in torch.argsort(shift_norm, descending=True) if c != 0][:args.k_classes]

        # covariance eigenbasis of the pool (randomized SVD)
        Qe, sig = topk_eigbasis(Xp, args.k_eig, device)
        lamb = (sig ** 2 + args.lam)

        # ---- A. PER-CLASS DECODER ALIGNMENT (full Sigma^-1 via CG) ----
        B_pool = V[suspicious].t().contiguous()          # d x K
        B_orac = Delta[suspicious].t().contiguous()
        D_pool = solve_whitened(Xp, B_pool, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        D_orac = solve_whitened(Xp, B_orac, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        align = {}
        recover = {}
        for i, c in enumerate(suspicious):
            dp = D_pool[:, i]; dor = D_orac[:, i]
            align[str(c)] = cos_(dp, dor)
            recover[str(c)] = float((dp * dor).sum().item() / (dor.norm().item() ** 2 + 1e-12))

        # ---- B. PAIRWISE DECODER ALIGNMENT (top-3 pairs of suspicious) ----
        pair_align = {}
        combos = []
        for i in range(len(suspicious)):
            for j in range(i + 1, len(suspicious)):
                combos.append((suspicious[i], suspicious[j]))
        for (a, b) in combos[:3]:
            vab = V[a] - V[b]
            d_oracle_ab = (Ws[:, a] - Ws[:, b]).detach().cpu() - (W0c[:, a] - W0c[:, b])
            d_pool_ab = solve_whitened(Xp, vab.unsqueeze(1), args.lam, args.cg_iters,
                                       args.nystrom_m, device).cpu()[:, 0]
            pair_align[f"{a}-{b}"] = cos_(d_pool_ab, d_oracle_ab)

        # ---- C. EIGEN-SPECTRUM: where do v_c and Delta_mu_c live? ----
        spec = {}
        for c in suspicious:
            av = Qe.t() @ V[c]                 # a_j
            bv = Qe.t() @ Delta[c]             # b_j
            # whitened coefficients in the top-k basis
            aw = av / lamb
            bw = bv / lamb
            spec[str(c)] = {
                'a_frac_top_half': float(av[:args.k_eig // 2].norm().item() /
                                         (av.norm().item() + 1e-12)),
                'b_frac_top_half': float(bv[:args.k_eig // 2].norm().item() /
                                         (bv.norm().item() + 1e-12)),
                'cos_whitened_coef': cos_(aw, bw),
            }

        # ---- D. FRACTIONAL WHITENING alignment (eigenbasis, per class) ----
        frac = {}
        for beta in beta_sweep:
            al = []
            for c in suspicious:
                wv = frac_whiten(Qe, sig, args.lam, V[c], beta)
                wd = frac_whiten(Qe, sig, args.lam, Delta[c], beta)
                al.append(cos_(wv, wd))
            frac[str(beta)] = sum(al) / len(al) if al else None

        # ---- E. RANK SWEEP: per-class alignment + decoder gc(r) ----
        rank_align = {}
        rank_gc = {}
        for r in rank_sweep:
            rr = min(r, args.k_eig)
            Qr = Qe[:, :rr]; lr = lamb[:rr]
            al = []
            for c in suspicious:
                wv = Qr @ ((Qr.t() @ V[c]) / lr)
                wd = Qr @ ((Qr.t() @ Delta[c]) / lr)
                al.append(cos_(wv, wd))
            rank_align[str(r)] = sum(al) / len(al) if al else None
            # decoder: W0 + Sigma_r^-1 (V masked to suspicious classes)
            M_r = M0.clone()
            for c in suspicious:
                M_r[c] = M0[c] + (Qr @ ((Qr.t() @ V[c]) / lr))
            W_r = solve_whitened(Xp, (M_r * C0.unsqueeze(1)).t().contiguous(),
                                 args.lam, args.cg_iters, args.nystrom_m, device)
            rank_gc[str(r)] = gc(mw(W_r, Xv, vl))

        # ---- F. PER-CLASS SCALAR CORRECTION (alpha on class c only) ----
        scalar = {}
        for c in suspicious:
            dc = solve_whitened(Xp, V[c].unsqueeze(1), args.lam, args.cg_iters,
                                args.nystrom_m, device).cpu()[:, 0]
            out = {}
            for a in alpha_sweep:
                W_a = W0c.clone()
                W_a[:, c] = W0c[:, c] + a * dc
                out[str(a)] = gc(mw(W_a, Xv, vl))
            scalar[str(c)] = out

        # ---- G. PER-CLASS RESIDUAL DECOMPOSITION ----
        dec = {}
        for c in suspicious:
            Rc = R[:, c]
            R_mean_c = solve_whitened(Xp, (C0[c] * Delta[c]).unsqueeze(1),
                                      args.lam, args.cg_iters, args.nystrom_m, device).cpu()[:, 0]
            dec[str(c)] = {
                'frac_mean_norm': float(R_mean_c.norm().item() / (Rc.norm().item() + 1e-12)),
                'align_Rc': cos_(R_mean_c, Rc),
            }

        # ---- H. CORRUPTION CONTROL IN MEAN SPACE (then decode) ----
        corr = {}
        torch.manual_seed(21)
        for rho in rho_sweep:
            M_r = M0.clone()
            for c in suspicious:
                dmu = Delta[c]
                N = torch.randn_like(dmu)
                N = N / (N.norm().item() + 1e-12) * dmu.norm().item()
                M_r[c] = M0[c] + (1 - rho ** 2) ** 0.5 * dmu + rho * N
            W_r = solve_whitened(Xp, (M_r * C0.unsqueeze(1)).t().contiguous(),
                                 args.lam, args.cg_iters, args.nystrom_m, device)
            corr[str(rho)] = gc(mw(W_r, Xv, vl))

        # ---- I. ALTERNATIVE PSEUDO-MEANS: per-class decoder alignment ----
        M_soft = soft_means(Xp, p0, NUM_CLASSES)
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xp) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xp, Xp) @ W0c, dim=1))
        M_tta = soft_means(Xp, torch.stack(draws).mean(dim=0), NUM_CLASSES)
        M_high = M0.clone(); M_core = M0.clone()
        for c in range(1, NUM_CLASSES):
            idx = torch.nonzero(pseudo == c).squeeze(1)
            if len(idx) < 10:
                continue
            hc = idx[pseudo_conf[idx] > args.conf_thresh]
            if len(hc) >= 5:
                M_high[c] = Xp[hc].float().mean(dim=0)
            sim = F_normalize(Xp[idx].float()) @ F_normalize(M_hard[c].unsqueeze(0)).t()
            core = idx[torch.argsort(sim[:, 0], descending=True)[:max(len(idx) // 2, 5)]]
            M_core[c] = Xp[core].float().mean(dim=0)
        schemes = {'hard': M_hard, 'soft': M_soft, 'tta': M_tta,
                   'highconf': M_high, 'core': M_core}
        scheme_align = {}
        for sname, Ms in schemes.items():
            Bs = (Ms - M0)[suspicious].t().contiguous()
            Ds = solve_whitened(Xp, Bs, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
            al = [cos_(Ds[:, i], D_orac[:, i]) for i in range(len(suspicious))]
            scheme_align[sname] = sum(al) / len(al) if al else None

        cond_res = {'refs': refs, 'gap': float(gap),
                    'suspicious': suspicious,
                    'A_perclass_align': align,
                    'A_recoverability': recover,
                    'B_pair_align': pair_align,
                    'C_eigspec': spec,
                    'D_fractional': frac,
                    'E_rank_align': rank_align,
                    'E_rank_gc': rank_gc,
                    'F_scalar': scalar,
                    'G_perclass_resid': dec,
                    'H_corruption_meanspace': corr,
                    'I_scheme_align': scheme_align}
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, M_star, Qe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    suspicious: {suspicious}")
        print("    A per-class align: " + " ".join(f"c{c}:{v:+.2f}" for c, v in align.items()))
        print("    B pair align: " + " ".join(f"{k}:{v:+.2f}" for k, v in pair_align.items()))
        print("    D fractional: " + " ".join(f"b{k}:{v:+.2f}" if v is not None else f"b{k}:NA"
                                              for k, v in frac.items()))
        print("    E rank align: " + " ".join(f"r{k}:{v:+.2f}" for k, v in rank_align.items()))
        print("    E rank gc:    " + " ".join(f"r{k}:{v:+.2f}" for k, v in rank_gc.items()))
        print("    G perclass resid: " + " ".join(
            f"c{k}:frac{v['frac_mean_norm']:.2f}al{v['align_Rc']:+.2f}" for k, v in dec.items()))
        print("    H corruption(mean-space): " + " ".join(f"r{k}:{v:+.2f}" for k, v in corr.items()))
        print("    I scheme align: " + " ".join(f"{k}:{v:+.2f}" if v is not None else f"{k}:NA"
                                                for k, v in scheme_align.items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. PER-CLASS decoder alignment is THE decisive test (vs the global")
    print("   resid_rel ~ 0.001 of Iteration 5 which mixes all classes):")
    print("   G_cc >> 0 -> direction survives decoder geometry (H2: the global")
    print("   residual comparison was misleading). G_cc ~ 0 -> close the line.")
    print("B. Pairwise alignment (w_a - w_b is the decision-relevant object).")
    print("D/E. Fractional / rank-truncated whitening: if -1/2 or low rank is good")
    print("   but full -1 is bad, over-whitening / irrelevant eigendirections (H3).")
    print("G. Per-class residual: is the mean-shift the dominant part of class c's")
    print("   OWN residual column?")
    print("H. Corruption in MEAN space then decode: the correct tolerance test.")
    print("I. Alternative pseudo-means: does any scheme get positive diagonal")


def F_normalize(x):
    return torch.nn.functional.normalize(x, p=2, dim=1)


if __name__ == "__main__":
    main()
