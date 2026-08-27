"""al_uest_diag.py: estimate the residual subspace U -- with and without oracle
labels -- and evaluate the FULL update chain for both goals:

  GOAL A (TTA, label-free U): a label-free TTA method that gives meaningful
      improvement on the unhealthy conditions (fog/crosstalk). U is estimated
      with NO labels; C is fit from the frozen probe's pseudo-labels in the
      U-subspace. The hypothesis: the Iterations 9-12 label-free failure was the
      FULL-space inverse amplifying poisoned T; restricting to the low-rank U
      (where the real shift lives) may let even pseudo-label C work.
  GOAL B (AL, few-label U): an AL method that gets near the ceiling. U is
      estimated from a small label budget (sub-fit / shift-sub), then C is fit
      from the same leverage-selected labels in that U.

U estimators tested (all give directions in the 10000-d CODE space, comparable
to the oracle left-singular subspace of R = W* - W0):

  label-free (GOAL A):
    oracle        reference: top-r SVD of (Ws - W0)                    [needs full labels]
    softshift     corrupted class means from the frozen probe's SOFT
                  assignment, shift = soft_mean - clean_mean, mapped
                  through (S+lI)^-1 via CG, then SVD                    [no labels]
    poolcov       top-r eigenvectors of the corrupted pool covariance
                  (the C21/C22 old failure, re-tested at low rank)      [no labels]
    ccameans      PCA-whitened CCA between the CLEAN class-mean matrix
                  and the soft corrupted class-mean matrix; the
                  corrupted-side canonical directions (N7)              [no labels]
  few-label (GOAL B):
    subfit        SVD of (W_sub - W0) where W_sub is fit on b labeled
                  points (leverage-in-frozen-influence selection)       [b labels]
    shiftsub      corrupted class means from b labeled points, shift =
                  labeled_mean - clean_mean, mapped through geometry    [b labels]

Evaluation per U estimator:
  - alignment: cos(U_hat[:, :r], U_oracle[:, :r]) for r in {2,4,8}
  - GOAL A chain: W_res = W0 + U_hat C, C fit on frozen PSEUDO-labels in the
    U-subspace -> delta vs frozen (must be POSITIVE on fog/crosstalk to be a
    meaningful TTA)
  - GOAL B chain: W_res = W0 + U_hat C, C fit on b leverage-in-U_hat TRUE
    labels -> delta and gap-closed vs the oracle-U reference (near-ceiling?)

Usage:
  uv run python robust_diagnostic/al_uest_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_uest_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
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


def extract_features(model, parser, device, num_frames=100):
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
        p.append((codes[s:s + chunk].float() @ W).argmax(1))
    return torch.cat(p)


def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)


def cg_solve(X, T, lam, device, iters=8, x0=None):
    X = X.to(device)
    d = X.shape[1]; C = T.shape[1]
    x = x0.to(device).clone() if x0 is not None else torch.zeros(d, C, device=device)
    def A(v):
        return X.T @ (X @ v)
    b = T.to(device)
    r = b - A(x); p = r.clone(); rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p); alpha = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha.unsqueeze(0) * p; r = r - alpha.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0); beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p; rs_old = rs_new
    return x.float()


def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device); torch.manual_seed(SKETCH_SEED); m = min(m, X.shape[1])
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P; Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device); That = XP.t() @ Yd
    x = P @ torch.linalg.solve(Shat, That); b = X.t() @ Yd
    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); a = rs / ((p * Ap).sum(0) + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0); be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p; rs = rsn
    return x.float()


def lsq_residual(X_lab, Y_lab, W0, U, device):
    Xd = X_lab.to(device).float(); Yd = Y_lab.to(device).float(); Ud = U.to(device)
    r = Ud.shape[1]; XU = Xd @ Ud
    A = XU.t() @ XU + 1e-6 * torch.eye(r, device=device)
    b = XU.t() @ (Yd - Xd @ W0.to(device))
    return torch.linalg.solve(A, b).cpu()


def topk_svd(M, r):
    U, S, _ = torch.linalg.svd(M.double(), full_matrices=False)
    return U[:, :r].float(), S.float()


def subspace_cos(U_hat, U_oracle, r):
    """cos of the r-dim subspaces: mean singular value of U_hat^T U_oracle."""
    uh = U_hat[:, :r]; uo = U_oracle[:, :r]
    S = torch.linalg.svd((uh.t() @ uo).double(), compute_uv=False)
    return float(S.mean().item())


def class_means_from_soft(X, sm, classes):
    """Soft-assignment corrupted class means (10000-d), per class."""
    means = torch.zeros(len(classes), X.shape[1])
    for i, c in enumerate(classes):
        w = sm[:, c]
        s = w.sum().clamp(min=1e-8)
        means[i] = (w.unsqueeze(1) * X).sum(dim=0) / s
    return means


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
    ap.add_argument("--pool_size", type=int, default=50000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=200000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--r_sweep", type=str, default="2,4,8")
    ap.add_argument("--budget_sweep", type=str, default="8,32")
    ap.add_argument("--cca_pca_k", type=int, default=8)
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    r_sweep = [int(x) for x in args.r_sweep.split(',')]
    budget_sweep = [int(x) for x in args.budget_sweep.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    results = {'label': args.label, 'method': args.method_b, 'conds': {}}

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
        proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
        Xc = hdc_codes(fa[ci], proj, device).float()
        Xp = hdc_codes(pool, proj, device).float()
        Xv = hdc_codes(val, proj, device).float()

        W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_oracle, S_or = topk_svd(R, max(r_sweep))

        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}

        # clean class means (from the CLEAN set, which we HAVE labels for)
        clean_means = {}
        clean_n = {}
        for c in classes:
            m = (la[ci] == c)
            if int(m.sum().item()) > 0:
                clean_means[c] = Xc[m].mean(dim=0)
                clean_n[c] = int(m.sum().item())

        # true corrupted class means (oracle, for the diagnostic decomposition)
        true_means = {}
        true_n = {}
        for c in classes:
            m = (pl == c)
            if int(m.sum().item()) > 0:
                true_means[c] = Xp[m].mean(dim=0)
                true_n[c] = int(m.sum().item())

        # frozen probe soft assignment on the corrupted pool
        pool_s = Xp.float() @ W0.cpu()
        sm = torch.softmax(pool_s, dim=1)
        ppred = sm.argmax(dim=1)
        sm_means = class_means_from_soft(Xp, sm, classes)

        # ---- U estimators ----
        Uhat = {}

        # oracle (reference)
        Uhat['oracle'] = U_oracle

        # softshift (label-free): corrupted soft means - clean means, via geometry.
        # M_shift is d x C (the RHS T of (S+lI)W = T): col c = (soft_mean_c - clean_mean_c).
        M_shift = torch.zeros(Xp.shape[1], NUM_CLASSES)
        for i, c in enumerate(classes):
            if c in clean_means:
                M_shift[:, c] = sm_means[i] - clean_means[c]
        R_soft = cg_solve(Xp, M_shift, args.lam, device, args.cg_iters)   # (S+lI)^-1 shift
        Uhat['softshift'], _ = topk_svd(R_soft.detach().cpu(), max(r_sweep))

        # poolcov (label-free): top eigenvectors of corrupted pool covariance.
        # Eigenvectors of Xp^T Xp = RIGHT singular vectors of Xp (d x d); the left
        # singular vectors live in the n-sample space and are the wrong orientation.
        Uc, Sc, Vc = torch.linalg.svd(Xp.double(), full_matrices=False)
        Uhat['poolcov'] = Vc[:, :max(r_sweep)].float()

        # ccameans (label-free, N7): CCA clean vs soft-corr class means. Both are
        # n_c x d (rows = classes, cols = 10000-d code). Whiten each with its RIGHT
        # singular vectors (the code-space directions) then CCA on the whitened
        # cross-cov; the corrupted-side canonical directions in the code space are U.
        try:
            A = torch.stack([clean_means[c] for c in classes]).double()   # n_c x d
            B = sm_means.double()                                         # n_c x d
            Ac = A - A.mean(0); Bc = B - B.mean(0)
            Ac = Ac / (Ac.norm(dim=1, keepdim=True) + 1e-8)
            Bc = Bc / (Bc.norm(dim=1, keepdim=True) + 1e-8)
            k = min(args.cca_pca_k, len(classes) - 1, Ac.shape[1])
            # NOTE: torch.linalg.svd returns Vh (= V^T) of shape (min(m,n), n). The
            # right singular vectors (code-space directions, d x n_c) are Va.t().
            Ua, Sa, Va = torch.linalg.svd(Ac, full_matrices=False)   # Va is Vh: (n_c, d)
            Ub, Sb, Vb = torch.linalg.svd(Bc, full_matrices=False)
            Wa = Va.t()[:, :k] @ torch.diag(1.0 / (Sa[:k] + 1e-6))   # d x k whitening (code space)
            Wb = Vb.t()[:, :k] @ torch.diag(1.0 / (Sb[:k] + 1e-6))
            Aw = Ac @ Wa                                              # n_c x k whitened classes
            Bw = Bc @ Wb
            Ccc = Aw.t() @ Bw                                         # k x k cross-cov
            _, _, Vh = torch.linalg.svd(Ccc, full_matrices=False)
            Ucca = (Wb @ Vh.t()).float()                              # corrupted-side canonical dirs (d x k)
            # orthonormalize in the code space so it is a proper U basis
            Ucca, _ = torch.linalg.qr(Ucca)
            if Ucca.shape[1] >= max(r_sweep):
                Uhat['ccameans'] = Ucca[:, :max(r_sweep)]
        except Exception as e:
            print(f"  [ccameans] failed: {e}")
            Uhat['ccameans'] = None

        # ---- few-label estimators ----
        # frozen-influence proxy selection (deployable, no oracle U needed):
        # leverage in the soft-shift geometry (what a label-free update could use).
        lev_frozen = torch.norm(Xp.float() @ Uhat['softshift'], p=2, dim=1)
        for b in budget_sweep:
            if b >= args.pool_size:
                continue
            sel = torch.argsort(lev_frozen, descending=True)[:b].long()
            # subfit: SVD of (W_sub - W0)
            W_sub = ridge_fit_soft(Xp[sel], onehot(pl[sel], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
            Uhat[f'subfit_b{b}'], _ = topk_svd((W_sub - W0).detach().cpu(), max(r_sweep))
            # shiftsub: corrupted class means from the labeled subset
            M_sub = torch.zeros(Xp.shape[1], NUM_CLASSES)
            for c in classes:
                m = (pl[sel] == c)
                if int(m.sum().item()) > 0 and c in clean_means:
                    M_sub[:, c] = Xp[sel][m].mean(dim=0) - clean_means[c]
            R_sub = cg_solve(Xp, M_sub, args.lam, device, args.cg_iters)
            Uhat[f'shiftsub_b{b}'], _ = topk_svd(R_sub.detach().cpu(), max(r_sweep))

        # ---- causal diagnostics (shared / per-estimator) ----
        # what the update mechanism actually needs, at each (r, b):
        #   s_oracle = U^T X^T (Y_true - X W0) projected residual signal that C fits.
        #   s_pseudo = the same with pseudo labels -> cos(s_pseudo, s_oracle) tells whether
        #              the label source points C in the right direction (the Iteration-11
        #              "wrong labels anti-align" check, now IN the subspace).
        #   C_oracle = the C that WOULD be fit if labels were right (what it needed).
        proj_sig = {}
        for r in r_sweep:
            for b in budget_sweep:
                if b >= args.pool_size:
                    continue
                proj_sig[f'r{r}_b{b}'] = {}
                # oracle U is the reference basis for "what the mechanism needed"
                for uname, Uh in Uhat.items():
                    if Uh is None:
                        continue
                    Ur = Uh[:, :r]
                    lev = torch.norm(Xp.float() @ Ur, p=2, dim=1)
                    sel = torch.argsort(lev, descending=True)[:b].long()
                    XU = Xp[sel].float() @ Ur
                    resid_true = onehot(pl[sel], NUM_CLASSES) - Xp[sel].float() @ W0.cpu()
                    resid_pseudo = onehot(ppred[sel], NUM_CLASSES) - Xp[sel].float() @ W0.cpu()
                    s_true = XU.t() @ resid_true
                    s_pseudo = XU.t() @ resid_pseudo
                    cos_ = F.cosine_similarity(s_pseudo.flatten(), s_true.flatten(), dim=0).item() \
                        if s_pseudo.numel() else None
                    # how much of the true signal lies in this U (vs what oracle U captures)
                    s_true_full = s_true.flatten()
                    norm_true = s_true_full.norm()
                    # C that WOULD be fit with true labels in this U (what the mechanism needed)
                    proj_sig[f'r{r}_b{b}'][uname] = {
                        'cos_pseudo_true_signal': cos_,
                        '||s_true||': float(norm_true),
                        '||s_pseudo||': float(s_pseudo.flatten().norm()),
                        'pseudo_true_signal_ratio': float(s_pseudo.flatten().norm() / (norm_true + 1e-12)),
                    }

        # per-estimator evaluation
        est = {}
        for uname, Uh in Uhat.items():
            if Uh is None:
                continue
            e = {'align': {}, 'residual_capture': {}, 'goalA_tta': {}, 'goalB_al': {}}
            # alignment vs oracle at each r: per-direction cos (which r-dirs are right)
            for r in r_sweep:
                uh = Uh[:, :r]; uo = U_oracle[:, :r]
                sv = torch.linalg.svd((uh.t() @ uo).double(), compute_uv=False)
                e['align'][str(r)] = {'mean': float(sv.mean().item()),
                                      'per_dir': [float(x) for x in sv.tolist()],
                                      'cos_span': float(F.cosine_similarity(
                                          uh.flatten(), uo.flatten(), dim=0).item())}
            # how much of the ORACLE residual R does this U actually capture?
            for r in r_sweep:
                ur = Uh[:, :r]
                cap = (ur.t() @ R.double()).norm().item() / (R.double().norm().item() + 1e-12)
                e['residual_capture'][str(r)] = float(cap)
            # GOAL A: TTA chain -- C from frozen PSEUDO-labels in the U-subspace
            # GOAL B: AL chain -- C from b leverage-in-U TRUE labels
            for r in r_sweep:
                Ur = Uh[:, :r]
                lev = torch.norm(Xp.float() @ Ur, p=2, dim=1)
                for b in budget_sweep:
                    if b >= args.pool_size:
                        continue
                    sel = torch.argsort(lev, descending=True)[:b].long()
                    for goal, ylab, store in [
                            ('goalA_tta', ppred, e['goalA_tta']),
                            ('goalB_al', pl, e['goalB_al'])]:
                        C = lsq_residual(Xp[sel], onehot(ylab[sel], NUM_CLASSES), W0, Ur, device)
                        W_res = W0.detach().cpu() + (Ur.cpu() @ C)
                        delta = mw(W_res, Xv, vl) - refs['frozen']
                        store[f'r{r}_b{b}'] = {'delta': float(delta),
                                               'gap_closed': float(delta / gap) if gap > 1e-9 else None,
                                               'cos_pseudo_true_signal':
                                                   proj_sig[f'r{r}_b{b}'][uname]['cos_pseudo_true_signal'] if goal == 'goalA_tta' else None}
            est[uname] = e

        # ---- construction-vs-estimation decomposition for shift-family estimators ----
        # Does the CONSTRUCTION (shift -> U) recover the residual at all, given TRUE means?
        # Build the true-shift U (oracle shift directions) as the "what it would have needed"
        # upper bound for the shift family, then compare est-shift-U vs true-shift-U vs oracle-U.
        M_true_shift = torch.zeros(Xp.shape[1], NUM_CLASSES)
        for c in classes:
            if c in clean_means and c in true_means:
                M_true_shift[:, c] = true_means[c] - clean_means[c]
        R_true_shift = cg_solve(Xp, M_true_shift, args.lam, device, args.cg_iters)
        U_true_shift, _ = topk_svd(R_true_shift.detach().cpu(), max(r_sweep))
        est['softshift']['construction_diag'] = {
            'true_shift_vs_oracle': {str(r): subspace_cos(U_true_shift, U_oracle, r) for r in r_sweep},
            'est_shift_vs_true_shift': {str(r): subspace_cos(Uhat['softshift'], U_true_shift, r) for r in r_sweep},
            'soft_mean_vs_true_mean': {},
            'shift_magnitude_ratio': {},
        }
        for i, c in enumerate(classes):
            if c in true_means:
                est['softshift']['construction_diag']['soft_mean_vs_true_mean'][str(c)] = float(
                    F.cosine_similarity(sm_means[i], true_means[c], dim=0).item())
                est['softshift']['construction_diag']['shift_magnitude_ratio'][str(c)] = float(
                    (sm_means[i] - clean_means[c]).norm() / ((true_means[c] - clean_means[c]).norm() + 1e-12))
        # soft-assignment quality
        pseudo_acc = float((ppred == pl).float().mean().item())
        est['softshift']['construction_diag']['pseudo_acc'] = pseudo_acc
        est['softshift']['construction_diag']['true_shift_singular'] = [
            float(x) for x in torch.linalg.svd(R_true_shift.double(), compute_uv=False)[:max(r_sweep)].tolist()]

        results['conds'][cond] = {'refs': refs, 'gap': float(gap), 'U_estimators': est,
                                  'proj_signal': proj_sig,
                                  'singular_values': S_or[:max(r_sweep)].tolist()}
        del Xc, Xp, Xv, W0, Ws, R, U_oracle, Uhat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        for uname, e in est.items():
            al = " ".join(f"r{r}:{v['mean']:.2f}({','.join(f'{x:.2f}' for x in v['per_dir'])})"
                          for r, v in e['align'].items())
            cap = " ".join(f"r{r}:{v:.2f}" for r, v in e['residual_capture'].items())
            ga = " ".join(f"{k}:{v['delta']:+.3f}(gc {v['gap_closed']:.2f}, sig {v['cos_pseudo_true_signal']:.2f})"
                          if v['cos_pseudo_true_signal'] is not None else f"{k}:{v['delta']:+.3f}(gc {v['gap_closed']:.2f})"
                          for k, v in e['goalA_tta'].items())
            gb = " ".join(f"{k}:{v['delta']:+.3f}(gc {v['gap_closed']:.2f})" for k, v in e['goalB_al'].items())
            print(f"  {uname:14s} align({al})")
            print(f"    residual_capture {cap}")
            print(f"    TTA(r_b:delta,gc,sig) {ga}")
            print(f"    AL (r_b:delta,gc)     {gb}")
        # construction diagnostics for shift family
        if 'softshift' in est:
            cd = est['softshift'].get('construction_diag', {})
            print(f"  [softshift construction] true_shift_vs_oracle=" +
                  " ".join(f"r{r}:{v:.2f}" for r, v in cd.get('true_shift_vs_oracle', {}).items()))
            print(f"      est_shift_vs_true_shift=" +
                  " ".join(f"r{r}:{v:.2f}" for r, v in cd.get('est_shift_vs_true_shift', {}).items()))
            print(f"      pseudo_acc={cd.get('pseudo_acc')}  soft_mean_vs_true_mean mean=" +
                  f"{np.mean(list(cd.get('soft_mean_vs_true_mean', {}).values())) if cd.get('soft_mean_vs_true_mean') else 'na':.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("U estimators: oracle (ref, full labels) | softshift, poolcov, ccameans (label-free)")
    print("              | subfit_b, shiftsub_b (b labels, frozen-influence selection)")
    print("align.per_dir: per-direction cos(U_hat_r, U_oracle_r) -- WHICH residual directions")
    print("              are recovered (top-2 right but rest wrong? all wrong?).")
    print("residual_capture: ||U_hat^T R||/||R|| -- how much of the actual residual the U holds.")
    print("goalA_tta.sig: cos(pseudo-residual signal, true-residual signal) IN THE SUBSPACE --")
    print("              does the label source point C the right way (Iteration-11 anti-align, now low-rank)?")
    print("proj_signal.*: ||s_pseudo||/||s_true|| -- does the update under- or over-shoot?")
    print("softshift.construction_diag: true_shift_vs_oracle (is the shift->U construction right?);")
    print("              est_shift_vs_true_shift (is the soft estimation the problem?);")
    print("              soft_mean_vs_true_mean per class (which classes' shifts are wrong);")
    print("              shift_magnitude_ratio per class (over/under-shift).")
    print("VERDICT RULE: split each failure into construction vs estimation vs label-source:")
    print("  - align good but TTA bad  -> label source (sig cos low) or C amplification, NOT U.")
    print("  - construction good but est-shift bad -> soft means wrong (per-class cos shows which).")
    print("  - construction bad (true_shift_vs_oracle low) -> shift->U is the wrong construction;")
    print("    the residual is NOT the shifted means, need a different U family.")
    print("  - residual_capture low -> the U is in the wrong part of the code space entirely.")


if __name__ == "__main__":
    main()
