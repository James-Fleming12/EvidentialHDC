"""al_uest_bdry_diag.py: decision-rule U estimation -- the NEW constructions that
the al_uest diagnostic could not (it showed pool covariance / class-mean shift /
CCA are all orthogonal to the oracle residual, because R = W*-W0 is a
DECISION-RULE object, not a distribution-shift object).

Three families, all built from the measured insight "the residual lives in
decision-boundary geometry":

FAMILY 1 (label-free, boundary-conditioned) -- ask "which feature directions
would change the frozen classifier's decisions", not "which directions differ":
  bdry_pca        : PCA of NEAR-BOUNDARY pool points (low |margin| from W0)
  bdry_outer      : top-r left singulars of sum_i x_i (g_a - g_b)_i^T, where
                    (a,b) = top-2 predicted classes of point i, g = W0 rows
                    (decision-weighted outer product: x along the boundary normal)
  bdry_margin_cov : top-r of sum_i x_i x_i^T / |margin_i| (inverse-margin
                    weighted covariance -- near-boundary points dominate)
  pair_ab         : PCA of the pool points whose top-2 = the most-confused pair
                    (confusion-pair-specific boundary, N8 made concrete)

FAMILY 2 (few-label, adaptation tangent space) -- the reframing: don't ask an
unlabeled statistic for U; let a few labels MEASURE the adaptation response:
  tangent_b       : split b labels into T tiny windows, fit a provisional ridge
                    W_t on each, stack dW_t = W_t - W0, right-SVD -> U.
                    PCA across noisy provisional updates recovers U even if each
                    is poor (r=2 makes this cheap).
  ensemble        : the same stack but for structurally DIFFERENT weak
                    classifiers (prototype, boundary-fit, high-ridge, window) --
                    their common low-dim change is a plausible U.

Evaluation per U (r in {2,4}): align cos vs oracle, residual_capture,
AL chain (leverage-in-U true labels -> delta/gap_closed), and for the label-free
family the TTA chain (leverage-in-U pseudo labels -> delta).

Usage:
  uv run python robust_diagnostic/al_uest_bdry_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_uest_bdry_<label>.json
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
    X = X.to(device); torch.manual_seed(SKETCH_SEED)
    # cap the Nystrom sketch dim by the sample count so tiny provisional fits
    # (e.g. 2-point windows) do not produce a singular Shat
    m = min(m, X.shape[1], max(1, X.shape[0] - 1))
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


def right_topk_svd(M, r):
    """Right singular vectors (code-space directions) of a C x d matrix M."""
    _, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    return Vh[:r].t().float(), S.float()


def subspace_cos(U_hat, U_oracle, r):
    uh = U_hat[:, :r]; uo = U_oracle[:, :r]
    S = torch.linalg.svdvals((uh.t() @ uo).double())
    return float(S.mean().item())


def rand_svd_gram(M_apply, d, r, oversample=5):
    """Randomized SVD: top-r LEFT singulars of an implicit d x d symmetric
    matrix M with M_apply(v) = M @ v (v is d x k). k = r + oversample.
    Runs in float32 on CPU to bound memory (the d x k intermediates are cheap;
    only the final k x k SVD is upcast). M_apply wraps CPU tensors (pool)."""
    k = r + oversample
    torch.manual_seed(0)
    Om = torch.randn(d, k)
    Y = M_apply(Om)
    Q, _ = torch.linalg.qr(Y)
    B = Q.t() @ M_apply(Q)
    UB, S, _ = torch.linalg.svd(B.double())
    U = (Q @ UB[:, :r]).float()
    return U, S[:r].float()


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
    ap.add_argument("--max_clean", type=int, default=100000,
                    help="cap on clean points for the frozen probe fit (200k=8GB CPU; "
                         "100k saturates the fit and halves memory)")
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--r_sweep", type=str, default="2,4")
    ap.add_argument("--budget_sweep", type=str, default="8,32")
    ap.add_argument("--bdry_frac", type=float, default=0.2, help="near-boundary fraction")
    ap.add_argument("--n_windows", type=int, default=4, help="tangent-space windows")
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
    rmax = max(r_sweep)

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
        # free the raw 128-d feature tensors (keep the label tensors la/pl/vl and
        # the codes Xc/Xp/Xv, which are used through the end of the condition)
        del fa, pool, val, f, l
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_oracle, S_or = topk_svd(R, rmax)

        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # ---- frozen probe logits on the pool: margin, top-2, boundary geometry ----
        pool_s = Xp.float() @ W0.cpu()
        sm = torch.softmax(pool_s, dim=1)
        ppred = sm.argmax(dim=1)
        top2 = torch.topk(sm, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()   # |margin| of each point
        a_idx = top2.indices[:, 0]; b_idx = top2.indices[:, 1]

        Uhat = {}

        # oracle (reference)
        Uhat['oracle'] = U_oracle

        # ---- FAMILY 1: label-free boundary-conditioned U ----
        # near-boundary points: lowest |margin| fraction
        nb = torch.argsort(margin)[:int(args.bdry_frac * len(Xp))].long()
        Xnb = Xp[nb]                       # n_bdry x d
        lev = torch.norm(Xnb.float(), p=2, dim=1)
        ord_l = torch.argsort(lev, descending=True)
        if len(Xnb) > 20000:
            ord_l = ord_l[:20000]
        Xnb = Xnb[ord_l]
        n_b = len(Xnb)
        nb_sorted = nb[ord_l]              # the SAME points, same order, as Xnb

        # bdry_pca: PCA of near-boundary points (code-space right singulars)
        Uc_b, Sc_b, Vc_b = torch.linalg.svd(Xnb.double(), full_matrices=False)
        Uhat['bdry_pca'] = Vc_b[:, :rmax].float()

        # bdry_outer: top-r of M = sum_i x_i (g_a - g_b)_i^T. W0 is d x C, so the
        # boundary normal for pair (a,b) is the COLUMN W0[:, a] - W0[:, b]
        # (a 10000-d code-space direction). Gnb rows = those normals (n_b x d).
        W0_cpu = W0.detach().cpu()
        Gnb = (W0_cpu[:, a_idx[nb_sorted]] - W0_cpu[:, b_idx[nb_sorted]]).t().float()
        Gnb = Gnb[:len(Xnb)]
        Xnb_f = Xnb.float(); Gnb_f = Gnb.float()
        def M_apply_outer(v):
            return Xnb_f.t() @ (Gnb_f @ v.float())
        try:
            Uo, _ = rand_svd_gram(M_apply_outer, Xnb.shape[1], rmax)
            Uhat['bdry_outer'] = Uo
        except Exception as e:
            print(f"  [bdry_outer] failed: {e}")

        # bdry_margin_cov: top-r of M = sum_i x_i x_i^T / |margin_i| (near-boundary dominates)
        w = (1.0 / (margin[nb_sorted].float() + 1e-6))
        def M_apply_mcov(v):
            return Xnb_f.t() @ (w.unsqueeze(1) * (Xnb_f @ v.float()))
        try:
            Um, _ = rand_svd_gram(M_apply_mcov, Xnb.shape[1], rmax)
            Uhat['bdry_margin_cov'] = Um
        except Exception as e:
            print(f"  [bdry_margin_cov] failed: {e}")

        # pair_ab: PCA of points whose top-2 = the most-confused pair
        pair = torch.stack([torch.minimum(a_idx, b_idx), torch.maximum(a_idx, b_idx)], dim=1)
        pcnt = torch.bincount(pair[:, 0] * NUM_CLASSES + pair[:, 1], minlength=NUM_CLASSES * NUM_CLASSES)
        top_pair_idx = int(torch.argmax(pcnt).item())
        pa, pb = divmod(top_pair_idx, NUM_CLASSES)
        sel_p = (pair[:, 0] == pa) & (pair[:, 1] == pb)
        if int(sel_p.sum().item()) > 100:
            Xpp = Xp[sel_p][:20000]
            Upc, Spc, Vpc = torch.linalg.svd(Xpp.double(), full_matrices=False)
            Uhat['pair_ab'] = Vpc[:, :rmax].float()
            print(f"  [pair_ab] confused pair ({pa},{pb}), n={int(sel_p.sum().item())}")

        # ---- FAMILY 2: few-label adaptation-tangent-space U ----
        # tangent_b: split b labeled points into n_windows provisional ridge fits
        for b in budget_sweep:
            if b < args.n_windows:
                continue
            # select b points (leverage in the frozen-margin sense: near-boundary high-lev)
            lev_p = torch.norm(Xp.float() @ Uhat.get('bdry_pca', U_oracle[:, :2]), p=2, dim=1) if 'bdry_pca' in Uhat else torch.norm(Xp.float(), p=2, dim=1)
            sel = torch.argsort(lev_p, descending=True)[:b].long()
            wins = torch.chunk(torch.randperm(b), args.n_windows)
            dW_stack = []
            for wi in wins:
                si = sel[wi]
                W_t = ridge_fit_soft(Xp[si], onehot(pl[si], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
                dW_stack.append((W_t - W0).detach().cpu().t())   # C x d (code-space rows)
            D = torch.cat([dw for dw in dW_stack], dim=0)          # (n_windows*C) x d
            U_tan, _ = right_topk_svd(D, rmax)
            Uhat[f'tangent_b{b}'] = U_tan

        # ensemble: stack structurally different weak classifiers' dW (all C x d)
        dW_ens = []
        # prototype (class-mean code)
        proto = torch.zeros(NUM_CLASSES, Xp.shape[1])
        for c in range(1, NUM_CLASSES):
            m = (pl == c)
            if int(m.sum().item()) > 100:
                proto[c] = Xp[m].mean(dim=0)
        dW_ens.append((proto - W0.cpu().t()).float())
        # boundary-fit: ridge on near-boundary points (true labels, small set)
        bnd_lab = pl[nb][:min(5000, len(nb))]
        if len(bnd_lab) > 100:
            W_bnd = ridge_fit_soft(Xp[nb][:min(5000, len(nb))], onehot(bnd_lab, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
            dW_ens.append((W_bnd - W0).detach().cpu().t())
        # high-ridge (aggressive regularization)
        W_hr = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), 1e-1, args.cg_iters, args.nystrom_m, device)
        dW_ens.append((W_hr - W0).detach().cpu().t())
        D_ens = torch.cat(dW_ens, dim=0)
        U_ens, _ = right_topk_svd(D_ens, rmax)
        Uhat['ensemble'] = U_ens

        # ---- evaluation ----
        est = {}
        for uname, Uh in Uhat.items():
            if Uh is None:
                continue
            e = {'align': {}, 'residual_capture': {}, 'goalB_al': {}, 'goalA_tta': {}}
            for r in r_sweep:
                uh = Uh[:, :r]; uo = U_oracle[:, :r]
                sv = torch.linalg.svdvals((uh.t() @ uo).double())
                e['align'][str(r)] = {'mean': float(sv.mean().item()),
                                      'per_dir': [float(x) for x in sv.tolist()]}
            for r in r_sweep:
                ur = Uh[:, :r].double()
                cap = (ur.t() @ R.double()).norm().item() / (R.double().norm().item() + 1e-12)
                e['residual_capture'][str(r)] = float(cap)
            # label-free family gets the TTA chain too (pseudo-label C)
            is_label_free = uname in ('bdry_pca', 'bdry_outer', 'bdry_margin_cov', 'pair_ab')
            for r in r_sweep:
                Ur = Uh[:, :r]
                lev = torch.norm(Xp.float() @ Ur, p=2, dim=1)
                for b in budget_sweep:
                    if b >= args.pool_size:
                        continue
                    sel = torch.argsort(lev, descending=True)[:b].long()
                    # AL chain: C from true labels
                    C = lsq_residual(Xp[sel], onehot(pl[sel], NUM_CLASSES), W0, Ur, device)
                    W_res = W0.detach().cpu() + (Ur.cpu() @ C)
                    delta = mw(W_res, Xv, vl) - refs['frozen']
                    e['goalB_al'][f'r{r}_b{b}'] = {'delta': float(delta),
                                                   'gap_closed': float(delta / gap) if gap > 1e-9 else None}
                    if is_label_free:
                        Cp = lsq_residual(Xp[sel], onehot(ppred[sel], NUM_CLASSES), W0, Ur, device)
                        W_p = W0.detach().cpu() + (Ur.cpu() @ Cp)
                        delta_p = mw(W_p, Xv, vl) - refs['frozen']
                        e['goalA_tta'][f'r{r}_b{b}'] = {'delta': float(delta_p),
                                                        'gap_closed': float(delta_p / gap) if gap > 1e-9 else None}
            est[uname] = e

        results['conds'][cond] = {'refs': refs, 'gap': float(gap), 'U_estimators': est,
                                  'singular_values': S_or[:rmax].tolist()}
        del Xc, Xp, Xv, W0, Ws, R, U_oracle, Uhat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        for uname, e in est.items():
            al = " ".join(f"r{r}:{v['mean']:.2f}({','.join(f'{x:.2f}' for x in v['per_dir'])})"
                          for r, v in e['align'].items())
            cap = " ".join(f"r{r}:{v:.2f}" for r, v in e['residual_capture'].items())
            gb = " ".join(f"{k}:{v['delta']:+.3f}(gc {v['gap_closed']:.2f})" for k, v in e['goalB_al'].items())
            ga = " ".join(f"{k}:{v['delta']:+.3f}(gc {v['gap_closed']:.2f})" for k, v in e['goalA_tta'].items())
            print(f"  {uname:16s} align({al}) resid_cap({cap})")
            print(f"    AL (r_b:delta,gc)  {gb}")
            if ga:
                print(f"    TTA(r_b:delta,gc) {ga}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("FAMILY 1 (label-free, decision-rule): bdry_pca (near-boundary PCA),")
    print("  bdry_outer (x along boundary normal), bdry_margin_cov (inverse-margin cov),")
    print("  pair_ab (most-confused-pair PCA). If align >> 0.1, decision-boundary")
    print("  geometry DOES contain the residual -> the label-free path reopens.")
    print("FAMILY 2 (few-label): tangent_b (PCA across provisional tiny updates),")
    print("  ensemble (stack of weak-classifier dW). If align high, repeated noisy")
    print("  supervised responses recover U -> the couple-of-points AL is deployable.")
    print("AL chain: does a few-label U + leverage true labels approach the ceiling?")
    print("TTA chain (label-free family): does boundary-U + pseudo-label C beat frozen?")


if __name__ == "__main__":
    main()
