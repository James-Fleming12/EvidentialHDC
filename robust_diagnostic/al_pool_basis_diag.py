"""al_pool_basis_diag.py: P1 -- POOL-DERIVED BASIS + FEW-LABEL COEFFICIENT
SELECTION (the last open parameter-update branch, new_iters.md Tier 1.5).

Iteration 3b closed the "few labels -> span(x_i) -> Delta W" family: 8 labels'
CE-gradient directions capture only 0.4-2.0% of the oracle residual R. But the
COEFFICIENT half is easy -- Iteration 1 showed few labels drive the step well
(+0.29-0.37 gc) GIVEN oracle U. So the one open question: can the UNLABELED POOL
provide a basis that CONTAINS R, and can a few labels SELECT/WEIGHT the right
combination from it?

   few labels -> span(x_i) -> Delta W                  (CLOSED, 3b)
   unlabeled pool -> basis V (rich dictionary)         <- THIS TEST
   + few labels -> coefficients c                      <- THIS TEST
   Delta W = V c

The diagnostic cleanly SEPARATES the basis half from the selection half:

A. BASIS QUALITY (does a pool-derived dictionary contain R?)
   - dict_span = ||P_span(D) R|| / ||R||  -- the decisive number, same metric as
     rank-1b but for the POOL dictionary. ~0 => no pool structure contains R and
     P1 is dead at the basis (fast, decisive).
   - oracle_coef = W0 + P_span(D) R -- the best classifier reachable with
     W1-W0 in span(D) (perfect coefficients). Compared to the oracle_U gc bound
     (top-2 of R, the Iteration-1 reference): if oracle_coef ~ oracle_U gc, the
     dictionary is as good as the true residual basis; if far below, the dict is
     the bottleneck.

B. SELECTION (can few labels pick the combination from a given dictionary?)
   With the dictionary fixed, only the label-to-coefficient rule varies:
   - firstorder: G = (X_lab D)^T (Y - X_lab W0), W1 = W0 + rho * D G/||G||  -- the
     exact Iteration-1 rule, with D in place of oracle U.
   - lsq: C = argmin_c ||(X_lab D) c - (Y - X_lab W0)|| (scale-adaptive ridge),
     W1 = W0 + D C  -- direct few-label coefficient fit.
   Compared to oracle_coef (perfect selection). If selection ~ oracle_coef gc, the
   few labels select as well as the true R itself -- P1 fully works. If selection
   << oracle_coef, the basis is fine but few labels cannot select it.

Dictionary elements (all label-free, from the UNLABELED pool + frozen probe):
   pool_cov    top-r eigenvectors of the pool code covariance (the KNOWN-FAILING
               control from pool_span -- kept to anchor the comparison)
   bdry_pca    top-r PCA of the boundary points (low frozen margin) -- does R live
               along the frozen boundary directions?
   bdry_disp   mean(boundary) - mean(pool) -- where the boundary mass sits
   conf_pair   (w_a - w_b) for the top-N confused pairs -- the confusion-margin
               directions of the frozen probe (decision-rule structure, R3)
   class_disp  per-class pseudo-label mean shifts vs pool mean -- where each
               class's mass moved under corruption
   tta_disp    mean(high-TTA-variance points) - mean(pool) -- the disagreement
               displacement direction

Read:
   dict_span > 0.5 & oracle_coef ~ oracle_U gc  -> pool basis CONTAINS R (basis ok)
   dict_span ~ 0                                -> no pool structure contains R;
                                                  P1 closed at the basis
   selection ~ oracle_coef gc                   -> few labels select the right combo
   selection << oracle_coef gc                  -> basis ok but labels can't select
                                                  (the coefficient half is NOT
                                                  actually easy for a non-oracle basis)

Usage:
  uv run python robust_diagnostic/al_pool_basis_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_pool_basis_<label>.json
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


def right_topk_svd(M, r):
    _, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    if not torch.isfinite(S).all():
        S = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
        Vh = torch.where(torch.isfinite(Vh), Vh, torch.zeros_like(Vh))
    return Vh[:r].t().float(), S.float()


def farthest_point(feats, cand_idx, b, device):
    cf = F.normalize(feats[cand_idx].float(), p=2, dim=1).to(device)
    torch.manual_seed(3)
    sel = [int(torch.randint(len(cand_idx), (1,)).item())]
    dist = (cf - cf[sel[0]]).norm(dim=1)
    for _ in range(b - 1):
        nxt = int(dist.argmax().item())
        sel.append(nxt)
        d2 = (cf - cf[nxt]).norm(dim=1)
        dist = torch.minimum(dist, d2)
    return cand_idx[torch.tensor(sel)]


def span_capture(D, R_flat):
    """Fraction of ||R|| captured by the span of the orthonormal columns of D.
    D: d x K orthonormal. R_flat: (d*C) flattened. Projection is on the flattened
    space via the Kronecker block structure: P_span(R) = D (D^T R_block) stacked.
    We compute it directly: R_block = R_flat.view(d, C); proj = D @ (D^T @ R_block);
    capture = ||proj|| / ||R_block||."""
    Dd = D.double()
    R_b = R_flat.view(-1, NUM_CLASSES)
    proj = Dd @ (Dd.t() @ R_b)
    return (proj.norm().item()) / (R_b.norm().item() + 1e-12)


def orthonormal(D, device):
    """Orthonormalize the d x K dictionary (QR on columns)."""
    D = D.float().to(device)
    Q, _ = torch.linalg.qr(D)
    keep = Q.norm(dim=0) > 1e-8
    return Q[:, keep]


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
    ap.add_argument("--budgets", type=str, default="2,4,8")
    ap.add_argument("--rho_sweep", type=str, default="0.05,0.2,0.8")
    ap.add_argument("--cand_frac", type=float, default=0.05)
    ap.add_argument("--tta_augs", type=int, default=5)
    ap.add_argument("--r", type=int, default=2, help="rank for pool_cov/bdry_pca per-element bases")
    ap.add_argument("--n_pairs", type=int, default=8, help="top-N confused pairs for conf_pair directions")
    ap.add_argument("--dicts", type=str, default="pool_cov,bdry_pca,bdry_disp,conf_pair,class_disp,tta_disp")
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
    budgets = [int(x) for x in args.budgets.split(',')]
    bmax = max(budgets)
    rho_sweep = [float(x) for x in args.rho_sweep.split(',')]
    dict_names = [x.strip() for x in args.dicts.split(',') if x.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'r': args.r,
               'n_pairs': args.n_pairs, 'budgets': budgets, 'rho_sweep': rho_sweep,
               'dicts': dict_names, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
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
        R = (Ws - W0).detach().cpu().float()
        R_flat = R.flatten().double()
        U_or, _ = right_topk_svd(R.t(), args.r)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # ---- label-free pool statistics (frozen probe) ----
        sm = torch.softmax(Xp.float() @ W0c, dim=1)
        top2 = torch.topk(sm, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        a_idx = top2.indices[:, 0]; b_idx = top2.indices[:, 1]
        pool_mean = Xp.mean(dim=0)
        pseudo = sm.argmax(dim=1)

        # ---- build the dictionary elements ----
        dict_elems = {}
        n = len(Xp)

        if 'pool_cov' in dict_names:
            # top-r eigenvectors of the pool code covariance (known-failing control)
            Xs = Xp[::5].double()                       # subsample for SVD cost
            Up, Sp, Vp = torch.linalg.svd(Xs, full_matrices=False)
            dict_elems['pool_cov'] = Vp[:args.r].t().float()

        if 'bdry_pca' in dict_names:
            nb = int(0.05 * n)
            bnd = torch.argsort(margin)[:nb]
            Xb = Xp[bnd].double()
            Ub, Sb, Vb = torch.linalg.svd(Xb, full_matrices=False)
            dict_elems['bdry_pca'] = Vb[:args.r].t().float()

        if 'bdry_disp' in dict_names:
            bnd = torch.argsort(margin)[:int(0.05 * n)]
            disp = Xp[bnd].mean(dim=0) - pool_mean
            dict_elems['bdry_disp'] = disp / (disp.norm() + 1e-8)

        if 'conf_pair' in dict_names:
            # top-N confused pairs by count of (top1, top2) among the pool
            pairs = a_idx * NUM_CLASSES + b_idx
            uniq, cnt = torch.unique(pairs, return_counts=True)
            order = torch.argsort(cnt, descending=True)
            dirs = []
            for k in order[:args.n_pairs]:
                a = int(uniq[k].item()) // NUM_CLASSES; b = int(uniq[k].item()) % NUM_CLASSES
                dvec = (W0c[a] - W0c[b])
                if dvec.norm().item() > 1e-8:
                    dirs.append(dvec / (dvec.norm() + 1e-8))
            if len(dirs) > 0:
                dict_elems['conf_pair'] = torch.stack(dirs, dim=0).t().float()

        if 'class_disp' in dict_names:
            # per-class pseudo-label mean shifts vs pool mean
            dirs = []
            for c in range(NUM_CLASSES):
                m = (pseudo == c)
                if int(m.sum().item()) > 100:
                    dvec = Xp[m].mean(dim=0) - pool_mean
                    if dvec.norm().item() > 1e-8:
                        dirs.append(dvec / (dvec.norm() + 1e-8))
            if len(dirs) > 0:
                dict_elems['class_disp'] = torch.stack(dirs, dim=0).t().float()

        if 'tta_disp' in dict_names:
            # disagreement displacement: mean of high-TTA-variance points vs pool mean
            n_cand = min(8000, n)
            cand = torch.argsort(margin)[:n_cand]
            Xcand = Xp[cand].float()
            draws = []
            for _ in range(args.tta_augs):
                torch.manual_seed(100 + _)
                flip = torch.rand_like(Xcand) < 0.02
                draws.append(torch.softmax(torch.where(flip, -Xcand, Xcand) @ W0c, dim=1))
            tta_var = torch.stack(draws).var(dim=0).mean(dim=1)
            hi = cand[torch.argsort(tta_var, descending=True)[:int(0.02 * n_cand)]]
            dvec = Xp[hi].mean(dim=0) - pool_mean
            dict_elems['tta_disp'] = dvec / (dvec.norm() + 1e-8)

        # ---- orthonormalize each element + the full dictionary ----
        elem_q = {k: orthonormal(D, device).cpu() for k, D in dict_elems.items()}
        D_full = torch.cat([Dq for Dq in elem_q.values()], dim=1)
        D_full_q = orthonormal(D_full, device).cpu()

        # ---- A. BASIS QUALITY ----
        span = {}
        for k, Dq in elem_q.items():
            span[k] = span_capture(Dq, R_flat)
        span['full'] = span_capture(D_full_q, R_flat)

        # oracle coefficients: W1 = W0 + P_span(D) R (the best classifier in span)
        R_b = R_flat.view(-1, NUM_CLASSES)
        coef_oracle = D_full_q.t() @ R_b
        W_or = W0c + (D_full_q @ coef_oracle).float()
        gc_oracle_coef = (mw(W_or, Xv, vl) - refs['frozen']) / gap if gap > 1e-9 else None

        # ---- B. SELECTION (dictionary fixed, only label->coefficient varies) ----
        # acquisition: margin_tta_div (Iteration-1 winner)
        cand_full = torch.argsort(margin)[:max(int(args.cand_frac * n), 8 * bmax)]
        cand_margin = margin[cand_full]
        Xcand = Xp[cand_full].float()
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xcand) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xcand, Xcand) @ W0c, dim=1))
        tta_var = torch.stack(draws).var(dim=0).mean(dim=1)
        m = cand_margin / (cand_margin.max() + 1e-8)
        v = tta_var / (tta_var.max() + 1e-8)
        topM = torch.argsort(-m + v, descending=True)[:8 * bmax]
        sel = farthest_point(pool_f, cand_full[topM], bmax, device).long()

        Dq_gpu = D_full_q.float().to(device)
        X_lab_cpu = Xp[sel].float()
        y_lab = pl[sel]
        X_lab_gpu = X_lab_cpu.to(device)

        sel_res = {'refs': refs, 'gap': float(gap),
                   'span_capture': span,
                   'oracle_coef': {'gc': gc_oracle_coef,
                                   'align_full': span['full']},
                   'budgets': {}}
        for b in budgets:
            X_lab = X_lab_gpu[:b]; Y_lab = onehot(y_lab[:b], NUM_CLASSES).float().to(device)
            resid_lab = Y_lab - X_lab @ W0c.to(device)
            entry = {}
            # oracle-U reference (Iteration-1 bound) with the SAME few labels:
            # first-order with U = top-2 of R. This is the +0.29-0.37 gc result.
            U_or_gpu = U_or.float().to(device)
            G_or = (X_lab @ U_or_gpu).t() @ resid_lab
            for rho in rho_sweep:
                G_or_n = G_or / (G_or.norm() + 1e-8)
                W1 = W0c + (U_or_gpu @ (rho * G_or_n)).cpu().float()
                gc = (mw(W1, Xv, vl) - refs['frozen']) / gap if gap > 1e-9 else None
                entry.setdefault('oracle_U', {})[str(rho)] = None if gc is None else float(gc)
            # firstorder: the Iteration-1 rule with D in place of U
            G = (X_lab @ Dq_gpu).t() @ resid_lab
            for rho in rho_sweep:
                G_n = G / (G.norm() + 1e-8)
                W1 = W0c + (Dq_gpu @ (rho * G_n)).cpu().float()
                gc = (mw(W1, Xv, vl) - refs['frozen']) / gap if gap > 1e-9 else None
                entry.setdefault('firstorder', {})[str(rho)] = None if gc is None else float(gc)
            # lsq: direct few-label coefficient fit (scale-adaptive ridge)
            A = X_lab @ Dq_gpu                       # b x K
            lam_ridge = 1e-2 * torch.diag(A.t() @ A).max().item() + 1e-6
            Kc = A.t() @ A + lam_ridge * torch.eye(A.shape[1], device=device)
            C = torch.linalg.solve(Kc, A.t() @ resid_lab)
            W1 = W0c + (Dq_gpu @ C).cpu().float()
            gc = (mw(W1, Xv, vl) - refs['frozen']) / gap if gap > 1e-9 else None
            entry['lsq'] = None if gc is None else float(gc)
            sel_res['budgets'][str(b)] = entry

        results['conds'][cond] = sel_res
        del Xp, Xv, Ws, R, pool_f, D_full_q, Dq_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print("    span_capture: " + " ".join(f"{k}:{v:.3f}" for k, v in span.items()))
        print(f"    oracle_coef gc {gc_oracle_coef:+.3f} (dict span {span['full']:.3f})")
        for b in budgets:
            e = sel_res['budgets'][str(b)]
            ou = " ".join(f"rho{k}:{v:+.3f}" for k, v in e['oracle_U'].items())
            fo = " ".join(f"rho{k}:{v:+.3f}" for k, v in e['firstorder'].items())
            print(f"    b{b}: oracle_U {ou} | firstorder {fo} | lsq {e['lsq']:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. BASIS: span_capture(full) is the decisive number -- does the pool")
    print("   dictionary span R? ~0 => P1 dead at the basis (no pool structure")
    print("   contains R). oracle_coef gc vs per-budget oracle_U gc: is the")
    print("   dictionary as good as R's top-2 (perfect coefficients)?")
    print("B. SELECTION (SAME few labels, dictionary fixed): firstorder / lsq gc vs")
    print("   oracle_U gc -- can few labels do as well with the pool basis as with")
    print("   the true residual basis? And vs oracle_coef gc (perfect selection):")
    print("   selection ~ oracle_coef => P1 fully works; selection << oracle_coef")
    print("   => labels can't select even a good basis (coefficient half NOT easy")
    print("   for non-oracle bases).")


if __name__ == "__main__":
    main()
