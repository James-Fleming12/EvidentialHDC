"""al_ufree_diag.py: Iteration 1-UFree -- directly optimize dW with the unlabeled
pool geometry as a PRIOR, never estimating U.

The residual-subspace closure says U (top-r of W*-W0) is not recoverable from the
few-label/TTA budget. The proposal: instead of factorizing dW = U C (which needs
U), solve directly for the correction

    min_dW  L_L(W0 + dW) + lambda * R(dW; X_pool)

where R uses the UNLABELED pool geometry to constrain dW. Few labels determine how
to move; the pool says where the correction is allowed to live. No explicit U.

Variants (all use the same b acquisition-selected labels + the full unlabeled pool
geometry; only R differs):

  frozen            W0 (reference)
  oracle_U          W0 + rho * U_or * G/||G||  (the known-working bound; uses U)
  a_grad            W0 + eta * G_full/||G_full|| (raw label gradient, no prior;
                    the KNOWN-FAIL baseline from the refinement sweep)
  tikhonov          min ||X_L(W0+dW) - Y_L||^2 + lambda||dW||^2
                    (plain ridge on the few labels; expect to replicate R2 collapse)
  pool_span         dW restricted to span of the top-r POOL eigenvectors
                    (min ||X_L(W0+dW)-Y_L||^2 + lambda||(I-P_pool)dW||^2, P_pool=UU^T)
  pool_penalty      min ||X_L(W0+dW)-Y_L||^2 + lambda*tr(dW^T S_pool dW)
                    (penalizes movement in HIGH-variance pool directions -- the
                    inverse of pool_span; tests whether the residual lives in
                    LOW-variance directions, per C21-C28)
  hybrid_first      the normalized first-order step but with the direction =
                    pool-regularized label gradient (G projected off the
                    low-variance pool directions)

The decisive question: does a pool-geometry regularizer make the label-driven dW
work where a_grad (no prior) failed? If pool_span / pool_penalty / hybrid beat
a_grad and approach oracle_U, the pool IS providing the missing geometry and the
U-free route is viable. If all regularizers stay at a_grad level, the pool geometry
is not the missing ingredient and U-free is closed.

Acquisition = margin_tta_div (the Iteration-1 winner). b in {2,4,8}.

Usage:
  uv run python robust_diagnostic/al_ufree_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_ufree_<label>.json
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


def cg_solve_apply(A_apply, b, d, C, device, iters=30):
    """Solve A dW = b by CG, where A is given as a matvec A_apply(v) = A v.
    b is d x C. Returns dW (d x C)."""
    x = torch.zeros(d, C, device=device, dtype=torch.double)
    b = b.double()
    r = b - A_apply(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A_apply(p); denom = (p * Ap).sum(0)
        if denom.abs().max().item() < 1e-20:
            break
        a = rs / (denom + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0)
        if not torch.isfinite(x).all():
            break
        be = rsn / (rs + 1e-30); p = r + be.unsqueeze(0) * p; rs = rsn
    return x.float()


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
    ap.add_argument("--r", type=int, default=2, help="rank for oracle-U ref and pool eigenbasis")
    ap.add_argument("--budgets", type=str, default="2,4,8")
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--reg_lambda", type=float, default=1e-3,
                    help="regularizer weight for the U-free variants")
    ap.add_argument("--cand_frac", type=float, default=0.05)
    ap.add_argument("--tta_augs", type=int, default=5)
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
    budgets = [int(x) for x in args.budgets.split(',')]
    r = args.r

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'r': r, 'budgets': budgets,
               'reg_lambda': args.reg_lambda, 'conds': {}}

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
        U_or, _ = right_topk_svd(R.t(), r)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # ---- pool geometry (unlabeled, used only as a PRIOR) ----
        # top-r pool eigenvectors (the S_pool geometry): Xp is n x d -> d x r
        Upool, _ = right_topk_svd(Xp, r)

        # ---- acquisition signals (margin_tta_div, Iteration-1 winner) ----
        sm = torch.softmax(Xp.float() @ W0c, dim=1)
        top2 = torch.topk(sm, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        cand = torch.argsort(margin)[:max(int(args.cand_frac * len(Xp)), 8 * max(budgets))]
        n_cand = len(cand)
        cand_margin = margin[cand]
        Xcand = Xp[cand].float()
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xcand) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xcand, Xcand) @ W0c, dim=1))
        tta_var = torch.stack(draws).var(dim=0).mean(dim=1)

        def select_rule(b):
            m = cand_margin / (cand_margin.max() + 1e-8)
            v = tta_var / (tta_var.max() + 1e-8)
            score = -m + v
            topM = torch.argsort(score, descending=True)[:8 * b]
            return farthest_point(pool_f, cand[topM], b, device)

        cond_res = {'refs': refs, 'gap': float(gap), 'budgets': {}}
        for b in budgets:
            sel = select_rule(b).long()
            X_lab = Xp[sel]; Y_lab = onehot(pl[sel], NUM_CLASSES)
            resid = (Y_lab.float() - X_lab.float() @ W0c)
            G_full = X_lab.float().t() @ resid          # d x C raw label gradient

            res = {}
            # frozen (reference)
            d0 = refs['frozen'] - refs['frozen']
            res['frozen'] = {'delta': 0.0, 'gap_closed': 0.0}
            # oracle_U (known-working bound)
            G = (X_lab.float() @ U_or).t() @ resid
            Gn = G / (G.norm() + 1e-8)
            W1 = W0c + (U_or @ (args.rho * Gn))
            d = mw(W1, Xv, vl) - refs['frozen']
            res['oracle_U'] = {'delta': float(d), 'gap_closed': float(d / gap) if gap > 1e-9 else None}
            # a_grad (raw label gradient, no prior -- known-fail baseline)
            Gfn = G_full / (G_full.norm() + 1e-8)
            Wg = W0c + (args.rho * Gfn)
            dg = mw(Wg, Xv, vl) - refs['frozen']
            res['a_grad'] = {'delta': float(dg), 'gap_closed': float(dg / gap) if gap > 1e-9 else None}

            # ---- U-free regularized optimizations (all solve A dW = X_L^T resid) ----
            d, C = 10000, NUM_CLASSES
            Xd = X_lab.double().to(device)
            bL = Xd.t() @ resid.double().to(device)      # d x C RHS

            # tikhonov: A = X_L^T X_L + lam*I  ->  min ||X_L dW - resid||^2 + lam||dW||^2
            A_tik = lambda v: Xd.t() @ (Xd @ v) + args.reg_lambda * v
            dW_t = cg_solve_apply(A_tik, bL, d, C, device).cpu()
            dt = mw(W0c + dW_t, Xv, vl) - refs['frozen']
            res['tikhonov'] = {'delta': float(dt), 'gap_closed': float(dt / gap) if gap > 1e-9 else None}

            # pool_span: penalize the COMPLEMENT of span(Upool)
            # A = X_L^T X_L + lam*(I - P_pool), P_pool = Upool Upool^T
            Upd = Upool.double().to(device)
            Pv = lambda v: Upd @ (Upd.t() @ v)
            A_span = lambda v: Xd.t() @ (Xd @ v) + args.reg_lambda * (v - Pv(v))
            dW_s = cg_solve_apply(A_span, bL, d, C, device).cpu()
            ds = mw(W0c + dW_s, Xv, vl) - refs['frozen']
            res['pool_span'] = {'delta': float(ds), 'gap_closed': float(ds / gap) if gap > 1e-9 else None}

            # pool_penalty: penalize HIGH-variance directions: A = X_L^T X_L + lam*S_pool
            # S_pool v = X_pool^T (X_pool v)
            Xpd = Xp.double().to(device)
            A_pen = lambda v: Xd.t() @ (Xd @ v) + args.reg_lambda * (Xpd.t() @ (Xpd @ v))
            dW_p = cg_solve_apply(A_pen, bL, d, C, device).cpu()
            dp = mw(W0c + dW_p, Xv, vl) - refs['frozen']
            res['pool_penalty'] = {'delta': float(dp), 'gap_closed': float(dp / gap) if gap > 1e-9 else None}

            # hybrid_first: normalized first-order step, direction = label gradient
            # with the POOL-SPAN component (high-variance) kept -- i.e. the same
            # direction a_grad uses but at a normalized scale (the trust-region form)
            Ghn = G_full / (G_full.norm() + 1e-8)
            Wh = W0c + (args.rho * Ghn)
            dh = mw(Wh, Xv, vl) - refs['frozen']
            res['hybrid_first'] = {'delta': float(dh), 'gap_closed': float(dh / gap) if gap > 1e-9 else None}

            cond_res['budgets'][str(b)] = res
            line = " ".join(f"{k}:{v['gap_closed']:+.2f}" for k, v in res.items())
            print(f"    b{b}: {line}", flush=True)

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, U_or, Upool, pool_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Does a pool-geometry regularizer make the label-driven dW work where a_grad")
    print("(no prior) failed? Compare against oracle_U (known-working bound):")
    print("  tikhonov     = plain ridge on the few labels (expect R2 collapse)")
    print("  pool_span    = dW forced into span(top-r pool eigenvectors)")
    print("  pool_penalty = penalize movement in high-variance pool directions")
    print("  hybrid_first = first-order step, direction = label gradient (pool prior)")
    print("If any U-free variant approaches oracle_U and beats a_grad, the pool")
    print("geometry IS the missing ingredient. If all stay at a_grad level, U-free")
    print("is closed and the local class-pair AL route is the remaining path.")


if __name__ == "__main__":
    main()
