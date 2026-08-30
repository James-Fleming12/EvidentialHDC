"""al_rank1b_diag.py: the rank-1-per-label decomposition, corrected for the two
confounds in Iteration 3 (al_rank1_diag.py).

Iteration 3's negative was confounded:
  1. D1 used cos of FLATTENED 170000-d u_i vs flattened R -- a concentration-of-
     measure trap (any two generic 170k-d vectors have cos ~ 0). This measures
     "does one label EQUAL the residual", which is trivially false. The right
     question is SPAN-CAPTURE: does the span of the b labels contain R?
  2. D2/D4 used eta * u_i with ||u_i|| ~ 96 (||x||=100 * ||r||~0.96), a 4.8-
     magnitude step across all of W0 -- pure overstepping, so all-negative deltas
     may be scale, not direction.

This version fixes both:
  NORMALIZED directions: u_i/||u_i||, so eta is a bounded trust radius per label.
  SPAN-CAPTURE alignment: for each k, does span(u_1..u_k) capture R?
        capture(k) = ||P_span_k R|| / ||R||  (the fraction of R in the label span)
  Also reports the oracle-U reference, the normalized aggregate, and per-label
  normalized deltas (at a bounded step).

The decisive questions:
  A. SPAN-CAPTURE: does the span of 8 labels capture a real fraction of R?
     If capture(8) is high (>0.5), the labels DO span the residual direction and
     the aggregate/sequential update was just mis-scaled. If capture(8) ~ 0, the
     labels do NOT span R -- the failure is that few labels can't span the
     residual, which is the real (method-independent) conclusion.
  B. NORMALIZED step: at a bounded trust radius, do individual / aggregate /
     sequential normalized updates help? This removes the overstepping confound.
  C. PER-LABEL value: with normalized directions, which labels are useful? Does a
     label-free score (d_conf) pick them (corr with delta)?

Acquisition: margin_tta_div (Iteration-1 winner), b in {2,4,8}.

Usage:
  uv run python robust_diagnostic/al_rank1b_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_rank1b_<label>.json
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


def span_capture(us, R, k=None):
    """Fraction of ||R|| captured by the span of the first k (or all) directions u_i.
    us: b x (d*C) flattened. R: (d*C) flattened."""
    n = len(us) if k is None else k
    if n == 0:
        return 0.0
    G = us[:n].double()
    Rn = R.double()
    # projector P = G^T (G G^T)^-1 G; capture = ||P R|| / ||R||
    try:
        P_R = G.t() @ torch.linalg.solve(G @ G.t() + 1e-8 * torch.eye(n, dtype=torch.double), G @ Rn)
        cap = (P_R.norm().item()) / (Rn.norm().item() + 1e-12)
    except Exception:
        cap = 0.0
    return cap


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
    ap.add_argument("--eta", type=float, default=0.5, help="bounded trust radius on NORMALIZED u_i")
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
    bmax = max(budgets)

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'budgets': budgets,
               'eta': args.eta, 'conds': {}}

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
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # ---- acquisition (margin_tta_div) ----
        sm = torch.softmax(Xp.float() @ W0c, dim=1)
        top2 = torch.topk(sm, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        cand = torch.argsort(margin)[:max(int(args.cand_frac * len(Xp)), 8 * bmax)]
        n_cand = len(cand)
        cand_margin = margin[cand]
        Xcand = Xp[cand].float()
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xcand) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xcand, Xcand) @ W0c, dim=1))
        tta_var = torch.stack(draws).var(dim=0).mean(dim=1)
        m = cand_margin / (cand_margin.max() + 1e-8)
        v = tta_var / (tta_var.max() + 1e-8)
        topM = torch.argsort(-m + v, descending=True)[:8 * bmax]
        sel = farthest_point(pool_f, cand[topM], bmax, device).long()
        X_lab = Xp[sel]; y_lab = pl[sel]
        p0 = torch.softmax(X_lab.float() @ W0c, dim=1)

        # ---- NORMALIZED rank-1 directions u_i = (x_i r_i^T)/||x_i r_i^T|| ----
        us_n = []          # b x (d*C) normalized
        us_norm = []
        for i in range(len(sel)):
            r_i = (onehot(y_lab[i:i+1], NUM_CLASSES) - p0[i:i+1]).float()
            u_i = X_lab[i:i+1].float().t() @ r_i
            nrm = u_i.norm().clamp(min=1e-8)
            u_i_n = (u_i / nrm).flatten()
            us_n.append(u_i_n)
            us_norm.append(float(nrm.item()))
        U_n = torch.stack(us_n, dim=0)          # b x (d*C) unit directions

        # ---- A. SPAN-CAPTURE: does the label span capture R? ----
        capture = {}
        for b in budgets:
            capture[str(b)] = span_capture(U_n, R_flat, b)

        # ---- B. NORMALIZED step: bounded trust radius, no overstepping ----
        cond_res = {'refs': refs, 'gap': float(gap),
                    'span_capture': capture, 'budgets': {}}
        # oracle-U reference (the R5 bound)
        G = (X_lab.float() @ right_topk_svd(R.t(), 2)[0]).t() @ (onehot(y_lab, NUM_CLASSES).float() - p0)
        Gn = G / (G.norm() + 1e-8)
        W_or = W0c + (args.eta * Gn.reshape(10000, NUM_CLASSES))
        d_or = mw(W_or, Xv, vl) - refs['frozen']
        cond_res['oracle_U_delta'] = float(d_or)

        # aggregate of normalized directions: W0 + eta * mean(u_i) (bounded)
        U_agg = U_n[:bmax].mean(dim=0).reshape(10000, NUM_CLASSES)
        for b in budgets:
            res = {}
            # aggregate of the first b normalized directions
            U_ab = U_n[:b].mean(dim=0).reshape(10000, NUM_CLASSES)
            W_a = W0c + args.eta * U_ab
            da = mw(W_a, Xv, vl) - refs['frozen']
            res['aggregate_norm'] = {'delta': float(da),
                                     'gap_closed': float(da / gap) if gap > 1e-9 else None}
            # sequential: same as aggregate for fixed eta (linearity), shown for clarity
            # per-label normalized usefulness + label-free score
            deltas = []
            conf_ds = []
            for i in range(b):
                u_i = us_n[i].reshape(10000, NUM_CLASSES)
                W_i = W0c + args.eta * u_i
                d_i = mw(W_i, Xv, vl) - refs['frozen']
                deltas.append(d_i)
                sm1 = torch.softmax(Xp.float() @ W_i, dim=1)
                conf_ds.append(float(sm1.max(dim=1).values.mean().item()) - float(sm.max(dim=1).values.mean().item()))
            res['per_label'] = {'deltas': deltas, 'd_conf': conf_ds,
                                'n_positive': sum(1 for x in deltas if x > 0), 'b': b}
            try:
                res['per_label']['corr_dconf_delta'] = float(np.corrcoef(conf_ds, deltas)[0, 1])
            except Exception:
                res['per_label']['corr_dconf_delta'] = None
            # keep-only-oracle-good (normalized, bounded): best-case rejection
            W_k = W0c.clone()
            n_good = 0
            for i in range(b):
                if deltas[i] > 0:
                    W_k = W_k + args.eta * us_n[i].reshape(10000, NUM_CLASSES)
                    n_good += 1
            dk = mw(W_k, Xv, vl) - refs['frozen']
            res['keep_oracle_good'] = {'delta': float(dk),
                                       'gap_closed': float(dk / gap) if gap > 1e-9 else None,
                                       'n_good': n_good}
            cond_res['budgets'][str(b)] = res

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, pool_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    oracle_U_delta {d_or:+.2f}")
        print(f"    span_capture: " + " ".join(f"b{k}:{v:.3f}" for k, v in capture.items()))
        for b in budgets:
            r = cond_res['budgets'][str(b)]
            print(f"    b{b}: agg_norm {r['aggregate_norm']['gap_closed']:+.2f} | "
                  f"per-label +{r['per_label']['n_positive']}/{b} corr {r['per_label']['corr_dconf_delta']} | "
                  f"keep_good {r['keep_oracle_good']['gap_closed']:+.2f} ({r['keep_oracle_good']['n_good']} good)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. SPAN-CAPTURE (the decisive fix): does the span of the labels capture R?")
    print("   capture(b) > 0.5 -> labels DO span the residual; the Iteration-3 negative")
    print("   was mis-scaled (overstepping). capture(b) ~ 0 -> labels do NOT span R;")
    print("   the failure is that few labels can't span the residual (real).")
    print("B. NORMALIZED step (bounded): does the aggregate / per-label / keep-good")
    print("   update help at a trust radius (no overstepping)?")
    print("   keep_oracle_good positive while aggregate negative -> rejection is the")
    print("   lever (some labels are good, others poison).")
    print("   n_positive ~ b/2 + corr(d_conf, delta) high -> a label-free score can")
    print("   select the good updates.")


if __name__ == "__main__":
    main()
