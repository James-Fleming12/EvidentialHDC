"""al_residual_al_diag.py: C21 -- sparse-label estimation of the low-rank
residual correction W = W0 + U_r C.

C20 showed the oracle residual R = W* - W0 is low-rank (eff-rank 4-5, r=8
reproduces the oracle exactly on every extractor). C21 tests the AL side: can
a sparse label budget estimate the residual coefficients C (17 x r) well
enough to beat frozen -- and how does the low-rank correction compare to the
full-probe estimation (the Iterations-7/8 T-synthesis that failed)?

Two basis choices:
  A. ORACLE basis: U_r from the SVD of the true residual R = W* - W0. This is
     the CEILING of any basis-estimation method (the basis is given). Answers:
     "if we knew the right r directions, how many labels estimate C?"
  B. LABEL-ESTIMATED basis: U_r from the SVD of the residual estimated from a
     SUBSET of labels (fit W_sub on the labeled pool, R_sub = W_sub - W0).
     This is the deployable version (no oracle basis). Answers: "can labels
     discover the directions themselves?"

For each basis, per r in {1,2,4,8}, per budget k (labels/class):
  - C_fit = (U_r^T X_lab^T X_lab U_r)^-1 U_r^T X_lab^T (Y_lab - X_lab W0)
    i.e. ridge-free LSQ on the residual: labels predict the RESIDUAL of the
    frozen probe, not the full probe.
  - W = W0 + U_r C_fit, report mIoU, delta vs frozen, and fraction of the
    oracle gap recovered.
Also report the full-probe ridge with the same budget (the Iteration-7/8
comparison) and the oracle-ceiling residual curve (C20 numbers) as the bound.

Usage:
  uv run python robust_diagnostic/al_residual_al_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_residual_al_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11
RS = [1, 2, 4, 8]
KS = [2, 4, 8]

def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)

def extract_features(model, parser, device, num_frames=100):
    feats, lbls = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol)
            z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(zf.cpu())
            lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(lbls)

def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)

def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc)
    y[torch.arange(len(lbls)), lbls.long()] = 1
    return y

def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    p = []
    for s in range(0, len(codes), chunk):
        p.append((codes[s:s + chunk].float() @ W).argmax(1))
    return torch.cat(p)

def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)

def cos_sim(a, b):
    a = a.detach().cpu().float().reshape(-1)
    b = b.detach().cpu().float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))

def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device)
    torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1])
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P
    Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    That = XP.t() @ Yd
    x = P @ torch.linalg.solve(Shat, That)
    b = X.t() @ Yd

    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x)
    p = r.clone()
    rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p)
        a = rs / ((p * Ap).sum(0) + 1e-30)
        x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap
        rsn = (r * r).sum(0)
        be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p
        rs = rsn
    return x.float()

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def tic():
    sync()
    return time.time()

def toc(t0):
    sync()
    return time.time() - t0

def lsq_residual(X_lab, Y_lab, W0, U, device):
    """Fit C: (U^T X^T X U) C = U^T X^T (Y - X W0), with a small ridge."""
    Xd = X_lab.to(device).float()
    Yd = Y_lab.to(device).float()
    U_d = U.to(device)
    r = U_d.shape[1]
    XU = Xd @ U_d                       # n x r
    A = XU.t() @ XU + 1e-6 * torch.eye(r, device=device)
    b = XU.t() @ (Yd - Xd @ W0.to(device))
    C = torch.linalg.solve(A, b)        # r x 17
    return C.cpu()

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
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="med")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config))
    ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)

    results = {'label': args.label, 'method': args.method_b, 'rs': RS, 'ks': KS,
               'conds': {}}

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]

        proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
        Xc = hdc_codes(fa[ci], proj, device).float()
        Xp = hdc_codes(pool, proj, device).float()
        Xv = hdc_codes(val, proj, device).float()

        W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam,
                            args.cg_iters, args.nystrom_m, device)
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam,
                            args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_full, s_full, Vh = torch.linalg.svd(R.double(), full_matrices=False)
        U_full = U_full.float()

        r_cond = {'refs': {}, 'basis_oracle': {}, 'basis_est': {}, 'full_probe': {}}
        r_cond['refs']['frozen'] = mw(W0, Xv, vl)
        r_cond['refs']['oracle'] = mw(Ws, Xv, vl)

        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}

        for k in KS:
            # labeled subset: k random points per class (the cheap budget)
            lab_idx = []
            for c in classes:
                idx = cls_idx[c]
                if len(idx) < max(50, k):
                    continue
                torch.manual_seed(2)
                lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx:
                continue
            lab_idx = torch.cat(lab_idx)
            X_lab = Xp[lab_idx]
            Y_lab = onehot(pl[lab_idx], NUM_CLASSES)
            n_labels = len(lab_idx)

            # ---- basis A: oracle U_r ----
            ra = {'n_labels': n_labels}
            for r in RS:
                U = U_full[:, :r]
                C = lsq_residual(X_lab, Y_lab, W0, U, device)
                W = W0.detach().cpu() + (U @ C)
                ra[str(r)] = {'miou': mw(W, Xv, vl),
                              'delta': mw(W, Xv, vl) - r_cond['refs']['frozen']}
            r_cond['basis_oracle'][str(k)] = ra

            # ---- basis B: label-estimated U_r (fit W_sub on the labels) ----
            rb = {'n_labels': n_labels}
            W_sub = ridge_fit_soft(X_lab, Y_lab, args.lam, args.cg_iters,
                                   args.nystrom_m, device)
            R_sub = (W_sub - W0).detach().cpu().float()
            U_sub_full, _, _ = torch.linalg.svd(R_sub.double(), full_matrices=False)
            U_sub_full = U_sub_full.float()
            for r in RS:
                U = U_sub_full[:, :r]
                C = lsq_residual(X_lab, Y_lab, W0, U, device)
                W = W0.detach().cpu() + (U @ C)
                rb[str(r)] = {'miou': mw(W, Xv, vl),
                              'delta': mw(W, Xv, vl) - r_cond['refs']['frozen']}
            r_cond['basis_est'][str(k)] = rb

            # ---- full-probe ridge on the labels (Iteration-7/8 comparison) ----
            W_full = ridge_fit_soft(X_lab, Y_lab, args.lam, args.cg_iters,
                                    args.nystrom_m, device)
            r_cond['full_probe'][str(k)] = {'miou': mw(W_full, Xv, vl),
                                            'delta': mw(W_full, Xv, vl) - r_cond['refs']['frozen']}

        results['conds'][cond] = r_cond
        del Xc, Xp, Xv, W0, Ws, R, U_full, s_full, Vh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r_cond['refs']['frozen']:.3f} / oracle {r_cond['refs']['oracle']:.3f}")
        for k in KS:
            if str(k) not in r_cond['basis_oracle']:
                continue
            bo = r_cond['basis_oracle'][str(k)]
            be = r_cond['basis_est'][str(k)]
            fp = r_cond['full_probe'][str(k)]
            print(f"  k={k} ({bo['n_labels']} labels):")
            print(f"    oracle-basis: " + " ".join(
                f"r{r}:{bo[str(r)]['delta']:+.3f}" for r in RS))
            print(f"    est-basis:    " + " ".join(
                f"r{r}:{be[str(r)]['delta']:+.3f}" for r in RS))
            print(f"    full-probe:   {fp['delta']:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Compare the oracle-basis (ceiling) vs est-basis (deployable) deltas:")
    print("  - oracle-basis r=8 delta > 0 on the hard conditions -> the low-rank")
    print("    residual IS estimable from labels when the directions are known.")
    print("  - est-basis ~ oracle-basis -> labels can discover the directions too")
    print("    -> C21 is deployable as-is (W = W0 + U_hat_r C).")
    print("  - full_probe vs oracle-basis: does restricting to the residual")
    print("    subspace beat the full T_hat estimation (the Iterations-7/8 fail)?")

if __name__ == "__main__":
    main()
