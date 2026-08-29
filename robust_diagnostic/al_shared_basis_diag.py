"""al_shared_basis_diag.py: Iteration 1 -- do the residuals across conditions
share a usable structure?

The canonical adapter and the efficient bank both assume ONE low-rank structure
U0 serves ALL corruption conditions (R_c ~ U0 C_c for fog, crosstalk, snow,
wet_ground). We know each condition's residual is individually low-rank
(rank 4-5) but have NEVER measured whether the FOG residual and the CROSSTALK
residual live in the SAME directions or in different ones. This test measures the
property directly.

Key measurements, per condition c:
  R_c = W*_c - W0    (d x C: oracle pool-fit decoder minus the clean decoder)
  U_c = top-r left singular vectors of R_c        (the condition's own basis)
  U_pool = top-r left singular vectors of the POOLED residual
          [R_fog | R_crosstalk | R_snow | R_wet]  (the shared basis)

Metrics:
  1. per-condition capture of the POOLED basis:
        capture_pool(c) = ||U_pool^T R_c|| / ||R_c||
     -- if high (~0.9+) for every condition, one shared structure explains all.
  2. per-condition capture of its OWN top-r basis (the ceiling reference):
        capture_own(c) = ||U_c^T R_c|| / ||R_c||
     -- this is the max a rank-r basis can capture for that condition.
  3. ratio capture_pool / capture_own  (how much of "own" the pool explains)
  4. pairwise subspace agreement: cos(U_c, U_d) for every (c, d) -- the top-r
     direction overlap between conditions.

The answer: if capture_pool ~ capture_own on every condition (ratio ~1) AND
pairwise cos is high, the conditions share a basis and a single adapter is
well-posed. If some condition has low capture_pool (a "left-out" condition), the
shared-adapter assumption is violated for it -- the canonical adapter is
structurally impossible and the bank must be condition-specific.

Also reports the pooled singular values (how low-rank the shared structure is) and
the effective rank of the pooled residual.

Usage:
  uv run python robust_diagnostic/al_shared_basis_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_shared_basis_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection

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


def subspace_cos(U_a, U_b, r):
    uh = U_a[:, :r]; uo = U_b[:, :r]
    S = torch.linalg.svdvals((uh.t() @ uo).double())
    return float(S.mean().item())


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
    ap.add_argument("--max_clean", type=int, default=50000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--r_sweep", type=str, default="2,4,8")
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
    rmax = max(r_sweep)

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    # ---- W0: frozen clean decoder (shared across conditions) ----
    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- per-condition residuals R_c = W*_c - W0 ----
    residuals = {}       # cond -> (d x C) float
    refs = {}
    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        fd, ld = extract_clean(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(fd))
        pool, pl = fd[perm[:args.pool_size]], ld[perm[:args.pool_size]]
        val, vl = fd[perm[-args.val_size:]], ld[perm[-args.val_size:]]
        Xp = hdc_codes(pool, proj, device).float()
        Xv = hdc_codes(val, proj, device).float()
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R_c = (Ws - W0).detach().cpu().float()
        residuals[cond] = R_c
        refs[cond] = {'W0_frozen': None, 'Ws_oracle': None, 'n_pool': len(pool)}
        del pool, pl, val, vl, fd, ld, Xp, Xv, Ws
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [extract] {cond}: R_c shape {tuple(R_c.shape)} ({toc(t0):.0f}s)", flush=True)

    # ---- the pooled residual (concatenate across conditions) ----
    R_pool = torch.cat([residuals[c] for c in conds], dim=1)   # d x (n_cond * C)

    # ---- measurements ----
    out = {'label': args.label, 'method': args.method_b, 'r_sweep': r_sweep,
           'pooled_singular': None, 'effective_rank_pooled': None,
           'per_cond': {}, 'pairwise': {}}

    # pooled singular values (how low-rank the shared structure is)
    _, S_pool, _ = torch.linalg.svd(R_pool.double(), full_matrices=False)
    out['pooled_singular'] = [float(x) for x in S_pool[:rmax].tolist()]
    sv_all = S_pool.tolist()
    energy = sum(x * x for x in sv_all) + 1e-30
    cum = 0; eff = 0
    for x in sv_all:
        cum += x * x
        eff += 1
        if cum / energy > 0.9:
            break
    out['effective_rank_pooled'] = eff

    for r in r_sweep:
        U_pool, _ = right_topk_svd(R_pool.t(), r)      # R_pool is d x (n_cond*C) -> rows x d
        entry = {'capture_pool': {}, 'capture_own': {}, 'ratio': {}, 'U_pool_align_own': {}}
        for c in conds:
            R_c = residuals[c]
            U_c, _ = right_topk_svd(R_c.t(), r)        # R_c is d x C -> rows x d
            capture_pool = (U_pool.t() @ R_c.double()).norm().item() / (R_c.double().norm().item() + 1e-12)
            capture_own = (U_c.t() @ R_c.double()).norm().item() / (R_c.double().norm().item() + 1e-12)
            entry['capture_pool'][c] = float(capture_pool)
            entry['capture_own'][c] = float(capture_own)
            entry['ratio'][c] = float(capture_pool / (capture_own + 1e-12))
            entry['U_pool_align_own'][c] = subspace_cos(U_pool, U_c, r)
        out['per_cond'][str(r)] = entry

        # pairwise subspace agreement between conditions
        pair = {}
        for i, c1 in enumerate(conds):
            for c2 in conds[i + 1:]:
                U1, _ = right_topk_svd(residuals[c1].t(), r)
                U2, _ = right_topk_svd(residuals[c2].t(), r)
                pair[f'{c1}~{c2}'] = subspace_cos(U1, U2, r)
        out['pairwise'][str(r)] = pair

        print(f"\n=== rank r={r} ===")
        print("  per-condition capture: pool / own / ratio | pool-vs-own align")
        for c in conds:
            e = entry
            print(f"    {c:12s} capture_pool {e['capture_pool'][c]:.3f} | capture_own {e['capture_own'][c]:.3f} "
                  f"| ratio {e['ratio'][c]:.2f} | align_pool_own {e['U_pool_align_own'][c]:.2f}")
        print(f"  pairwise cos: " + " ".join(f"{k}:{v:.2f}" for k, v in pair.items()))

    print(f"\n=== pooled singular (top-{rmax}) ===")
    print(f"  {[round(x, 1) for x in out['pooled_singular']]}  effective_rank(90%) {out['effective_rank_pooled']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("The answer: do the residuals share a usable structure?")
    print("  PASS: capture_pool ~ capture_own (ratio ~1) on EVERY condition AND")
    print("        pairwise cos is high (~0.8+) -- one shared basis explains all")
    print("        conditions; a single adapter (canonical U0 / one bank U) is well-posed.")
    print("  FAIL: some condition has capture_pool << capture_own (a 'left-out'")
    print("        condition) or low pairwise cos -- the shared-adapter assumption")
    print("        is violated; the canonical adapter is structurally impossible and")
    print("        the bank must be condition-specific.")
    print("  effective_rank_pooled: how low-rank the shared structure is (the r to use).")


if __name__ == "__main__":
    main()
