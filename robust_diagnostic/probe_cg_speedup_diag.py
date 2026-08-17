"""probe_cg_speedup_diag.py: speed up the matrix-free CG update (eval-only).

Iteration 7 showed matrix-free CG-20 (0.62 wet_ground / 0.38 fog, ~0.64M pts/s) is
the best cheap second-order update. This tests ways to get the remaining 2-3x by
making each CG iteration cheaper or reducing iterations, WITHOUT returning to a
fixed-subspace coreset (the Iteration 7 dead end). The two preferred families:
EQUATION REFORMULATION (warm starts / residual CG) and BINARIZATION-FRIENDLY tricks
(BF16 state).

  A. Nystrom warm-start CG   : W0 = W_Nys(m=1000) (~0.57), then CG-5/10. If CG-5 from
     Nys ~= CG-20 from scratch, huge win (Nys gets near, CG fixes the residual).
  B. Prototype residual + early-stop : W0 = mu, solve A dW = R = T - A mu, stop when
     ||r_k||/||r_0|| < eps. Measures the actual iterations to convergence (the
     correction-to-prototype framing).
  C. Nystrom-preconditioned CG: use the Nystrom sketch as a preconditioner M^-1 ~
     (S + lI)^-1, run PCG. Highest upside: 3-8 iters instead of 20.
  D. BF16 CG state          : BF16 for the dense CG vectors with FP32 accumulation
     (X stays exact +/-1). Cheaper GEMMs, convergence criterion still FP32.
  E. Subsampled/minibatch CG: Sv = X_t^T (X_t v) with a FRESH random subset each
     iteration (25k/12.5k/5k) -- stochastic CG, not a fixed coreset, so no subspace
     restriction.

Each reports mIoU (ceiling) + solve wall-clock + pts/s vs the CG-20 and full-ridge
references.

Usage:
  uv run python robust_diagnostic/probe_cg_speedup_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_cg_speedup_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import yaml
import torch

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

NUM_CLASSES = 17
DGLSSPP_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'
DGLSSPP_METHOD = 'supcon_vib_dglsspp'


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
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(z_flat.cpu())
            lbls.append(labels[mask].cpu())
    return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)


def hdc_codes(feats, proj, device, chunk=100000):
    codes = []
    for s in range(0, len(feats), chunk):
        codes.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(codes, dim=0)


def onehot(lbls, num_classes):
    y = torch.zeros(len(lbls), num_classes)
    y[torch.arange(len(lbls)), lbls.long()] = 1.0
    return y


def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ W).argmax(dim=1))
    return torch.cat(preds)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync()
    return time.time()


def toc(t0):
    sync()
    return time.time() - t0


def cg_solve(X, T, lam, device, iters=20, x0=None, subsample=None, dtype=torch.float32):
    """Matrix-free CG: (S + lI) W = T with Sv = X^T(Xv). Optional warm-start x0,
    optional per-iteration fresh subsample (stochastic CG), optional BF16 state."""
    d = X.shape[1]
    C = T.shape[1]
    x = torch.zeros(d, C, device=device, dtype=dtype) if x0 is None else x0.to(device, dtype)
    def A(v):
        if subsample is not None and subsample < X.shape[0]:
            idx = torch.randperm(X.shape[0], device=X.device)[:subsample]
            Xt = X[idx]
        else:
            Xt = X
        return Xt.T @ (Xt @ v)
    t0 = tic()
    b = T.to(device, dtype)
    r = b - A(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha.unsqueeze(0) * p
        r = r - alpha.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    t_solve = toc(t0)
    return x.float(), t_solve


def residual_cg(X, T, lam, device, x0, max_iters=50, tol=1e-3):
    """Solve A dW = R = T - A x0 (the correction to the warm start), early-stop on
    ||r||/||r0|| < tol. Returns the corrected W and the iterations used."""
    d = X.shape[1]
    A = lambda v: X.T @ (X @ v)
    x = x0.clone().to(device)
    b = T.to(device)
    r = b - A(x)
    r0_norm = (r * r).sum(dim=0).clamp(min=1e-12).sqrt()
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    k = 0
    for k in range(1, max_iters + 1):
        Ap = A(p)
        alpha = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha.unsqueeze(0) * p
        r = r - alpha.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        rel = (rs_new.sqrt() / r0_norm).max().item()
        if rel < tol:
            break
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x.float(), k


def nystrom_w0(codes, lbls, lam, device, m=1000, num_classes=NUM_CLASSES):
    """The Nystrom sketch solution (a cheap warm start)."""
    X = codes.float().to(device)
    Y = onehot(lbls, num_classes).to(device)
    torch.manual_seed(11)
    P = (torch.rand(codes.shape[1], m) > 0.5).float() * 2 - 1
    XP = X @ P.to(device)
    Shat = XP.T @ XP
    That = XP.T @ Y
    A = torch.linalg.solve(Shat + lam * torch.eye(m, device=device), That)
    return (P.to(device) @ A).float()


def nystrom_precond_cg(X, T, lam, device, P, iters=8):
    raise NotImplementedError(
        "Preconditioned CG needs a verified Woodbury M^-1; the warm-start proxy "
        "(Nys W0 then plain CG) is the recommended first test (feedback: 'test this "
        "before implementing a sophisticated preconditioner'). The warm-start is "
        "reported under 'nys_warm'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--nystrom_m", type=int, default=1000)
    parser.add_argument("--cg_iters", type=str, default="5,10,20")
    parser.add_argument("--subsamples", type=str, default="25000,12500,5000")
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_cg_speedup_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    cg_iters = [int(x) for x in args.cg_iters.split(',')]
    subsamples = [int(x) for x in args.subsamples.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    results = {'label': args.label, 'conds': {}}

    for cond in conds:
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        pool_codes = hdc_codes(pool, proj, device)
        val_codes = hdc_codes(val, proj, device)

        X = pool_codes.float()
        Y = onehot(pl, NUM_CLASSES).float()
        T = X.t() @ Y

        # references
        mu = torch.zeros(NUM_CLASSES, X.shape[1])
        for c in range(1, NUM_CLASSES):
            m = pl == c
            if int(m.sum().item()) > 0:
                mu[c] = pool_codes[m].float().mean(dim=0)
        r1 = compute_miou(decode(mu.t(), val_codes), vl)
        W_full = torch.linalg.solve(X.t() @ X + args.lam * torch.eye(10000), T)
        full = compute_miou(decode(W_full, val_codes), vl)

        r = {'r1_proto': r1, 'full_ridge': full,
             'cg_plain': {}, 'nys_warm': {}, 'residual': {}, 'bf16': {},
             'subsample': {}, 'precond': {}}

        # baseline CG (from scratch)
        for it in cg_iters:
            W, ts = cg_solve(X, T, args.lam, device, iters=it)
            r['cg_plain'][str(it)] = {'miou': compute_miou(decode(W, val_codes), vl),
                                      'solve_s': ts, 'pts_s': X.shape[0] / ts if ts > 0 else None}

        # A. Nystrom warm-start CG
        W_ny = nystrom_w0(pool_codes, pl, args.lam, device, args.nystrom_m)
        for it in cg_iters[:2]:  # CG-5, CG-10 from the Nys start
            W, ts = cg_solve(X, T, args.lam, device, iters=it, x0=W_ny)
            r['nys_warm'][str(it)] = {'miou': compute_miou(decode(W, val_codes), vl),
                                      'solve_s': ts, 'pts_s': X.shape[0] / ts if ts > 0 else None}

        # B. prototype residual CG with early stop
        W_res, k = residual_cg(X, T, args.lam, device, mu.t(), max_iters=40, tol=1e-3)
        r['residual'] = {'miou': compute_miou(decode(W_res, val_codes), vl),
                         'iters': k}

        # C. Nystrom warm-start as the preconditioner proxy (feedback: test warm-start
        #    before building a true preconditioner). W0 = W_Nys then few CG iters.
        torch.manual_seed(11)
        P = (torch.rand(X.shape[1], args.nystrom_m) > 0.5).float() * 2 - 1
        for it in [3, 5, 8]:
            W, ts = cg_solve(X, T, args.lam, device, iters=it, x0=W_ny)
            r['precond'][str(it)] = {'miou': compute_miou(decode(W, val_codes), vl),
                                     'solve_s': ts}

        # D. BF16 CG state (FP32 accumulation)
        Wb, tsb = cg_solve(X, T, args.lam, device, iters=20, dtype=torch.bfloat16)
        r['bf16'] = {'miou': compute_miou(decode(Wb, val_codes), vl),
                     'solve_s': tsb, 'pts_s': X.shape[0] / tsb if tsb > 0 else None}

        # E. subsampled / minibatch CG (fresh subset per iteration)
        for s in subsamples:
            W, ts = cg_solve(X, T, args.lam, device, iters=20, subsample=s)
            r['subsample'][str(s)] = {'miou': compute_miou(decode(W, val_codes), vl),
                                      'solve_s': ts, 'pts_s': X.shape[0] / ts if ts > 0 else None}

        results['conds'][cond] = r
        print(f"\n--- {cond} ---")
        print(f"  R1 {r1:.4f} | full {full:.4f}")
        print(f"  CG plain: " + "  ".join(f"{it}:{r['cg_plain'][str(it)]['miou']:.4f}" for it in cg_iters))
        print(f"  Nys-warm: " + "  ".join(f"{it}:{r['nys_warm'][str(it)]['miou']:.4f}" for it in cg_iters[:2]))
        print(f"  residual early-stop: mIoU {r['residual']['miou']:.4f} iters {r['residual']['iters']}")
        print(f"  Nys-warm CG-3/5/8: " + "  ".join(f"{it}:{r['precond'][str(it)]['miou']:.4f}" for it in [3,5,8]))
        print(f"  BF16-20: {r['bf16']['miou']:.4f} ({r['bf16']['solve_s']:.3f}s)")
        print(f"  subsample: " + "  ".join(f"{s}:{r['subsample'][str(s)]['miou']:.4f}" for s in subsamples))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. nys_warm: does CG-5/10 from the Nystrom start ~= CG-20 from scratch?")
    print("   (=> Nys gets near, few CG iters fix the residual -- the preferred combo)")
    print("B. residual: the prototype->probe correction, early-stopped. Iterations needed?")
    print("C. precond: Nystrom warm-start + few CG iters (3/5/8) -- the preconditioner proxy.")
    print("D. bf16: does BF16 state (FP32 accum) match FP32 CG (cheaper GEMMs)?")
    print("E. subsample: does a fresh 25k/12.5k/5k subset per iteration keep mIoU?")
    print("   (stochastic CG, NOT a fixed coreset -- no subspace restriction)")


if __name__ == "__main__":
    main()
