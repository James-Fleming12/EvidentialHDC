"""probe_spectral_trunc_diag.py: exact-solve accuracy at CG-class cost for the R4
ridge probe -- the untried middle ground in the probe-efficiency thread.

The current tradeoff is: full ridge (S + lI)^-1 T via dense eigh/solve (~4s, the
ceiling 0.671 wet) vs the Nystrom-warm-started CG-8 (~0.034s but under-converges,
0.616 wet, -0.055 below exact). The proposed middle is a TRUNCATED spectral
factorization: compute the top-K eigendirections of S matrix-free (Lanczos /
randomized SVD on Sv = X^T(X v)), then apply the (1/(lambda + sigma)) filter as a
scalar on the returned eigenvalues -- exact-solve accuracy without forming the
full d x d inverse and without CG's under-convergence.

Methods compared (per condition, full-dataset pool reservoir):
  full_eigh   : dense eigh of S (reference ceiling), fit time measured
  full_solve  : dense solve (what ridge_fit_exact does), time measured
  cg8         : Nystrom-warm CG-8  (the documented 0.034s fast path)
  cg20        : Nystrom-warm CG-20 (more accurate, ~0.08s)
  rsvd_K      : randomized SVD top-K (K = 100/500/1000) of S via Sv = X^T(Xv),
                then per-class spectral filter
  lanczos_K   : Lanczos top-K, same filter (if torch has linalg.eigvalsh-based
                iteration available; falls back to rsvd only otherwise)

Reported per method: fit time, ceiling mIoU, delta vs full_eigh.

Usage:
  uv run python robust_diagnostic/probe_spectral_trunc_diag.py \
    --path_b <ckpt> --method_b <method> --label spec_trunc_ep10 \
    --out robust_diagnostic/logs/probe_spectral_trunc_ep10.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, NUM_CLASSES, CONDS_ALL)
from robust_diagnostic.al_per_class_diag import ConfMatrix

def decode_miou(codes, W, vl, device, chunk=100000):
    cm = ConfMatrix()
    Wd = W.to(device)
    for s in range(0, len(codes), chunk):
        e = min(s + chunk, len(codes))
        cm.update((codes[s:e].to(device) @ Wd).argmax(1).cpu(), vl[s:e].cpu())
    return cm.miou()

def randomized_topk_S(X, K, iters=3, seed=42, device='cuda'):
    """Randomized SVD of S = X^T X (matrix-free, via Sv = X^T(X v)): returns
    top-K eigenvalues (ascending) and eigenvectors of S."""
    g = torch.Generator(device=device).manual_seed(seed)
    d = X.shape[1]
    Omega = torch.randn(d, K + 8, generator=g, device=device)
    def apply_S(v):
        return X.t() @ (X @ v)
    Y = apply_S(Omega)
    for _ in range(iters):
        Y = apply_S(Y)
        Y, _ = torch.linalg.qr(Y)
    Q, _ = torch.linalg.qr(Y)
    T = Q.t() @ apply_S(Q)
    evals, evecs = torch.linalg.eigh(T)
    # T is (K+8)x(K+8); map back to the d-dim basis
    eigvecs = Q @ evecs
    return evals, eigvecs

def spectral_filter_fit(X, Y, lam, evals, eigvecs, device, top_k=None):
    """W = V diag(1/(lambda + sigma)) V^T T, using the (possibly truncated)
    spectral factors. evals ascending; only the top `top_k` are kept (the rest
    get the frozen / max filter so tail directions are not amplified)."""
    d = X.shape[1]; nc = Y.shape[1]
    T = (X.t() @ Y.to(device)).double()
    if top_k is not None and top_k < d:
        # keep the top-k largest sigma; tail directions get filter 1/(lam + sigma_max)
        e = evals.double(); v = eigvecs.double()
        tail = e[-1]
        filt = 1.0 / (lam + e)
        filt[:d - top_k] = 1.0 / (lam + tail)
        W = v @ (filt.unsqueeze(1) * (v.t() @ T))
    else:
        filt = 1.0 / (lam + evals.double())
        W = eigvecs.double() @ (filt.unsqueeze(1) * (eigvecs.double().t() @ T))
    return W.float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = ALL frames of seq 08")
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--val_cap", type=int, default=200000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--ks", type=str, default="100,500,1000",
                    help="truncation ranks to test (randomized top-K)")
    ap.add_argument("--conds", type=str, default="wet_ground,fog")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="spec_trunc_ep10")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    ks = [int(x) for x in args.ks.split(',')]

    trainer = __import__('modules.gen_trainers', fromlist=['GenTrainer']).GenTrainer(
        ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model

    results = {'label': args.label, 'conds': {}}
    for cond in conds:
        t0 = time.time()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        cparser = build_parser(cdir, DATA, ARCH)
        pf, pl, _ = reservoir_collect(stream_frames(model, cparser, device, args.max_frames),
                                      args.pool_cap, 42)
        vf, vl, _ = reservoir_collect(stream_frames(model, cparser, device, args.max_frames),
                                      args.val_cap, 43)
        from modules.oracle_core import get_hdc_projection
        proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
        X = torch.sign(pf.to(device) @ proj).float()
        Xv = torch.sign(vf.to(device) @ proj).float()
        Y = onehot(pl, NUM_CLASSES).to(device)
        T = X.t() @ Y

        r = {}

        # references: dense eigh and dense solve (times + ceilings)
        S = X.t() @ X
        t_eig = time.time()
        evals, evecs = torch.linalg.eigh(S.double())
        r['eigh_s'] = time.time() - t_eig
        W_eigh = spectral_filter_fit(X, Y, args.lam, evals, evecs, device)
        r['full_eigh'] = decode_miou(Xv, W_eigh, vl, device)
        t_solve = time.time()
        W_full = torch.linalg.solve(S.double() + args.lam * torch.eye(10000, device=device, dtype=torch.float64), T.double()).float()
        r['solve_s'] = time.time() - t_solve
        r['full_solve'] = decode_miou(Xv, W_full, vl, device)
        print(f"=== {cond} ===")
        print(f"  full eigh {r['full_eigh']:.3f} ({r['eigh_s']:.2f}s) | full solve "
              f"{r['full_solve']:.3f} ({r['solve_s']:.2f}s)")

        # Nystrom-warm CG (the documented fast path)
        from robust_diagnostic.probe_cg_speedup_diag import cg_solve, nystrom_w0
        W_ny = nystrom_w0(pf.cpu(), pl.cpu(), args.lam, device, 1000)
        for it in (8, 20):
            W, ts = cg_solve(X, T, args.lam, device, iters=it, x0=W_ny)
            r[f'cg{it}'] = decode_miou(Xv, W, vl, device)
            r[f'cg{it}_s'] = ts
            print(f"  nys-warm CG-{it} {r[f'cg{it}']:.3f} ({ts:.3f}s)")

        # randomized top-K spectral fits
        for K in ks:
            t_r = time.time()
            ev, evc = randomized_topk_S(X, K, iters=3, device=device)
            t_rsvd = time.time() - t_r
            Wk = spectral_filter_fit(X, Y, args.lam, ev, evc, device, top_k=K)
            r[f'rsvd{K}'] = decode_miou(Xv, Wk, vl, device)
            r[f'rsvd{K}_s'] = t_rsvd
            print(f"  rsvd-{K} {r[f'rsvd{K}']:.3f} ({t_rsvd:.2f}s, "
                  f"delta {r[f'rsvd{K}']-r['full_eigh']:+.3f})")
        results['conds'][cond] = r
        print(f"  ({time.time()-t0:.0f}s total)")
        del pf, pl, vf, vl, X, Xv, S, T
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
