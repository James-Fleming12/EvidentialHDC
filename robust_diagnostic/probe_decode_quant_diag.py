"""probe_decode_quant_diag.py: inference-side efficiency of the R4 linear probe
decode -- is there a gap, and does W quantization close it without losing ceiling?

Current decode: codes @ W with W (10000x17) fp32, ~0.22-0.23s per 100k pts
(~4.4e5 pts/s, already ~1.5x faster than the prototype decode). The HDC codes are
+-1 (binary), so the matmul can be an integer dot product if W is quantized. The
old block-ridge work measured +-1 W decode at prototype-class speed but lost
~2-4 points ceiling. The untried middle is int8 W (keeps accuracy, removes the
fp32 matmul).

Methods compared (per condition, frozen cov-shift extractor):
  fp32 : codes @ W_fp32            (the current decode)
  int8 : codes @ W_int8 via integer matmul (W quantized per-channel, dequant)
  pm1  : W binarized to +-1 (the old block-sign form)
  lowrank : codes @ (W0 + U8 C) factored as codes@W0 + (codes@U8)@C  -- smaller
            matmul (10k x 8 instead of 10k x 17 for the delta part)
Each reports decode time (s per 100k pts), pts/s, and ceiling mIoU delta vs fp32.

Usage:
  uv run python robust_diagnostic/probe_decode_quant_diag.py \
    --path_b <ckpt> --method_b <method> --label decode_quant_ep10 \
    --out robust_diagnostic/logs/probe_decode_quant_ep10.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, NUM_CLASSES, CONDS_ALL)
from robust_diagnostic.al_per_class_diag import ConfMatrix
from modules.oracle_core import get_hdc_projection

def decode_fp32(codes, W, device, chunk=100000):
    Wd = W.float().to(device)
    preds = []
    for s in range(0, len(codes), chunk):
        e = min(s + chunk, len(codes))
        preds.append((codes[s:e].to(device) @ Wd).argmax(1).cpu())
    return torch.cat(preds)

def decode_int8(codes, W, device, chunk=100000):
    """W int8 quantized per output column (scale per class); codes stay +-1 fp32
    but the matmul runs in int8 where possible. Keep it simple: quantize W to
    int8 with a per-class scale, matmul in fp32 with int8 W (no float W)."""
    Wd = W.float()
    amax = Wd.abs().amax(dim=0, keepdim=True).clamp(min=1e-8)
    Wq = (Wd / amax * 127).round().to(torch.int8)
    scale = (amax / 127).to(device)
    preds = []
    for s in range(0, len(codes), chunk):
        e = min(s + chunk, len(codes))
        sim = (codes[s:e].to(device) @ Wq.to(device).float()) * scale
        preds.append(sim.argmax(1).cpu())
    return torch.cat(preds)

def decode_pm1(codes, W, device, chunk=100000):
    """W binarized to +-1 (the old block-sign form): integer dot product."""
    Wd = W.float()
    Wp = torch.where(Wd >= 0, 1.0, -1.0).to(device)
    preds = []
    for s in range(0, len(codes), chunk):
        e = min(s + chunk, len(codes))
        preds.append((codes[s:e].to(device) @ Wp).argmax(1).cpu())
    return torch.cat(preds)

def decode_lowrank(codes, W0, U8, C, device, chunk=100000):
    """W = W0 + U8 C factored: codes@W0 + (codes@U8)@C (delta matmul is 10k x 8)."""
    W0d = W0.float().to(device); U8d = U8.float().to(device); Cd = C.float().to(device)
    preds = []
    for s in range(0, len(codes), chunk):
        e = min(s + chunk, len(codes))
        c = codes[s:e].to(device)
        sim = c @ W0d + (c @ U8d) @ Cd
        preds.append(sim.argmax(1).cpu())
    return torch.cat(preds)

def bench(name, fn, codes, vl, device):
    t0 = time.time()
    preds = fn()
    dt = time.time() - t0
    cm = ConfMatrix(); cm.update(preds, vl)
    n = len(codes)
    print(f"  {name:<10s} ceiling {cm.miou():.3f} | {dt:.3f}s / {n//1000}k pts "
          f"= {n/dt/1e5:.1f}e5 pts/s")
    return {'miou': cm.miou(), 'decode_s': dt, 'pts_s': n / dt if dt > 0 else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--val_cap", type=int, default=200000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--conds", type=str, default="fog,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="decode_quant_ep10")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = __import__('modules.gen_trainers', fromlist=['GenTrainer']).GenTrainer(
        ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)

    results = {'label': args.label, 'conds': {}}
    for cond in conds:
        t0 = time.time()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        cparser = build_parser(cdir, DATA, ARCH)
        print(f"=== {cond} ===", flush=True)
        pf, pl, _ = reservoir_collect(stream_frames(model, cparser, device, args.max_frames,
                                                    progress=cond), args.pool_cap, 42)
        vf, vl, _ = reservoir_collect(stream_frames(model, cparser, device, args.max_frames,
                                                    progress=cond), args.val_cap, 43)
        X = torch.sign(pf.to(device) @ proj).float()
        Xv = torch.sign(vf.to(device) @ proj).float()
        Y = onehot(pl, NUM_CLASSES).to(device)
        W = ridge_fit_exact(X, Y, args.lam, device).cpu()

        # low-rank factor of W for the factored decode
        R = W - W.mean(0)
        U8 = torch.linalg.svd(R.double(), full_matrices=False)[0][:, :8].float()
        C = U8.t() @ R
        W0r = W - U8 @ C

        r = {}
        r['fp32'] = bench('fp32', lambda: decode_fp32(Xv, W, device), Xv, vl.cpu(), device)
        r['int8'] = bench('int8', lambda: decode_int8(Xv, W, device), Xv, vl.cpu(), device)
        r['pm1'] = bench('pm1', lambda: decode_pm1(Xv, W, device), Xv, vl.cpu(), device)
        r['lowrank'] = bench('lowrank', lambda: decode_lowrank(Xv, W0r, U8, C, device),
                             Xv, vl.cpu(), device)
        results['conds'][cond] = r
        print(f"  ({time.time()-t0:.0f}s total)")
        del pf, pl, vf, vl, X, Xv, W
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
