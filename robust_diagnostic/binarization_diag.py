"""binarization_diag.py: diagnose what the current HDC sign-binarization loses on the
cov-shift extractor, and test alternative binarization / encoding schemes.

Iteration C6 found that on the HEALTHY conditions (snow, wet_ground) the cov-shift
extractor keeps the continuous class structure (LP kept, dir_ret kept) but loses the
binarized-space packing (corr_tight drops -> HDC-oracle drops). This diagnostic
answers two things on frozen features (no training):

  A. WHAT THE CURRENT BINARIZATION LOSES:
     - per-coordinate PRE-SIGN margin distribution: |z @ R| near 0 means the coordinate
       is threshold-hugging and flips sign on small feature noise. If the cov-shift
       healthy features have MORE coordinates near 0 than DGLSS++, that is the
       quantitative signature of the packing loss. (margin-frac near 0)
     - the continuous-to-binarized information retention: how much class-mean
       separation survives sign() vs the alternatives.

  B. WHAT A GOOD BINARIZATION WOULD NEED TO DO (tested on the same frozen features):
     1. sign(z @ R)            : current, threshold 0.
     2. sign(z @ R - b)        : per-coordinate bias b = median/mean of the CLEAN
                                 projection, so each coordinate binarizes around its
                                 clean-typical value, not an arbitrary 0.
     3. sign((z @ R - b)/s)    : per-coordinate bias AND scale (z-score the clean
                                 projection), making each coordinate equally weighted
                                 before binarization.
     4. fourier                : the OTHER standard HDC style -- encode as
                                 [cos(z @ w), sin(z @ w)] with random frequencies w
                                 (no sign, continuous codes). Tests whether a smooth
                                 periodic encoding retains the packing that sign()
                                 loses.

Reports, per condition and per encoding: the pre-sign margin fraction (near-0
coordinates), the frozen-prototype zs decode, and the oracle decode (re-estimate from
corrupted labeled points).

Run:
  uv run python robust_diagnostic/binarization_diag.py
      --path_b robust_diagnostic/logs/ep10_.../supcon_vib_dglsspp_inputin_in_chan
      --method_b supcon_vib_dglsspp_inputin_in_chan --label_b covshift_ep10
      --conds snow,wet_ground,fog,crosstalk
"""
import os
import sys
import json
import argparse
import yaml
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer

CONDS_DEFAULT = ['snow', 'wet_ground', 'fog', 'crosstalk']
NUM_CLASSES = 17
DIM_OUT = 10000
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

def per_class_iou(preds, lbls, classes):
    out = {}
    for c in classes:
        tp = int(((preds == c) & (lbls == c)).sum().item())
        fp = int(((preds == c) & (lbls != c)).sum().item())
        fn = int(((preds != c) & (lbls == c)).sum().item())
        denom = tp + fp + fn
        out[c] = tp / denom if denom > 0 else 0.0
    return out

def mean_iou(d, classes):
    vs = [d[c] for c in classes if d[c] == d[c]]
    return sum(vs) / len(vs) if vs else float('nan')

def encode(feats, proj, mode, bias=None, scale=None, device='cuda', chunk=100000):
    """Encode features into the HDC space, CHUNKED to bound memory (the clean pool is
    millions of points x 10000 dims; one matmul is hundreds of GB). proj is the
    projection/frequency matrix. Returns the code (n, code_dim) on cpu.
    mode:
      'sign'    : sign(z @ R), threshold 0
      'bias'    : sign(z @ R - b), b per-coordinate clean bias
      'zscore'  : sign((z @ R - b) / s), b and s per-coordinate clean stats
      'fourier' : [cos(z @ w), sin(z @ w)] with random frequencies w
    """
    out_chunks = []
    bias_d = bias.to(device) if bias is not None else None
    scale_d = scale.to(device) if scale is not None else None
    for i in range(0, len(feats), chunk):
        f = feats[i:i + chunk].to(device)
        p = f @ proj
        if mode == 'sign':
            out_chunks.append(torch.sign(p).cpu())
        elif mode == 'bias':
            out_chunks.append(torch.sign(p - bias_d).cpu())
        elif mode == 'zscore':
            out_chunks.append(torch.sign((p - bias_d) / (scale_d + 1e-6)).cpu())
        elif mode == 'fourier':
            out_chunks.append(torch.cat([torch.cos(p), torch.sin(p)], dim=1).cpu())
        else:
            raise ValueError(mode)
    return torch.cat(out_chunks, dim=0)

def build_protos_stream(feats, lbls, enc_fn, num_classes=17, chunk=200000, device='cuda'):
    """Build per-class mean codes from the (possibly huge) clean pool in a streaming
    pass: encode each chunk, accumulate per-class sums, then discard the chunk. This
    avoids ever materializing the full n x code_dim code (which is hundreds of GB for
    the 8M-point clean pool). enc_fn maps a chunk (n x d) -> codes (n x code_dim) on cpu."""
    code_dim = None
    protos = None
    counts = None
    for i in range(0, len(feats), chunk):
        codes = enc_fn(feats[i:i + chunk])            # (c, code_dim) on cpu
        cl = lbls[i:i + chunk]
        if protos is None:
            code_dim = codes.shape[1]
            protos = torch.zeros(num_classes, code_dim, device=codes.device)
            counts = torch.zeros(num_classes, device=codes.device)
        for c in range(num_classes):
            m = cl == c
            if m.sum() > 0:
                protos[c] += codes[m].sum(dim=0)
                counts[c] += m.sum()
        del codes
    for c in range(num_classes):
        if counts[c] > 0:
            protos[c] /= counts[c].clamp(min=1)
    valid = counts > 0
    base = F.normalize(protos[valid], p=2, dim=1)
    return base, torch.arange(num_classes, device=protos.device)[valid]

def decode(codes, protos, proto_lbls, device, chunk=50000):
    protos = F.normalize(protos, p=2, dim=1)
    preds = []
    protos_d = protos.to(device)
    for i in range(0, len(codes), chunk):
        c = codes[i:i + chunk].to(device)
        sims = c @ protos_d.T
        idx = sims.argmax(dim=1)
        preds.append(proto_lbls[idx.cpu()])
    return torch.cat(preds)

def margin_fraction(pre_proj, eps=0.1):
    """Fraction of coordinates with |pre-sign projection| < eps -- the
    threshold-hugging coordinates that flip sign on small noise."""
    p = pre_proj
    return float((p.abs() < eps).float().mean().item())

def margin_frac_chunked(feats, proj, eps=0.1, device='cuda', chunk=100000):
    """margin_fraction computed on the FULL clean pool without OOM (chunked)."""
    tot, near = 0.0, 0.0
    for i in range(0, len(feats), chunk):
        p = (feats[i:i + chunk].to(device) @ proj).cpu()
        tot += p.numel()
        near += int((p.abs() < eps).sum().item())
    return near / tot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label_b", type=str, default="covshift")
    parser.add_argument("--conds", type=str, default=",".join(CONDS_DEFAULT))
    parser.add_argument("--eps", type=float, default=0.1,
                        help="pre-sign margin threshold for the 'near-0 coordinate' metric")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/binarization_diag_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    eps = args.eps

    def load(path, method):
        return GenTrainer(ARCH, DATA, args.kitti_dir, path, path=path, method=method).model

    model_a = load(DGLSSPP_PATH, DGLSSPP_METHOD)
    model_b = load(args.path_b, args.method_b)

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model_a, clean_parser, device, args.frames)
    fb, lb = extract_features(model_b, clean_parser, device, args.frames)
    # Cap the clean pool: the full 8M-point set is unnecessary for prototypes and is
    # the main CPU-memory driver. Use a bounded sample for the clean prototypes too.
    MAX_CLEAN = 1000000
    ca_idx = torch.randperm(len(fa))[:MAX_CLEAN]
    fa, la = fa[ca_idx], la[ca_idx]
    cb_idx = torch.randperm(len(fb))[:MAX_CLEAN]
    fb, lb = fb[cb_idx], lb[cb_idx]
    print(f"clean pool capped to {MAX_CLEAN} points per extractor")

    # projections: sign-style R (dim_in x DIM_OUT) and fourier frequencies w (dim_in x DIM_OUT)
    torch.manual_seed(42)
    R = (torch.rand(fa.shape[1], DIM_OUT) > 0.5).float() * 2 - 1
    torch.manual_seed(7)
    w = torch.randn(fa.shape[1], DIM_OUT) * 0.5
    R, w = R.to(device), w.to(device)

    # per-coordinate clean bias / scale for the adaptive binarizations. Uses a BOUNDED
    # sample of the clean pool (the full 8M-point projection is hundreds of GB) and
    # incremental mean/var (no torch.cat of the full projection). Bias = the per-coord
    # mean (a robust central tendency for re-centering; exact median is not worth 40GB).
    def clean_stats(f_clean, proj, max_pts=1000000):
        idx = torch.randperm(len(f_clean))[:max_pts]
        f_sub = f_clean[idx]
        n = len(f_sub)
        mean = torch.zeros(DIM_OUT)
        sq = torch.zeros(DIM_OUT)
        for i in range(0, n, 200000):
            p = (f_sub[i:i + 200000].to(device) @ proj).cpu().float()
            k = p.shape[0]
            mean += p.sum(dim=0)
            sq += (p ** 2).sum(dim=0)
        mean = mean / n
        var = (sq / n - mean ** 2).clamp(min=1e-6)
        return mean.cpu(), var.sqrt().cpu()

    bias_a, scale_a = clean_stats(fa, R)
    bias_b, scale_b = clean_stats(fb, R)

    MODES = ['sign', 'bias', 'zscore', 'fourier']
    results = {}
    print(f"\n{'='*100}\n=== binarization diagnostic: DGLSS++ (A) vs {args.label_b} (B) ===\n{'='*100}")
    for cond in conds:
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        pa, pl = extract_features(model_a, build_parser(cdir, DATA, ARCH), device, args.frames)
        pb, plb = extract_features(model_b, build_parser(cdir, DATA, ARCH), device, args.frames)
        assert torch.equal(pl, plb), f"{cond} labels must align"
        # keep only pool + val points; the full 8M-point pool is unnecessary and the
        # main CPU-memory driver
        keep = min(len(pa), args.pool_size + args.val_size)
        keep_idx = torch.randperm(len(pa))[:keep]
        pa, pb, pl = pa[keep_idx], pb[keep_idx], pl[keep_idx]
        torch.manual_seed(42)
        perm = torch.randperm(len(pa))
        pa, pb = pa[perm], pb[perm]
        pl = pl[perm]
        va, vb, vl = pa[perm[-args.val_size:]], pb[perm[-args.val_size:]], pl[perm[-args.val_size:]]
        po_a, po_b = pa[perm[:args.pool_size]], pb[perm[:args.pool_size]]
        po_l = pl[perm[:args.pool_size]]

        results[cond] = {}
        # pre-sign margin (current sign binarization) -- the "what is lost" quantity
        ma = margin_frac_chunked(fa, R, eps, device)
        mb = margin_frac_chunked(fb, R, eps, device)
        results[cond]['margin_frac_clean_A'] = ma
        results[cond]['margin_frac_clean_B'] = mb
        print(f"\n--- {cond} ---")
        print(f"  clean pre-sign near-0 fraction (eps={eps}): A {ma:.4f}  B {mb:.4f}")

        for mode in MODES:
            # encode helper for this mode + extractor (clean/val/pool all same scheme)
            if mode == 'fourier':
                def enc_a(f):
                    return encode(f, w, mode, device=device)
                def enc_b(f):
                    return encode(f, w, mode, device=device)
            else:
                def enc_a(f):
                    return encode(f, R, mode, bias_a if mode != 'sign' else None,
                                  scale_a if mode == 'zscore' else None, device)
                def enc_b(f):
                    return encode(f, R, mode, bias_b if mode != 'sign' else None,
                                  scale_b if mode == 'zscore' else None, device)

            # clean prototypes: streaming (the full clean pool is hundreds of GB)
            proto_a, lbl_a = build_protos_stream(fa, la, enc_a, device=device)
            proto_b, lbl_b = build_protos_stream(fb, lb, enc_b, device=device)

            # frozen-prototype decode on val (bounded, 100k)
            va_c = enc_a(va)
            vb_c = enc_b(vb)
            zs_a = mean_iou(per_class_iou(decode(va_c, proto_a, lbl_a, device), vl, range(1, NUM_CLASSES)), range(1, NUM_CLASSES))
            zs_b = mean_iou(per_class_iou(decode(vb_c, proto_b, lbl_b, device), vl, range(1, NUM_CLASSES)), range(1, NUM_CLASSES))

            # oracle: re-estimate from corrupted labeled POOL points (bounded, 100k)
            ora_a, ol_a = build_protos_stream(po_a, po_l, enc_a, device=device)
            ora_b, ol_b = build_protos_stream(po_b, po_l, enc_b, device=device)
            or_a = mean_iou(per_class_iou(decode(va_c, ora_a, ol_a, device), vl, range(1, NUM_CLASSES)), range(1, NUM_CLASSES))
            or_b = mean_iou(per_class_iou(decode(vb_c, ora_b, ol_b, device), vl, range(1, NUM_CLASSES)), range(1, NUM_CLASSES))
            # free the large val codes before the next mode (fourier codes are 8GB each)
            del va_c, vb_c

            results[cond][mode] = {'zs_A': zs_a, 'zs_B': zs_b, 'oracle_A': or_a, 'oracle_B': or_b}
            print(f"  {mode:<8} zs A {zs_a:.4f} B {zs_b:.4f} | oracle A {or_a:.4f} B {or_b:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== WHAT TO LOOK FOR ===")
    print("A. margin_frac: if B (cov-shift) > A (DGLSS++) on snow/wet_ground, the cov-shift")
    print("   healthy features sit closer to the sign threshold -- the packing-loss signature.")
    print("B. On the healthy conditions, does 'bias', 'zscore', or 'fourier' recover the")
    print("   B oracle toward A's level (or above) WITHOUT losing the fog/crosstalk B gain?")
    print("   That encoding is the fix for the cov-shift healthy-ceiling regression.")

if __name__ == "__main__":
    main()
