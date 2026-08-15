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


def encode(feats, proj, mode, bias=None, scale=None, device='cuda'):
    """Encode features into the HDC space. proj is the projection/frequency matrix.
    Returns the code (n, code_dim) on cpu.
    mode:
      'sign'    : sign(z @ R), threshold 0
      'bias'    : sign(z @ R - b), b per-coordinate clean bias
      'zscore'  : sign((z @ R - b) / s), b and s per-coordinate clean stats
      'fourier' : [cos(z @ w), sin(z @ w)] with random frequencies w
    """
    f = feats.to(device)
    p = f @ proj
    if mode == 'sign':
        return torch.sign(p).cpu()
    if mode == 'bias':
        return torch.sign(p - bias.to(device)).cpu()
    if mode == 'zscore':
        return torch.sign((p - bias.to(device)) / (scale.to(device) + 1e-6)).cpu()
    if mode == 'fourier':
        # dim_out here is the number of frequencies; code is 2*dim_out
        c = torch.cos(p)
        s = torch.sin(p)
        return torch.cat([c, s], dim=1).cpu()
    raise ValueError(mode)


def build_protos(codes, lbls, num_classes=17, chunk=50000, fourier=False):
    """Per-class mean of codes, L2-normalized (drop the sign-normalization assumption
    for the fourier continuous codes)."""
    D = codes.shape[1]
    protos = torch.zeros(num_classes, D, device=codes.device)
    counts = torch.zeros(num_classes, device=codes.device)
    for i in range(0, len(codes), chunk):
        ch = codes[i:i + chunk]
        cl = lbls[i:i + chunk].to(codes.device)
        for c in range(num_classes):
            m = cl == c
            if m.sum() > 0:
                protos[c] += ch[m].sum(dim=0)
                counts[c] += m.sum()
    for c in range(num_classes):
        if counts[c] > 0:
            protos[c] /= counts[c].clamp(min=1)
    valid = counts > 0
    base = F.normalize(protos[valid], p=2, dim=1)
    return base, torch.arange(num_classes, device=codes.device)[valid]


def decode(codes, protos, proto_lbls, device, chunk=50000):
    protos = F.normalize(protos, p=2, dim=1)
    preds = []
    for i in range(0, len(codes), chunk):
        c = codes[i:i + chunk].to(device)
        sims = c @ protos.T
        preds.append(proto_lbls[sims.argmax(dim=1)].cpu())
    return torch.cat(preds)


def margin_fraction(pre_proj, eps=0.1):
    """Fraction of coordinates with |pre-sign projection| < eps -- the
    threshold-hugging coordinates that flip sign on small noise."""
    p = pre_proj
    return float((p.abs() < eps).float().mean().item())


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

    # projections: sign-style R (dim_in x DIM_OUT) and fourier frequencies w (dim_in x DIM_OUT)
    torch.manual_seed(42)
    R = (torch.rand(fa.shape[1], DIM_OUT) > 0.5).float() * 2 - 1
    torch.manual_seed(7)
    w = torch.randn(fa.shape[1], DIM_OUT) * 0.5
    R, w = R.to(device), w.to(device)

    # per-coordinate clean bias / scale for the adaptive binarizations
    def clean_stats(f_clean, proj):
        p = f_clean.to(device) @ proj
        b = p.median(dim=0).values
        s = p.std(dim=0) + 1e-6
        return b.cpu(), s.cpu()

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
        torch.manual_seed(42)
        perm = torch.randperm(len(pa))
        pa, pb = pa[perm], pb[perm]
        pl = pl[perm]
        va, vb, vl = pa[perm[-args.val_size:]], pb[perm[-args.val_size:]], pl[perm[-args.val_size:]]
        po_a, po_b = pa[perm[:args.pool_size]], pb[perm[:args.pool_size]]
        po_l = pl[perm[:args.pool_size]]

        results[cond] = {}
        # pre-sign margin (current sign binarization) -- the "what is lost" quantity
        ma = float(((fa.to(device) @ R).abs() < eps).float().mean().item())
        mb = float(((fb.to(device) @ R).abs() < eps).float().mean().item())
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

            ca_c = enc_a(fa)
            cb_c = enc_b(fb)

            # prototypes from the CLEAN features
            proto_a, lbl_a = build_protos(ca_c, la, fourier=(mode == 'fourier'))
            proto_b, lbl_b = build_protos(cb_c, lb, fourier=(mode == 'fourier'))

            # frozen-prototype decode on val
            va_c = enc_a(va)
            vb_c = enc_b(vb)
            zs_a = mean_iou(per_class_iou(decode(va_c, proto_a, lbl_a, device), vl, range(1, NUM_CLASSES)), range(1, NUM_CLASSES))
            zs_b = mean_iou(per_class_iou(decode(vb_c, proto_b, lbl_b, device), vl, range(1, NUM_CLASSES)), range(1, NUM_CLASSES))

            # oracle: re-estimate from corrupted labeled POOL points
            poa_c = enc_a(po_a)
            pob_c = enc_b(po_b)
            ora_a, ol_a = build_protos(poa_c, po_l, fourier=(mode == 'fourier'))
            ora_b, ol_b = build_protos(pob_c, po_l, fourier=(mode == 'fourier'))
            or_a = mean_iou(per_class_iou(decode(va_c, ora_a, ol_a, device), vl, range(1, NUM_CLASSES)), range(1, NUM_CLASSES))
            or_b = mean_iou(per_class_iou(decode(vb_c, ora_b, ol_b, device), vl, range(1, NUM_CLASSES)), range(1, NUM_CLASSES))

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
