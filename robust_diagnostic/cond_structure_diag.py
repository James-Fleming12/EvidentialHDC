"""cond_structure_diag.py: why does the cov-shift normalization hurt healthy-condition
ceilings (snow, wet_ground, etc.) even though it fixes fog/crosstalk?

The frozen-ceiling comparison showed: LP accuracy is mostly preserved on the healthy
conditions (snow 79->82, wet_ground 85->77) but the HDC-oracle drops (wet_ground 51->
37, beam_missing 51->44). So the continuous class structure survives; the recoverable
structure through the HDC sign-binarization degrades. This diagnostic measures the
PER-CLASS feature structure on a healthy condition (default snow + wet_ground) for
two checkpoints -- the plain DGLSS++ reference and the cov-shift model -- to see
whether the normalization erases a class-specific signal (feat_cos / dir_retention /
corr_tightness) that recoverability needs.

Run: uv run python robust_diagnostic/cond_structure_diag.py
  --path_b robust_diagnostic/logs/ep10_.../supcon_vib_dglsspp_inputin_in_chan
  --method_b supcon_vib_dglsspp_inputin_in_chan --label_b covshift_ep10
  --conds snow,wet_ground
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
from modules.oracle_core import (get_hdc_projection, build_hdc_prototypes,
                                 weighted_mean_update)

CONDS_DEFAULT = ['snow', 'wet_ground']
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

def extract_features_pair(model_a, model_b, parser, device, num_frames=100):
    """Extract features from BOTH models on the SAME points in one pass.

    The parser randomly drops points per scan (`drop_points = random.uniform(0, 0.5)`
    in Parser), so two separate extract_features calls can consume different points and
    produce MISALIGNED label streams (the C8 scalereg gate crashed on fog with
    "labels must align"). Extracting both models from one shared pass guarantees A and
    B are evaluated on identical points."""
    fa, la, fb = [], [], []
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            za = model_a(in_vol)
            if len(za) == 3:
                za = za[2]
            else:
                za = za[1]
            zb = model_b(in_vol)
            if len(zb) == 3:
                zb = zb[2]
            else:
                zb = zb[1]
            za_flat = za.permute(0, 2, 3, 1).reshape(-1, za.shape[1])[mask]
            zb_flat = zb.permute(0, 2, 3, 1).reshape(-1, zb.shape[1])[mask]
            fa.append(za_flat.cpu())
            fb.append(zb_flat.cpu())
            la.append(labels[mask].cpu())
    return (torch.cat(fa, dim=0), torch.cat(la, dim=0),
            torch.cat(fb, dim=0), torch.cat(la, dim=0))

def per_class_iou(preds, lbls, classes):
    out = {}
    for c in classes:
        tp = int(((preds == c) & (lbls == c)).sum().item())
        fp = int(((preds == c) & (lbls != c)).sum().item())
        fn = int(((preds != c) & (lbls == c)).sum().item())
        denom = tp + fp + fn
        out[c] = tp / denom if denom > 0 else 0.0
    return out

def decode(protos, feats, proto_lbls, proj, device, chunk=50000):
    protos = F.normalize(protos, p=2, dim=1)
    preds = []
    for s in range(0, len(feats), chunk):
        hc = F.normalize(torch.sign(feats[s:s + chunk].to(device) @ proj), p=2, dim=1)
        sims = hc @ protos.T
        preds.append(proto_lbls[sims.argmax(dim=1)].cpu())
    return torch.cat(preds)

def clean_class_means(feats, lbls):
    means = {}
    for c in range(1, NUM_CLASSES):
        m = feats[lbls == c]
        if len(m):
            means[c] = F.normalize(m.mean(0), p=2, dim=0)
    return means

def class_structure(pool, pool_l, clean_means, classes):
    col_of = {c: j for j, c in enumerate(classes)}
    means_mat = torch.stack([clean_means[c] for c in classes])
    zn = F.normalize(pool, p=2, dim=1)
    cos = zn @ means_mat.t()
    rows = {}
    for c in classes:
        m = pool_l == c
        n = int(m.sum())
        if n == 0:
            rows[c] = {'freq': 0, 'feat_cos': float('nan'), 'dir_ret': float('nan'),
                       'corr_tight': float('nan')}
            continue
        pts = zn[m]
        corr_mean = F.normalize(pts.mean(0), p=2, dim=0)
        rows[c] = {'freq': n,
                   'feat_cos': float(cos[m, col_of[c]].mean().item()),
                   'dir_ret': float((corr_mean * means_mat[col_of[c]]).sum().item()),
                   'corr_tight': float((pts @ corr_mean).mean().item())}
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH,
                        help="the model to compare against plain DGLSS++ (A)")
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label_b", type=str, default="covshift")
    parser.add_argument("--conds", type=str, default=",".join(CONDS_DEFAULT))
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/cond_structure_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    def load(path, method):
        return GenTrainer(ARCH, DATA, args.kitti_dir, path, path=path, method=method).model

    model_a = load(DGLSSPP_PATH, DGLSSPP_METHOD)
    model_b = load(args.path_b, args.method_b)

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model_a, clean_parser, device, args.frames)
    fb, lb = extract_features(model_b, clean_parser, device, args.frames)
    ca, cb = clean_class_means(fa, la), clean_class_means(fb, lb)
    # only classes present in BOTH extractors' clean means (rare classes can be
    # missing from one extractor's sampled clean features, e.g. tiny dry runs)
    classes = sorted(set(ca) & set(cb))

    results = {}
    for cond in conds:
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        # single shared pass so A and B are evaluated on the SAME points (the parser
        # randomly drops points per scan; two separate passes can misalign labels)
        pa, pl, pb, plb = extract_features_pair(model_a, model_b,
                                                build_parser(cdir, DATA, ARCH),
                                                device, args.frames)
        assert torch.equal(pl, plb), f"{cond} labels must align"
        torch.manual_seed(42)
        perm = torch.randperm(len(pa))
        pa, pb = pa[perm], pb[perm]
        pl = pl[perm]
        va, vb, vl = pa[perm[-args.val_size:]], pb[perm[-args.val_size:]], pl[perm[-args.val_size:]]
        pool_a, pool_b = pa[perm[:args.pool_size]], pb[perm[:args.pool_size]]
        pool_l = pl[perm[:args.pool_size]]

        sa = class_structure(pool_a, pool_l, ca, classes)
        sb = class_structure(pool_b, pool_l, cb, classes)

        # HDC oracle on the val split for both (same prototypes pipeline as frozen ceiling)
        proj_a = get_hdc_projection(dim_in=va.shape[1], dim_out=10000, device=device)
        proj_b = get_hdc_projection(dim_in=vb.shape[1], dim_out=10000, device=device)
        base_a, plbl_a = build_hdc_prototypes(fa, la, proj_a, device=device)
        base_b, plbl_b = build_hdc_prototypes(fb, lb, proj_b, device=device)
        iou_a = per_class_iou(decode(base_a, va, plbl_a, proj_a, device), vl, classes)
        iou_b = per_class_iou(decode(base_b, vb, plbl_b, proj_b, device), vl, classes)

        results[cond] = {'per_class': {}, 'aggregate': {}}
        for c in classes:
            results[cond]['per_class'][str(c)] = {
                'freq': sa[c]['freq'],
                'feat_cos_A': sa[c]['feat_cos'], 'feat_cos_B': sb[c]['feat_cos'],
                'dir_ret_A': sa[c]['dir_ret'], 'dir_ret_B': sb[c]['dir_ret'],
                'corr_tight_A': sa[c]['corr_tight'], 'corr_tight_B': sb[c]['corr_tight'],
                'zs_A': iou_a[c], 'zs_B': iou_b[c],
            }
        print(f"\n{'='*80}\n=== {cond}: plain DGLSS++ (A) vs {args.label_b} (B) ===\n{'='*80}")
        print(f"{'cls':>3} {'fcA':>6} {'fcB':>6} {'dirA':>6} {'dirB':>6} {'tA':>5} {'tB':>5} | {'zsA':>6} {'zsB':>6}")
        for c in classes:
            r = results[cond]['per_class'][str(c)]
            g = lambda k: float('nan') if r[k] != r[k] else r[k]
            print(f"{int(c):>3} {g('feat_cos_A'):>6.2f} {g('feat_cos_B'):>6.2f} "
                  f"{g('dir_ret_A'):>6.2f} {g('dir_ret_B'):>6.2f} "
                  f"{g('corr_tight_A'):>5.2f} {g('corr_tight_B'):>5.2f} | "
                  f"{g('zs_A'):>6.3f} {g('zs_B'):>6.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("Look for: a class whose dir_ret or corr_tight drops under B (cov-shift) while")
    print("zs also drops -- that is the recoverable structure the normalization erases.")

if __name__ == "__main__":
    main()
