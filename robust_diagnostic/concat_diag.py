"""concat_diag.py: the teacher-premise falsification test (cheap, eval-only).

The teacher framework's core claim is that the ROBUST and PLAIN DGLSS++ feature
spaces are complementary: robust carries the TTA, plain DGLSS++ carries the higher
labeled ceiling. Before spending a training run distilling one into the other, test
whether the two FROZEN representations actually combine in one HDC decoder:
extract features for the same points from both checkpoints, concatenate them into
one vector, and measure the oracle / naive TTA / BN on the concat vs each alone.

If the frozen concat already beats both parts (ceiling up, TTA held), the teacher
premise is validated cheaply and the distillation is worth training. If even the
free concatenation fails, no training objective can rescue it -- close the
representation direction and rely on the AL framework.

Usage:
  uv run python robust_diagnostic/concat_diag.py
      --path_a robust_21ep --path_b plain_dglsspp_med [--inv_ch 128]
"""
import os
import sys
import json
import argparse
import yaml
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import (get_hdc_projection, build_hdc_prototypes,
                                 weighted_mean_update)

CONDS = ['fog', 'crosstalk', 'snow']
NUM_CLASSES = 17
PATH_A = 'robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon'   # robust
PATH_B = 'robust_diagnostic/logs/supcon_vib_dglsspp'                                 # plain DGLSS++
METHOD_A = 'supcon_vib_dglsspp_corsupcon'
METHOD_B = 'supcon_vib_dglsspp'

def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)

def extract_aligned(model_a, model_b, parser, device, num_frames=100):
    """Run BOTH models on the SAME batches in ONE parser pass, so the per-frame masks
    and labels are identical by construction (a separate pass per model is not
    guaranteed to yield the same frame/mask sequence). Returns (feat_a, feat_b, lbl)
    aligned point-for-point."""
    fa, fb, lbls = [], [], []
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            oa = model_a(in_vol)
            ob = model_b(in_vol)
            za = oa[2] if len(oa) == 3 else oa[1]
            zb = ob[2] if len(ob) == 3 else ob[1]
            fa.append(za.permute(0, 2, 3, 1).reshape(-1, za.shape[1])[mask].cpu())
            fb.append(zb.permute(0, 2, 3, 1).reshape(-1, zb.shape[1])[mask].cpu())
            lbls.append(labels[mask].cpu())
    return (torch.cat(fa, dim=0), torch.cat(fb, dim=0), torch.cat(lbls, dim=0))

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--path_a", type=str, default=PATH_A)
    parser.add_argument("--method_a", type=str, default=METHOD_A)
    parser.add_argument("--path_b", type=str, default=PATH_B)
    parser.add_argument("--method_b", type=str, default=METHOD_B)
    parser.add_argument("--label_a", type=str, default="robust_21ep")
    parser.add_argument("--label_b", type=str, default="dglsspp_med")
    parser.add_argument("--inv_ch", type=int, default=128,
                        help="split point in A's features when A is a two-branch model "
                             "(use only the inv slice for the A-alone / concat rows)")
    parser.add_argument("--out", type=str,
                        default="robust_diagnostic/logs/concat_diag_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ---- load both frozen models, extract clean features for the same points ----
    def load(path, method):
        tr = GenTrainer(ARCH, DATA, args.kitti_dir, path, path=path, method=method)
        return tr.model

    model_a = load(args.path_a, args.method_a)   # robust (256D two-branch, inv first)
    model_b = load(args.path_b, args.method_b)   # plain DGLSS++ (128D)
    print(f"{args.label_a}: model loaded, {args.method_a}")
    print(f"{args.label_b}: model loaded, {args.method_b}")

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, fb, la = extract_aligned(model_a, model_b, clean_parser, device, args.frames)
    print(f"clean feats: A {fa.shape}, B {fb.shape}, aligned labels {la.shape}")

    # A-alone uses the inv slice only (the TTA branch); concat = [inv_A, B]
    fa_inv = fa[:, :args.inv_ch] if fa.shape[1] > args.inv_ch else fa
    fcat = torch.cat([fa_inv, fb], dim=1)
    print(f"A-alone dim {fa_inv.shape[1]}, B-alone dim {fb.shape[1]}, concat dim {fcat.shape[1]}")

    feats_map = {
        f"{args.label_a}_inv": (fa_inv, la),
        f"{args.label_b}": (fb, la),
        "concat": (fcat, la),
    }

    results = {}
    header = (f"{'row':<16} {'cond':<10} {'zs':>6} {'naive':>6} {'bn':>6} {'oracle':>7}")
    print(header)
    for name, (f_all, l_all) in feats_map.items():
        proj = get_hdc_projection(dim_in=f_all.shape[1], dim_out=10000, device=device)
        base_protos, proto_lbls = build_hdc_prototypes(f_all, l_all, proj, device=device)
        # per-row logistic classifier on THIS row's clean features (the naive TTA
        # pseudo-labels must match the pool dimension; a clf fit on the 128D inv
        # slice cannot predict 256D concat features).
        clf = LogisticRegression(max_iter=1000)
        clf.fit(f_all[:min(100000, len(f_all))].numpy(),
                l_all[:min(100000, len(l_all))].numpy())
        results[name] = {}
        for cond in CONDS:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            pa, pb, pl = extract_aligned(model_a, model_b, build_parser(cdir, DATA, ARCH),
                                         device, args.frames)

            pa = pa[:, :args.inv_ch] if pa.shape[1] > args.inv_ch else pa
            a_name = f"{args.label_a}_inv"
            b_name = f"{args.label_b}"
            pool_f = {a_name: pa, b_name: pb, "concat": torch.cat([pa, pb], dim=1)}[name]

            torch.manual_seed(42)
            perm = torch.randperm(len(pool_f))
            pool, val = pool_f[perm[:args.pool_size]], pool_f[perm[-args.val_size:]]
            pool_l, vl = pl[perm[:args.pool_size]], pl[perm[-args.val_size:]]

            classes = sorted(cids)
            def preds(protos):
                return decode(protos, val, proto_lbls, proj, device)

            iou_zs = per_class_iou(preds(base_protos), vl, classes)
            lp_preds = torch.tensor(clf.predict(pool.numpy())).to(device)
            ones = torch.ones(len(pool), device=device)
            iou_na = per_class_iou(preds(weighted_mean_update(base_protos, proto_lbls, pool,
                                                              lp_preds, ones, proj, device)), vl, classes)
            iou_or = per_class_iou(preds(weighted_mean_update(base_protos, proto_lbls, pool,
                                                              pool_l.to(device), ones, proj, device)), vl, classes)
            zs, na, orc = mean_iou(iou_zs, classes), mean_iou(iou_na, classes), mean_iou(iou_or, classes)
            results[name][cond] = {'zs': zs, 'naive': na, 'oracle': orc,
                                   'per_class_oracle': {str(c): iou_or[c] for c in classes}}
            print(f"{name:<16} {cond:<10} {zs:>6.3f} {na:>6.3f} {'--':>6} {orc:>7.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== TEACHER-PREMISE VERDICT ===")
    print("If concat oracle > max(A, B) oracle on fog/crosstalk AND concat naive >= A naive:")
    print("  the two frozen representations combine -> the teacher distillation is worth training.")
    print("Else: close the representation direction, use the AL framework on the robust extractor.")

def decode(protos, feats, proto_lbls, proj, device, chunk=50000):
    protos = F.normalize(protos, p=2, dim=1)
    preds = []
    for s in range(0, len(feats), chunk):
        hc = F.normalize(torch.sign(feats[s:s + chunk].to(device) @ proj), p=2, dim=1)
        sims = hc @ protos.T
        preds.append(proto_lbls[sims.argmax(dim=1)].cpu())
    return torch.cat(preds)

cids = list(range(1, NUM_CLASSES))

if __name__ == "__main__":
    main()
