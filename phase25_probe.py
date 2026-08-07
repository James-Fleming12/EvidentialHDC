"""phase25_probe.py: micro-scale probe for the Phase 25 training levers.

De-risks the 10-hour run (Phase 25, gen_iterations.md) before committing. Trains each
method at micro scale (cutoff 0.1, few epochs, ~2 min/epoch) and measures the diagnostics
that decide whether the levers are worth scaling:

  1. Per-class fog LP corrupt accuracy on the casualties (2, 7, 13, 14, 15).
     Target: `supcon_vib_fragile` moves them off ~0 vs plain (Iteration 4B).
  2. Clean zero-shot proto mIoU + acc: the weighted SupCon must NOT regress the
     healthy manifold (the additive regimen's failure mode, Phase 24.6).
  3. Fog proto mIoU and snow proto mIoU (sanity / collapse check).

Usage:
  uv run python phase25_probe.py --methods supcon_vib,supcon_vib_fragile --epochs 3
"""
import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

FRAGILE = [2, 7, 13, 14, 15]


def extract_features(model, parser, device, num_frames=40):
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


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)


def per_class_lp_acc(clf, feats, lbls):
    preds = clf.predict(feats.numpy())
    out = {}
    present = set(lbls.tolist())
    for c in range(1, 17):
        if c not in present:
            continue
        m = (lbls == c).numpy()
        out[c] = float((preds[m] == c).mean()) if m.sum() > 0 else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="logs/phase25_probe")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=0.1)
    parser.add_argument("--methods", type=str, default="supcon_vib,supcon_vib_fragile")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    fog_dir = os.path.join(args.kittic_dir, 'fog', 'heavy')
    if not os.path.exists(fog_dir):
        fog_dir = os.path.join(args.kittic_dir, 'fog', 'moderate')
    snow_dir = os.path.join(args.kittic_dir, 'snow', 'heavy')
    if not os.path.exists(snow_dir):
        snow_dir = os.path.join(args.kittic_dir, 'snow', 'moderate')

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fog_parser = build_parser(fog_dir, DATA, ARCH)
    snow_parser = build_parser(snow_dir, DATA, ARCH)

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    results = {}
    for method in methods:
        log_dir = os.path.join(args.log_dir, method)
        os.makedirs(log_dir, exist_ok=True)
        print(f"\n=== Training {method} (cutoff {args.cutoff}, {args.epochs} epochs) ===")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, method=method,
                             cutoff_percent=args.cutoff)
        trainer.train(epochs=args.epochs)

        print(f"=== Evaluating {method} ===")
        clean_f, clean_l = extract_features(trainer.model, clean_parser, device)
        fog_f, fog_l = extract_features(trainer.model, fog_parser, device)
        snow_f, snow_l = extract_features(trainer.model, snow_parser, device)

        proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)
        base_norm = F.normalize(base_protos, p=2, dim=1)

        def proto_miou(feats, lbls):
            h = F.normalize(torch.sign(feats.to(device) @ proj), p=2, dim=1)
            preds = proto_lbls[(h @ base_norm.T).argmax(dim=1)]
            return compute_miou(preds, lbls.to(device))

        clf = LogisticRegression(max_iter=1000)
        clf.fit(clean_f.numpy(), clean_l.numpy())

        r = {
            'clean_proto_miou': proto_miou(clean_f, clean_l),
            'fog_proto_miou': proto_miou(fog_f, fog_l),
            'snow_proto_miou': proto_miou(snow_f, snow_l),
            'clean_lp_acc': float(clf.score(clean_f.numpy(), clean_l.numpy())),
            'fog_lp_acc': float(clf.score(fog_f.numpy(), fog_l.numpy())),
            'fog_lp_per_class': per_class_lp_acc(clf, fog_f, fog_l),
        }
        results[method] = r
        print(f"  clean proto mIoU {r['clean_proto_miou']:.4f} | fog proto mIoU "
              f"{r['fog_proto_miou']:.4f} | snow {r['snow_proto_miou']:.4f} | "
              f"clean LP {r['clean_lp_acc']:.4f} | fog LP {r['fog_lp_acc']:.4f}")
        frag_acc = {c: r['fog_lp_per_class'].get(c) for c in FRAGILE}
        print("  fog LP per-class (casualties): " + ", ".join(
            f"{c}:{v:.3f}" if v is not None else f"{c}:n/a" for c, v in frag_acc.items()))

    print("\n=== Comparison (fog LP per-class on casualties) ===")
    print(f"{'method':<22}" + "".join(f"{c:>8}" for c in FRAGILE))
    for method, r in results.items():
        row = "".join(f"{(r['fog_lp_per_class'].get(c) or 0.0):>8.3f}" for c in FRAGILE)
        print(f"{method:<22}{row}")
    print("clean proto mIoU: " + ", ".join(
        f"{m}:{r['clean_proto_miou']:.4f}" for m, r in results.items()))
    print("fog proto mIoU:   " + ", ".join(
        f"{m}:{r['fog_proto_miou']:.4f}" for m, r in results.items()))


if __name__ == "__main__":
    main()
