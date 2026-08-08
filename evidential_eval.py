"""evidential_eval.py: post-run diagnostics for the uncertainty heads (supcon_vib_evidential
and supcon_vib_losspred checkpoints).

Loads the checkpoint via GenTrainer (which reconstructs the head from the optimizer state,
the same path logvar_head uses), then measures whether the head learned USEFUL uncertainty:

  1. Mean uncertainty on clean vs corrupted conditions. A calibrated head should give
     higher uncertainty on fog/crosstalk than on clean.
  2. AUROC of uncertainty for separating CORRECT from WRONG predictions (correctness taken
     from the model's semantic output) on each condition. AUROC > 0.5 means the head's
     uncertainty predicts the model's own errors.
  3. Per-class mean uncertainty on fog (casualties should be more uncertain than survivors).

Usage:
  uv run python evidential_eval.py --load_path logs/... --method supcon_vib_evidential
  uv run python evidential_eval.py --load_path logs/... --method supcon_vib_losspred
"""
import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer

CONDS = {'fog': 'fog', 'crosstalk': 'crosstalk', 'snow': 'snow',
         'wet_ground': 'wet_ground', 'cross_sensor': 'cross_sensor'}


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract(model, head, parser, device, num_frames=50):
    """Returns z (128D), head output (per-pixel), semantic pred, labels."""
    zs, ho, ps, lbls = [], [], [], []
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
                output, _, z8 = out_tuple
            else:
                output, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            ho_flat = head(z8).permute(0, 2, 3, 1).reshape(-1, head.out_channels)[mask]
            pred = output.argmax(dim=1).reshape(-1)[mask]
            zs.append(z_flat.cpu())
            ho.append(ho_flat.cpu())
            ps.append(pred.cpu())
            lbls.append(labels[mask].cpu())
    return (torch.cat(zs, dim=0), torch.cat(ho, dim=0),
            torch.cat(ps, dim=0), torch.cat(lbls, dim=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="logs/med_pretrain_supcon_vib_evidential")
    parser.add_argument("--method", type=str, default="supcon_vib_evidential",
                        choices=['supcon_vib_evidential', 'supcon_vib_losspred'])
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.method} checkpoint from {args.load_path}...")
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.load_path,
                         method=args.method, path=args.load_path)
    model = trainer.model
    is_losspred = args.method == 'supcon_vib_losspred'
    head = trainer.losspred_head if is_losspred else trainer.evidence_head
    assert head is not None, f"checkpoint did not reconstruct the {args.method} head"
    model.eval()
    head.eval()

    def uncertainty(ho):
        if is_losspred:
            return F.softplus(ho).squeeze(1)          # predicted loss: higher = more uncertain
        S = F.softplus(ho).sum(dim=1) + float(head.out_channels)
        return float(head.out_channels) / S          # K/S: higher = more uncertain

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)

    print("Extracting clean...")
    _, ec, pc, lc = extract(model, head, clean_parser, device, args.frames)
    unc_c = uncertainty(ec)
    correct_c = (pc == lc).float()
    print(f"  clean: n {len(lc)} | mean uncertainty {unc_c.mean().item():.4f} | "
          f"semantic acc {correct_c.mean().item():.4f}")
    if correct_c.std() > 0 and unc_c.std() > 0:
        print(f"  clean correct-vs-wrong uncertainty AUROC: "
              f"{roc_auc_score(correct_c.numpy(), -unc_c.numpy()):.3f}")

    for name, cond in CONDS.items():
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        print(f"Extracting {name}...")
        _, ef, pf, lf = extract(model, head, build_parser(cdir, DATA, ARCH), device, args.frames)
        unc_f = uncertainty(ef)
        correct_f = (pf == lf).float()
        print(f"  {name}: n {len(lf)} | mean uncertainty {unc_f.mean().item():.4f} "
              f"(clean {unc_c.mean().item():.4f}) | semantic acc {correct_f.mean().item():.4f}")
        if correct_f.std() > 0 and unc_f.std() > 0:
            print(f"  {name} correct-vs-wrong uncertainty AUROC: "
                  f"{roc_auc_score(correct_f.numpy(), -unc_f.numpy()):.3f}")
        if name == 'fog':
            per_class = {}
            for c in range(1, 17):
                m = (lf == c)
                if m.sum() > 20:
                    per_class[c] = float(unc_f[m].mean().item())
            top = sorted(per_class.items(), key=lambda kv: -kv[1])
            print("  fog per-class mean uncertainty (highest first): " +
                  ", ".join(f"{c}:{v:.3f}" for c, v in top[:6]))


if __name__ == "__main__":
    main()
