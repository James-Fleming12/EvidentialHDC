"""evidential_eval.py: post-run diagnostics for the `supcon_vib_evidential` checkpoint.

Loads the checkpoint via GenTrainer (which reconstructs the evidence_head from the
optimizer state, the same path logvar_head uses), then measures whether the evidential
head learned USEFUL uncertainty:

  1. Mean uncertainty (K/S) on clean vs corrupted conditions. A calibrated head should
     give higher uncertainty on fog/crosstalk than on clean.
  2. AUROC of uncertainty for separating CORRECT from WRONG predictions on each
     condition (clean and each corruption). AUROC > 0.5 means the head's uncertainty
     predicts the model's own errors, which is the diagnostic for whether the KL weight
     is in the right regime: a collapsed head (KL too strong) gives ~0.5, a well-trained
     one gives > 0.6-0.7.
  3. Per-class mean uncertainty on fog (the casualties should be more uncertain than the
     survivors if the head calibrated on corruption-hard points).

Usage:
  uv run python evidential_eval.py --load_path logs/med_pretrain_supcon_vib_evidential
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


def extract(model, evidence_head, parser, device, num_frames=50):
    zs, ev, lbls = [], [], []
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
            ev_flat = evidence_head(z8).permute(0, 2, 3, 1).reshape(-1, evidence_head.out_channels)[mask]
            zs.append(z_flat.cpu())
            ev.append(ev_flat.cpu())
            lbls.append(labels[mask].cpu())
    return (torch.cat(zs, dim=0), torch.cat(ev, dim=0), torch.cat(lbls, dim=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="logs/med_pretrain_supcon_vib_evidential")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading supcon_vib_evidential checkpoint from {args.load_path}...")
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.load_path,
                         method='supcon_vib_evidential', path=args.load_path)
    model = trainer.model
    evidence_head = trainer.evidence_head
    assert evidence_head is not None, "checkpoint did not reconstruct the evidence_head"
    model.eval()
    evidence_head.eval()

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)

    # clean baseline
    print("Extracting clean...")
    zc, ec, lc = extract(model, evidence_head, clean_parser, device, args.frames)
    S_c = F.softplus(ec).sum(dim=1) + float(evidence_head.out_channels)
    unc_c = float(evidence_head.out_channels) / S_c
    correct_c = (ec.argmax(dim=1) == lc).float()

    print(f"  clean: n {len(lc)} | mean uncertainty {unc_c.mean().item():.4f} | "
          f"acc {correct_c.mean().item():.4f}")
    if correct_c.std() > 0 and unc_c.std() > 0:
        auc = roc_auc_score(correct_c.numpy(), -unc_c.numpy())
        print(f"  clean correct-vs-wrong uncertainty AUROC: {auc:.3f}")

    for name, cond in CONDS.items():
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        print(f"Extracting {name}...")
        zf, ef, lf = extract(model, evidence_head, build_parser(cdir, DATA, ARCH), device, args.frames)
        S_f = F.softplus(ef).sum(dim=1) + float(evidence_head.out_channels)
        unc_f = float(evidence_head.out_channels) / S_f
        correct_f = (ef.argmax(dim=1) == lf).float()
        print(f"  {name}: n {len(lf)} | mean uncertainty {unc_f.mean().item():.4f} "
              f"(clean {unc_c.mean().item():.4f}) | acc {correct_f.mean().item():.4f}")
        if correct_f.std() > 0 and unc_f.std() > 0:
            auc = roc_auc_score(correct_f.numpy(), -unc_f.numpy())
            print(f"  {name} correct-vs-wrong uncertainty AUROC: {auc:.3f}")
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
