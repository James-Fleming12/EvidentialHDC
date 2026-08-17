"""d3ctta_diag.py: overnight diagnostic for WHY D3CTTA achieves fog/crosstalk robustness.

D3CTTA (Liu et al., T-ITS 2023) reports good results on fog/crosstalk, but its feature
extractor (a 3D sparse MinkUNet) and pretraining set (Synth4D) are unavailable here
(no MinkowskiEngine, no checkpoint). This isolates the transferable part: does the
D3CTTA MECHANISM (per-class entropy/prob pseudo-label selection + kNN-consistency +
per-domain ridge-classifier adaptation) work on OUR features?

Diagnosis logic:
  - If the mechanism lifts fog/crosstalk full-scene mIoU above our zero-shot (and its
    confident pseudo-label selection is accurate), the mechanism transfers and we should
    adopt it.
  - If it is flat and its SELECTED pseudo-labels are mostly wrong, then D3CTTA's wins
    came from the backbone/pretraining (the features), not the mechanism, which
    localizes the difference to the feature extractor.

Reports per condition: zero-shot LP mIoU, D3CTTA-mechanism mIoU, the selected-pseudo
accuracy (the mechanism-reliability signal), and the full-label oracle as the ceiling.

Usage:
  uv run python d3ctta_diag.py --load_path logs/med_pretrain_supcon_vib
  uv run python d3ctta_diag.py --load_path logs/med_pretrain_supcon_vib_additive --enc additive
"""
import os
import sys

# Script lives in robust_diagnostic/, one level below the repo root (modules/,
# dataset/ are relative to the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou
from modules.D3CTTA import D3CTTA_Decoder

CONDS = ['fog', 'crosstalk', 'snow', 'wet_ground', 'incomplete_echo',
         'beam_missing', 'motion_blur', 'cross_sensor']

def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)

def extract_frames(model, parser, device, num_frames=100):
    """Returns per-frame (feat, lbl) 128D chunks, preserving frame boundaries for the
    online (sequential) adaptation."""
    frames = []
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
            frames.append((z_flat.cpu(), labels[mask].cpu()))
    return frames

def proto_miou_10k(feats, lbls, base_protos, proto_lbls, proj, device):
    feats_d = feats.to(device)
    protos = F.normalize(base_protos, p=2, dim=1)
    sims = []
    for start in range(0, len(feats_d), 50000):
        hc = F.normalize(torch.sign(feats_d[start:start + 50000] @ proj), p=2, dim=1)
        sims.append(hc @ protos.T)
    sims = torch.cat(sims, dim=0)
    return compute_miou(proto_lbls[sims.argmax(dim=1)], lbls.to(device))

def clean_centroids(clean_z, clean_l):
    """Normalized 128D clean class means (aligned to class index)."""
    means = {}
    for c in range(1, 17):
        m = clean_z[clean_l == c]
        if len(m):
            means[c] = F.normalize(m.mean(0), p=2, dim=0)
    return means

def feature_recoverability(z_all, l_all, clean_means, k=3):
    """Feature-extractor issue check: is the recoverable information IN the features?

    For each corrupt point, rank the clean class centroids by 128D cosine. A
    zero-shot-WRONG point is 'recoverable' if its true class sits in the top-k (the
    feature space still localizes the right answer) or if its cosine to its true
    centroid is high (the features still look like the class). 

    Reading:
      - rec_of_wrong LOW  => the encoder has already destroyed the class identity of
        the artifact points: NO mechanism (D3CTTA or otherwise) can recover them.
        That is a FEATURE-EXTRACTOR issue, not a TTA-mechanism issue.
      - rec_of_wrong HIGH but mechanism mIoU low => the features carry the answer;
        the failure is the adaptation mechanism.
      - true_cos LOW => corrupt-wrong points have drifted far from their own class;
        consistent with the feature extractor being corruption-invariant.
    """
    dev = z_all.device
    sims = F.normalize(z_all, p=2, dim=1) @ F.normalize(
        torch.stack([clean_means[c] for c in sorted(clean_means)]), p=2, dim=1).T
    zs_pred = torch.tensor(sorted(clean_means), device=dev)[sims.argmax(1)]
    wrong = zs_pred != l_all.to(dev)
    wrong_n = int(wrong.sum().item())

    classes_sorted = torch.tensor(sorted(clean_means), device=dev)[
        sims.argsort(descending=True, dim=1)]
    true_in_topk = (classes_sorted[:, :k] == l_all.to(dev).unsqueeze(1)).any(1)

    true_cos = []
    for c in sorted(clean_means):
        m = (wrong & (l_all.to(dev) == c))
        if m.sum() > 0:
            true_cos.append(float(sims[m, sorted(clean_means).index(c)].mean().item()))
    rec_of_wrong = (float((wrong & true_in_topk).float().sum().item() / wrong_n)
                    if wrong_n else float('nan'))
    return {'wrong_n': wrong_n,
            'rec_of_wrong': rec_of_wrong,
            'true_cos': float(np.mean(true_cos)) if true_cos else float('nan')}


ENCODERS = {
    'plain': ('logs/med_pretrain_supcon_vib', 'supcon_vib'),
    'additive': ('logs/micro_pretrain_additive_retrain/supcon_vib_additive', 'supcon_vib_additive'),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", type=str, default="plain,additive",
                        help="comma-separated subset of {plain, additive}")
    parser.add_argument("--select_ratios", type=str, default="0.05,0.2")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--w_dim", type=int, default=256)
    parser.add_argument("--conditions", type=str, default="",
                        help="comma-separated subset; default = all 8")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratios = [float(r) for r in args.select_ratios.split(',')]
    encoders = [e.strip() for e in args.encoders.split(',') if e.strip()]

    for enc in encoders:
        load_path, method = ENCODERS[enc]
        print(f"\n{'='*80}\n=== D3CTTA-mechanism diagnostic on the {enc} encoder "
              f"({load_path}) ===\n{'='*80}")
        print(f"Loading {method} encoder...")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, load_path,
                             method=method, path=load_path)
        model = trainer.model
        model.eval()

        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        print("Extracting clean (for the source LP)...")
        clean_frames = extract_frames(model, clean_parser, device, args.frames)
        clean_z = torch.cat([f for f, _ in clean_frames], dim=0)
        clean_l = torch.cat([l for _, l in clean_frames], dim=0)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(clean_z.numpy(), clean_l.numpy())
        print(f"  clean LP acc {clf.score(clean_z.numpy(), clean_l.numpy()):.4f}")

        proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
        base_protos, proto_lbls = build_hdc_prototypes(clean_z, clean_l, proj, device=device)
        cmeans = clean_centroids(clean_z, clean_l)
        from modules.oracle_core import weighted_mean_update

        conds = [c.strip() for c in args.conditions.split(',')] if args.conditions else CONDS
        header = (f"{'cond':<16} {'zs-LP':>7} {'d3ctta':>7} {'no-cons':>8} "
                  f"{'selftrain':>9} {'sel-acc':>8} {'domains':>7} {'oracle':>7} "
                  f"{'rec@3':>7} {'cosT':>6}")
        print(header)
        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            frames = extract_frames(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            z_all = torch.cat([f for f, _ in frames], dim=0)
            l_all = torch.cat([l for _, l in frames], dim=0)

            lp_preds = torch.tensor(clf.predict(z_all.numpy())).to(device)
            zs_miou = compute_miou(lp_preds, l_all.to(device))
            w_one = torch.ones(len(z_all), device=device)
            protos_oracle = weighted_mean_update(base_protos, proto_lbls, z_all,
                                                 l_all.to(device), w_one, proj, device)
            oracle_miou = proto_miou_10k(z_all, l_all, protos_oracle, proto_lbls, proj, device)

            best = {}
            for ratio in ratios:
                dec_full = D3CTTA_Decoder(num_classes=17, feat_dim=128, w_dim=args.w_dim,
                                          select_ratio=ratio, use_consistency=True)
                dec_noc = D3CTTA_Decoder(num_classes=17, feat_dim=128, w_dim=args.w_dim,
                                         select_ratio=ratio, use_consistency=False)
                preds_f, preds_n, sel_accs, n_sel_t = [], [], [], 0
                self_train_preds = []
                for zf, lf in frames:
                    logits = torch.tensor(clf.predict_proba(zf.numpy()),
                                          dtype=torch.float32).to(device)
                    p, n_sel, sa = dec_full.fit_predict(zf.to(device), logits, lf.to(device))
                    preds_f.append(p.cpu())
                    pn, _, _ = dec_noc.fit_predict(zf.to(device), logits, lf.to(device))
                    preds_n.append(pn.cpu())
                    if sa is not None:
                        sel_accs.append(sa)
                    n_sel_t += n_sel
                    # self-train reference: refit the LP on the selected confident points
                    if n_sel > 10:
                        sel_pts = dec_full._select_pseudo(logits).cpu()
                        psel = logits[sel_pts.to(device)].argmax(1).cpu()
                        fit = LogisticRegression(max_iter=1000)
                        fit.fit(zf[sel_pts][:20000].numpy(), psel[:20000].numpy())
                        self_train_preds.append(torch.tensor(fit.predict(zf.numpy())).to(device))
                    else:
                        self_train_preds.append(logits.argmax(1))
                best[ratio] = {
                    'd3': compute_miou(torch.cat(preds_f, dim=0), l_all.to(device)),
                    'noc': compute_miou(torch.cat(preds_n, dim=0), l_all.to(device)),
                    'st': compute_miou(torch.cat(self_train_preds, dim=0), l_all.to(device)),
                    'sel_acc': float(np.mean(sel_accs)) if sel_accs else float('nan'),
                    'domains': len(dec_full.domain_params),
                }
            # pick the best ratio for the headline row (by d3ctta mIoU)
            br = max(best, key=lambda r: best[r]['d3'])
            b = best[br]
            feat = feature_recoverability(z_all, l_all, cmeans)
            print(f"{cond:<16} {zs_miou:>7.4f} {b['d3']:>7.4f} {b['noc']:>8.4f} "
                  f"{b['st']:>9.4f} {b['sel_acc']:>8.3f} {b['domains']:>7d} {oracle_miou:>7.4f} "
                  f"{feat['rec_of_wrong']:>7.3f} {feat['true_cos']:>6.3f}"
                  + (f"  [r={br}]" if len(ratios) > 1 else ""))
            if feat['rec_of_wrong'] < 0.5:
                print(f"    [features] rec@3 {feat['rec_of_wrong']:.3f} < 0.5: "
                      f"FEATURE-EXTRACTOR ISSUE ({feat['wrong_n']} zs-wrong pts, "
                      f"true-cos {feat['true_cos']:.3f})")
            else:
                print(f"    [features] rec@3 {feat['rec_of_wrong']:.3f} >= 0.5: "
                      f"features carry the answer -> mechanism-limited "
                      f"(true-cos {feat['true_cos']:.3f})")
            for ratio, r in best.items():
                if len(ratios) > 1 and ratio != br:
                    print(f"{'':<16} {'':>7} {r['d3']:>7.4f} {r['noc']:>8.4f} "
                          f"{r['st']:>9.4f} {r['sel_acc']:>8.3f} {r['domains']:>7d}  [r={ratio}]")

        if enc != encoders[-1]:
            del model, trainer
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
