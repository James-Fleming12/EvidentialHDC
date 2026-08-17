"""probe_gap_diag.py: Iteration-0 TTA/AL gap diagnostic for the linear-probe decoder.

The headline C10 result: a pool-refit linear probe (R4-oracle) beats the frozen
clean-fit probe (R4-zs) on every condition. Before designing the TTA/AL update, this
characterizes WHAT the zero-shot -> labeled gap is, per condition, so we know what a
method must improve on:

  1. SHIFT TYPE     : cosine of W_zs (clean-fit) vs W_oracle (pool-refit), per class
                      and mean. cos ~ 1 = a clean translation (bias-only update
                      suffices); cos << 1 = the decision boundary must ROTATE
                      (weight update needed, not just the intercept).
  2. BIAS-ONLY SHARE: how much of the zs->oracle gap a bias-only update closes
                      (freeze W_zs, re-center the intercept to the pool class
                      proportions -- the gradient-free option-1 update). This is the
                      cheapest TTA and the diagnostic's key gradient-free answer.
  3. MARGIN         : the frozen probe's top-2 decision margin on val, for
                      zs-correct vs zs-wrong points, and the margin of the points the
                      oracle FIXES. If the fixed points are low-margin, a margin-gated
                      TTA has signal; if they are high-margin (confident but wrong),
                      margins won't help -- the boundary itself must move.
  4. OUTLIER        : code-space norm of zs-correct vs zs-wrong vs oracle-fixed
                      points. If the oracle-fixed points are high-norm, they are
                      outliers a norm-gate would veto; if low-norm, they are inlier
                      boundary points.
  5. PER-CLASS      : which classes the oracle lifts most (the TTA/AL target classes).
  6. POOL-SIZE CURVE: oracle mIoU vs labeled pool size (1k/5k/10k/50k/100k) -- how
                      many labels close the gap (the active-learning budget).

Usage:
  uv run python robust_diagnostic/probe_gap_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds snow,wet_ground,fog,crosstalk \
    --out robust_diagnostic/logs/probe_gap_covshift_ep10.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

CONDS_DEFAULT = ['snow', 'wet_ground', 'fog', 'crosstalk']
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

def hdc_codes(feats, proj, device, chunk=100000):
    codes = []
    for s in range(0, len(feats), chunk):
        codes.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(codes, dim=0)

def fit_probe(codes, lbls, max_fit=100000, C=1.0, max_iter=1000):
    clf = LogisticRegression(max_iter=max_iter, C=C)
    n = min(max_fit, len(codes))
    clf.fit(codes[:n].numpy(), lbls[:n].numpy())
    return clf

def predict_probe(codes, clf, chunk=100000):
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append(torch.tensor(clf.predict(codes[s:s + chunk].numpy())))
    return torch.cat(preds)

def probe_scores(codes, clf, chunk=100000):
    """decision_function = W @ code + b, shape (n, n_classes)."""
    scores = []
    for s in range(0, len(codes), chunk):
        scores.append(torch.tensor(clf.decision_function(codes[s:s + chunk].numpy())))
    return torch.cat(scores, dim=0)

def margin(scores):
    """top-2 margin: scores.max(1) - scores.sort(descending)[0][:, 1]."""
    top2 = torch.topk(scores, 2, dim=1).values
    return top2[:, 0] - top2[:, 1]

def norm_per_class(clf, classes):
    """Per-class L2 norm of the probe weight rows (how much each class boundary moved)."""
    W = torch.tensor(clf.coef_)
    lbls = torch.tensor(clf.classes_)
    out = {}
    for j, c in enumerate(classes):
        m = lbls == c
        if m.any():
            out[int(c)] = float(W[m].norm(dim=1).mean().item())
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--max_clean", type=int, default=200000)
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="lbfgs iterations for the probe fits; 1000 = same as C10")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--conds", type=str, default=",".join(CONDS_DEFAULT))
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_gap_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    # bounded clean sample for the frozen (zero-shot) probe
    max_clean = min(args.max_clean, len(fa))
    ci = torch.randperm(len(fa))[:max_clean]
    fa_s, la_s = fa[ci], la[ci]
    clean_codes = hdc_codes(fa_s, proj, device)
    clf_zs = fit_probe(clean_codes, la_s, max_iter=args.max_iter)
    classes = sorted(set(int(c) for c in clf_zs.classes_))

    results = {}
    print(f"\n{'='*100}")
    print(f"=== {args.label}: probe-gap diagnostic (zero-shot vs labeled) ===")
    print(f"{'='*100}")

    for cond in conds:
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        pool_codes = hdc_codes(pool, proj, device)
        val_codes = hdc_codes(val, proj, device)

        # oracle probe: pool-refit
        clf_or = fit_probe(pool_codes, pl, max_iter=args.max_iter)

        # zero-shot / oracle mIoU
        zs_preds = predict_probe(val_codes, clf_zs)
        or_preds = predict_probe(val_codes, clf_or)
        zs_miou = compute_miou(zs_preds, vl)
        or_miou = compute_miou(or_preds, vl)
        gap = or_miou - zs_miou

        # 1. SHIFT TYPE: W_zs vs W_oracle cosine (per class + mean)
        Wz = torch.tensor(clf_zs.coef_)
        Wo = torch.tensor(clf_or.coef_)
        cl = torch.tensor(clf_zs.classes_)
        cos_per_class = {}
        for j, c in enumerate(cl.tolist()):
            wz = F.normalize(Wz[j], p=2, dim=0)
            wo = F.normalize(Wo[j], p=2, dim=0)
            cos_per_class[int(c)] = float((wz * wo).sum().item())
        mean_cos = float(np.mean(list(cos_per_class.values())))

        # 2. BIAS-ONLY: freeze W_zs, re-center intercept to pool class proportions
        pool_counts = torch.bincount(pl, minlength=NUM_CLASSES).float()
        pool_frac = pool_counts / pool_counts.sum().clamp(min=1)
        clf_bias = LogisticRegression(max_iter=1)  # reuse structure; we'll override coef_/intercept_
        clf_bias.classes_ = clf_zs.classes_
        clf_bias.coef_ = clf_zs.coef_
        # intercept from log prior ratio (pool vs clean proportions)
        clean_counts = torch.bincount(la_s, minlength=NUM_CLASSES).float()
        clean_frac = clean_counts / clean_counts.sum().clamp(min=1)
        prior_shift = torch.zeros(len(clf_zs.classes_))
        for j, c in enumerate(clf_zs.classes_.tolist()):
            pc = pool_frac[int(c)].item()
            cc = clean_frac[int(c)].item()
            prior_shift[j] = float(np.log((pc + 1e-8) / (cc + 1e-8)))
        clf_bias.intercept_ = clf_zs.intercept_ + prior_shift.numpy()
        bias_preds = predict_probe(val_codes, clf_bias)
        bias_miou = compute_miou(bias_preds, vl)
        bias_share = (bias_miou - zs_miou) / gap if gap > 0 else 0.0

        # 3. MARGIN decomposition on the frozen probe
        zs_scores = probe_scores(val_codes, clf_zs)
        zs_marg = margin(zs_scores)
        correct = zs_preds == vl
        wrong = ~correct
        fixed = (or_preds == vl) & wrong  # oracle fixes these
        mar_c = float(zs_marg[correct].mean().item())
        mar_w = float(zs_marg[wrong].mean().item())
        mar_f = float(zs_marg[fixed].mean().item()) if fixed.any() else float('nan')

        # 4. OUTLIER: code-space norm of correct / wrong / fixed
        nrm = val_codes.float().norm(p=2, dim=1)
        nrm_c = float(nrm[correct].mean().item())
        nrm_w = float(nrm[wrong].mean().item())
        nrm_f = float(nrm[fixed].mean().item()) if fixed.any() else float('nan')

        # 5. PER-CLASS oracle gain (the TTA/AL target classes)
        per_class = {}
        for c in classes:
            m = vl == c
            if int(m.sum().item()) < 100:
                continue
            zs_c = float((zs_preds[m] == vl[m]).float().mean().item())
            or_c = float((or_preds[m] == vl[m]).float().mean().item())
            per_class[int(c)] = {'zs_acc': zs_c, 'oracle_acc': or_c, 'gain': or_c - zs_c}

        # 6. POOL-SIZE CURVE: oracle mIoU vs labeled pool size (AL budget). The 100k
        # point is the already-fit oracle (reused, no refit); smaller pools show how
        # many labels close the gap. 1k/10k/50k cover the AL-budget range.
        pool_sizes = [1000, 10000, 50000]
        curve = {}
        for n in pool_sizes:
            ci_p = torch.randperm(len(pool_codes))[:n]
            clf_n = fit_probe(pool_codes[ci_p], pl[ci_p], max_iter=args.max_iter)
            curve[str(n)] = compute_miou(predict_probe(val_codes, clf_n), vl)
        if len(pool_codes) >= 100000:
            curve['100000'] = or_miou

        results[cond] = {
            'zs_miou': zs_miou, 'oracle_miou': or_miou, 'gap': gap,
            'shift': {'mean_cos_W': mean_cos, 'per_class_cos_W': cos_per_class},
            'bias_only': {'miou': bias_miou, 'share_of_gap': bias_share},
            'margin': {'zs_correct_mean': mar_c, 'zs_wrong_mean': mar_w,
                       'oracle_fixed_mean': mar_f},
            'norm': {'zs_correct_mean': nrm_c, 'zs_wrong_mean': nrm_w,
                     'oracle_fixed_mean': nrm_f},
            'per_class': per_class,
            'pool_curve': curve,
        }
        print(f"\n=== {cond}: zs {zs_miou:.4f} -> oracle {or_miou:.4f} (gap {gap:+.4f}) ===")
        print(f"  shift:   mean cos(W_zs, W_oracle) = {mean_cos:.4f}  (1 = pure translation, <<1 = rotation)")
        print(f"  bias-only miou {bias_miou:.4f} = {bias_share:.0%} of the gap (freeze W, re-center)")
        print(f"  margin:  zs-correct {mar_c:.3f}  zs-wrong {mar_w:.3f}  oracle-fixed {mar_f:.3f}")
        print(f"  norm:    zs-correct {nrm_c:.1f}  zs-wrong {nrm_w:.1f}  oracle-fixed {nrm_f:.1f}")
        print(f"  pool curve: " + "  ".join(f"{k}:{v:.4f}" for k, v in curve.items()))
        top = sorted(per_class.items(), key=lambda kv: -kv[1]['gain'])[:4]
        print(f"  top oracle-gain classes: " + ", ".join(
            f"{c}(+{v['gain']:.3f})" for c, v in top))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== WHAT TO LOOK FOR ===")
    print("1. shift.mean_cos_W ~1   -> the gap is a clean translation; bias-only TTA suffices.")
    print("2. bias_only.share_of_gap ~1 -> gradient-free option-1 (intercept re-center) closes")
    print("   most of the gap with zero weights update. If small, weights must move (options 2/3).")
    print("3. margin.oracle_fixed vs zs_wrong: if fixed points are LOW-margin, a margin-gated TTA")
    print("   has signal; if HIGH-margin, they are confident-but-wrong -> margins won't gate them.")
    print("4. norm.oracle_fixed vs zs_wrong: high-norm fixed = outliers (norm-gate veto); low-norm")
    print("   fixed = inlier boundary points (need boundary movement, not veto).")
    print("5. per_class: the largest-gain classes are the TTA/AL target.")
    print("6. pool_curve: how many labeled points close the gap (the AL budget).")

if __name__ == "__main__":
    main()
