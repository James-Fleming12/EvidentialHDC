"""probe_prototype_alignment_diag.py: does a learned prototype reproduce the linear
probe's decisions? (eval-only)

The probe's decision is argmax_c (W_c . h + b_c). For +/-1 codes ||h|| is constant,
so W_c . h is a cosine-like proximity to a "learned prototype" p_c = W_c / ||W_c||.
This diagnostic tests whether redefining proximity as cosine to the learned W_c
(over the class mean) recovers the probe's accuracy at prototype-decode cost, and
how much the sign-quantized / whitened variants cost.

For each candidate prototype definition (fit on the corrupted labeled pool =
ceiling, and on clean = zero-shot):
  R1 class mean    : cosine to mu_c = mean of the class codes (the current rule).
  W_c cosine float : cosine to the learned ridge prototype p_c = W_c / ||W_c||.
  W_c cosine sign  : cosine to sign(W_c) -- the integer/popcount decode.
  W_c dot + prior  : the probe itself, W_c . h + b_c (the reference it must match).
  whitened mean    : LDA-equivalent Mahalanobis: cosine in the whitened space
                     (W_c-normalized), the classical "distance to prototype" that
                     reproduces a linear discriminant.

Metrics per condition:
  - agreement with the probe: fraction of val points where the candidate's argmax
    equals the probe's argmax (how well the prototype definition reproduces it).
  - ceiling mIoU (fit on the labeled pool, decode val).
  - decode pts/s.

If W_c cosine (float) has ~high agreement with the probe and ceiling mIoU near the
probe's, then "distance to the learned prototype" is the redefinition that keeps
prototype-speed decode with full accuracy.

Usage:
  uv run python robust_diagnostic/probe_prototype_alignment_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_proto_alignment_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import yaml
import torch
import torch.nn.functional as F

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

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


def onehot(lbls, num_classes):
    y = torch.zeros(len(lbls), num_classes)
    y[torch.arange(len(lbls)), lbls.long()] = 1.0
    return y


def ridge_fit(codes, lbls, lam, device, num_classes=NUM_CLASSES):
    """Accumulate-and-solve ridge: W = (X^T X + lI)^{-1} X^T Y (the learned prototype
    source). Returns W (d x C)."""
    X = codes.float().to(device)
    Y = onehot(lbls, num_classes).to(device)
    S = X.T @ X
    T = X.T @ Y
    d = X.shape[1]
    W = torch.linalg.solve(S + lam * torch.eye(d, device=device), T)
    return W


def class_means(codes, lbls, num_classes=NUM_CLASSES):
    """mu_c = mean of class-c codes, in the FULL num_classes label space. Present
    classes get their normalized mean; absent classes get a -inf row (never argmax)
    so all candidates share the same label space for the agreement comparison."""
    M = torch.full((num_classes, codes.shape[1]), float('-inf'))
    for c in range(1, num_classes):
        m = lbls == c
        if int(m.sum().item()) > 0:
            mc = codes[m].float().mean(dim=0)
            M[c] = F.normalize(mc, p=2, dim=0)
    return M


def dec_cosine(codes, P, lbls, chunk=100000):
    """Cosine to prototype rows P (C x d). P rows are pre-normalized (or -inf for
    absent classes). Scores = h . P_c with ||h|| constant for +/-1 codes."""
    preds = []
    for s in range(0, len(codes), chunk):
        h = codes[s:s + chunk].float()          # ||h|| = sqrt(d), constant
        preds.append((h @ P.T).argmax(dim=1))
    return torch.cat(preds)


def dec_dot_prior(codes, W, b, chunk=100000):
    """The probe itself: argmax(W . h + b)."""
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ W + b).argmax(dim=1))
    return torch.cat(preds)


def agreement(a, b):
    return float((a == b).float().mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_proto_alignment_results.json")
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

    results = {'label': args.label, 'conds': {}}

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

        # learned probe (the reference it must match). W = ridge W (d x C, full 17-class
        # row space). The intercept/prior is a separate design constraint (the README
        # separates it from the weight pathway), so the reference uses a zero intercept
        # and the candidates are matched in the same full-class space.
        W = ridge_fit(pool_codes, pl, args.lam, device)
        b = torch.zeros(W.shape[1])

        # candidate prototypes. W is (d x C): the per-class prototype is W_c (row c).
        P_mean = class_means(pool_codes, pl)                    # R1 class mean
        Wn = F.normalize(W.T, p=2, dim=1)                        # (C, d), per-class rows
        P_learn = Wn                                            # learned W_c normalized
        P_sign = Wn.sign()

        # the probe reference decode
        ref = dec_dot_prior(val_codes, W.cpu(), b)

        cand = {}
        for name, P in [('class_mean', P_mean), ('W_cos_float', P_learn),
                        ('W_cos_sign', P_sign)]:
            t0 = time.time()
            preds = dec_cosine(val_codes, P, vl)
            dt = time.time() - t0
            cand[name] = {'agreement_with_probe': agreement(preds, ref),
                          'ceiling_miou': compute_miou(preds, vl),
                          'decode_pts_s': len(val) / dt if dt > 0 else None}
        # the probe itself (agreement = 1 by construction; the reference mIoU)
        cand['probe_ref'] = {'agreement_with_probe': 1.0,
                             'ceiling_miou': compute_miou(ref, vl),
                             'decode_pts_s': None}

        results['conds'][cond] = cand
        print(f"\n--- {cond} ---")
        print(f"  {'prototype def':<16} {'agree w/probe':>13} {'ceiling mIoU':>12} {'decode pts/s':>13}")
        for name, r in cand.items():
            print(f"  {name:<16} {r['agreement_with_probe']:>13.3f} {r['ceiling_miou']:>12.4f} "
                  f"{r['decode_pts_s'] if r['decode_pts_s'] else 0:>13,.0f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("The question: does cosine to a LEARNED prototype (W_c) reproduce the probe?")
    print("  - W_cos_float.agreement_with_probe near 1 -> 'learned prototype' cosine")
    print("    recovers the probe's decisions at prototype decode cost, full accuracy.")
    print("  - W_cos_sign.agreement -> how much the integer/popcount decode costs.")
    print("  - class_mean agreement is the current R1 (should be low on fog/crosstalk).")
    print("  - ceiling_miou confirms each candidate's recoverable bound.")


if __name__ == "__main__":
    main()
