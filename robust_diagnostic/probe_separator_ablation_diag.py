"""probe_separator_ablation_diag.py: separator-form ablation (eval-only).

Iteration 5 showed the low-rank correction (W = mu + VA, k=32/64) and the tiny gauge
do NOT capture the probe's rotation -- the useful rotation is not low-rank in a small
random subspace. This ablation tests DIFFERENT separator FORMS whose sufficient
statistics are first-order (or much cheaper than the full X^T X), to isolate WHERE
the probe's gain comes from:

  R1 prototype      : s_c(h) = mu_c . h. Class sums (O(Cd)). Baseline.
  Diagonal probe    : s_c(h) = (mu_c * q) . h, shared coordinate weighting q.
                     Tests if a class-INDEPENDENT reweighting captures the gain.
  Diagonal LDA      : s_c(h) = sum_j h_j mu_cj / sigma^2_cj. Per-class per-coordinate
                     weights. For +/-1 codes sigma^2_cj = 1 - mu_cj^2 (a closed form
                     of the first-order mean!) -- so it needs ONLY class sums. Tests
                     if the gain is coordinate-wise reweighting.
  Shared diagonal   : s_c(h) = sum_j h_j mu_cj / sigma^2_j (pooled sigma^2_j, O(d)).
                     The domain-wide rotation hypothesis (shared transform).
  Batch perceptron  : init W = mu, then W_y += eta h, W_yhat -= eta h on mistakes,
                     batched as a matmul (O(n d C) per epoch, first-order). Tests if
                     a genuine separator can be learned WITHOUT covariance.
  Passive-aggressive: like perceptron but margin-based step eta = loss/||h||^2 = loss/d.
  Nystrom (m=1000)  : the current best cheap approximation (sketch covariance).
  Full ridge        : W = (X^T X + lI)^{-1} X^T Y (the ceiling, O(nd^2)).

For each: mIoU (ceiling = fit on the corrupted labeled pool, decode val), update
wall-clock, update pts/s, and the statistic order (first vs second). The two key
questions the feedback poses:
  A. Does DIAGONAL LDA capture most of the R4 gain? (=> the gain is coordinate-wise
     reweighting, learnable with prototype-like statistics.)
  B. Does the BATCH PERCEPTRON / PA produce the boundary rotation from first-order
     mistake stats? (=> linear separability does NOT need second-order statistics.)

Usage:
  uv run python robust_diagnostic/probe_separator_ablation_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_separator_ablation_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import resource
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

def class_sums(codes, lbls, num_classes=NUM_CLASSES):
    """Per-class coordinate sums s_cj = sum_{i:y_i=c} h_ij and counts. FIRST-ORDER
    statistics -- the same accumulator as the prototype update."""
    S = torch.zeros(num_classes, codes.shape[1])
    n = torch.zeros(num_classes)
    for c in range(1, num_classes):
        m = lbls == c
        if int(m.sum().item()) > 0:
            S[c] = codes[m].float().sum(dim=0)
            n[c] = int(m.sum().item())
    return S, n

def decode(W, codes, chunk=100000):
    """argmax over the full class space of codes @ W (W: d x C)."""
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ W).argmax(dim=1))
    return torch.cat(preds)

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
    parser.add_argument("--eta", type=float, default=1e-2, help="perceptron/PA step")
    parser.add_argument("--perceptron_epochs", type=int, default=5)
    parser.add_argument("--nystrom_m", type=int, default=1000)
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_separator_ablation_results.json")
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
        S, n = class_sums(pool_codes, pl)
        mu = S / n.clamp(min=1).unsqueeze(1)              # (C, d) class means

        r = {}
        # ---- R1 prototype (baseline): W = mu ----
        t0 = time.time()
        W_proto = mu.t()                                   # (d, C)
        r['r1_proto'] = {'miou': compute_miou(decode(W_proto, val_codes), vl),
                         'update_s': time.time() - t0, 'order': '1st'}

        # ---- Diagonal LDA: W_cj = mu_cj / (1 - mu_cj^2) (first-order closed form) ----
        # For +/-1 codes sigma^2_cj = E[h^2] - mu^2 = 1 - mu_cj^2, so the diagonal-LDA
        # weight is mu_cj/sigma^2_cj, a pure function of the class mean. Class-specific.
        t0 = time.time()
        W_diag_lda = (mu / (1 - mu ** 2 + args.lam)).t()  # (d, C)
        r['diag_lda'] = {'miou': compute_miou(decode(W_diag_lda, val_codes), vl),
                         'update_s': time.time() - t0, 'order': '1st'}

        # ---- Diagonal probe: shared coordinate weighting q, W_c = q * mu_c ----
        # q_j = 1/var_j (pooled across classes) -- the shared-diagonal transform.
        var_j = (mu ** 2).mean(dim=0)                      # pooled per-coord "variance"
        q = 1.0 / (var_j + args.lam)                       # (d,)
        W_shared = (mu * q).t()                            # (d, C) W_c = q .* mu_c
        r['shared_diag'] = {'miou': compute_miou(decode(W_shared, val_codes), vl),
                            'update_s': 0.0, 'order': '1st'}

        # ---- Batch perceptron: init W = mu, correct mistakes by adding/subtracting h ----
        X = pool_codes.float()
        Y = onehot(pl, NUM_CLASSES).float()
        Wp = mu.t().clone()                                # (d, C)
        t0 = time.time()
        for _ in range(args.perceptron_epochs):
            scores = X @ Wp
            preds = scores.argmax(dim=1)
            bad = preds != pl
            if not bad.any():
                break
            # W_y += eta h, W_yhat -= eta h  -> batched as a matmul
            target = onehot(pl[bad], NUM_CLASSES).float() - onehot(preds[bad], NUM_CLASSES).float()
            Wp = Wp + args.eta * (X[bad].t() @ target)
        r['perceptron'] = {'miou': compute_miou(decode(Wp, val_codes), vl),
                           'update_s': time.time() - t0, 'order': '1st'}

        # ---- Passive-aggressive: margin-based step, eta_i = loss/||h||^2 = loss/d ----
        Wpa = mu.t().clone()
        t0 = time.time()
        for _ in range(args.perceptron_epochs):
            scores = X @ Wpa
            true_score = scores.gather(1, pl.long().unsqueeze(1)).squeeze(1)
            margin = (scores - true_score.unsqueeze(1) + 1.0)
            margin[torch.arange(len(pl)), pl.long()] = 0.0
            worst = margin.argmax(dim=1)
            loss = margin.gather(1, worst.unsqueeze(1)).squeeze(1).clamp(min=0)
            bad = loss > 0
            if not bad.any():
                break
            step = loss[bad] / 10000.0                      # /||h||^2 = /d
            nb = int(bad.sum().item())
            tgt = torch.zeros(nb, NUM_CLASSES)
            tgt[torch.arange(nb), pl[bad].long()] = step
            tgt[torch.arange(nb), worst[bad].long()] = -step
            Wpa = Wpa + X[bad].t() @ tgt
        r['passive_agg'] = {'miou': compute_miou(decode(Wpa, val_codes), vl),
                            'update_s': time.time() - t0, 'order': '1st'}

        # ---- Nystrom sketch (current best cheap approx) ----
        torch.manual_seed(11)
        P = (torch.rand(10000, args.nystrom_m) > 0.5).float() * 2 - 1
        t0 = time.time()
        XP = X @ P
        Shat = XP.t() @ XP
        That = XP.t() @ Y
        A = torch.linalg.solve(Shat + args.lam * torch.eye(args.nystrom_m), That)
        W_ny = P @ A                                       # (d, C)
        r['nystrom'] = {'miou': compute_miou(decode(W_ny, val_codes), vl),
                        'update_s': time.time() - t0, 'order': '2nd(sketch)'}

        # ---- Full ridge (the ceiling) ----
        t0 = time.time()
        W_full = torch.linalg.solve(X.t() @ X + args.lam * torch.eye(10000), X.t() @ Y)
        r['full_ridge'] = {'miou': compute_miou(decode(W_full, val_codes), vl),
                           'update_s': time.time() - t0, 'order': '2nd'}

        results['conds'][cond] = r
        print(f"\n--- {cond} ---")
        print(f"  {'separator':<16} {'mIoU':>7} {'order':>10} {'update_s':>9}")
        for name, rr in r.items():
            print(f"  {name:<16} {rr['miou']:>7.4f} {rr['order']:>10} {rr['update_s']:>9.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. Does DIAG-LDA capture most of the R4 gain? (W_cj = mu_cj/(1-mu_cj^2),")
    print("   FIRST-ORDER class sums only.) If yes, the gain is coordinate reweighting.")
    print("B. Does BATCH PERCEPTRON / PA produce the rotation from first-order mistake")
    print("   stats? If yes, linear separability does NOT need second-order statistics.")
    print("Compare each to R1 (baseline) and full_ridge (ceiling): which first-order")
    print("separator gets closest to the ceiling, and at what update cost.")

if __name__ == "__main__":
    main()
