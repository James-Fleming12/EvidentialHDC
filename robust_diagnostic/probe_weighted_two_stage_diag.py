"""probe_weighted_two_stage_diag.py: the Iteration-9 follow-up (eval-only).

Iteration 9 showed pseudo-label GATING is closed under the probe update: every hard
admit/veto gate stays at or below no_gate, because the gate that cleans the labels
also starves the Nystrom+CG update's covariance. Two levers that avoid the hard gate:

  A. WEIGHTED UPDATE: each pool point contributes to the covariance/prototype scaled
     by its confidence (soft weighting, not a hard admit/veto). Wrong pseudo-labels
     (low confidence) still contribute but weakly -- the covariance keeps ALL points'
     structure while down-weighting the poison. Implemented as the weighted ridge
     W = (X^T D X + lI)^-1 X^T D Y (D = diag(conf)), via weighted Nystrom warm start +
     weighted matrix-free CG.
  B. TWO-STAGE: fit the probe on the frozen pseudo-labels FIRST (no gate), then use
     the UPDATED probe's confidence to re-gate the pool and refit a second time. The
     first update may already clean the pseudo-labels enough that a second-round gate
     (which failed from the frozen start) now works.

References: frozen (no update), oracle (true labels), no_gate (all frozen
pseudo-labels, no weight). Gate = the best hard gate from Iteration 9 (selfcal
conf_top30%).

Weight schemes for A:
  w=conf        : weight = softmax confidence (soft)
  w=conf^2      : sharpen (more aggressive down-weighting of wrong points)
  w=margin      : weight = top-2 margin (normalized)

Two-stage for B:
  stage1: fit on all frozen pseudo-labels -> W1
  stage2: re-gate the pool by W1's confidence (conf_top30% or a selfcal quantile),
          refit on the gated pool -> W2
  Also a "soft two-stage": stage1 weighted, then stage2 weighted on the updated conf.

Reports mIoU (ceiling = pool-refit, decode val) for each, vs frozen/oracle/no_gate.

Usage:
  uv run python robust_diagnostic/probe_weighted_two_stage_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_weighted_2stage_covshift_ep10.json
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
from modules.oracle_core import get_hdc_projection, compute_miou

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


def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ W).argmax(dim=1))
    return torch.cat(preds)


def scores(W, codes, chunk=100000):
    W = W.detach().cpu()
    outs = []
    for s in range(0, len(codes), chunk):
        outs.append(codes[s:s + chunk].float() @ W)
    return torch.cat(outs, dim=0)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------- weighted probe update (Nystrom warm start + CG) ----------------

def nystrom_w0(codes, lbls, weights, lam, device, m=1000):
    """Weighted Nystrom warm start: W = P (P^T X^T D X P + lI)^-1 P^T X^T D Y.
    The sketch S_hat = P^T X^T D X P (m x m); the RHS is P^T X^T D Y (m x C)."""
    X = codes.float().to(device)
    Y = onehot(lbls, NUM_CLASSES).to(device)
    w = weights.float().to(device)
    torch.manual_seed(11)
    P = (torch.rand(codes.shape[1], m) > 0.5).float() * 2 - 1
    Xw = X * w.unsqueeze(1)                        # each row scaled by weight
    Shat = (X @ P.to(device)).t() @ (Xw @ P.to(device))   # P^T X^T D X P (m x m)
    That = (Xw @ P.to(device)).t() @ Y             # P^T X^T D Y (m x C)
    A = torch.linalg.solve(Shat + lam * torch.eye(m, device=device), That)
    return (P.to(device) @ A).float()


def weighted_cg(codes, lbls, weights, lam, device, iters=8, x0=None):
    """Weighted ridge W = (X^T D X + lI)^-1 X^T D Y via CG with matrix-free
    (X^T D X) v = X^T (D (X v))."""
    X = codes.float().to(device)
    w = weights.float().to(device)
    d = X.shape[1]
    C = NUM_CLASSES
    D = w.unsqueeze(1)
    x = x0.to(device).clone() if x0 is not None else torch.zeros(d, C, device=device)
    b = (X * D).t() @ onehot(lbls, C).to(device)   # X^T D Y
    def A(v):
        return X.t() @ (D * (X @ v))               # X^T D X v
    r = b - A(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha.unsqueeze(0) * p
        r = r - alpha.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x.float()


def weighted_probe_fit(codes, lbls, weights, lam, device, iters=8):
    x0 = nystrom_w0(codes, lbls, weights, lam, device)
    return weighted_cg(codes, lbls, weights, lam, device, iters=iters, x0=x0)


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
    parser.add_argument("--max_clean", type=int, default=200000)
    parser.add_argument("--nystrom_m", type=int, default=1000)
    parser.add_argument("--gate_frac", type=float, default=0.3,
                        help="selfcal gate fraction for the two-stage / hard-gate refs")
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_weighted_2stage_results.json")
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

        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]
        clean_codes = hdc_codes(fa[ci], proj, device)
        ones = torch.ones(len(clean_codes))
        W_clean = weighted_probe_fit(clean_codes, la[ci], ones, args.lam, device)

        # frozen pseudo-labels + signals
        sm = torch.softmax(scores(W_clean, pool_codes), dim=1)
        pconf = sm.max(dim=1).values
        ppred = sm.argmax(dim=1)
        top2 = torch.topk(sm, 2, dim=1).values
        pmargin = (top2[:, 0] - top2[:, 1]).clamp(min=0)
        pool_ones = torch.ones(len(pool))

        r = {'frozen': None, 'oracle': None, 'no_gate': None,
             'weighted': {}, 'two_stage': {}}

        r['frozen'] = compute_miou(decode(W_clean, val_codes), vl)
        r['oracle'] = compute_miou(decode(weighted_probe_fit(pool_codes, pl, pool_ones,
                                                             args.lam, device), val_codes), vl)
        r['no_gate'] = compute_miou(decode(weighted_probe_fit(pool_codes, ppred, pool_ones,
                                                              args.lam, device), val_codes), vl)

        # A. weighted update (soft weighting, no hard gate)
        for wname, w in [('w_conf', pconf), ('w_conf2', pconf ** 2),
                         ('w_margin', pmargin / pmargin.max().clamp(min=1e-8))]:
            Ww = weighted_probe_fit(pool_codes, ppred, w, args.lam, device)
            r['weighted'][wname] = compute_miou(decode(Ww, val_codes), vl)

        # B. two-stage: fit on frozen pseudo-labels, re-gate by updated conf, refit
        W1 = weighted_probe_fit(pool_codes, ppred, pool_ones, args.lam, device)
        sm1 = torch.softmax(scores(W1, pool_codes), dim=1)
        conf1 = sm1.max(dim=1).values
        pred1 = sm1.argmax(dim=1)
        # hard re-gate (selfcal conf_top frac)
        keep = conf1 >= torch.quantile(conf1, 1 - args.gate_frac)
        W2 = weighted_probe_fit(pool_codes[keep], pred1[keep], torch.ones(int(keep.sum())),
                                args.lam, device)
        r['two_stage']['hard_regate_%d' % int(100 * args.gate_frac)] = compute_miou(
            decode(W2, val_codes), vl)
        # soft two-stage: weighted by stage-1 conf, refit (no gate)
        W2s = weighted_probe_fit(pool_codes, pred1, conf1, args.lam, device)
        r['two_stage']['soft_weighted'] = compute_miou(decode(W2s, val_codes), vl)
        # hard two-stage with stage-2 conf (three rounds, gate again)
        sm2 = torch.softmax(scores(W2s, pool_codes), dim=1)
        conf2 = sm2.max(dim=1).values
        pred2 = sm2.argmax(dim=1)
        keep2 = conf2 >= torch.quantile(conf2, 1 - args.gate_frac)
        W3 = weighted_probe_fit(pool_codes[keep2], pred2[keep2], torch.ones(int(keep2.sum())),
                                args.lam, device)
        r['two_stage']['soft_then_hard'] = compute_miou(decode(W3, val_codes), vl)

        results['conds'][cond] = r
        print(f"\n--- {cond} ---")
        print(f"  frozen {r['frozen']:.4f} | oracle {r['oracle']:.4f} | no_gate {r['no_gate']:.4f}")
        print(f"  A weighted: " + " ".join(f"{k}:{v:.4f}" for k, v in r['weighted'].items()))
        print(f"  B two-stage: " + " ".join(f"{k}:{v:.4f}" for k, v in r['two_stage'].items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Iteration 9: hard gates stay <= no_gate (they starve the covariance). These")
    print("avoid the hard gate:")
    print("  A weighted (w=conf / conf^2 / margin): wrong points contribute weakly, all")
    print("    points' covariance is kept. If > no_gate toward oracle, soft weighting is")
    print("    the label-free lever.")
    print("  B two-stage: fit on frozen pseudo-labels, then re-gate / reweight by the")
    print("    UPDATED probe's confidence. If the second round climbs (vs Iteration 9's")
    print("    failed first-round gate), the first update cleaned the pseudo-labels.")


if __name__ == "__main__":
    main()
