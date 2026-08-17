"""probe_pseudo_gate_diag.py: pseudo-label gating for the Nystrom+CG probe update
(eval-only).

The old geometric-gate line was closed under the PROTOTYPE-EMA decoder. The decoder
is now a learned probe with the Nystrom-warm-start + matrix-free-CG update, so the
gate question deserves a fresh look: which standard pseudo-label gates, applied to
the corrupted pool, let the probe update recover the oracle gain label-free?

Pipelines per gate mode (pool = corrupted labeled points; pseudo-labels from the
FROZEN CLEAN probe; the gate decides which pool points are admitted to the update):
  frozen       : decode val with the frozen clean probe (no update) -- reference.
  oracle       : update with TRUE labels (the ceiling) -- reference.
  no_gate      : update with ALL pseudo-labels.
  conf_{t}     : keep points with top-1 probe confidence >= t.
  margin_{t}   : keep points with top-2 margin >= t.
  norm_{t}     : keep points with low 128-d feature norm (z-scored) <= t.
  uncer_{t}    : fuse_uncertainties(epistemic=1-conf, geometric=norm-z), keep w >= t.
  prior        : bias-only update (freeze W, re-center b to pool class proportions).

The update is the Iteration-8 winner: Nystrom warm start (m=1000) + matrix-free
CG-8, fit on the gated pseudo-labeled points.

DIAGNOSTICS (where methods go wrong / what the features need to filter):
  1. GATE AUROC: for each gate signal (conf/margin/norm/uncer), the AUROC for
     separating CORRECT from WRONG pseudo-labels on the pool. If ~0.5 the signal
     cannot discriminate (no gate can help); if high, a threshold exists.
  2. Per-class pseudo-label accuracy: which classes the frozen probe gets right on
     the corrupted pool (the TTA-relevant target classes vs the fragile ones).
  3. Retain-vs-precision: for a confidence/margin threshold sweep, the fraction of
     pool retained vs the precision of the retained pseudo-labels -- the gate
     operating curve.
  4. Wrong-label feature profile: for the wrongly-pseudo-labeled points, their
     confidence / margin / norm distributions vs the correct ones (what a filter
     would need to separate them).

Usage:
  uv run python robust_diagnostic/probe_pseudo_gate_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_pseudo_gate_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import yaml
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou
from modules.HDC_utils import fuse_uncertainties

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

def tic():
    sync()
    return time.time()

def toc(t0):
    sync()
    return time.time() - t0

def cg_solve(X, T, lam, device, iters=8, x0=None):
    X = X.to(device)
    d = X.shape[1]
    C = T.shape[1]
    x = x0.to(device).clone() if x0 is not None else torch.zeros(d, C, device=device)
    def A(v):
        return X.T @ (X @ v)
    b = T.to(device)
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

def nystrom_w0(codes, lbls, lam, device, m=1000):
    X = codes.float().to(device)
    Y = onehot(lbls, NUM_CLASSES).to(device)
    torch.manual_seed(11)
    P = (torch.rand(codes.shape[1], m) > 0.5).float() * 2 - 1
    XP = X @ P.to(device)
    Shat = XP.T @ XP
    That = XP.T @ Y
    A = torch.linalg.solve(Shat + lam * torch.eye(m, device=device), That)
    return (P.to(device) @ A).float()

def probe_fit(codes, lbls, lam, device):
    """Nystrom warm start + matrix-free CG-8 (the Iteration-8 update)."""
    x0 = nystrom_w0(codes, lbls, lam, device)
    T = codes.float().to(device).t() @ onehot(lbls, NUM_CLASSES).to(device)
    return cg_solve(codes, T, lam, device, iters=8, x0=x0)

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
    parser.add_argument("--nystrom_m", type=int, default=1000)
    parser.add_argument("--max_clean", type=int, default=200000,
                        help="cap on the clean points used for the frozen probe fit "
                             "(binarizing the full 8M-point clean pool is 320GB)")
    parser.add_argument("--conf_sweep", type=str, default="0.05,0.1,0.15,0.2")
    parser.add_argument("--margin_sweep", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--norm_sweep", type=str, default="60,80,95")
    parser.add_argument("--selfcal_fracs", type=str, default="0.5,0.3,0.1",
                        help="self-calibrating gates: keep the top-K% of pool points by "
                             "each signal (K from the pool's own distribution, no manual "
                             "threshold -- the no-heavy-tuning requirement)")
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_pseudo_gate_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    conf_sweep = [float(x) for x in args.conf_sweep.split(',')]
    margin_sweep = [float(x) for x in args.margin_sweep.split(',')]
    norm_sweep = [float(x) for x in args.norm_sweep.split(',')]
    selfcal_fracs = [float(x) for x in args.selfcal_fracs.split(',')]

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

        # frozen clean probe (the pseudo-label source + frozen reference). BOUNDED
        # clean sample (binarizing the full 8M clean pool is 320GB).
        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]
        W_clean = probe_fit(hdc_codes(fa[ci], proj, device), la[ci], args.lam, device)

        # pseudo-labels + gate signals on the corrupted pool
        pool_s = scores(W_clean, pool_codes)
        sm = torch.softmax(pool_s, dim=1)
        pconf = sm.max(dim=1).values
        ppred = sm.argmax(dim=1)
        top2 = torch.topk(sm, 2, dim=1).values
        pmargin = top2[:, 0] - top2[:, 1]
        pnorm = torch.norm(pool.float(), p=2, dim=1)
        # epistemic = 1 - conf; geometric = norm z-score (higher = worse)
        u_epi = 1.0 - pconf
        z_geom = (pnorm - pnorm.mean()) / (pnorm.std() + 1e-8)

        pseudo_correct = (ppred == pl)

        r = {'frozen': None, 'oracle': None, 'no_gate': None,
             'conf': {}, 'margin': {}, 'norm': {}, 'uncer': {}, 'selfcal': {},
             'diag': {}}

        # references
        r['frozen'] = compute_miou(decode(W_clean, val_codes), vl)
        r['oracle'] = compute_miou(decode(probe_fit(pool_codes, pl, args.lam, device), val_codes), vl)
        r['no_gate'] = compute_miou(decode(probe_fit(pool_codes, ppred, args.lam, device), val_codes), vl)

        # gates: keep pool points, refit the probe on the gated pseudo-labels
        def gated_fit(mask):
            if mask.sum() < 100:
                return None
            return compute_miou(decode(probe_fit(pool_codes[mask], ppred[mask],
                                                 args.lam, device), val_codes), vl)

        for t in conf_sweep:
            r['conf'][str(t)] = gated_fit(pconf >= t)
        for t in margin_sweep:
            r['margin'][str(t)] = gated_fit(pmargin >= t)
        for t in norm_sweep:
            r['norm'][str(t)] = gated_fit(pnorm <= t)
        for t in [0.3, 0.5, 0.7]:
            w = fuse_uncertainties(u_epi, z_geom, method='soft_dual_weight',
                                   cfg={"u_th": 0.5, "u_coef": 1.5, "z_th": 0.5, "z_coef": 1.0})
            r['uncer'][str(t)] = gated_fit(w >= t)

        # SELF-CALIBRATING gates: keep the top-K% of pool points by each signal, where
        # K is a QUANTILE of the pool's own distribution (no hand-tuned absolute
        # threshold -- the no-heavy-tuning requirement). Works per condition.
        for f in selfcal_fracs:
            n_keep = int(len(pool) * f)
            # top-K by confidence / margin, bottom-K by norm, top-K by uncer-decay
            r['selfcal'][f'conf_top{f}'] = gated_fit(
                pconf >= torch.quantile(pconf, 1 - f))
            r['selfcal'][f'margin_top{f}'] = gated_fit(
                pmargin >= torch.quantile(pmargin, 1 - f))
            r['selfcal'][f'norm_bot{f}'] = gated_fit(
                pnorm <= torch.quantile(pnorm, f))
            w = fuse_uncertainties(u_epi, z_geom, method='soft_dual_weight',
                                   cfg={"u_th": 0.5, "u_coef": 1.5, "z_th": 0.5, "z_coef": 1.0})
            r['selfcal'][f'uncer_top{f}'] = gated_fit(
                w >= torch.quantile(w, 1 - f))

        # ---- diagnostics ----
        diag = r['diag']
        # 1. gate AUROC: can each signal separate correct from wrong pseudo-labels?
        diag['auroc'] = {}
        for name, sig in [('conf', pconf), ('margin', pmargin), ('norm', -pnorm),
                          ('uncer', 1.0 - u_epi)]:
            try:
                diag['auroc'][name] = float(roc_auc_score(pseudo_correct.numpy(), sig.numpy()))
            except Exception:
                diag['auroc'][name] = None
        # 2. per-class pseudo-label accuracy on the pool
        per_class = {}
        for c in range(1, NUM_CLASSES):
            m = pl == c
            if int(m.sum().item()) > 100:
                per_class[str(c)] = float(pseudo_correct[m].float().mean().item())
        diag['per_class_pseudo_acc'] = per_class
        # 3. retain-vs-precision for the confidence gate
        retain_prec = {}
        for t in conf_sweep + [0.6, 0.8, 0.95]:
            keep = pconf >= t
            nk = int(keep.sum().item())
            prec = float(pseudo_correct[keep].float().mean().item()) if nk > 0 else 0.0
            retain_prec[str(t)] = {'retain': nk / len(pool), 'precision': prec}
        diag['retain_vs_precision'] = retain_prec
        # 4. wrong-label feature profile (conf/margin/norm of wrong vs correct)
        wrong = ~pseudo_correct
        diag['wrong_profile'] = {
            'conf': {'correct': float(pconf[pseudo_correct].mean().item()),
                     'wrong': float(pconf[wrong].mean().item())},
            'margin': {'correct': float(pmargin[pseudo_correct].mean().item()),
                       'wrong': float(pmargin[wrong].mean().item())},
            'norm': {'correct': float(pnorm[pseudo_correct].mean().item()),
                     'wrong': float(pnorm[wrong].mean().item())},
        }

        results['conds'][cond] = r
        print(f"\n--- {cond} ---")
        print(f"  frozen {r['frozen']:.4f} | oracle {r['oracle']:.4f} | no_gate {r['no_gate']:.4f}")
        print(f"  conf: " + " ".join(f"{t}:{v:.4f}" if v else f"{t}:skip" for t, v in r['conf'].items()))
        print(f"  margin: " + " ".join(f"{t}:{v:.4f}" if v else f"{t}:skip" for t, v in r['margin'].items()))
        print(f"  norm: " + " ".join(f"{t}:{v:.4f}" if v else f"{t}:skip" for t, v in r['norm'].items()))
        print(f"  uncer: " + " ".join(f"{t}:{v:.4f}" if v else f"{t}:skip" for t, v in r['uncer'].items()))
        print(f"  SELFCAL (top-K%, no manual threshold): " + " ".join(
            f"{k}:{v:.4f}" if v else f"{k}:skip" for k, v in r['selfcal'].items()))
        print(f"  DIAG gate AUROC: " + " ".join(f"{k}:{v:.3f}" if v else f"{k}:na" for k, v in diag['auroc'].items()))
        print(f"  DIAG pseudo-label acc: correct {pseudo_correct.float().mean():.3f} | "
              f"wrong conf {diag['wrong_profile']['conf']['wrong']:.3f} vs correct {diag['wrong_profile']['conf']['correct']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Pipelines: does gating the pseudo-labeled pool (conf/margin/norm/uncertainty)")
    print("let the Nystrom+CG update climb from no_gate toward the oracle ceiling?")
    print("  - If a gate reaches near-oracle: a simple pseudo-label filter works.")
    print("  - SELFCAL rows are the no-tuning version: keep the top-K% by each signal, K")
    print("    a quantile of the pool's own distribution (works per condition, no manual")
    print("    threshold). If a selfcal gate reaches near-oracle, the method is adaptive.")
    print("  - If ALL gates stay near no_gate: the wrong pseudo-labels poison the update")
    print("    and no basic gate separates them (the old closure, re-tested on the probe).")
    print("DIAGNOSTICS (where it goes wrong / what the features need to filter):")
    print("  auroc: can each gate signal separate correct from wrong pseudo-labels?")
    print("    ~0.5 = no filter can help; high = a threshold exists (then why does the")
    print("    gate sweep fail to exploit it?).")
    print("  per_class_pseudo_acc: which classes the frozen probe gets right on the")
    print("    corrupted pool (the TTA-target vs fragile classes).")
    print("  retain_vs_precision: the gate operating curve (retain vs pseudo precision).")
    print("  wrong_profile: conf/margin/norm of wrong vs correct pseudo-labels -- what a")
    print("    filter would need to separate them (are they LOW-margin, HIGH-norm, etc.).")

if __name__ == "__main__":
    main()
