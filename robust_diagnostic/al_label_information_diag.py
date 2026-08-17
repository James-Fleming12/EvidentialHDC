"""al_label_information_diag.py: what does ONE true label tell us about the
decision rule? (eval-only, no plots). Iteration 3.

Iteration 2 showed the bottleneck is ANCHOR DENSITY, not the propagation rule:
with 8-68 queried anchors, neither cluster-grounding nor graph diffusion beats
the frozen probe. So this diagnostic stops trying to expand labels and instead
asks, for a tiny queried set (1-2 labels per class by the validated influence
class-floor rule): which expansion of K labels into T is the most trustworthy,
and can the labels directly estimate the decision-rule correction?

The clean experiment: compare THREE ways of turning K labels into T, as
T-label precision vs T coverage (BEFORE any ridge update):

  A. NEAREST-ANCHOR propagation: x gets the label of its highest-cosine
     queried anchor if the cosine exceeds a threshold (swept). Assumption:
     local geometry carries labels.
  B. CLASS-CENTROID cosine: x_c = mean of the queried anchors for class c;
     x gets argmax_c cos(x, x_c) (with a margin threshold sweep). Assumption:
     class-level geometry carries labels (the near-unimodality claim).
  C. DECISION correction: from the queried points build the lookup
     (frozen_pred, margin_bin) -> true label; apply to the pool where the bin
     matches. Assumption: the probe's failures are locally systematic
     (the decision-rule rotation).

Plus, per condition:
  - direct-sparse ridge: K labels in T, NO expansion (the fallback).
  - soft confusion matrix: Y_i = C[hat_y_i, :] with C_ij = P(y=j | hat_y=i)
    estimated from the queried points (the cheapest possible expansion; the
    ridge already supports soft Y), plus the ORACLE-C ceiling (C from all pool
    labels -- if even that cannot beat frozen, the family is closed).
  - confusion stability: how far the queried-estimated C is from the oracle C
    (per-class row cosine) -- the corruption-induced confusion structure.
  - oracle-shift-prototype decode as a reference (Iteration-2 finding).

All operations are cosine decodes / lookup tables / the existing ridge: no
graph, no clustering, no diffusion.

Usage:
  uv run python robust_diagnostic/al_label_information_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_label_information_covshift_ep10.json
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

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
DGLSSPP_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'
DGLSSPP_METHOD = 'supcon_vib_dglsspp'
SKETCH_SEED = 11

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

# ---------------- Nystrom influence (the validated query signal) ----------------

def nystrom_influence(Xd, lam, m, device):
    """I_i ~= ||(S + lI)^-1 x_i|| in the Nystrom subspace."""
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    M = torch.linalg.inv(Shat)
    C = Xd @ P
    MC = C @ M
    return (MC.norm(dim=1) * (Xd.shape[1] ** 0.5)).cpu()

# ---------------- the ridge (soft Y supported) ----------------

def ridge_fit_soft(Xd, Y, lam, iters, m, device):
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    That = XP.t() @ Yd
    x = P @ torch.linalg.solve(Shat, That)
    b = Xd.t() @ Yd
    def A(v):
        return Xd.t() @ (Xd @ v)
    r = b - A(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha_k = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha_k.unsqueeze(0) * p
        r = r - alpha_k.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x.float()

# ---------------- main ----------------

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
    parser.add_argument("--cg_iters", type=int, default=8)
    parser.add_argument("--labels_per_class", type=int, default=1,
                        help="queried labels per class (1 or 2)")
    parser.add_argument("--margin_bins", type=str, default="0.1,0.2,0.3,0.5,0.8")
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_label_information_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    margin_bins = [float(x) for x in args.margin_bins.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    results = {'label': args.label, 'conds': {}}

    for cond in conds:
        t_cond = tic()
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

        proto_pairs = []
        for c in range(1, NUM_CLASSES):
            m = la[ci] == c
            if int(m.sum().item()) > 0:
                proto_pairs.append((c, clean_codes[m].float().mean(dim=0)))
        proto_ids = torch.tensor([c for c, _ in proto_pairs])
        proto_mat = torch.stack([p for _, p in proto_pairs])
        proto_mat = proto_mat / (proto_mat.norm(dim=1, keepdim=True) + 1e-8)

        Xc = clean_codes.float().to(device)
        W_clean = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam,
                                 args.cg_iters, args.nystrom_m, device)
        del clean_codes
        Xd = pool_codes.float().to(device)

        sm = torch.softmax(scores(W_clean, pool_codes), dim=1)
        pconf = sm.max(dim=1).values
        ppred = sm.argmax(dim=1)
        pseudo_correct = (ppred == pl)
        proto_scores = pool_codes.float() @ proto_mat.t()
        proto_pred = proto_ids[proto_scores.argmax(dim=1)]
        top2p = torch.topk(proto_scores, 2, dim=1).values
        proto_margin = (top2p[:, 0] - top2p[:, 1]).clamp(min=0)
        I = nystrom_influence(Xd, args.lam, args.nystrom_m, device)

        r = {'refs': {}, 'query': {}, 'expansion': {}, 'ridge': {},
             'confusion': {}, 'synthesis': []}

        def mw(W):
            return compute_miou(decode(W, val_codes), vl)

        W_oracle = ridge_fit_soft(Xd, onehot(pl, NUM_CLASSES), args.lam,
                                  args.cg_iters, args.nystrom_m, device)
        r['refs'] = {'frozen': mw(W_clean), 'oracle': mw(W_oracle),
                     'pseudo_acc': float(pseudo_correct.float().mean().item())}

        # ---- query K labels per class by the influence class-floor rule ----
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        qidx = []
        for c in classes:
            m = pl == c
            if int(m.sum().item()) == 0:
                continue
            j = int(I[m].argmax().item())
            qidx.append(int(m.nonzero().squeeze(1)[j]))
            if args.labels_per_class > 1:
                m2 = m.clone()
                m2[qidx[-1]] = False
                if int(m2.sum().item()) > 0:
                    j2 = int(I[m2].argmax().item())
                    qidx.append(int(m2.nonzero().squeeze(1)[j2]))
        qidx = torch.tensor(qidx)
        qlbl = pl[qidx]
        r['query'] = {'n': int(len(qidx)),
                      'labels_per_class': args.labels_per_class,
                      'anchor_acc': float((ppred[qidx] == qlbl).float().mean().item())}

        qcodes = pool_codes[qidx].float()
        qlbl_onehot = onehot(qlbl, NUM_CLASSES)

        # ---- A. nearest-anchor propagation: precision vs coverage ----
        sims_a = pool_codes.float() @ qcodes.t()          # n x K
        best_sim, best_a = sims_a.max(dim=1)
        best_lbl = qlbl[best_a]
        a_curve = []
        for tau in [0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            m_keep = best_sim >= tau
            nk = int(m_keep.sum().item())
            prec = float((best_lbl[m_keep] == pl[m_keep]).float().mean().item()) if nk > 0 else None
            a_curve.append({'tau': tau, 'coverage': nk / len(pool),
                            'precision': prec, 'n': nk})
        r['expansion']['A_nearest_anchor'] = a_curve

        # ---- B. class-centroid cosine: precision vs coverage ----
        centroids = []
        for c in classes:
            m = qlbl == c
            if int(m.sum().item()) > 0:
                centroids.append(qcodes[m].mean(dim=0))
        cid = torch.tensor([c for c in classes if int((qlbl == c).sum().item()) > 0])
        c_mat = torch.stack(centroids)
        c_mat = c_mat / (c_mat.norm(dim=1, keepdim=True) + 1e-8)
        sims_b = pool_codes.float() @ c_mat.t()
        best_b, bi = sims_b.max(dim=1)
        top2b = torch.topk(sims_b, 2, dim=1).values
        b_margin = (top2b[:, 0] - top2b[:, 1]).clamp(min=0)
        b_lbl = cid[bi]
        b_curve = []
        for t in [0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            m_keep = best_b >= t
            nk = int(m_keep.sum().item())
            prec = float((b_lbl[m_keep] == pl[m_keep]).float().mean().item()) if nk > 0 else None
            b_curve.append({'tau': t, 'coverage': nk / len(pool),
                            'precision': prec, 'n': nk})
        r['expansion']['B_class_centroid'] = b_curve
        r['expansion']['B_margin_curve'] = []
        for t in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
            m_keep = b_margin >= t
            nk = int(m_keep.sum().item())
            prec = float((b_lbl[m_keep] == pl[m_keep]).float().mean().item()) if nk > 0 else None
            r['expansion']['B_margin_curve'].append({'tau': t, 'coverage': nk / len(pool),
                                                     'precision': prec, 'n': nk})

        # ---- C. decision correction: (frozen_pred, margin_bin) -> true label ----
        pmargin = (pconf * 0 + (sm.topk(2, dim=1).values[:, 0] - sm.topk(2, dim=1).values[:, 1])).clamp(min=0)
        bins = [0.0] + margin_bins + [1e9]
        lookup = {}
        for i in range(len(bins) - 1):
            for c in range(NUM_CLASSES):
                lookup[(c, i)] = []
        for j in range(len(qidx)):
            c = int(ppred[qidx[j]].item())
            m = float(pmargin[qidx[j]].item())
            bi_ = int(np.searchsorted(bins, m, side='right') - 1)
            bi_ = max(0, min(len(bins) - 2, bi_))
            lookup[(c, bi_)].append(int(qlbl[j].item()))
        c_lbl = torch.zeros(len(pool), dtype=torch.long)
        c_conf = torch.zeros(len(pool))
        c_curve = []
        for j in range(len(pool)):
            c = int(ppred[j].item())
            m = float(pmargin[j].item())
            bi_ = int(np.searchsorted(bins, m, side='right') - 1)
            bi_ = max(0, min(len(bins) - 2, bi_))
            ents = lookup[(c, bi_)]
            if ents:
                vals = torch.tensor(ents)
                c_lbl[j] = int(vals.mode().values.item()) if len(vals) > 0 else c
                c_conf[j] = vals.float().mean().item() if len(vals) > 0 else 0.0
        # precision vs coverage by confidence in the lookup entry
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            m_keep = c_conf >= t
            nk = int(m_keep.sum().item())
            prec = float((c_lbl[m_keep] == pl[m_keep]).float().mean().item()) if nk > 0 else None
            c_curve.append({'tau': t, 'coverage': nk / len(pool),
                            'precision': prec, 'n': nk})
        r['expansion']['C_decision_correction'] = c_curve
        # plain confusion lookup (no margin conditioning)
        conf_plain = {}
        for j in range(len(qidx)):
            c = int(ppred[qidx[j]].item())
            conf_plain.setdefault(c, []).append(int(qlbl[j].item()))
        c2_lbl = torch.zeros(len(pool), dtype=torch.long)
        for j in range(len(pool)):
            c = int(ppred[j].item())
            ents = conf_plain.get(c, [])
            c2_lbl[j] = int(torch.tensor(ents).mode().values.item()) if ents else c
        nk = len(pool)
        r['expansion']['C_confusion_only'] = {
            'coverage': 1.0,
            'precision': float((c2_lbl == pl).float().mean().item()),
            'n': nk}

        # ---- ridge mIoU for the main expansions + the direct-sparse baseline ----
        rd = r['ridge']
        # direct sparse: K labels only in T (no expansion)
        Y_sp = torch.zeros(len(pool), NUM_CLASSES)
        Y_sp[qidx] = qlbl_onehot
        rd['direct_sparse'] = mw(ridge_fit_soft(Xd, Y_sp, args.lam, args.cg_iters,
                                                args.nystrom_m, device))
        # A: nearest-anchor at the best-precision threshold with >5% coverage
        best_a = max((e for e in a_curve if e['precision'] is not None and e['coverage'] > 0.05),
                     key=lambda e: e['precision'], default=None)
        if best_a is not None:
            Y_a = torch.zeros(len(pool), NUM_CLASSES)
            m_keep = best_sim >= best_a['tau']
            Y_a[m_keep] = onehot(best_lbl[m_keep], NUM_CLASSES)
            rd['A_best'] = mw(ridge_fit_soft(Xd, Y_a, args.lam, args.cg_iters,
                                             args.nystrom_m, device))
            rd['A_best_tau'] = best_a['tau']
        # B: class-centroid at the best-precision threshold with >5% coverage
        best_b = max((e for e in b_curve if e['precision'] is not None and e['coverage'] > 0.05),
                     key=lambda e: e['precision'], default=None)
        if best_b is not None:
            Y_b = torch.zeros(len(pool), NUM_CLASSES)
            m_keep = best_b >= best_b['tau']
            Y_b[m_keep] = onehot(b_lbl[m_keep], NUM_CLASSES)
            rd['B_best'] = mw(ridge_fit_soft(Xd, Y_b, args.lam, args.cg_iters,
                                             args.nystrom_m, device))
            rd['B_best_tau'] = best_b['tau']
        # C: decision correction (full coverage, best lookup confidence >= 0.5)
        Y_c = torch.zeros(len(pool), NUM_CLASSES)
        m_keep = c_conf >= 0.5
        Y_c[m_keep] = onehot(c_lbl[m_keep], NUM_CLASSES)
        rd['C_best'] = mw(ridge_fit_soft(Xd, Y_c, args.lam, args.cg_iters,
                                         args.nystrom_m, device))
        # soft confusion matrix: Y_i = C[hat_y_i, :] (estimated from queries)
        C_est = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        for j in range(len(qidx)):
            C_est[ppred[qidx[j]], qlbl[j]] += 1.0
        C_est = C_est / (C_est.sum(dim=1, keepdim=True) + 1e-9)
        Y_soft = C_est[ppred]
        rd['C_soft_estimated'] = mw(ridge_fit_soft(Xd, Y_soft, args.lam, args.cg_iters,
                                                   args.nystrom_m, device))
        # ORACLE confusion matrix ceiling (all pool labels)
        C_or = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        for j in range(len(pool)):
            C_or[ppred[j], pl[j]] += 1.0
        C_or = C_or / (C_or.sum(dim=1, keepdim=True) + 1e-9)
        rd['C_soft_oracle'] = mw(ridge_fit_soft(Xd, C_or[ppred], args.lam, args.cg_iters,
                                                args.nystrom_m, device))

        # ---- confusion stability: estimated C vs oracle C ----
        cf = r['confusion']
        rows = []
        for c in range(NUM_CLASSES):
            if C_or[c].sum().item() > 0 and C_est[c].sum().item() > 0:
                a = C_est[c]
                b = C_or[c]
                rows.append(float((a * b).sum().item() /
                                  (a.norm().item() * b.norm().item() + 1e-9)))
        cf['row_cos_mean'] = float(np.mean(rows)) if rows else None
        cf['row_cos_n'] = len(rows)
        cf['est_vs_oracle_l1'] = float((C_est - C_or).abs().sum().item())
        # top confusion pairs in the oracle (the systematic errors to correct)
        pairs = []
        for i in range(1, NUM_CLASSES):
            for j in range(1, NUM_CLASSES):
                if i != j and C_or[i, j].item() > 0.02:
                    pairs.append((i, j, float(C_or[i, j].item())))
        pairs.sort(key=lambda t: -t[2])
        cf['top_oracle_pairs'] = [{'pred': i, 'true': j, 'frac': v}
                                  for i, j, v in pairs[:8]]

        # ---- synthesis ----
        syn = r['synthesis']
        syn.append(f"COND {cond}: frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} "
                   f"| query {r['query']['n']} labels (per-class {args.labels_per_class}), "
                   f"anchor acc {r['query']['anchor_acc']:.3f}")
        for name, curve in [('A', a_curve), ('B', b_curve)]:
            pts = [f"{e['tau']}:{e['precision']:.3f}@{e['coverage']:.2f}" if e['precision'] is not None
                   else f"{e['tau']}:na" for e in curve]
            syn.append(f"  {name} prec@cov: " + " ".join(pts))
        syn.append(f"  C decision correction: " + " ".join(
            f"{e['tau']}:{e['precision']:.3f}@{e['coverage']:.2f}" if e['precision'] is not None
            else f"{e['tau']}:na" for e in c_curve))
        syn.append(f"  ridge: direct_sparse {rd['direct_sparse']:.3f} | A_best "
                   f"{rd.get('A_best', float('nan')):.3f} | B_best "
                   f"{rd.get('B_best', float('nan')):.3f} | C_best {rd['C_best']:.3f} | "
                   f"C_soft_est {rd['C_soft_estimated']:.3f} | C_soft_ORACLE "
                   f"{rd['C_soft_oracle']:.3f}")
        syn.append(f"  confusion: est-vs-oracle row-cos {cf['row_cos_mean']:.3f} (n={cf['row_cos_n']}) | "
                   f"top oracle pairs: " + " ".join(
                       f"{p['pred']}->{p['true']}({p['frac']:.2f})" for p in cf['top_oracle_pairs'][:4]))

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("expansion A/B/C: T-label precision vs coverage BEFORE any ridge. The")
    print("  winner says which assumption is true: A local geometry, B class-level")
    print("  geometry, C systematic probe errors. The contamination-free operating")
    print("  point is the highest-precision row with usable coverage.")
    print("ridge: direct_sparse (K labels, no expansion) is the fallback baseline;")
    print("  A_best/B_best/C_best use each expansion at its best-precision point;")
    print("  C_soft_estimated is the confusion matrix from the queries;")
    print("  C_soft_ORACLE is the confusion family's ceiling (all pool labels).")
    print("confusion: row-cos of estimated vs oracle C says whether the queried")
    print("  confusion structure is stable; top_oracle_pairs are the systematic")
    print("  probe errors a correction table would need to learn.")
    print("If even C_soft_ORACLE cannot beat frozen: the confusion family is")
    print("  closed and the direct-sparse route (or a new supervision source) is")
    print("  the only path. If C_soft_ORACLE beats frozen but C_soft_estimated")
    print("  does not: the structure exists but 1-2 labels/class cannot estimate")
    print("  it -- raise labels_per_class and re-run.")

if __name__ == "__main__":
    main()
