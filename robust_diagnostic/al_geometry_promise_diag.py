"""al_geometry_promise_diag.py: the intrinsic information content of the feature
space's geometric properties for ultra-cheap labeling (eval-only, no plots).

Iterations 1-3 closed the expansion mechanisms built on clusters / HDC-code
similarity / sparse confusion, but the A/B failure had a systematic cause: the
expansions computed similarity in the SATURATED 10k-d HDC code space (all
cosines ~0.98+), while the packing we keep citing (1-NN purity 51-77%, intra vs
inter cosine 0.62-0.70 vs 0.004-0.055) was measured in the 128-d features.
This diagnostic measures the PROMISE of each geometric property in the 128-d
space, with robustness: every quantity is averaged over R repeated RANDOM
anchor draws (mean +- std), so no selection rule can hide a weak property and
no lucky draw can fake a strong one. No ridge, no clustering, no diffusion.

  A. NEAREST-ANCHOR (128-d): 1 random queried point per class; precision vs
     coverage over a cosine threshold (the "label if close to a queried point"
     rule, in the space where closeness is informative).
  B. CLASS-CENTROID (128-d): centroid of the 1 queried anchor per class;
     argmax-cosine decode with a top-cosine threshold (the near-unimodality
     claim). Plus the ORACLE centroid (all class points) as the class-level
     geometry ceiling.
  C. MULTI-ANCHOR agreement: 2 queried anchors per class; label only where the
     MIN cosine to both exceeds tau (the membership certificate). The
     contamination-free operating curve.
  D. SPATIAL adjacency (LiDAR projection): from the label grids directly (no
     features) -- P(same class | 4-/8-neighbor), per class; the superpixel
     grounding promise: label one point, its projection neighbors inherit.
  E. PER-CLASS packing (128-d): per-class 1-NN same-class purity on the
     corrupted pool -- which classes are packable at all (the budget-allocation
     signal from Iteration 0).
  F. CONFIDENCE-conditioned packing: does the frozen probe's confidence gate
     the 128-d anchor similarity? (conf >= q) & (cos >= tau) precision --
     whether the free confidence signal can sharpen the geometry gate.

Everything is cosine matmuls on the 128-d features + label-grid ops: no HDC
codes except the frozen-probe confidence in F.

Usage:
  uv run python robust_diagnostic/al_geometry_promise_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_geometry_promise_covshift_ep10.json
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

def extract_full(model, parser, device, num_frames=100):
    """Features (128-d, valid points), labels, AND the per-frame label grids +
    masks for the spatial promise (no flattening loss)."""
    feats, lbls = [], []
    grids = []            # (labels (H,W), mask (H,W)) per frame
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device)
            lab_grid = batch[2].to(device).squeeze(0)          # (H, W)
            msk_grid = (batch[1].to(device) > 0).squeeze(0)    # (H, W)
            labels = lab_grid.view(-1)
            mask = msk_grid.view(-1)
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(z_flat.cpu())
            lbls.append(labels[mask].cpu())
            grids.append((lab_grid.cpu(), msk_grid.cpu()))
    return torch.cat(feats, dim=0), torch.cat(lbls, dim=0), grids

def hdc_codes(feats, proj, device, chunk=100000):
    codes = []
    for s in range(0, len(feats), chunk):
        codes.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(codes, dim=0)

def onehot(lbls, num_classes):
    y = torch.zeros(len(lbls), num_classes)
    y[torch.arange(len(lbls)), lbls.long()] = 1.0
    return y

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

def prec_cov(keep, pred_lbl, true_lbl):
    nk = int(keep.sum().item())
    if nk == 0:
        return {'coverage': 0.0, 'precision': None, 'n': 0}
    return {'coverage': nk / len(true_lbl),
            'precision': float((pred_lbl[keep] == true_lbl[keep]).float().mean().item()),
            'n': nk}

# ---------------- spatial promise (label grids only) ----------------

def _roll_grid(g, H, W, di, dj):
    """Roll a grid by (di, dj) with zero fill (the 'neighbor value' view)."""
    rolled = torch.zeros_like(g)
    si, ei = max(0, -di), H - max(0, di)
    sj, ej = max(0, -dj), W - max(0, dj)
    rolled[si:ei, sj:ej] = g[max(0, di):H - max(0, -di),
                             max(0, dj):W - max(0, -dj)]
    return rolled

def spatial_promise(grids, classes):
    """P(same class | neighbor) for 4- and 8-connectivity over the valid points,
    plus per-class 4-neighbor coherence. Pure grid ops, no features. Each
    directed neighbor contributes one (point, neighbor) pair to the count."""
    acc4 = {'num': 0, 'den': 0}
    acc8 = {'num': 0, 'den': 0}
    per_class = {c: {'num': 0, 'den': 0} for c in classes}
    dirs4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dirs8 = dirs4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for lab, msk in grids:
        H, W = lab.shape
        lab_m = lab * msk.long()                 # masked labels, 0 = invalid
        for di, dj in dirs8:
            rolled = _roll_grid(lab_m, H, W, di, dj)
            nmsk = _roll_grid(msk.long(), H, W, di, dj).bool()
            rmask = msk & nmsk                   # point and neighbor both valid
            same = (lab_m == rolled) & rmask
            if (di, dj) in dirs4:
                acc4['num'] += int(same.sum().item())
                acc4['den'] += int(rmask.sum().item())
            acc8['num'] += int(same.sum().item())
            acc8['den'] += int(rmask.sum().item())
        # per-class 4-neighbor coherence: fraction of the class's directed
        # (point, valid 4-neighbor) pairs that are same-class
        for c in classes:
            mc = (lab == c) & msk
            if int(mc.sum().item()) == 0:
                continue
            pc_num = 0
            pc_den = 0
            for di, dj in dirs4:
                rolled = _roll_grid(lab_m, H, W, di, dj)
                nmsk = _roll_grid(msk.long(), H, W, di, dj).bool()
                rmask = mc & nmsk
                pc_num += int(((lab_m == rolled) & rmask).sum().item())
                pc_den += int(rmask.sum().item())
            per_class[c]['num'] += pc_num
            per_class[c]['den'] += pc_den
    out = {
        'p4': acc4['num'] / acc4['den'] if acc4['den'] else None,
        'p8': acc8['num'] / acc8['den'] if acc8['den'] else None,
    }
    for c, v in per_class.items():
        out[str(c)] = v['num'] / v['den'] if v['den'] else None
    return out

# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50000)
    parser.add_argument("--max_clean", type=int, default=200000)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--nystrom_m", type=int, default=1000)
    parser.add_argument("--cg_iters", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10,
                        help="random anchor draws per metric (robustness)")
    parser.add_argument("--taus", type=str, default="0.3,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_geometry_promise_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    taus = [float(x) for x in args.taus.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    # clean features + frozen probe ONCE (shared across conditions; used only by
    # the F confidence-conditioning section)
    cf, cl, _ = extract_full(model, build_parser(args.kitti_dir, DATA, ARCH),
                             device, args.frames)
    proj_clean = get_hdc_projection(dim_in=cf.shape[1], dim_out=10000, device=device)
    mc = min(args.max_clean, len(cf))
    cc = hdc_codes(cf[:mc], proj_clean, device)
    W_clean = ridge_fit_soft(cc.float().to(device), onehot(cl[:mc], NUM_CLASSES),
                             args.lam, args.cg_iters, args.nystrom_m, device)
    del cc, cf, cl

    results = {'label': args.label, 'conds': {}}

    for cond in conds:
        t_cond = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l, grids = extract_full(model, build_parser(cdir, DATA, ARCH), device,
                                   args.frames)
        proj = get_hdc_projection(dim_in=f.shape[1], dim_out=10000, device=device)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]

        zn = pool.float()
        zn = zn / (zn.norm(dim=1, keepdim=True) + 1e-8)
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}

        r = {'promise': {}, 'spatial': {}, 'per_class': {}, 'synthesis': []}
        pr = r['promise']

        # ---- A/B/C: anchor-based promise over R random draws ----
        # A: 1 anchor/class, nearest-anchor cosine gate
        # B: class-centroid from those anchors, top-cos gate (+ oracle centroid)
        # C: 2 anchors/class, min-cos agreement gate
        for name, k_anch in [('A_nearest_1', 1), ('C_min_agreement_2', 2)]:
            rows = {str(t): [] for t in taus}
            for rep in range(args.repeats):
                torch.manual_seed(rep)
                aids = []
                for c in classes:
                    idx = cls_idx[c]
                    if len(idx) < 50:
                        continue
                    picks = idx[torch.randperm(len(idx))[:k_anch]]
                    aids.append(picks)
                aids = torch.cat(aids)
                albl = pl[aids]
                sim = zn @ zn[aids].t()                       # n x K
                best, bi = sim.max(dim=1)
                best_lbl = albl[bi]
                if k_anch == 2:
                    # min over the k anchors of the chosen class (membership
                    # certificate: the point must resemble ALL queried examples).
                    # The assigned class is the one with the highest min-sim.
                    class_min = {}
                    for c in classes:
                        m = (albl == c)
                        if int(m.sum().item()) == 0:
                            continue
                        class_min[c] = sim[:, m].min(dim=1).values
                    ckeys = list(class_min)
                    min_mat = torch.stack([class_min[c] for c in ckeys], dim=1)
                    gate_sim, ci = min_mat.max(dim=1)
                    best_lbl = torch.tensor(ckeys)[ci]
                else:
                    gate_sim = best
                for t in taus:
                    keep = gate_sim >= t
                    rows[str(t)].append(prec_cov(keep, best_lbl, pl))
            pr[name] = {}
            for t in taus:
                entries = [e for e in rows[str(t)] if e['precision'] is not None]
                if entries:
                    precs = np.array([e['precision'] for e in entries])
                    covs = np.array([e['coverage'] for e in entries])
                    pr[name][str(t)] = {'prec_mean': float(precs.mean()),
                                        'prec_std': float(precs.std()),
                                        'cov_mean': float(covs.mean()),
                                        'n': int(entries[0]['n'])}
                else:
                    pr[name][str(t)] = {'prec_mean': None, 'prec_std': None,
                                        'cov_mean': 0.0, 'n': 0}

        # B: class-centroid (1 anchor/class) + oracle centroid ceiling
        rows_b = {str(t): [] for t in taus}
        rows_or = {str(t): [] for t in taus}
        for rep in range(args.repeats):
            torch.manual_seed(rep)
            cents = []
            for c in classes:
                idx = cls_idx[c]
                if len(idx) < 50:
                    continue
                cents.append(zn[idx[torch.randperm(len(idx))[0]]])
            cid = torch.tensor([c for c in classes if len(cls_idx[c]) >= 50])
            c_mat = torch.stack(cents)
            c_mat = c_mat / (c_mat.norm(dim=1, keepdim=True) + 1e-8)
            simb = zn @ c_mat.t()
            best, bi = simb.max(dim=1)
            best_lbl = cid[bi]
            for t in taus:
                keep = best >= t
                rows_b[str(t)].append(prec_cov(keep, best_lbl, pl))
            # oracle centroids (all class points) -- the class-geometry ceiling
            c_or = torch.stack([zn[cls_idx[c]].mean(dim=0) for c in classes
                                if len(cls_idx[c]) >= 50])
            c_or = c_or / (c_or.norm(dim=1, keepdim=True) + 1e-8)
            simo = zn @ c_or.t()
            besto, bio = simo.max(dim=1)
            besto_lbl = cid[bio]
            for t in taus:
                keep = besto >= t
                rows_or[str(t)].append(prec_cov(keep, besto_lbl, pl))
        pr['B_centroid_1'] = {}
        pr['B_centroid_oracle'] = {}
        for t in taus:
            for name, rows in [('B_centroid_1', rows_b),
                               ('B_centroid_oracle', rows_or)]:
                entries = [e for e in rows[str(t)] if e['precision'] is not None]
                if entries:
                    pr[name][str(t)] = {
                        'prec_mean': float(np.mean([e['precision'] for e in entries])),
                        'prec_std': float(np.std([e['precision'] for e in entries])),
                        'cov_mean': float(np.mean([e['coverage'] for e in entries])),
                        'n': int(entries[0]['n'])}
                else:
                    pr[name][str(t)] = {'prec_mean': None, 'prec_std': None,
                                        'cov_mean': 0.0, 'n': 0}

        # ---- D: spatial promise ----
        r['spatial'] = spatial_promise(grids, classes)

        # ---- E: per-class 128-d NN purity (against the FULL pool, chunked) ----
        pc = {}
        for c in classes:
            idx = cls_idx[c]
            n_c = len(idx)
            if n_c < 50:
                continue
            sub = zn[idx].to(device)
            nn_same = 0
            for s in range(0, n_c, 4096):
                e = min(s + 4096, n_c)
                sim_c = sub[s:e] @ zn.to(device).t()       # (m, n) full pool
                sim_c[torch.arange(e - s), idx[s:e]] = -1e9  # exclude self
                nn = sim_c.argmax(dim=1)
                nn_same += int((pl[nn.cpu()] == c).sum().item())
            pc[str(c)] = {'nn1_purity': nn_same / n_c, 'n': n_c}
        r['per_class'] = pc
        nn1_all = [v['nn1_purity'] for v in pc.values()]
        r['per_class']['_mean'] = float(np.mean(nn1_all)) if nn1_all else None

        # ---- F: confidence-conditioned packing (frozen probe on the pool) ----
        # the frozen probe comes from the CLEAN features (extracted once in main)
        pool_codes = hdc_codes(pool, proj, device)
        smf = torch.softmax(scores(W_clean, pool_codes), dim=1)
        pconf = smf.max(dim=1).values
        del pool_codes
        pr['F_conf_cond'] = {}
        # B_centroid_1 per-rep similarities are gone; recompute a quick version:
        # oracle centroid gate intersected with the probe confidence gate
        c_or = torch.stack([zn[cls_idx[c]].mean(dim=0) for c in classes
                            if len(cls_idx[c]) >= 50])
        c_or = c_or / (c_or.norm(dim=1, keepdim=True) + 1e-8)
        simo = zn @ c_or.t()
        besto, bio = simo.max(dim=1)
        besto_lbl = torch.tensor([c for c in classes if len(cls_idx[c]) >= 50])[bio]
        # F: confidence-conditioned packing uses RELATIVE confidence quantiles
        # (the frozen probe's absolute max-softmax is < 0.3 everywhere on the
        # corrupted pool -- Iteration-11 calibration finding -- so absolute
        # gates are vacuous; relative top-q% gates test whether confidence can
        # still RANK the geometry gate)
        for q in [0.3, 0.5, 0.7]:
            keep_conf = pconf >= torch.quantile(pconf, 1 - q)
            pr['F_conf_cond'][str(q)] = {}
            for t in taus:
                keep = (besto >= t) & keep_conf
                pr['F_conf_cond'][str(q)][str(t)] = prec_cov(keep, besto_lbl, pl)

        # ---- synthesis ----
        syn = r['synthesis']
        syn.append(f"COND {cond}: classes {len(classes)}, pool {len(pool)}")
        for name in ['A_nearest_1', 'B_centroid_1', 'B_centroid_oracle',
                     'C_min_agreement_2']:
            line = f"  {name}: " + " ".join(
                f"t{t}:{pr[name][str(t)]['prec_mean']:.3f}±{pr[name][str(t)]['prec_std']:.3f}"
                f"@{pr[name][str(t)]['cov_mean']:.2f}"
                if pr[name][str(t)]['prec_mean'] is not None else f"t{t}:na"
                for t in taus)
            syn.append(line)
        sp = r['spatial']
        syn.append(f"  spatial: P4 {sp['p4']:.3f} / P8 {sp['p8']:.3f} | "
                   f"per-class 4-neighbor min {min(v for k, v in sp.items() if k not in ('p4', 'p8') and v is not None):.3f}")
        syn.append(f"  per-class nn1: mean {r['per_class']['_mean']:.3f} | "
                   f"min {min(v['nn1_purity'] for k, v in pc.items() if k != '_mean'):.3f} "
                   f"(class {min((k for k in pc if k != '_mean'), key=lambda k: pc[k]['nn1_purity'])}) | "
                   f"max {max(v['nn1_purity'] for k, v in pc.items() if k != '_mean'):.3f} "
                   f"(class {max((k for k in pc if k != '_mean'), key=lambda k: pc[k]['nn1_purity'])})")
        for q in [0.3, 0.5, 0.7]:
            line = f"  F conf>={q}: " + " ".join(
                f"t{t}:{pr['F_conf_cond'][str(q)][str(t)]['precision']:.3f}@"
                f"{pr['F_conf_cond'][str(q)][str(t)]['coverage']:.2f}"
                if pr['F_conf_cond'][str(q)][str(t)]['precision'] is not None
                else f"t{t}:na" for t in taus)
            syn.append(line)

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("A/B/C are the 128-d expansion promises, mean+-std over R random draws:")
    print("  A_nearest_1      : 1 queried point/class, cosine gate to nearest")
    print("  B_centroid_1     : class centroid from 1 queried point/class")
    print("  B_centroid_oracle: class centroid from ALL points (ceiling of the")
    print("                     class-level geometry -- the near-unimodality")
    print("                     claim's max).")
    print("  C_min_agreement_2: 2 queried points/class, MIN-cosine gate")
    print("                     (the membership certificate).")
    print("The operating point is the highest-tau row with usable coverage; a")
    print("precision ~0.9+ at 0.2-0.5 coverage means the property carries labels")
    print("safely; ~0.5-0.7 means it is a soft signal; ~class-balance means the")
    print("property is useless in 128-d too.")
    print("spatial: P4/P8 = P(same class | projection neighbor) from the label")
    print("  grids -- the superpixel grounding promise (label one point, its")
    print("  neighbors inherit). Per-class rows show which classes are spatially")
    print("  coherent.")
    print("per_class nn1: which classes are packable in 128-d at all (budget")
    print("  allocation); min-class is where labels must be spent.")
    print("If A/B/C stay at class balance in 128-d AND spatial P4 is low, the")
    print("geometric promises are exhausted and the AL story must lean on the")
    print("decision-correction family (labels_per_class >= 2) or direct sparse")
    print("labels. If spatial P4 is high, the projection adjacency is the")
    print("unused sensor geometry.")

if __name__ == "__main__":
    main()
