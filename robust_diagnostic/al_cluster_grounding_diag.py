"""al_cluster_grounding_diag.py: AL framework preliminary diagnostics (eval-only,
no plots). Verifies the dense-per-class-cluster packing on the CURRENT cov-shift
setup and measures how few labels the space actually needs.

Pillar-3's leverage is that one queried point per cluster grounds the whole
cluster by distance, so the budget scales with the number of clusters, not
points. This diagnostic measures that claim and the label-reduction properties:

  A. PACKING (does the cluster structure survive corruption?):
     - 1-NN / k-NN same-class purity per class (128-d), corrupted pool AND a
       clean reference -- the README's 75-87% claim was on the un-pretrained
       model; re-verify on cov-shift ep10/ep21.
     - k-means cluster purity at K = #classes: the dominant-class fraction per
       cluster (what "one label per cluster" would actually ground).
     - separation: mean intra-class cosine vs mean inter-class cosine.
  B. ONE-LABEL-PER-CLUSTER GROUNDING (the distance-gated labeling rule):
     For K in {17, 34, 68, 136, 272} k-means clusters, representative = the
     point nearest the centroid (the single queried point), label = its true
     label. For every pool point, distance to its cluster's representative:
     - grounding coverage: fraction whose representative has the SAME true
       label (the budget-vs-coverage curve: labels needed vs labels saved).
     - distance-gated coverage: coverage restricted to points within the
       distance quantile q of their representative, q in {0.5, 0.75, 0.9} --
       the "label if close, else ask" decision rule's operating curve.
     - the radius needed to cover 90% of each cluster's correctly-grounded
       points (the distance threshold the gate would use).
  C. WAYS TO REDUCE LABELS FURTHER (properties of the space):
     - per-class shift alignment: the corrupted-vs-clean class-mean shift
       vectors; if they are strongly aligned across classes, the corruption is
       a near-global transform and few classes' labels estimate the shift for
       all (carry-over). Reports mean pairwise cosine of the shifts and the
       shift-vs-mean correlation.
     - confidence-representativeness: within a cluster, does the frozen
       probe's confidence pick the centroid-near point? (corr(conf, dist to
       centroid) -- if negative, high-confidence points ARE the
       representatives and no extra selection signal is needed).
     - multi-modality: within-class k-means at K_c in {2, 4, 8}; the dominant
       fraction measures how many subclusters per class are needed to reach
       ~90% purity (labels scale with modes, not classes).
     - pseudo-label agreement as a packing proxy: does the frozen probe's
       prediction agree with the true label more on centroid-near points?
       (pseudo acc vs distance quantile -- can the frozen probe self-select
       the query point?)

Outputs: JSON + log, doc-ready synthesis per condition.

Usage:
  uv run python robust_diagnostic/al_cluster_grounding_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_cluster_grounding_covshift_ep10.json
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

# ---------------- k-means (GPU, subsample fit, chunked assignment) ----------------

def kmeans_labels(X, K, iters=30, n_init=2, seed=0, device='cuda', fit_size=20000):
    X = X.float()
    torch.manual_seed(seed)
    sub = X[torch.randperm(len(X))[:fit_size]].to(device)
    best = None
    for init in range(n_init):
        torch.manual_seed(seed + init)
        idx = torch.randint(0, len(sub), (K,))
        cents = sub[idx].clone()
        for _ in range(iters):
            d2 = ((sub.unsqueeze(1) - cents.unsqueeze(0)) ** 2).sum(dim=2)
            labs = d2.argmin(dim=1)
            new_cents = []
            for c in range(K):
                m = labs == c
                if int(m.sum().item()) == 0:
                    new_cents.append(cents[c])
                else:
                    new_cents.append(sub[m].mean(dim=0))
            cents = torch.stack(new_cents)
        d2 = ((sub.unsqueeze(1) - cents.unsqueeze(0)) ** 2).sum(dim=2)
        labs = d2.argmin(dim=1)
        cost = float(d2.min(dim=1).values.sum().item())
        if best is None or cost < best[0]:
            best = (cost, cents)
    cents = best[1]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    outs = []
    Xd = X.to(device)
    for s in range(0, len(Xd), 20000):
        chunk = Xd[s:s + 20000]
        d2 = ((chunk.unsqueeze(1) - cents.unsqueeze(0)) ** 2).sum(dim=2)
        outs.append(d2.argmin(dim=1).cpu())
    return torch.cat(outs)

# ---------------- packing metrics ----------------

def nn_purities(feats, lbls, classes, k=10, max_pts=20000, chunk=4096):
    """Per-class 1-NN and k-NN same-class purity on a seeded subsample (128-d)."""
    torch.manual_seed(0)
    if len(feats) > max_pts:
        idx = torch.randperm(len(feats))[:max_pts]
        feats, lbls = feats[idx], lbls[idx]
    zn = feats.float()
    zn = zn / (zn.norm(dim=1, keepdim=True) + 1e-8)
    n = len(feats)
    nn1 = torch.zeros(n, dtype=torch.bool)
    nnk = torch.zeros(n, dtype=torch.bool)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        m = e - s
        sim = zn[s:e] @ zn.t()
        sim[torch.arange(m), torch.arange(s, e)] = -1e9
        vals, inds = torch.topk(sim, k, dim=1)
        knn_lbl = lbls[inds]
        nn1[s:e] = knn_lbl[:, 0] == lbls[s:e]
        nnk[s:e] = (knn_lbl == lbls[s:e].unsqueeze(1)).float().mean(dim=1) > 0.5
    out = {}
    means1, meansk = [], []
    for c in classes:
        m = lbls == c
        n_c = int(m.sum().item())
        if n_c == 0:
            continue
        p1 = float(nn1[m].float().mean().item())
        pk = float(nnk[m].float().mean().item())
        out[str(c)] = {'nn1': p1, 'nnk': pk, 'n': n_c}
        means1.append(p1)
        meansk.append(pk)
    return out, (sum(means1) / len(means1), sum(meansk) / len(meansk))

def class_separation(feats, lbls, classes):
    """Mean intra-class cosine vs mean inter-class cosine (128-d, normalized)."""
    zn = feats.float()
    zn = zn / (zn.norm(dim=1, keepdim=True) + 1e-8)
    means = {c: zn[lbls == c].mean(dim=0) for c in classes}
    intra, inter = [], []
    for i, c in enumerate(classes):
        m = zn[lbls == c]
        if len(m) == 0:
            continue
        intra.append(float((m @ means[c]).mean().item()))
        for d in classes[i + 1:]:
            inter.append(float((m @ means[d]).mean().item()))
    return {'intra': float(np.mean(intra)) if intra else None,
            'inter': float(np.mean(inter)) if inter else None}

# ---------------- the ridge (for the frozen probe: pseudo-acc context) ----------------

def ridge_fit(Xd, Y, lam, iters, m, device):
    """Nystrom warm start + matrix-free CG (the established probe update)."""
    torch.manual_seed(11)
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
    parser.add_argument("--max_clean", type=int, default=200000)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--nystrom_m", type=int, default=1000)
    parser.add_argument("--cg_iters", type=int, default=8)
    parser.add_argument("--cluster_ks", type=str, default="17,34,68,136,272")
    parser.add_argument("--dist_quantiles", type=str, default="0.5,0.75,0.9")
    parser.add_argument("--within_ks", type=str, default="2,4,8")
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_cluster_grounding_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    cluster_ks = [int(x) for x in args.cluster_ks.split(',')]
    dist_qs = [float(x) for x in args.dist_quantiles.split(',')]
    within_ks = [int(x) for x in args.within_ks.split(',')]

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
        pool_codes = hdc_codes(pool, proj, device)

        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]
        clean_codes = hdc_codes(fa[ci], proj, device)
        W_clean = ridge_fit(clean_codes.float().to(device), onehot(la[ci], NUM_CLASSES),
                            args.lam, args.cg_iters, args.nystrom_m, device)
        del clean_codes

        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        r = {'packing': {}, 'grounding': {}, 'reduction': {}, 'synthesis': []}

        # ---- A. packing ----
        pk = r['packing']
        pk['pool_nn'], pk['pool_nn_mean'] = nn_purities(pool, pl, classes)
        pk['clean_nn'], pk['clean_nn_mean'] = nn_purities(fa, la, classes)
        pk['separation'] = class_separation(pool, pl, classes)
        # k-means cluster purity at K = #classes (what 1 label/cluster grounds)
        labs17 = kmeans_labels(pool, len(classes), device=device)
        cl_purity = {}
        for c in range(len(classes)):
            m = labs17 == c
            if int(m.sum().item()) < 10:
                continue
            cnt = pl[m].bincount(minlength=NUM_CLASSES)
            dom, n_dom = int(cnt.argmax().item()), int(cnt.max().item())
            cl_purity[str(c)] = {'n': int(m.sum().item()), 'dom_class': dom,
                                 'purity': n_dom / int(m.sum().item())}
        pk['cluster_purity_K_classes'] = cl_purity
        pk['cluster_purity_mean'] = float(np.mean(
            [v['purity'] for v in cl_purity.values()])) if cl_purity else None

        # ---- B. one-label-per-cluster grounding ----
        gr = r['grounding']
        sm = torch.softmax(scores(W_clean, pool_codes), dim=1)
        pconf = sm.max(dim=1).values
        ppred = sm.argmax(dim=1)
        pool_f = pool.float()
        for K in cluster_ks:
            labs = kmeans_labels(pool_f, K, device=device)
            dists = torch.zeros(len(pool))
            rep_lbl = torch.zeros(len(pool), dtype=torch.long)
            dist2cent = torch.zeros(len(pool))
            for c in range(K):
                m = labs == c
                if int(m.sum().item()) < 1:
                    continue
                cents = pool_f[m].mean(dim=0)
                d2 = (pool_f[m] - cents).pow(2).sum(dim=1)
                dist2cent[m] = d2.sqrt()
                rep = m.nonzero().squeeze(1)[int(d2.argmin().item())]
                rep_lbl[m] = pl[rep]
                dists[m] = (pool_f[m] - pool_f[rep]).norm(dim=1)
            grounded = (rep_lbl == pl)
            cov_total = float(grounded.float().mean().item())
            entry = {'K': K, 'budget': K,
                     'coverage_total': cov_total,
                     'per_class': {},
                     'dist_gated': {}}
            for q in dist_qs:
                thr = torch.quantile(dists, q)
                m_close = dists <= thr
                entry['dist_gated'][str(q)] = {
                    'n': int(m_close.sum().item()),
                    'coverage': float(grounded[m_close].float().mean().item()) if int(
                        m_close.sum().item()) > 0 else None,
                }
            # radius to cover 90% of the correctly-grounded points
            gd = dists[grounded]
            if len(gd) > 0:
                entry['radius_q90'] = float(torch.quantile(gd, 0.9).item())
                entry['radius_median'] = float(torch.quantile(gd, 0.5).item())
            for cl in classes:
                m = pl == cl
                if int(m.sum().item()) < 100:
                    continue
                entry['per_class'][str(cl)] = {
                    'coverage': float(grounded[m].float().mean().item()),
                    'n': int(m.sum().item())}
            gr[f'K{K}'] = entry

        # ---- C. label-reduction properties ----
        red = r['reduction']
        # per-class shift alignment (corrupted mean - clean mean, normalized)
        zn_pool = pool_f / (pool_f.norm(dim=1, keepdim=True) + 1e-8)
        shift_c, shift_norm = [], []
        for cl in classes:
            m_p = pool_f[pl == cl]
            m_c = fa[la == cl]
            if len(m_p) < 50 or len(m_c) < 50:
                continue
            mu_p = m_p.mean(dim=0)
            mu_c = m_c.mean(dim=0)
            s = mu_p - mu_c
            shift_norm.append(float(s.norm().item()))
            if s.norm().item() > 1e-8:
                shift_c.append(s / s.norm())
        red['shift_norm_mean'] = float(np.mean(shift_norm)) if shift_norm else None
        if len(shift_c) >= 2:
            M = torch.stack(shift_c)
            coss = (M @ M.t())
            ii = torch.eye(len(shift_c), dtype=torch.bool)
            red['shift_pairwise_cos'] = {
                'mean': float(coss[~ii].abs().mean().item()),
                'std': float(coss[~ii].abs().std().item()),
                'n_classes': len(shift_c)}
        # confidence-representativeness: corr(conf, dist to cluster centroid)
        labs17b = labs17
        d2c = torch.zeros(len(pool))
        for c in range(len(classes)):
            m = labs17b == c
            if int(m.sum().item()) < 10:
                continue
            cents = pool_f[m].mean(dim=0)
            d2c[m] = (pool_f[m] - cents).norm(dim=1)
        a = pconf.float(); b = d2c.float()
        a = a - a.mean(); b = b - b.mean()
        red['corr_conf_dist2cent'] = float((a * b).sum().item() /
                                           (a.norm().item() * b.norm().item() + 1e-30))
        # multi-modality: within-class k-means dominant fraction at K_c
        red['within_class'] = {}
        for Kc in within_ks:
            fracs = []
            for cl in classes:
                m = pl == cl
                idx = m.nonzero().squeeze(1)
                if len(idx) < 200:
                    continue
                sub = pool_f[idx]
                labs = kmeans_labels(sub, Kc, device=device, fit_size=min(10000, len(sub)))
                frac = 0.0
                for c in range(Kc):
                    mm = labs == c
                    if int(mm.sum().item()) < 5:
                        continue
                    cnt = pl[idx][mm].bincount(minlength=NUM_CLASSES)
                    frac += int(cnt.max().item())
                fracs.append(frac / len(sub))
            red['within_class'][str(Kc)] = {
                'mean_dom_frac': float(np.mean(fracs)) if fracs else None,
                'n_classes': len(fracs)}
        # pseudo-label agreement as a packing proxy: acc vs distance quantile
        acc_by_q = {}
        for q in [0.25, 0.5, 0.75, 0.9]:
            thr = torch.quantile(d2c, q)
            m = d2c <= thr
            acc_by_q[str(q)] = float((ppred[m] == pl[m]).float().mean().item()) if int(
                m.sum().item()) > 0 else None
        red['pseudo_acc_vs_dist2cent'] = acc_by_q
        red['pseudo_acc_total'] = float((ppred == pl).float().mean().item())

        # ---- synthesis (doc-ready) ----
        syn = r['synthesis']
        nn1m = pk['pool_nn_mean'][0]
        nnkm = pk['pool_nn_mean'][1]
        syn.append(f"COND {cond}: pool nn1 {nn1m:.3f} / nnk {nnkm:.3f} | "
                   f"clean nn1 {pk['clean_nn_mean'][0]:.3f} | "
                   f"sep intra {pk['separation']['intra']:.3f} vs inter {pk['separation']['inter']:.3f} | "
                   f"cluster purity K=#classes {pk.get('cluster_purity_mean') if pk.get('cluster_purity_mean') else 0:.3f}")
        syn.append(f"grounding budget-coverage: " + " ".join(
            f"K{k}:{gr[k]['coverage_total']:.3f}" for k in gr))
        syn.append(f"dist-gated coverage (K=68): " + " ".join(
            f"q{q}:{gr['K68']['dist_gated'][q]['coverage']:.3f}" for q in dist_qs))
        syn.append(f"radius: q90 {gr['K68'].get('radius_q90', float('nan')):.2f} "
                   f"(128-d units), median {gr['K68'].get('radius_median', float('nan')):.2f}")
        syn.append(f"shift: pairwise-cos mean {red.get('shift_pairwise_cos', {}).get('mean')} | "
                   f"corr(conf, dist2cent) {red['corr_conf_dist2cent']:.3f} | "
                   f"within-class K=4 dom-frac {red['within_class'].get('4', {}).get('mean_dom_frac')} | "
                   f"pseudo acc total {red['pseudo_acc_total']:.3f}")

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("A. packing: nn1/nnk on the CORRUPTED pool (vs clean reference) re-verifies")
    print("   the dense per-class packing on the CURRENT extractor; cluster purity at")
    print("   K=#classes is what one-label-per-cluster would actually ground.")
    print("B. grounding: coverage_total is the fraction of the pool a K-label budget")
    print("   grounds correctly (labels saved per label spent). dist_gated q is the")
    print("   coverage restricted to points within distance quantile q of their")
    print("   representative -- the 'label if close, else ask' operating curve.")
    print("   radius_q90 is the distance threshold the gate would use.")
    print("C. reduction: shift pairwise-cos ~1 means the corruption is a near-global")
    print("   transform (few classes' labels estimate the shift for all);")
    print("   corr(conf, dist2cent) < 0 means high-confidence points ARE the")
    print("   representatives (the frozen probe self-selects the query point);")
    print("   within_class K=4 dom-frac near 1 means ~4 labels/class suffice;")
    print("   pseudo_acc_vs_dist2cent: the probe is more right near centroids ->")
    print("   its own confidence can rank the query candidates.")

if __name__ == "__main__":
    main()
