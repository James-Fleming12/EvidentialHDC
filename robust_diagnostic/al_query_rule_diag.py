"""al_query_rule_diag.py: the Iteration-1 query-rule comparison for the
one-label-per-cluster AL framework (eval-only, no plots). Oracle-simulated.

Grounding mechanism (Iteration 0): cluster the corrupted pool, query ONE point
per cluster (the representative), label it TRUE (simulated oracle), ground the
cluster by distance (points within the gate radius of the representative inherit
its label; beyond it they are not grounded -- "label if close, else ask").
The probe update is the established ridge with S = ALL pool points and
T = grounded points only (the S_all, T_oracle-gated construction).

This iteration compares FOUR query rules for spending the budget, per condition
and checkpoint, as budget (labels) -> mIoU curves plus efficiency:

  R1 cluster-influence : rank clusters by J_c = sum over cluster of the per-point
      influence I_i = ||(S + lI)^-1 x_i|| (Nystrom-subspace; the exact magnitude
      of the point's contribution to W). Query descending J_c.
  R2 pure confidence   : rank clusters by the representative's frozen-probe
      confidence, query ASCENDING (uncertainty sampling -- the standard AL
      baseline; confidence is free, influence needs the sketch solve).
  R3 influence + disagree-gate : as R1, but only clusters whose representative
      DISAGREES between the prototype and probe decoders are eligible (the
      disagreement-gate: agree -> the cluster is likely already decoded right,
      do not spend budget). Fewer labels spent if few clusters disagree.
  R4 confidence + disagree-gate : as R2 with the same eligibility gate.

References per condition: frozen (no labels), oracle (ALL pool points labeled --
the label ceiling), grounded_all (every cluster queried+grounded at K clusters --
the grounding scheme's own ceiling).

Efficiency: wall time of one 50k-pool ridge fit (the per-budget update cost,
~0.03s) and of the k-means clustering at each K -- the two costs of the AL loop;
reported per condition for the README table.

Usage:
  uv run python robust_diagnostic/al_query_rule_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_query_rule_covshift_ep10.json
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

# ---------------- Nystrom-subspace influence (the R1 signal) ----------------

def nystrom_influence(Xd, lam, m, device):
    """I_i ~= ||(S + lI)^-1 x_i|| in the Nystrom subspace (same sketch as the
    warm start): M = (S_hat + lI)^-1 (m x m), c_i = P^T x_i,
    I_i = sqrt(d) * ||M c_i||. This is the magnitude of point i's contribution
    to W (label-direction independent)."""
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    M = torch.linalg.inv(Shat)
    C = Xd @ P
    MC = C @ M
    return (MC.norm(dim=1) * (Xd.shape[1] ** 0.5)).cpu()
# ---------------- the ridge (S_all + T_gated, matrix-free CG) ----------------

def ridge_fit_t_gated(Xd, Y_gated, lam, iters, m, device):
    """W = (S_all + lI)^-1 X^T Y_gated: S from ALL pool points, T only from the
    gated/labeled subset. Nystrom warm start + matrix-free CG."""
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Yd = Y_gated.float().to(device)
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
    parser.add_argument("--cluster_ks", type=str, default="17,68,136")
    parser.add_argument("--gate_mode", type=str, default="auto",
                        help="grounding gate radius: 'auto' = quantile of the "
                             "dist-to-representative distribution, or a float")
    parser.add_argument("--gate_quantile", type=float, default=0.6)
    parser.add_argument("--budgets", type=str, default="",
                        help="comma list of label budgets (default: per-K "
                             "quarters/halves/full)")
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_query_rule_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    cluster_ks = [int(x) for x in args.cluster_ks.split(',')]

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

        # prototype (R1) predictions from the clean class means (HDC); computed
        # BEFORE clean_codes is freed (they are the pseudo-label context too)
        proto_pairs = []
        for c in range(1, NUM_CLASSES):
            m = la[ci] == c
            if int(m.sum().item()) > 0:
                proto_pairs.append((c, clean_codes[m].float().mean(dim=0)))
        proto_ids = torch.tensor([c for c, _ in proto_pairs])
        proto_mat = torch.stack([p for _, p in proto_pairs])
        proto_mat = proto_mat / (proto_mat.norm(dim=1, keepdim=True) + 1e-8)

        # frozen probe (the pseudo-label source + frozen reference)
        Xc = clean_codes.float().to(device)
        W_clean = ridge_fit_t_gated(Xc, onehot(la[ci], NUM_CLASSES), args.lam,
                                    args.cg_iters, args.nystrom_m, device)
        del clean_codes
        Xd = pool_codes.float().to(device)

        # per-point signals: confidence, influence, disagreement (proto vs probe)
        sm = torch.softmax(scores(W_clean, pool_codes), dim=1)
        pconf = sm.max(dim=1).values
        ppred = sm.argmax(dim=1)
        pseudo_correct = (ppred == pl)
        I = nystrom_influence(Xd, args.lam, args.nystrom_m, device)
        proto_pred = proto_ids[(pool_codes.float() @ proto_mat.t()).argmax(dim=1)]
        del pool_codes

        r = {'refs': {}, 'rules': {}, 'efficiency': {}, 'signals': {}, 'synthesis': []}

        def mw(W):
            return compute_miou(decode(W, val_codes), vl)

        # references
        W_oracle = ridge_fit_t_gated(Xd, onehot(pl, NUM_CLASSES), args.lam,
                                     args.cg_iters, args.nystrom_m, device)
        r['refs'] = {'frozen': mw(W_clean),
                     'oracle': mw(W_oracle),
                     'pseudo_acc': float(pseudo_correct.float().mean().item())}
        r['signals'] = {
            'corr_conf_influence': None,
            'disagree_rate': float((proto_pred != ppred).float().mean().item()),
        }
        a = pconf.float(); b = I.float()
        a = a - a.mean(); b = b - b.mean()
        r['signals']['corr_conf_influence'] = float((a * b).sum().item() /
                                                    (a.norm().item() * b.norm().item() + 1e-30))

        pool_f = pool.float()

        for K in cluster_ks:
            t_k = tic()
            labs = kmeans_labels(pool_f, K, device=device)
            t_kmeans = toc(t_k)
            n = len(pool)

            # per-cluster bookkeeping
            rep_idx = torch.zeros(K, dtype=torch.long)
            rep_conf = torch.zeros(K)
            cluster_I = torch.zeros(K)
            cluster_n = torch.zeros(K, dtype=torch.long)
            d_rep = torch.zeros(n)
            for c in range(K):
                m = labs == c
                idx = m.nonzero().squeeze(1)
                if len(idx) == 0:
                    rep_idx[c] = -1
                    continue
                cents = pool_f[idx].mean(dim=0)
                d2 = (pool_f[idx] - cents).pow(2).sum(dim=1)
                rep_idx[c] = idx[int(d2.argmin().item())]
                rep_conf[c] = pconf[rep_idx[c]]
                cluster_I[c] = float(I[idx].sum().item())
                cluster_n[c] = len(idx)
                d_rep[idx] = (pool_f[idx] - pool_f[rep_idx[c]]).norm(dim=1)
            # grounding gate radius
            if args.gate_mode == 'auto':
                radius = float(torch.quantile(d_rep, args.gate_quantile).item())
            else:
                radius = float(args.gate_mode)

            # per-cluster grounding: points within radius of the rep
            grounded = d_rep <= radius
            cluster_disagree = torch.zeros(K, dtype=torch.bool)
            for c in range(K):
                if rep_idx[c] < 0:
                    continue
                cluster_disagree[c] = proto_pred[rep_idx[c]] != ppred[rep_idx[c]]

            # the four query rules: cluster order (eligibility applied by the gate)
            def cluster_order(rank_signal, desc=True, gate=None):
                order = torch.argsort(rank_signal, descending=desc)
                if gate is not None:
                    order = order[gate[order]]
                return order.tolist()

            rules = [
                ('influence', cluster_I, True, None),
                ('confidence', rep_conf, False, None),
                ('influence_gated', cluster_I, True, cluster_disagree),
                ('confidence_gated', rep_conf, False, cluster_disagree),
            ]
            r['rules'][f'K{K}'] = {'radius': radius, 'n_grounded': int(grounded.sum().item()),
                                   'n_disagree_clusters': int(cluster_disagree.sum().item()),
                                   'kmeans_time_s': t_kmeans, 'rules': {}}

            def run_budget(order, budget):
                spent = 0
                Y_g = torch.zeros(n, NUM_CLASSES)
                for ci in order:
                    if spent >= budget:
                        break
                    if rep_idx[ci] < 0:
                        continue
                    idx = (labs == ci) & grounded
                    if int(idx.sum().item()) < 1:
                        spent += 1
                        continue
                    Y_g[idx] = onehot(pl[rep_idx[ci]], NUM_CLASSES)
                    spent += 1
                return Y_g, spent

            # per-budget update cost: one full-pool ridge fit (the AL loop fits
            # once after accumulating the grounded labels)
            t_fit = tic()
            ridge_fit_t_gated(Xd, onehot(pl, NUM_CLASSES), args.lam,
                              args.cg_iters, args.nystrom_m, device)
            t_fit = toc(t_fit)

            for rname, sig, desc, gate in rules:
                order = cluster_order(sig, desc, gate)
                # budgets: full K and (if explicit) the CLI list
                if args.budgets:
                    budgets = [int(x) for x in args.budgets.split(',')]
                else:
                    budgets = sorted(set([max(1, K // 4), max(1, K // 2), K]))
                r['rules'][f'K{K}']['rules'][rname] = {}
                for b in budgets:
                    Y_g, spent = run_budget(order, b)
                    if spent == 0:
                        r['rules'][f'K{K}']['rules'][rname][str(b)] = {
                            'miou': None, 'labels_spent': 0, 'n_grounded': 0,
                            'insufficient': True}
                        continue
                    W = ridge_fit_t_gated(Xd, Y_g, args.lam, args.cg_iters,
                                          args.nystrom_m, device)
                    r['rules'][f'K{K}']['rules'][rname][str(b)] = {
                        'miou': mw(W),
                        'labels_spent': spent,
                        'n_grounded': int((Y_g.sum(dim=1) > 0).sum().item())}
            # grounded-all ceiling (every cluster queried+grounded)
            Y_all = torch.zeros(n, NUM_CLASSES)
            for c in range(K):
                if rep_idx[c] < 0:
                    continue
                idx = (labs == c) & grounded
                if int(idx.sum().item()) > 0:
                    Y_all[idx] = onehot(pl[rep_idx[c]], NUM_CLASSES).unsqueeze(0)
            W_gall = ridge_fit_t_gated(Xd, Y_all, args.lam, args.cg_iters,
                                       args.nystrom_m, device)
            r['refs'][f'grounded_all_K{K}'] = mw(W_gall)

            r['efficiency'][f'K{K}'] = {
                'kmeans_time_s': t_kmeans,
                'fit_time_s': t_fit,
                'update_pts_s': float(len(pool) / (t_fit + 1e-9)),
                'labels_grounded_per_label': float(grounded.sum().item() / max(1, K)),
            }

            syn = r['synthesis']
            syn.append(f"COND {cond} K={K}: frozen {r['refs']['frozen']:.3f} / oracle "
                       f"{r['refs']['oracle']:.3f} / grounded_all {r['refs'][f'grounded_all_K{K}']:.3f} "
                       f"| radius {radius:.2f}, grounded {int(grounded.sum().item())}/{n}")
            for rname in rules:
                line = f"  {rname}: " + " ".join(
                    f"b{b}:{v['miou']:.3f}({v['labels_spent']}l,{v['n_grounded']}g)"
                    if v.get('miou') is not None else f"b{b}:skip"
                    for b, v in r['rules'][f'K{K}']['rules'][rname].items())
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
    print("refs: frozen (0 labels) / oracle (all pool labeled, the label ceiling) /")
    print("  grounded_all (K labels, every cluster grounded -- the grounding")
    print("  scheme's own ceiling). The gap grounded_all -> oracle is the loss from")
    print("  grounding by distance instead of labeling everything.")
    print("rules (budget -> mIoU, labels_spent, n_grounded):")
    print("  influence      : rank clusters by J_c = sum I_i (the exact W-magnitude)")
    print("  confidence     : rank by representative confidence, ascending")
    print("                   (uncertainty sampling; free signal)")
    print("  *_gated        : only clusters whose representative DISAGREES between")
    print("                   prototype and probe are eligible; fewer labels spent")
    print("efficiency: fit_time_s (one 50k ridge fit, the per-budget update cost),")
    print("  kmeans_time_s, labels_grounded_per_label (the leverage).")
    print("signals: corr_conf_influence, disagree_rate -- the ranking context.")

if __name__ == "__main__":
    main()
