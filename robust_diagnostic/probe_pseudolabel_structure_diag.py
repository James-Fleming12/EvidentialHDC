"""probe_pseudolabel_structure_diag.py: decompose WHY the label-free probe update
fails, and what a TTA method must do with pseudo labels (eval-only, no plots).

The Iteration 9-10 finding: gating/weighting pseudo-labels never beats the frozen
decode. The hypothesis sharpened by that result: the update has TWO independent
objects --
    S = X^T D_s X   (geometry / covariance of the target pool)
    T = X^T D_t Y   (label assignment; Y one-hot or soft)
-- and Iteration 9 gated BOTH at once, starving the covariance. This diagnostic
holds D_s and D_t independent so the decomposition is measured, not assumed.

Diagnostics produced per condition (JSON + log, doc-ready tables):

  A. S/T DECOMPOSITION (the core): mIoU + W-to-oracle cosine for
       S_all,T_all (no_gate) | S_all,T_gated (hard/soft/margin) | S_gated,T_all
       S_all,T_soft (frozen probe distribution, with temperature)
       S_all,T_oracle (upper bound; validates the formulation) | S_gated,T_gated
  B. W DECOMPOSITION: W_correct (correct pseudo-labels only) vs W_wrong vs
       W_pseudo; norms, cosines to W_oracle, linearity check
       W_pseudo == W_correct + W_wrong. Diagnoses whether wrong points are
       noise (A), systematic rotation (B), or coverage-limited (C).
  C. INFLUENCE/LEVERAGE: per-point Nystrom-subspace influence
       I_i = ||(S+lI)^-1 x_i y_i^T|| and leverage G_i = x_i^T(S+lI)^-1 x_i;
       correlation with confidence/margin, correct-vs-wrong split, bins by
       confidence quantile, and the top-influence gate mIoU.
  D. RELIABILITY: confidence quantile -> precision; per-predicted-class
       precision_c(q) at within-class quantiles; calibration bins; pseudo
       confusion matrix (top confused class pairs).
  E. REGION/AGREEMENT: prototype-region precision; prototype-vs-probe
       disagreement populations; agree-only and disagree-high-conf gates.
  F. COVERAGE: per gate, trace ratio, Hutchinson-Frobenius diff ratio,
       participation-ratio effective-rank, class coverage.
  G. COVERAGE-PRESERVING GATES: within-cluster top-confidence (K-means on the
       128-d features, K in {10,50,100,500}) and per-class top-conf.
  H. ORACLE GATE DECOMPOSITION: for target label precision p, build the largest
       subset with precision p by ordering wrong points (random / conf desc /
       conf asc / leverage desc / leverage asc), fit S_all,T_gated, mIoU vs p.
       The p=1 curve is the perfect-gate ceiling.

Usage:
  uv run python robust_diagnostic/probe_pseudolabel_structure_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/probe_pseudolabel_struct_covshift_ep10.json
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

# ---------------- the generalized weighted update (S and T decoupled) ----------------

def nystrom_w0(Xd, Yd, ws, wt, lam, m, device):
    """Nystrom warm start W0 = P (S_hat + lI)^-1 T_hat with independent S/T weights:
       S_hat = (XP)^T D_s (XP), T_hat = (XP)^T D_t Y."""
    torch.manual_seed(11)
    P = (torch.rand(Xd.shape[1], m) > 0.5).float() * 2 - 1
    XP = Xd @ P.to(device)
    Shat = XP.t() @ (ws.unsqueeze(1) * XP)
    That = XP.t() @ (wt.unsqueeze(1) * Yd)
    A = torch.linalg.solve(Shat + lam * torch.eye(m, device=device), That)
    return P.to(device) @ A

def cg_solve(Xd, Yd, ws, wt, lam, device, iters=8, x0=None):
    """Exact (X^T D_s X + lI)^-1 X^T D_t Y via matrix-free CG; Y may be soft."""
    d, C = Xd.shape[1], Yd.shape[1]
    x = x0.clone() if x0 is not None else torch.zeros(d, C, device=device)
    b = Xd.t() @ (wt.unsqueeze(1) * Yd)
    def A(v):
        return Xd.t() @ (ws.unsqueeze(1) * (Xd @ v))
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
    return x

def fit_probe(Xd, Y, w_s=None, w_t=None, lam=1e-3, iters=8, m=1000, device=None):
    """Generalized probe fit. Xd on device; Y (n x C, one-hot or soft) on cpu;
    w_s/w_t optional n-vectors on cpu (None = all ones). Returns W (d x C)."""
    n = Xd.shape[0]
    ws = torch.ones(n) if w_s is None else w_s
    wt = torch.ones(n) if w_t is None else w_t
    Yd = Y.float().to(device)
    x0 = nystrom_w0(Xd, Yd, ws.to(device), wt.to(device), lam, m, device)
    return cg_solve(Xd, Yd, ws.to(device), wt.to(device), lam, device,
                    iters=iters, x0=x0).float()

def cos_sim(Wa, Wb):
    a = Wa.detach().cpu().float().reshape(-1)
    b = Wb.detach().cpu().float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))

def rank_auc(y_bool, score):
    """Rank-based AUROC for separating y_bool=1 from y_bool=0 by score."""
    y = y_bool.float()
    n_pos = int(y.sum().item())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(score, descending=False)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(y) + 1, dtype=torch.float64)
    r_pos = ranks[y.bool()].sum().item()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def spearman(a, b):
    """Spearman rank correlation of two cpu tensors (ties broken by mean rank)."""
    a = a.float(); b = b.float()
    if a.numel() < 10:
        return None
    def rankize(x):
        order = torch.argsort(x)
        r = torch.empty_like(order, dtype=torch.float64)
        r[order] = torch.arange(1, len(x) + 1, dtype=torch.float64)
        return r
    ra, rb = rankize(a), rankize(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra * rb).sum().item() / (denom.item() + 1e-30))

def kmeans_labels(X, K, iters=30, n_init=2, seed=0, device='cuda', fit_size=20000):
    """Lloyd k-means on a GPU subsample, then assign all points by nearest
    centroid. X (n x d, cpu); returns cpu int labels."""
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

# ---------------- Nystrom-subspace influence / leverage ----------------

def nystrom_influence(Xd, lam, m, device):
    """Per-point influence and leverage in the Nystrom subspace (the same sketch
    as the warm start): with M = (S_hat + lI)^-1 (m x m, S_hat = (XP)^T XP),
    leverage G_i = c_i^T M c_i and influence I_i ~ ||M c_i|| * sqrt(d) for
    c_i = P^T x_i. Returns (I, G) torch tensors on cpu, plus the m x m M."""
    torch.manual_seed(11)
    P = (torch.rand(Xd.shape[1], m) > 0.5).float() * 2 - 1
    XP = Xd @ P.to(device)
    Shat = XP.t() @ XP
    M = torch.linalg.inv(Shat + lam * torch.eye(m, device=device))
    C = Xd @ P.to(device)                       # n x m (c_i rows)
    MC = C @ M                                  # n x m
    G = (MC * C).sum(dim=1)                     # leverage
    I = MC.norm(dim=1) * (Xd.shape[1] ** 0.5)   # influence
    return I.cpu(), G.cpu(), M

# ---------------- coverage metrics (matvec-only) ----------------

def hutch_frob2(Xd, device, vecs=8):
    """E_z ||S z||^2 = ||S||_F^2 for S = X^T X, via Hutchinson. Returns float."""
    d = Xd.shape[1]
    est = 0.0
    with torch.no_grad():
        for _ in range(vecs):
            z = torch.randn(d, 1, device=device)
            Sz = Xd.t() @ (Xd @ z)
            est += (Sz * Sz).sum().item()
    return est / vecs

def coverage_stats(Xd, mask, device, trace_all=None):
    """Coverage of the gated subset: trace ratio, Frobenius diff ratio
    ||S_g - S_all||_F / ||S_all||_F, effective-rank (participation) ratio,
    class coverage (count of predicted classes among gated points)."""
    n = Xd.shape[0]
    gated = Xd[mask.to(torch.bool)]
    trace_g = float((gated * gated).sum().item())       # trace(S_g) = sum ||x||^2
    frob_g2 = hutch_frob2(gated, device)
    frob_all2 = hutch_frob2(Xd, device)
    # diff via Hutchinson on S_g - S_all
    d = Xd.shape[1]
    diff_est = 0.0
    with torch.no_grad():
        for _ in range(6):
            z = torch.randn(d, 1, device=device)
            Sz = (gated.t() @ (gated @ z)) - (Xd.t() @ (Xd @ z))
            diff_est += (Sz * Sz).sum().item()
    diff_est /= 6
    fg, fa = frob_g2 ** 0.5, frob_all2 ** 0.5
    return {
        'trace_ratio': float(trace_g / trace_all) if trace_all else None,
        'frob_ratio': float(fg / fa),
        'frob_diff_ratio': float((diff_est ** 0.5) / fa),
        'eff_rank_ratio': float((trace_g ** 2 / fg) / (trace_all ** 2 / fa)) if trace_all else None,
        'n_gated': int(mask.sum().item()),
    }

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
    parser.add_argument("--decomp_iters", type=int, default=20,
                        help="CG iters for the W-correct/W-wrong linearity check")
    parser.add_argument("--gate_fracs", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--soft_temps", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--cluster_ks", type=str, default="10,50,100,500")
    parser.add_argument("--oracle_precisions", type=str, default="0.7,0.8,0.9,0.95,1.0")
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_pseudolabel_struct_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    gate_fracs = [float(x) for x in args.gate_fracs.split(',')]
    soft_temps = [float(x) for x in args.soft_temps.split(',')]
    cluster_ks = [int(x) for x in args.cluster_ks.split(',')]
    oracle_precs = [float(x) for x in args.oracle_precisions.split(',')]

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

        # clean probe (the pseudo-label source + frozen reference). Fitted BEFORE the
        # pool codes are moved to device (200k x 10k is 8GB; keep peak usage low).
        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]
        clean_codes = hdc_codes(fa[ci], proj, device)
        W_clean = fit_probe(clean_codes.float().to(device), onehot(la[ci], NUM_CLASSES),
                            lam=args.lam, iters=args.cg_iters, m=args.nystrom_m, device=device)

        # prototype (R1) predictions on the pool, from the clean class means
        protos = []
        for c in range(1, NUM_CLASSES):
            m = la[ci] == c
            if int(m.sum().item()) > 0:
                protos.append((clean_codes[m].float().mean(dim=0)))
        del clean_codes
        proto_mat = torch.stack(protos)             # K x d
        proto_mat = proto_mat / (proto_mat.norm(dim=1, keepdim=True) + 1e-8)
        proto_scores = pool_codes.float() @ proto_mat.t()
        proto_pred = proto_scores.argmax(dim=1) + 1  # classes stored from 1

        Xd = pool_codes.float().to(device)          # kept on device: 50k x 10k

        # ---- pseudo-labels and gate signals on the corrupted pool ----
        pool_s = scores(W_clean, pool_codes)
        sm = torch.softmax(pool_s, dim=1)
        pconf = sm.max(dim=1).values
        ppred = sm.argmax(dim=1)
        top2 = torch.topk(sm, 2, dim=1).values
        pmargin = (top2[:, 0] - top2[:, 1]).clamp(min=0)
        pnorm = torch.norm(pool.float(), p=2, dim=1)
        u_epi = 1.0 - pconf
        z_geom = (pnorm - pnorm.mean()) / (pnorm.std() + 1e-8)
        u_fuse = fuse_uncertainties(u_epi, z_geom, method='soft_dual_weight',
                                    cfg={"u_th": 0.5, "u_coef": 1.5, "z_th": 0.5, "z_coef": 1.0})
        pseudo_correct = (ppred == pl)
        ones = torch.ones(len(pool))

        # ---- references ----
        Y_oracle = onehot(pl, NUM_CLASSES)
        Y_pseudo = onehot(ppred, NUM_CLASSES)
        W_oracle = fit_probe(Xd, Y_oracle, lam=args.lam, iters=args.cg_iters,
                             m=args.nystrom_m, device=device)
        W_nogate = fit_probe(Xd, Y_pseudo, lam=args.lam, iters=args.cg_iters,
                             m=args.nystrom_m, device=device)

        # ---- per-point influence / leverage (Nystrom subspace) ----
        t_inf = tic()
        I, G, _ = nystrom_influence(Xd, args.lam, args.nystrom_m, device)
        t_inf = toc(t_inf)

        r = {'refs': {}, 'st': {}, 'w_decomp': {}, 'influence': {}, 'reliability': {},
             'agreement': {}, 'coverage': {}, 'corruption': {}, 'synthesis': []}

        def mw(W):
            return compute_miou(decode(W, val_codes), vl)

        r['refs'] = {
            'frozen': mw(W_clean),
            'oracle_s_all_t_oracle': mw(W_oracle),
            'no_gate_s_all_t_all': mw(W_nogate),
            'w_oracle_cos_nogate': cos_sim(W_oracle, W_nogate),
        }

        # ---- A. S/T decomposition ----
        st = r['st']
        def st_entry(W):
            return {'miou': mw(W), 'cos_oracle': cos_sim(W, W_oracle)}

        st['s_all_t_all'] = st_entry(W_nogate)
        st['s_all_t_oracle'] = st_entry(W_oracle)

        # S=all, T gated hard by confidence quantile
        for fr in gate_fracs:
            thr = torch.quantile(pconf, 1 - fr)
            mask = pconf >= thr
            W = fit_probe(Xd, Y_pseudo, w_t=mask, lam=args.lam,
                          iters=args.cg_iters, m=args.nystrom_m, device=device)
            st[f's_all_t_conf_top{fr}'] = st_entry(W)
            st[f's_all_t_conf_top{fr}']['retain'] = float(mask.float().mean().item())
            st[f's_all_t_conf_top{fr}']['precision'] = float(
                pseudo_correct[mask].float().mean().item())
        # S=all, T gated hard by margin quantile
        for fr in [0.3, 0.5]:
            thr = torch.quantile(pmargin, 1 - fr)
            mask = pmargin >= thr
            W = fit_probe(Xd, Y_pseudo, w_t=mask, lam=args.lam,
                          iters=args.cg_iters, m=args.nystrom_m, device=device)
            st[f's_all_t_margin_top{fr}'] = st_entry(W)
            st[f's_all_t_margin_top{fr}']['retain'] = float(mask.float().mean().item())
            st[f's_all_t_margin_top{fr}']['precision'] = float(
                pseudo_correct[mask].float().mean().item())
        # S=all, T weighted soft by confidence
        for wname, w in [('conf', pconf), ('conf2', pconf ** 2)]:
            W = fit_probe(Xd, Y_pseudo, w_t=w, lam=args.lam,
                          iters=args.cg_iters, m=args.nystrom_m, device=device)
            st[f's_all_t_w_{wname}'] = st_entry(W)
        # S=all, T weighted by the fused uncertainty (low uncertainty = high weight)
        W = fit_probe(Xd, Y_pseudo, w_t=(1.0 - u_fuse).clamp(min=0), lam=args.lam,
                      iters=args.cg_iters, m=args.nystrom_m, device=device)
        st['s_all_t_w_uncer'] = st_entry(W)
        # S=all, T soft (frozen probe distribution), with temperature
        for t in soft_temps:
            Psoft = torch.softmax(pool_s / t, dim=1)
            W = fit_probe(Xd, Psoft, lam=args.lam, iters=args.cg_iters,
                          m=args.nystrom_m, device=device)
            st[f's_all_t_soft_t{t}'] = st_entry(W)
        # S=all, T soft and conf-weighted
        W = fit_probe(Xd, torch.softmax(pool_s, dim=1), w_t=pconf, lam=args.lam,
                      iters=args.cg_iters, m=args.nystrom_m, device=device)
        st['s_all_t_soft_w_conf'] = st_entry(W)
        # S gated, T all  (the reverse decomposition)
        thr = torch.quantile(pconf, 0.7)
        mask = pconf >= thr
        W = fit_probe(Xd, Y_pseudo, w_s=mask, lam=args.lam, iters=args.cg_iters,
                      m=args.nystrom_m, device=device)
        st['s_gated_t_all'] = st_entry(W)
        st['s_gated_t_all']['retain'] = float(mask.float().mean().item())
        # S gated, T gated (the Iteration 9/10 failure mode, reference)
        W = fit_probe(Xd, Y_pseudo, w_s=mask, w_t=mask, lam=args.lam,
                      iters=args.cg_iters, m=args.nystrom_m, device=device)
        st['s_gated_t_gated'] = st_entry(W)
        st['s_gated_t_gated']['retain'] = float(mask.float().mean().item())
        # S=all, T = correct-only / wrong-only (oracle-informed; the purity probes)
        W = fit_probe(Xd, Y_pseudo, w_t=pseudo_correct.float(), lam=args.lam,
                      iters=args.cg_iters, m=args.nystrom_m, device=device)
        st['s_all_t_correct_only'] = st_entry(W)
        W = fit_probe(Xd, Y_pseudo, w_t=(~pseudo_correct).float(), lam=args.lam,
                      iters=args.cg_iters, m=args.nystrom_m, device=device)
        st['s_all_t_wrong_only'] = st_entry(W)

        # ---- B. W decomposition (linearity + noise-vs-rotation diagnosis) ----
        W_corr = fit_probe(Xd, Y_pseudo, w_t=pseudo_correct.float(), lam=args.lam,
                           iters=args.decomp_iters, m=args.nystrom_m, device=device)
        W_wrong = fit_probe(Xd, Y_pseudo, w_t=(~pseudo_correct).float(), lam=args.lam,
                            iters=args.decomp_iters, m=args.nystrom_m, device=device)
        W_sum = W_corr + W_wrong
        wd = r['w_decomp']
        wd['w_correct'] = {'norm': float(W_corr.norm().item()),
                           'cos_oracle': cos_sim(W_corr, W_oracle),
                           'miou': mw(W_corr)}
        wd['w_wrong'] = {'norm': float(W_wrong.norm().item()),
                         'cos_oracle': cos_sim(W_wrong, W_oracle),
                         'cos_correct': cos_sim(W_wrong, W_corr),
                         'miou': mw(W_wrong)}
        wd['linearity'] = {'cos_sum_vs_pseudo': cos_sim(W_sum, W_nogate),
                           'rel_err': float((W_sum - W_nogate).norm().item() /
                                            (W_nogate.norm().item() + 1e-30))}
        wd['norm_ratio_wrong_correct'] = float(W_wrong.norm().item() /
                                               (W_corr.norm().item() + 1e-30))
        wd['pseudo_to_oracle_err'] = float((W_nogate - W_oracle).norm().item() /
                                           (W_oracle.norm().item() + 1e-30))
        wd['correct_to_oracle_err'] = float((W_corr - W_oracle).norm().item() /
                                            (W_oracle.norm().item() + 1e-30))
        if wd['w_correct']['cos_oracle'] > 0.95:
            wd['diagnosis'] = ('C/PURITY: correct points alone recover the oracle; '
                               'the gap is pure label noise. Gating T toward clean '
                               'labels suffices.')
        elif wd['w_wrong']['cos_oracle'] > 0.3:
            wd['diagnosis'] = ('B/ROTATION: wrong points systematically align with '
                               'the oracle direction; confidence gating likely '
                               'insufficient, soft/reweighted T needed.')
        elif wd['correct_to_oracle_err'] > 0.3:
            wd['diagnosis'] = ('C/COVERAGE: even correct-only T does not recover the '
                               'oracle; the pool needs broad label coverage, not just '
                               'purity.')
        else:
            wd['diagnosis'] = ('A/NOISE: wrong points add misaligned noise; gating T '
                               'while keeping S=all is the right lever.')

        # ---- C. influence / leverage ----
        inf = r['influence']
        inf['nystrom_subspace'] = True
        try:
            inf['corr_conf'] = spearman(I, pconf)
            inf['corr_margin'] = spearman(I, pmargin)
        except Exception:
            inf['corr_conf'] = inf['corr_margin'] = None
        inf['mean_i_correct'] = float(I[pseudo_correct].mean().item())
        inf['mean_i_wrong'] = float(I[~pseudo_correct].mean().item())
        inf['mean_g_correct'] = float(G[pseudo_correct].mean().item())
        inf['mean_g_wrong'] = float(G[~pseudo_correct].mean().item())
        bins = []
        qs = torch.quantile(pconf, torch.linspace(0, 1, 11))
        for k in range(10):
            m = (pconf >= qs[k]) & (pconf < qs[k + 1])
            if int(m.sum().item()) < 10:
                continue
            bins.append({
                'conf_lo': float(qs[k].item()), 'conf_hi': float(qs[k + 1].item()),
                'n': int(m.sum().item()),
                'precision': float(pseudo_correct[m].float().mean().item()),
                'mean_i': float(I[m].mean().item()),
                'mean_g': float(G[m].mean().item()),
            })
        inf['conf_bins'] = bins
        # top-influence gate (S=all, T = top-30% by I)
        thr = torch.quantile(I, 0.7)
        mask = I >= thr
        W = fit_probe(Xd, Y_pseudo, w_t=mask, lam=args.lam, iters=args.cg_iters,
                      m=args.nystrom_m, device=device)
        inf['top_i_gate'] = {'miou': mw(W),
                             'cos_oracle': cos_sim(W, W_oracle),
                             'precision': float(pseudo_correct[mask].float().mean().item()),
                             'retain': float(mask.float().mean().item())}
        inf['time_s'] = t_inf

        # ---- D. reliability / calibration / confusion ----
        rel = r['reliability']
        rel['quantile_precision'] = [
            {'q': float(q), 'precision': float(
                pseudo_correct[pconf >= torch.quantile(pconf, q)].float().mean().item())}
            for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]]
        # per-predicted-class precision at within-class confidence quantiles
        per_class = {}
        for c in range(1, NUM_CLASSES):
            m = ppred == c
            if int(m.sum().item()) < 100:
                continue
            row = {'n': int(m.sum().item()),
                   'overall_prec': float(pseudo_correct[m].float().mean().item())}
            for q in [0.5, 0.8, 0.9]:
                thr = torch.quantile(pconf[m], q)
                mm = m & (pconf >= thr)
                row[f'prec_q{q}'] = float(pseudo_correct[mm].float().mean().item())
            per_class[str(c)] = row
        rel['per_class'] = per_class
        # calibration: P(correct | p_max) in absolute probability bins
        cal_bins = []
        edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
        for k in range(len(edges) - 1):
            m = (pconf >= edges[k]) & (pconf < edges[k + 1])
            if int(m.sum().item()) < 10:
                continue
            cal_bins.append({'p_lo': edges[k], 'p_hi': edges[k + 1],
                             'n': int(m.sum().item()),
                             'mean_p': float(pconf[m].mean().item()),
                             'acc': float(pseudo_correct[m].float().mean().item())})
        rel['calibration'] = cal_bins
        # pseudo confusion matrix: top pairs P(hat=a, y=b)
        pairs = []
        for a in range(1, NUM_CLASSES):
            ma = ppred == a
            if int(ma.sum().item()) < 100:
                continue
            for b in range(1, NUM_CLASSES):
                nb = int((ma & (pl == b)).sum().item())
                if nb > 0:
                    pairs.append((a, b, nb, nb / int(ma.sum().item())))
        pairs.sort(key=lambda t: -t[2])
        rel['confusion_top'] = [{'pred': a, 'true': b, 'n': n, 'frac': fr}
                                for a, b, n, fr in pairs[:10]]

        # ---- E. regions and agreement ----
        ag = r['agreement']
        agree = (proto_pred == ppred)
        pops = {}
        for name, m in [('agree_hi', agree & (pconf >= 0.9)),
                        ('agree_lo', agree & (pconf < 0.9)),
                        ('disagree_hi', ~agree & (pconf >= 0.9)),
                        ('disagree_lo', ~agree & (pconf < 0.9))]:
            if int(m.sum().item()) < 50:
                continue
            pops[name] = {'n': int(m.sum().item()),
                          'precision': float(pseudo_correct[m].float().mean().item()),
                          'mean_conf': float(pconf[m].mean().item()),
                          'mean_i': float(I[m].mean().item())}
        ag['populations'] = pops
        # agree-only gate and disagree-high-conf gate (S=all, T gated)
        W = fit_probe(Xd, Y_pseudo, w_t=agree.float(), lam=args.lam,
                      iters=args.cg_iters, m=args.nystrom_m, device=device)
        ag['agree_only'] = {'miou': mw(W), 'cos_oracle': cos_sim(W, W_oracle),
                            'retain': float(agree.float().mean().item())}
        m_dis = (~agree) & (pconf >= 0.9)
        W = fit_probe(Xd, Y_pseudo, w_t=m_dis.float(), lam=args.lam,
                      iters=args.cg_iters, m=args.nystrom_m, device=device)
        ag['disagree_hi_conf'] = {'miou': mw(W), 'cos_oracle': cos_sim(W, W_oracle),
                                  'retain': float(m_dis.float().mean().item())}
        # region-conditional gate: within each prototype region, keep top-30% conf
        mask_region = torch.zeros(len(pool), dtype=torch.bool)
        for c in proto_pred.unique():
            m = proto_pred == c
            thr = torch.quantile(pconf[m], 0.7)
            mask_region |= m & (pconf >= thr)
        W = fit_probe(Xd, Y_pseudo, w_t=mask_region, lam=args.lam, iters=args.cg_iters,
                      m=args.nystrom_m, device=device)
        ag['region_cond_top30'] = {'miou': mw(W), 'cos_oracle': cos_sim(W, W_oracle),
                                   'retain': float(mask_region.float().mean().item()),
                                   'precision': float(
                                       pseudo_correct[mask_region].float().mean().item())}
        # per-class-conditional gate: within each predicted class, keep top-30% conf
        mask_class = torch.zeros(len(pool), dtype=torch.bool)
        for c in range(1, NUM_CLASSES):
            m = ppred == c
            if int(m.sum().item()) < 100:
                continue
            thr = torch.quantile(pconf[m], 0.7)
            mask_class |= m & (pconf >= thr)
        W = fit_probe(Xd, Y_pseudo, w_t=mask_class, lam=args.lam, iters=args.cg_iters,
                      m=args.nystrom_m, device=device)
        ag['class_cond_top30'] = {'miou': mw(W), 'cos_oracle': cos_sim(W, W_oracle),
                                  'retain': float(mask_class.float().mean().item()),
                                  'precision': float(
                                      pseudo_correct[mask_class].float().mean().item())}

        # ---- F. coverage per gate ----
        trace_all = float((Xd * Xd).sum().item())
        cov = r['coverage']
        conf_top30_mask = pconf >= torch.quantile(pconf, 0.7)
        conf_top10_mask = pconf >= torch.quantile(pconf, 0.9)
        for gname, mask in [('all', torch.ones(len(pool), dtype=torch.bool)),
                            ('conf_top30', conf_top30_mask),
                            ('conf_top10', conf_top10_mask),
                            ('correct_only', pseudo_correct),
                            ('class_cond_top30', mask_class),
                            ('region_cond_top30', mask_region)]:
            stats = coverage_stats(Xd, mask, device, trace_all)
            stats['classes_covered'] = int(ppred[mask].unique().numel())
            cov[gname] = stats
        # effective-rank of S_all itself (participation ratio)
        frob_all2 = hutch_frob2(Xd, device)
        cov['s_all'] = {'trace': trace_all, 'frob': float(frob_all2 ** 0.5),
                        'eff_rank': float(trace_all ** 2 / frob_all2)}

        # ---- G. coverage-preserving (per-cluster top-conf) gates ----
        pool_feats = pool.float()
        for K in cluster_ks:
            labs = kmeans_labels(pool_feats, K, device=device)
            mask = torch.zeros(len(pool), dtype=torch.bool)
            for c in range(K):
                m = labs == c
                if int(m.sum().item()) < 10:
                    continue
                thr = torch.quantile(pconf[m], 0.7)
                mask |= m & (pconf >= thr)
            W = fit_probe(Xd, Y_pseudo, w_t=mask, lam=args.lam, iters=args.cg_iters,
                          m=args.nystrom_m, device=device)
            ag[f'cluster_k{K}_top30'] = {'miou': mw(W), 'cos_oracle': cos_sim(W, W_oracle),
                                         'retain': float(mask.float().mean().item()),
                                         'precision': float(
                                             pseudo_correct[mask].float().mean().item())}

        # ---- H. oracle gate decomposition: precision -> mIoU per strategy ----
        corr = r['corruption']
        wrong_idx = torch.nonzero(~pseudo_correct).squeeze(1)
        I_wrong = I[~pseudo_correct]
        torch.manual_seed(7)
        orderings = {
            'random': torch.randperm(len(wrong_idx)),
            'conf_desc': torch.argsort(pconf[~pseudo_correct], descending=True),
            'conf_asc': torch.argsort(pconf[~pseudo_correct], descending=False),
            'lev_desc': torch.argsort(I_wrong, descending=True),
            'lev_asc': torch.argsort(I_wrong, descending=False),
        }
        n_corr = int(pseudo_correct.sum().item())
        for sname, order in orderings.items():
            corr[sname] = {}
            for p in oracle_precs:
                n_w = int(np.ceil(n_corr * (1 - p) / p))
                keep = pseudo_correct.clone()
                if n_w > 0:
                    keep[wrong_idx[order[:n_w]]] = True
                W = fit_probe(Xd, Y_pseudo, w_t=keep.float(), lam=args.lam,
                              iters=args.cg_iters, m=args.nystrom_m, device=device)
                corr[sname][str(p)] = {'miou': mw(W),
                                       'cos_oracle': cos_sim(W, W_oracle),
                                       'n': int(keep.sum().item()),
                                       'precision': float(
                                           pseudo_correct[keep].float().mean().item())}

        # ---- per-gate AUROC (which signal separates correct from wrong) ----
        r['auroc'] = {}
        for name, sig in [('conf', pconf), ('margin', pmargin), ('norm', -pnorm),
                          ('influence', I), ('leverage', G)]:
            r['auroc'][name] = rank_auc(pseudo_correct, sig)

        # ---- per-condition doc-ready synthesis ----
        st_best = max((v['miou'], k) for k, v in st.items() if k != 's_all_t_oracle')
        syn = r['synthesis']
        syn.append(f"COND {cond} (pseudo acc {pseudo_correct.float().mean():.3f}, "
                   f"n {len(pool)})")
        syn.append(f"refs: frozen {r['refs']['frozen']:.3f} / no_gate {r['refs']['no_gate_s_all_t_all']:.3f} "
                   f"/ oracle {r['refs']['oracle_s_all_t_oracle']:.3f}")
        syn.append(f"st best: {st_best[1]} {st_best[0]:.3f} "
                   f"(vs no_gate {st['s_all_t_all']['miou']:.3f}, "
                   f"w-cos {st[st_best[1]]['cos_oracle']:.3f})")
        for k in ['s_all_t_conf_top0.3', 's_all_t_conf_top0.1', 's_all_t_soft_t1.0',
                  's_all_t_correct_only', 's_gated_t_gated', 's_gated_t_all']:
            if k in st:
                e = st[k]
                syn.append(f"  {k}: {e['miou']:.3f} (cos {e['cos_oracle']:.3f}, "
                           f"retain {e.get('retain', 1.0):.3f}, "
                           f"prec {e.get('precision', 'na')})")
        syn.append(f"w_decomp: {wd['diagnosis']}")
        syn.append(f"  ||W_wrong||/||W_correct|| {wd['norm_ratio_wrong_correct']:.3f} | "
                   f"cos(Wc,oracle) {wd['w_correct']['cos_oracle']:.3f} | "
                   f"cos(Ww,oracle) {wd['w_wrong']['cos_oracle']:.3f} | "
                   f"linearity {wd['linearity']['cos_sum_vs_pseudo']:.4f}")
        syn.append(f"influence: corr_conf {inf.get('corr_conf')} / corr_margin {inf.get('corr_margin')} | "
                   f"mean_i correct {inf['mean_i_correct']:.3f} vs wrong {inf['mean_i_wrong']:.3f} | "
                   f"top_i gate {inf['top_i_gate']['miou']:.3f} (prec {inf['top_i_gate']['precision']:.3f})")
        pc_vals = [v['overall_prec'] for v in rel['per_class'].values()]
        pc_hi = [v['prec_q0.9'] for v in rel['per_class'].values() if 'prec_q0.9' in v]
        if pc_vals and pc_hi:
            syn.append(f"per-class spread: overall prec {min(pc_vals):.3f}..{max(pc_vals):.3f} | "
                       f"q0.9 prec {min(pc_hi):.3f}..{max(pc_hi):.3f}")
        for p in [0.7, 0.8, 0.9, 1.0]:
            syn.append(f"  oracle-gate p={p}: random {corr['random'][str(p)]['miou']:.3f} / "
                       f"conf_desc {corr['conf_desc'][str(p)]['miou']:.3f} / "
                       f"conf_asc {corr['conf_asc'][str(p)]['miou']:.3f} / "
                       f"lev_desc {corr['lev_desc'][str(p)]['miou']:.3f}")

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        print(f"  pseudo-label acc {pseudo_correct.float().mean():.3f} | "
              f"frozen {r['refs']['frozen']:.4f} | oracle {r['refs']['oracle_s_all_t_oracle']:.4f} "
              f"| no_gate {r['refs']['no_gate_s_all_t_all']:.4f}")
        print(f"  S/T: " + " ".join(
            f"{k.split('s_all_t_')[-1]}:{v['miou']:.4f}"
            for k, v in st.items() if k.startswith('s_all_t_')))
        print(f"  W decomp: {wd['diagnosis']}")
        print(f"  influence: corr_conf {inf.get('corr_conf')} | mean_i corr {inf['mean_i_correct']:.3f} "
              f"vs wrong {inf['mean_i_wrong']:.3f} | top_i gate {inf['top_i_gate']['miou']:.4f}")
        print(f"  agreement: " + " ".join(f"{k}:{v['miou']:.4f}" for k, v in ag.items()
                                          if k.endswith('top30') or k.endswith('only')
                                          or k == 'disagree_hi_conf'))
        print(f"  corruption: " + " ".join(
            f"{s[:4]}(p={p}):{corr[s][str(p)]['miou']:.4f}"
            for s in ['random', 'conf_desc', 'lev_desc'] for p in [0.7, 0.9, 1.0]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("A. st: does S_all,T_gated beat S_all,T_all? If yes, gating T while keeping")
    print("   the geometry is the lever (Iteration 9 gated both and failed). If S_gated")
    print("   rows also stay low, the covariance needs ALL points.")
    print("B. w_decomp: linearity cos ~1 confirms W_pseudo = W_correct + W_wrong; the")
    print("   diagnosis letter (A/B/C) says whether wrong points are noise, rotation,")
    print("   or whether correct-only T lacks coverage.")
    print("C. influence: top_i_gate mIoU vs conf gates; corr_conf ~1 means influence")
    print("   and confidence rank the same way (a confidence gate cannot be beaten),")
    print("   ~0 means a leverage-aware gate is a genuinely different selection.")
    print("D. reliability.per_class: the spread of prec_q0.9 across classes says whether")
    print("   a GLOBAL confidence threshold is right (small spread) or a per-class gate")
    print("   is needed (large spread).")
    print("E. agreement: disagree_hi_conf population precision and its gate mIoU test")
    print("   the rotation-aware-gate hypothesis.")
    print("F. coverage: frob_diff_ratio near 0 for a gate means it preserves the")
    print("   covariance; conf_top10 should be large vs class_cond_top30.")
    print("G. cluster_k* / class_cond / region_cond: coverage-preserving gates.")
    print("H. corruption: at what precision does T-only gating cross the frozen decode?")
    print("   If even p=1.0 (perfect gate) is far below oracle, purity alone is not the")
    print("   story and the oracle-T construction is the upper bound to trust.")

if __name__ == "__main__":
    main()
