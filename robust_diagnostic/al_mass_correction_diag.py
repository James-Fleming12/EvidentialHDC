"""al_mass_correction_diag.py: Iteration-8 -- the mass-calibration and
mean-estimation routes (eval-only, no plots).

Iteration 7 narrowed the problem to two measurable pieces: the soft-mass class
COUNTS are catastrophically wrong (rel err 8-611x) and the within-class
ASSIGNMENT calibration (confusion matrix) needs more labels than the budget.
The T-synthesis framework is proven sound (T_oracle through the T-ridge = the
oracle exactly); the means ARE estimable (8-32 random points -> cos 0.95-0.99);
and the ridge is extraordinarily sensitive to the last few percent of T
(7A: t_cos 0.95-0.97 but w_cos ~0.03-0.13).

This iteration tests the two routes with progressively narrower questions.

SECTION 1 (the deciding experiment, run FIRST): 8D oracle-count ceiling.
  T_8D = X^T diag(N_oracle / N_soft) Q_frozen.
  If 8D ~ oracle: the problem is almost entirely class-mass calibration.
  If 8D substantially improves 7D but stays below oracle: mass + assignment
    error, both attackable.
  If 8D ~ 7D: the soft assignment itself is wrong; abandon the soft-Q route.

SECTION 2: the mass-correction family (soft-Q, 17-parameter calibration):
  8A raw alpha    : Q' = diag(alpha) Q, alpha_c = #(y_hat=c in L)/#(y=c in L)
  8E normalized   : Q'_ic = alpha_c Q_ic / sum_j alpha_j Q_ij (preserves the
                    per-point normalization -- a prior correction, not a
                    reweighting of the ridge targets)
  8F source-count : alpha_c = (N_source_c / N_soft_c)^rho *
                    (N_queried_true_c / N_queried_pred_c)^(1-rho), rho sweep;
                    uses the source class prevalence as the low-dim prior
  8G top-K        : L_c = top-K points by q_c(x) with K = N_hat_c (the 8F
                    count), T_c = sum over L_c of q_c(x) x -- corrected mass
                    + soft weights + no confusion model
  each: t_cos / w_cos / mIoU.

SECTION 3: the mean-estimation route (variance reduction, not expansion):
  3a mean-estimator comparison: strategies {random, confidence-top,
     influence-inverted, diversity-greedy} x k {8, 16, 32} -> cos(mu_hat,
     mu_oracle) -- can deterministic bulk sampling beat random (0.98->0.995+)?
  3b source-count synthesis: T_c = N_c x mu_hat_c with N_c in {N_source,
     N_oracle} x the best mean estimator (point 15: bypass the soft counts
     entirely).
  3c control-variate shrinkage: mu_hat(rho) = (1-rho) mu_clean + rho x_bar
     (rho sweep), evaluated by the RIDGE-RELEVANT (whitened) error
     ||(S+lI)^-1 (T_hat_c - T_oracle_c)|| / ||(S+lI)^-1 T_oracle_c|| via CG --
     the quantity that actually controls W (points 10/14), plus t_cos/mIoU.

All label-free selection strategies; oracle labels used only for evaluation.

Usage:
  uv run python robust_diagnostic/al_mass_correction_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_mass_correction_covshift_ep10.json
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

def cos_sim(a, b):
    a = a.detach().cpu().float().reshape(-1)
    b = b.detach().cpu().float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))

def nystrom_influence(Xd, lam, m, device):
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    M = torch.linalg.inv(Shat)
    C = Xd @ P
    MC = C @ M
    return (MC.norm(dim=1) * (Xd.shape[1] ** 0.5)).cpu()

def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device)
    torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1])
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P
    Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    That = XP.t() @ Yd
    x = P @ torch.linalg.solve(Shat, That)
    b = X.t() @ Yd
    def A(v):
        return X.t() @ (X @ v)
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

def t_ridge(W, T_hat, Xd, lam, iters, m, device):
    """Solve W = (S + lI)^-1 T_hat with the Nystrom warm start + CG."""
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    That = P.t() @ T_hat.to(device)
    x = P @ torch.linalg.solve(Shat, That)
    b = T_hat.to(device)
    def A(v):
        return Xd.t() @ (Xd @ v)
    res = b - A(x)
    p = res.clone()
    rs_old = (res * res).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha_k = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha_k.unsqueeze(0) * p
        res = res - alpha_k.unsqueeze(0) * Ap
        rs_new = (res * res).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = res + beta.unsqueeze(0) * p
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
    parser.add_argument("--labels_per_class", type=int, default=2)
    parser.add_argument("--rho_sweep", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--mean_ks", type=str, default="8,16,32")
    parser.add_argument("--mean_repeats", type=int, default=5)
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_mass_correction_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    rho_sweep = [float(x) for x in args.rho_sweep.split(',')]
    mean_ks = [int(x) for x in args.mean_ks.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)

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

        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]

        proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
        Xc = torch.sign(fa[ci].to(device) @ proj).cpu().float()
        Xp = torch.sign(pool.to(device) @ proj).cpu().float()
        Xv = torch.sign(val.to(device) @ proj).cpu().float()
        Xd = Xp.to(device)

        W_clean = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam,
                                 args.cg_iters, args.nystrom_m, device)
        W_oracle = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam,
                                  args.cg_iters, args.nystrom_m, device)

        def mw(W):
            return compute_miou(decode(W, Xv), vl)

        r = {'refs': {}, 'deciding': {}, 'mass': {}, 'mean_route': {},
             'synthesis': []}
        r['refs'] = {'frozen': mw(W_clean), 'oracle': mw(W_oracle)}

        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}
        cls_idx_c = {c: (la[ci] == c).nonzero().squeeze(1) for c in classes}

        # oracle T and per-class means
        T_or = torch.zeros(10000, NUM_CLASSES)
        mu_or = {}
        for c in classes:
            idx = cls_idx[c]
            if len(idx) == 0:
                continue
            T_or[:, c] = Xp[idx].sum(dim=0)
            mu_or[c] = Xp[idx].mean(dim=0)
        m0 = pl == 0
        if int(m0.sum().item()) > 0:
            T_or[:, 0] = Xp[m0].sum(dim=0)

        # source counts (clean) and oracle counts (pool)
        N_source = {c: len(cls_idx_c[c]) for c in classes}
        N_or = {c: len(cls_idx[c]) for c in classes}

        def tcos(T_hat):
            coss = []
            for c in classes:
                if T_or[:, c].norm().item() < 1e-9:
                    continue
                coss.append(cos_sim(T_hat[:, c], T_or[:, c]))
            return float(np.mean(coss)) if coss else None

        def evalm(T_hat):
            W = t_ridge(W_oracle, T_hat, Xd, args.lam, args.cg_iters,
                        args.nystrom_m, device)
            return {'t_cos': tcos(T_hat), 'w_cos': cos_sim(W, W_oracle),
                    'miou': mw(W)}

        # frozen soft assignments
        logits = scores(W_clean, Xp)
        Q_frozen = torch.softmax(logits, dim=1)
        ppred = logits.argmax(dim=1)
        N_soft = Q_frozen.sum(dim=0)                      # per-class soft counts

        # queried labels (influence class-floor)
        I = nystrom_influence(Xd, args.lam, args.nystrom_m, device)
        qidx = []
        for c in classes:
            idx = cls_idx[c]
            if len(idx) == 0:
                continue
            picks = [int(idx[int(I[idx].argmax().item())].item())]
            if args.labels_per_class > 1:
                m2 = torch.ones(len(idx), dtype=torch.bool)
                m2[picks[0] - int(idx[0].item())] = False
                if int(m2.sum().item()) > 0:
                    picks.append(int(idx[m2][int(I[idx[m2]].argmax().item())].item()))
            qidx.extend(picks)
        qidx = torch.tensor(qidx)
        qlbl = pl[qidx]

        # ---- SECTION 1: 8D oracle-count ceiling (the deciding experiment) ----
        dec = r['deciding']
        alpha_or = torch.zeros(NUM_CLASSES)
        for c in classes:
            alpha_or[c] = N_or[c] / (N_soft[c].item() + 1e-9)
        alpha_or[0] = 1.0
        T_8d = Xp.t() @ (Q_frozen * alpha_or.unsqueeze(0))
        dec['8D_oracle_count'] = evalm(T_8d)
        # verdict
        gap = r['refs']['oracle'] - r['refs']['frozen']
        d8_gain = dec['8D_oracle_count']['miou'] - r['refs']['frozen']
        if dec['8D_oracle_count']['miou'] >= r['refs']['oracle'] - 0.02:
            dec['verdict'] = 'STRONG: count correction alone reaches the oracle -- the problem is almost entirely class-mass calibration.'
        elif d8_gain > 0.5 * gap:
            dec['verdict'] = 'PARTIAL: mass + assignment error, both attackable.'
        elif dec['8D_oracle_count']['miou'] <= r['refs']['frozen'] + 0.02:
            dec['verdict'] = 'WEAK: the soft assignment itself is wrong -- abandon the soft-Q route.'
        else:
            dec['verdict'] = 'MODEST: count correction helps but the assignment error dominates.'

        # ---- SECTION 2: mass-correction family ----
        ms = r['mass']
        # 8A raw alpha from the queried count ratio
        Nq_pred = torch.zeros(NUM_CLASSES)
        Nq_true = torch.zeros(NUM_CLASSES)
        for j in range(len(qidx)):
            Nq_pred[ppred[qidx[j]]] += 1.0
            Nq_true[qlbl[j]] += 1.0
        alpha_8a = torch.ones(NUM_CLASSES)
        for c in classes:
            if Nq_pred[c].item() > 0:
                alpha_8a[c] = Nq_true[c].item() / Nq_pred[c].item()
        ms['8A_raw'] = evalm(Xp.t() @ (Q_frozen * alpha_8a.unsqueeze(0)))
        # 8E normalized alpha (preserves per-point normalization)
        Q_8e = Q_frozen * alpha_8a.unsqueeze(0)
        Q_8e = Q_8e / (Q_8e.sum(dim=1, keepdim=True) + 1e-9)
        ms['8E_normalized'] = evalm(Xp.t() @ Q_8e)
        # 8F source-count alpha with rho sweep
        ms['8F_source'] = {}
        for rho in rho_sweep:
            alpha_f = torch.ones(NUM_CLASSES)
            for c in classes:
                qr = (Nq_true[c].item() / (Nq_pred[c].item() + 1e-9)) \
                    if Nq_pred[c].item() > 0 else 1.0
                sr = N_source[c] / (N_soft[c].item() + 1e-9)
                alpha_f[c] = (sr ** rho) * (qr ** (1 - rho))
            ms['8F_source'][str(rho)] = evalm(Xp.t() @ (Q_frozen * alpha_f.unsqueeze(0)))
        # 8G top-K with corrected counts (best 8F alpha, soft weights)
        best_rho = max(rho_sweep, key=lambda x:
                       ms['8F_source'][str(x)]['t_cos'] or 0)
        alpha_g = torch.ones(NUM_CLASSES)
        for c in classes:
            qr = (Nq_true[c].item() / (Nq_pred[c].item() + 1e-9)) \
                if Nq_pred[c].item() > 0 else 1.0
            sr = N_source[c] / (N_soft[c].item() + 1e-9)
            alpha_g[c] = (sr ** best_rho) * (qr ** (1 - best_rho))
        T_8g = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            K = max(1, int(round(alpha_g[c].item() * N_soft[c].item())))
            K = min(K, len(pool))
            idx_top = torch.argsort(Q_frozen[:, c], descending=True)[:K]
            T_8g[:, c] = Xp[idx_top].t() @ Q_frozen[idx_top, c]
        ms['8G_topK'] = evalm(T_8g)

        # ---- SECTION 3: mean-estimation route ----
        mr = r['mean_route']
        zn = pool.float()
        zn = zn / (zn.norm(dim=1, keepdim=True) + 1e-8)
        pconf = Q_frozen.max(dim=1).values

        # 3a: mean-estimator comparison
        strategies = {}
        strategies['random'] = lambda idx, k: idx[torch.randperm(len(idx))[:k]]
        strategies['conf_top'] = lambda idx, k: idx[torch.argsort(
            pconf[idx], descending=True)[:k]]
        strategies['inf_inv'] = lambda idx, k: idx[torch.argsort(
            (1.0 / (I[idx] + 1e-6)), descending=True)[:k]]
        def div_greedy(idx, k):
            torch.manual_seed(1)
            sel = [int(idx[int(torch.argmax(pconf[idx]).item())].item())]
            remain = idx[idx != sel[0]]
            for _ in range(k - 1):
                if len(remain) == 0:
                    break
                sims = zn[remain] @ zn[sel].t()
                worst = sims.max(dim=1).values
                j = int(torch.argmax(worst).item())
                sel.append(int(remain[j].item()))
                remain = remain[remain != sel[-1]]
            return torch.tensor(sel)
        strategies['diversity'] = div_greedy

        est = {}
        for sname, sel_fn in strategies.items():
            for k in mean_ks:
                coss = []
                for c in classes:
                    idx = cls_idx[c]
                    if len(idx) < max(50, k):
                        continue
                    mu_true = zn[idx].mean(dim=0)
                    mu_true = mu_true / (mu_true.norm() + 1e-8)
                    cc = []
                    for rep in range(args.mean_repeats):
                        torch.manual_seed(rep)
                        sel = sel_fn(idx, k)
                        mu_h = zn[sel].mean(dim=0)
                        mu_h = mu_h / (mu_h.norm() + 1e-8)
                        cc.append(float((mu_h * mu_true).sum().item()))
                    coss.append(float(np.mean(cc)))
                est[f'{sname}_k{k}'] = float(np.mean(coss)) if coss else None
        mr['mean_estimators'] = est

        # 3b: source-count / oracle-count synthesis with the best mean estimator.
        # The selection is done in the CODE space (the means T consumes live
        # there; 128-d selection would be a different, cheaper variant).
        best_est = max(est, key=lambda x: est[x] or 0)
        k_best = max(mean_ks)
        sel_fn = strategies[best_est.split('_k')[0]]
        mu_hat_c = {}
        for c in classes:
            idx = cls_idx[c]
            if len(idx) >= max(50, k_best):
                torch.manual_seed(2)
                sel = sel_fn(idx, k_best)
                mu_hat_c[c] = Xp[sel].mean(dim=0)
        T_src = torch.zeros(10000, NUM_CLASSES)
        T_orc = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            if c in mu_hat_c:
                T_src[:, c] = N_source[c] * mu_hat_c[c]
                T_orc[:, c] = N_or[c] * mu_hat_c[c]
        mr['3b_source_counts'] = evalm(T_src)
        mr['3b_oracle_counts'] = evalm(T_orc)

        # 3c: control-variate shrinkage rho sweep with ridge-relevant error
        mu_clean_c = {c: Xc[cls_idx_c[c]].mean(dim=0) for c in classes
                      if len(cls_idx_c[c]) > 0}
        mr['3c_shrinkage'] = {}
        # whitened error helper: ||(S+lI)^-1 (T_hat_c - T_or_c)|| via CG
        def whitened_err(T_hat):
            errs = []
            for c in classes:
                if T_or[:, c].norm().item() < 1e-9:
                    continue
                dT = (T_hat[:, c] - T_or[:, c]).to(device)
                # solve (S + lI) z = dT
                torch.manual_seed(SKETCH_SEED)
                P = (torch.rand(Xd.shape[1], args.nystrom_m, device=device) > 0.5).float() * 2 - 1
                XP = Xd @ P
                Shat = XP.t() @ XP + args.lam * torch.eye(args.nystrom_m, device=device)
                That = P.t() @ dT.unsqueeze(1)
                z = P @ torch.linalg.solve(Shat, That)
                b = dT.unsqueeze(1)
                def A(v):
                    return Xd.t() @ (Xd @ v)
                res = b - A(z)
                p = res.clone()
                rs_old = (res * res).sum(dim=0)
                for _ in range(args.cg_iters):
                    Ap = A(p)
                    ak = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
                    z = z + ak.unsqueeze(0) * p
                    res = res - ak.unsqueeze(0) * Ap
                    rsn = (res * res).sum(dim=0)
                    be = rsn / (rs_old + 1e-30)
                    p = res + be.unsqueeze(0) * p
                    rs_old = rsn
                # oracle-side whitened norm
                zor = None
                torch.manual_seed(SKETCH_SEED)
                P2 = (torch.rand(Xd.shape[1], args.nystrom_m, device=device) > 0.5).float() * 2 - 1
                XP2 = Xd @ P2
                Shat2 = XP2.t() @ XP2 + args.lam * torch.eye(args.nystrom_m, device=device)
                That2 = P2.t() @ T_or[:, c].to(device).unsqueeze(1)
                zor = P2 @ torch.linalg.solve(Shat2, That2)
                b2 = T_or[:, c].to(device).unsqueeze(1)
                res2 = b2 - (Xd.t() @ (Xd @ zor))
                p2 = res2.clone()
                rs2 = (res2 * res2).sum(dim=0)
                for _ in range(args.cg_iters):
                    Ap2 = Xd.t() @ (Xd @ p2)
                    ak2 = rs2 / ((p2 * Ap2).sum(dim=0) + 1e-30)
                    zor = zor + ak2.unsqueeze(0) * p2
                    res2 = res2 - ak2.unsqueeze(0) * Ap2
                    rsn2 = (res2 * res2).sum(dim=0)
                    be2 = rsn2 / (rs2 + 1e-30)
                    p2 = res2 + be2.unsqueeze(0) * p2
                    rs2 = rsn2
                errs.append(float(z.norm().item() / (zor.norm().item() + 1e-9)))
            return float(np.mean(errs)) if errs else None

        for rho in rho_sweep:
            T_h = torch.zeros(10000, NUM_CLASSES)
            for c in classes:
                if c in mu_clean_c and c in mu_hat_c:
                    T_h[:, c] = N_or[c] * ((1 - rho) * mu_clean_c[c] +
                                           rho * mu_hat_c[c])
            e = evalm(T_h)
            e['whitened_err'] = whitened_err(T_h)
            mr['3c_shrinkage'][str(rho)] = e

        # ---- synthesis ----
        syn = r['synthesis']
        syn.append(f"COND {cond}: frozen {r['refs']['frozen']:.3f} / oracle "
                   f"{r['refs']['oracle']:.3f} | labels {len(qidx)}")
        syn.append(f"  8D oracle-count: {dec['8D_oracle_count']['miou']:.3f} "
                   f"(t {dec['8D_oracle_count']['t_cos']:.3f}, w "
                   f"{dec['8D_oracle_count']['w_cos']:.3f}) | verdict: "
                   f"{dec['verdict']}")
        syn.append(f"  8A raw {ms['8A_raw']['miou']:.3f} | 8E norm "
                   f"{ms['8E_normalized']['miou']:.3f} | 8G topK "
                   f"{ms['8G_topK']['miou']:.3f} | 8F: " + " ".join(
                       f"r{r}:{ms['8F_source'][r]['miou']:.3f}" for r in rho_sweep))
        syn.append(f"  mean-est: " + " ".join(
            f"{k}:{est[k]:.3f}" for k in
            [f'random_k{mean_ks[-1]}', f'conf_top_k{mean_ks[-1]}',
             f'inf_inv_k{mean_ks[-1]}', f'diversity_k{mean_ks[-1]}']))
        syn.append(f"  3b: source-count {mr['3b_source_counts']['miou']:.3f} "
                   f"(t {mr['3b_source_counts']['t_cos']:.3f}) | oracle-count "
                   f"{mr['3b_oracle_counts']['miou']:.3f} (t "
                   f"{mr['3b_oracle_counts']['t_cos']:.3f})")
        syn.append(f"  3c shrink: " + " ".join(
            f"r{r}:{mr['3c_shrinkage'][r]['miou']:.3f}"
            f"(w {mr['3c_shrinkage'][r]['whitened_err']:.3f})" for r in rho_sweep))

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("SECTION 1 (deciding): 8D oracle-count ceiling. STRONG -> the problem")
    print("  is almost entirely class-mass calibration (8A/8E/8F/8G path is very")
    print("  promising). PARTIAL -> mass + assignment error, both attackable.")
    print("  WEAK -> the soft assignment itself is wrong; abandon soft-Q and go")
    print("  the mean route.")
    print("SECTION 2: 8A raw alpha (noisy at 2 labels/class), 8E normalized,")
    print("  8F source-count prior (rho sweep), 8G top-K with corrected counts.")
    print("SECTION 3 (mean route): 3a which bulk-sampling strategy beats random")
    print("  (the sample-complexity 0.98-0.99 baseline); 3b source-count +")
    print("  target-mean synthesis (bypasses the soft counts entirely); 3c")
    print("  control-variate shrinkage with the WHITENED (ridge-relevant) error")
    print("  -- the quantity that actually controls W, per points 10/14.")

if __name__ == "__main__":
    main()
