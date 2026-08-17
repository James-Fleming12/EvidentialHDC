"""al_tsynthesis_diag.py: Iteration-7 -- estimate T, not labels (eval-only, no
plots). The escape from the coverage ceiling.

The diagnosis (Iterations 5-6): the AL gap is the missing-mass term
W_labeled = W_oracle - (S + lI)^-1 (sum over UNLABELED points). You do NOT need
a label on every point -- the oracle requires the 17 class-wise vector sums
T_c = sum_{i: y_i = c} x_i. If T can be estimated from ALL points with soft
assignment probabilities (calibrated by a few true labels), the mass problem is
solved without labeling the mass.

Section A. CLASS-MEAN SAMPLE COMPLEXITY (the deciding diagnostic):
  for each class, k random points (k = 1..256) estimate the class mean; report
  cos(mu_hat_c(k), mu_c_oracle) mean +- std over R draws, in the 128-d space
  AND the 10k-d code space. If ~32-64 points suffice, the mean-estimation
  family is viable; if even 100+ fail, abandon mean estimation and use the
  soft-mass family (which uses all 50k unlabeled points to reduce variance).

Section B. T-SYNTHESIS ablation (7A-7F), evaluated on the three-level chain:
  per-class cos(T_hat_c, T_oracle_c) -> cos(W_hat, W_oracle) -> mIoU.

  7A clean-mean   : T_c = N_c * mu_c_clean        (the extrapolation baseline)
  7B shift        : T_c = N_c * (mu_c_clean + d_hat_c), d_hat from k labeled
                    classes carried to all (the proposed shift model)
  7C shrink-shift : mu = mu_clean + alpha * d_hat_global (robust shrinkage)
  7D soft-frozen  : T = X^T Q_frozen              (soft-mass, no labels)
  7E calibrated   : T = X^T (Q_frozen C_hat)      (soft-mass + confusion
                    correction from the queried labels -- the star)
  7F shift-Q      : q_c(x) propto q_frozen,c(x) * exp(beta * cos128(x, mu_prior_c))
                    (decision info from the probe, geometry from the 128-d
                    shift prior -- the hybrid)

  Oracle-shift ceiling (all classes' true shifts) reported for 7B. All
  synthesis in the 10k-d CODE space (where the ridge lives); the shift prior
  means and cosines in the 128-d space (where the geometry is not saturated).
  N_c = oracle class counts for the mean-synthesis arms (isolates the MEAN
  accuracy); the soft-mass arms estimate N via the assignments naturally.
  Also reports the soft-mass count estimate N_hat_c vs oracle (the mass
  correction quality).

Usage:
  uv run python robust_diagnostic/al_tsynthesis_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_tsynthesis_covshift_ep10.json
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
    parser.add_argument("--labels_per_class", type=int, default=2,
                        help="queried labels per class (1/2/4)")
    parser.add_argument("--shrink_alpha", type=float, default=0.5,
                        help="7C shrinkage: mu = mu_clean + alpha * d_global")
    parser.add_argument("--shift_beta", type=float, default=1.0,
                        help="7F: exp(beta * cos128(x, mu_prior))")
    parser.add_argument("--mean_ks", type=str, default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--mean_repeats", type=int, default=10)
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_tsynthesis_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
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

        Yc = onehot(la[ci], NUM_CLASSES)
        W_clean = ridge_fit_soft(Xc, Yc, args.lam, args.cg_iters, args.nystrom_m, device)
        W_oracle = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam,
                                  args.cg_iters, args.nystrom_m, device)

        def mw(W):
            return compute_miou(decode(W, Xv), vl)

        r = {'refs': {}, 'sample_complexity': {}, 't_synthesis': {},
             'synthesis': []}
        r['refs'] = {'frozen': mw(W_clean), 'oracle': mw(W_oracle)}

        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}
        cls_idx_c = {c: (la[ci] == c).nonzero().squeeze(1) for c in classes}

        # oracle T (code space) and class means
        T_or = torch.zeros(10000, NUM_CLASSES)
        mu_or = {}
        for c in classes:
            idx = cls_idx[c]
            if len(idx) == 0:
                continue
            T_or[:, c] = Xp[idx].sum(dim=0)
            mu_or[c] = Xp[idx].mean(dim=0)
        mu_clean = {c: Xc[cls_idx_c[c]].mean(dim=0) for c in classes
                    if len(cls_idx_c[c]) > 0}

        # ---- Section A: class-mean sample complexity ----
        sc = r['sample_complexity']
        for space_name, Xmat in [('128d', pool.float()), ('code', Xp)]:
            zn = Xmat
            zn = zn / (zn.norm(dim=1, keepdim=True) + 1e-8)
            rows = {}
            for c in classes:
                idx = cls_idx[c]
                n_c = len(idx)
                if n_c < 100:
                    continue
                mu_true = zn[idx].mean(dim=0)
                mu_true = mu_true / (mu_true.norm() + 1e-8)
                for k in mean_ks:
                    if k > n_c:
                        continue
                    coss = []
                    for rep in range(args.mean_repeats):
                        torch.manual_seed(rep)
                        sub = zn[idx[torch.randperm(n_c)[:k]]].mean(dim=0)
                        sub = sub / (sub.norm() + 1e-8)
                        coss.append(float((sub * mu_true).sum().item()))
                    rows.setdefault(str(k), []).append((n_c, float(np.mean(coss)),
                                                        float(np.std(coss))))
            sc[space_name] = {k: {'n_class': rows[k][0][0],
                                  'cos_mean': float(np.mean([x[1] for x in rows[k]])),
                                  'cos_std': float(np.mean([x[2] for x in rows[k]]))}
                              for k in rows}

        # ---- queried labels (influence class-floor) ----
        I = nystrom_influence(Xd, args.lam, args.nystrom_m, device)
        qidx = []
        for c in classes:
            idx = cls_idx[c]
            if len(idx) == 0:
                continue
            j = int(I[idx].argmax().item())
            qidx.append(int(idx[j].item()))
            if args.labels_per_class > 1:
                m2 = torch.ones(len(idx), dtype=torch.bool)
                m2[j] = False
                if int(m2.sum().item()) > 0:
                    j2 = int(I[idx[m2]].argmax().item())
                    qidx.append(int(idx[m2][j2].item()))
                if args.labels_per_class > 2:
                    m3 = m2.clone()
                    m3[j2] = False
                    if int(m3.sum().item()) > 0:
                        j3 = int(I[idx[m3]].argmax().item())
                        qidx.append(int(idx[m3][j3].item()))
        qidx = torch.tensor(qidx)
        qlbl = pl[qidx]
        r['refs']['n_labels'] = int(len(qidx))

        # frozen soft assignments on the pool
        logits = scores(W_clean, Xp)
        Q_frozen = torch.softmax(logits, dim=1)
        ppred = logits.argmax(dim=1)

        # confusion correction from the queried pairs
        C_hat = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        for j in range(len(qidx)):
            C_hat[ppred[qidx[j]], qlbl[j]] += 1.0
        row_s = C_hat.sum(dim=1, keepdim=True)
        C_hat = C_hat / (row_s + 1e-9)
        # rows without evidence -> identity (no correction for that class)
        C_hat[row_s.squeeze(1) == 0] = torch.eye(NUM_CLASSES)[
            (row_s.squeeze(1) == 0).nonzero().squeeze(1)]

        # shift estimates from the queried labels (code space, per queried class)
        shift_q = {}
        for c in classes:
            m = qlbl == c
            if int(m.sum().item()) == 0 or c not in mu_clean:
                continue
            shift_q[c] = Xp[qidx[m]].mean(dim=0) - mu_clean[c]
        d_global = torch.stack(list(shift_q.values())).mean(dim=0) if shift_q \
            else torch.zeros(10000)

        # ---- Section B: T-synthesis variants ----
        ts = r['t_synthesis']
        N_or = {c: len(cls_idx[c]) for c in classes}

        def tcos(T_hat):
            coss = []
            for c in classes:
                if c >= NUM_CLASSES or T_or[:, c].norm().item() < 1e-9:
                    continue
                coss.append(cos_sim(T_hat[:, c], T_or[:, c]))
            return float(np.mean(coss)) if coss else None

        variants = {}
        # 7A clean-mean: T_c = N_c * mu_clean_c (code space)
        T_a = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            if c in mu_clean:
                T_a[:, c] = N_or[c] * mu_clean[c]
        variants['7A_clean_mean'] = T_a
        # 7B shift: T_c = N_c * (mu_clean_c + d_global) (carry-over)
        T_b = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            if c in mu_clean:
                T_b[:, c] = N_or[c] * (mu_clean[c] + d_global)
        variants['7B_shift_carry'] = T_b
        # 7C shrink: mu = mu_clean + alpha * d_global
        T_cv = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            if c in mu_clean:
                T_cv[:, c] = N_or[c] * (mu_clean[c] + args.shrink_alpha * d_global)
        variants['7C_shrink'] = T_cv
        # 7B-oracle ceiling: true shift per class
        T_bo = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            if c in mu_clean and c in mu_or:
                T_bo[:, c] = N_or[c] * mu_or[c]
        variants['7B_oracle_shift_ceiling'] = T_bo
        # 7D soft-frozen: T = X^T Q_frozen
        variants['7D_soft_frozen'] = Xp.t() @ Q_frozen
        # 7E calibrated: T = X^T (Q_frozen C_hat)
        variants['7E_soft_calibrated'] = Xp.t() @ (Q_frozen @ C_hat)
        # 7F shift-Q: q_c propto q_frozen,c * exp(beta * cos128(x, mu_prior_c))
        # where mu_prior = clean 128-d mean + 128-d shift (estimated from the
        # queried labels, carried to all classes)
        zn128 = pool.float()
        zn128 = zn128 / (zn128.norm(dim=1, keepdim=True) + 1e-8)
        zc128 = fa[ci].float()
        zc128 = zc128 / (zc128.norm(dim=1, keepdim=True) + 1e-8)
        mu128_clean = {c: zc128[cls_idx_c[c]].mean(dim=0) for c in classes
                       if len(cls_idx_c[c]) > 0}
        shift128_q = {}
        for c in classes:
            m = qlbl == c
            if int(m.sum().item()) == 0 or c not in mu128_clean:
                continue
            shift128_q[c] = zn128[qidx[m]].mean(dim=0) - mu128_clean[c]
        d128_global = torch.stack(list(shift128_q.values())).mean(dim=0) \
            if shift128_q else torch.zeros(fa.shape[1])
        mu_prior = {}
        for c in classes:
            if c in mu128_clean:
                mu_prior[c] = mu128_clean[c] + d128_global
        Q_shift = Q_frozen.clone()
        if len(mu_prior) >= 2:
            prior_mat = torch.stack([mu_prior[c] for c in classes if c in mu_prior])
            prior_lbl = torch.tensor([c for c in classes if c in mu_prior])
            prior_mat = prior_mat / (prior_mat.norm(dim=1, keepdim=True) + 1e-8)
            sim = zn128 @ prior_mat.t()                       # n x K
            adj = torch.zeros(len(pool), NUM_CLASSES)
            for j, c in enumerate(prior_lbl.tolist()):
                adj[:, c] = args.shift_beta * sim[:, j]
            Q_shift = Q_frozen * torch.exp(adj)
            Q_shift = Q_shift / (Q_shift.sum(dim=1, keepdim=True) + 1e-9)
        variants['7F_shift_Q'] = Xp.t() @ Q_shift

        for vname, T_hat in variants.items():
            # fit W = (S + lI)^-1 T_hat directly (the sufficient statistic is
            # T, not Y): CG on the columns of the 10000 x C system
            b = T_hat.to(device)
            def A(v):
                return Xd.t() @ (Xd @ v)
            x = b.clone()
            r = b - A(x)
            p = r.clone()
            rs_old = (r * r).sum(dim=0)
            for _ in range(args.cg_iters):
                Ap = A(p)
                alpha_k = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
                x = x + alpha_k.unsqueeze(0) * p
                r = r - alpha_k.unsqueeze(0) * Ap
                rs_new = (r * r).sum(dim=0)
                beta = rs_new / (rs_old + 1e-30)
                p = r + beta.unsqueeze(0) * p
                rs_old = rs_new
            W = x.float()
            ts[vname] = {'t_cos': tcos(T_hat),
                         'w_cos': cos_sim(W, W_oracle),
                         'miou': mw(W)}

        # soft-mass count estimates (the mass correction quality)
        N_hat = Q_frozen.sum(dim=0)
        mass_err = {}
        for c in classes:
            mass_err[str(c)] = float(abs(N_hat[c].item() - N_or[c]) / N_or[c])
        ts['_mass_est'] = {'n_hat': {str(c): float(N_hat[c].item()) for c in classes},
                           'rel_err_mean': float(np.mean(list(mass_err.values()))),
                           'rel_err': mass_err}

        # ---- synthesis ----
        syn = r['synthesis']
        syn.append(f"COND {cond}: frozen {r['refs']['frozen']:.3f} / oracle "
                   f"{r['refs']['oracle']:.3f} | labels {len(qidx)} (per-class "
                   f"{args.labels_per_class})")
        sc128 = sc['128d']
        syn.append(f"  mean-complexity 128-d: " + " ".join(
            f"k{k}:{sc128[k]['cos_mean']:.3f}" for k in ['2', '8', '32', '64', '128', '256']
            if k in sc128))
        sc_c = sc['code']
        syn.append(f"  mean-complexity code: " + " ".join(
            f"k{k}:{sc_c[k]['cos_mean']:.3f}" for k in ['8', '32', '128', '256']
            if k in sc_c))
        for vname in ['7A_clean_mean', '7B_shift_carry', '7C_shrink',
                      '7B_oracle_shift_ceiling', '7D_soft_frozen',
                      '7E_soft_calibrated', '7F_shift_Q']:
            e = ts[vname]
            syn.append(f"  {vname}: t_cos {e['t_cos']:.3f} | w_cos {e['w_cos']:.3f} "
                       f"| miou {e['miou']:.3f}")
        syn.append(f"  soft-mass count est: mean rel err "
                   f"{ts['_mass_est']['rel_err_mean']:.3f}")

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("A. sample_complexity: how many random points per class estimate the")
    print("   class mean to cos ~0.9? In 128-d AND the code space. If ~32-64 in")
    print("   128-d but 100+ in code, the mean-estimation family is viable only")
    print("   in low-dim; if even 128-d needs 100+, favor the soft-mass family.")
    print("B. t_synthesis (evaluate in order): per-class cos(T_hat_c, T_oracle_c)")
    print("   is the sufficient-statistic accuracy BEFORE the inverse covariance")
    print("   obscures it; then cos(W_hat, W_oracle); then mIoU.")
    print("   7A clean-mean baseline | 7B shift carry-over | 7C shrinkage |")
    print("   7B-oracle ceiling (true shifts) | 7D soft-frozen (no labels) |")
    print("   7E soft+confusion-corrected (the star) | 7F shift-informed Q.")
    print("   If 7E/7F beat 7D on t_cos and w_cos with 34-68 labels, the mass")
    print("   problem is solved without labeling the mass: the unlabeled pool")
    print("   supplies the mass, the labels calibrate the assignment, and the")
    print("   ridge does the rest. _mass_est reports the count correction.")

if __name__ == "__main__":
    main()
