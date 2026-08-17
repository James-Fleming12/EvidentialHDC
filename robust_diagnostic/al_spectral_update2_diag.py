"""al_spectral_update2_diag.py: Iteration-10 -- normalized spectrum + the
beta-continuum residual (eval-only, no plots).

Iteration 9 showed the fractional mechanism works (robustness up at beta=0.25,
oracle-ceiling peak moves to beta=0.75) and the residual anchor works (9D
reaches parity with frozen, never worse; oracle direction reaches 0.92), but
no variant beat frozen under T_hat -- and the spectral reconstruction was
confounded by the UNNORMALIZED S (gains ~0.001 everywhere, clips never bound).

This iteration fixes the normalization EXACTLY and tests the combination:

 1. NORMALIZATION: scale S, T and lambda all by 1/N. Then
    (S/N + l/N I)^-1 (T/N) = (S + lI)^-1 T exactly -- the ridge solution is
    unchanged, but the eigenvalues are O(1), the gains 1/(sig_hat + l_hat) are
    interpretable, and the clip/drop thresholds bind. The normalized ridge
    must reproduce W_oracle (w_cos ~1.0) -- the validation of the fix.
 2. 9A-frac RETEST under the normalized spectrum (beta sweep): was the muted
    Iteration-9 magnitude a normalization artifact or real?
 3. 10-COMB: the beta-continuum residual
    W = W_frozen + eta (W_beta - W_frozen),
    beta in {0.25, 0.5, 0.75} x eta sweep. This gives the residual family a
    CORRECTED direction (the fractional W_beta, not the full ridge W_hat) and
    the fractional family a SAFETY ANCHOR (eta=0 reproduces frozen). The
    combination the Iteration-9 data implicates.
 4. 9B-clip and 9E-unstable RETEST under normalized gains (the clip binds now).
 5. LABEL-BUDGET sweep on the T_hat construction: random-k means per class,
    k in {8, 16, 32} -- the method's actual label cost (128-512 total) and its
    effect on the combined update.

Efficiency recorded: eigh time (once per condition), per-update time. The
method stays: a few labels/class + a spectral solve + matrix-free decode.

Verdict framing: if the combined update lifts hat-mIoU above frozen with
oracle retention ~0.9 at k=8-32 labels/class, it is the Pillar-3 mechanism. If
it does not, the likely NEXT STEP is a constrained residual/prototype update
on the frozen decoder (not "the oracle is required"): the method must remain
efficient and low-label-cost, or it is not viable.

Usage:
  uv run python robust_diagnostic/al_spectral_update2_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_spectral_update2_covshift_ep10.json
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
    parser.add_argument("--betas", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--combo_betas", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--gammas", type=str, default="0.5,1,2,4,8")
    parser.add_argument("--etas", type=str, default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--drop_ps", type=str, default="1,5,10,20,40")
    parser.add_argument("--mean_ks", type=str, default="8,16,32")
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_spectral_update2_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    betas = [float(x) for x in args.betas.split(',')]
    combo_betas = [float(x) for x in args.combo_betas.split(',')]
    gammas = [float(x) for x in args.gammas.split(',')]
    etas = [float(x) for x in args.etas.split(',')]
    drop_ps = [float(x) for x in args.drop_ps.split(',')]
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
        N = Xp.shape[0]

        W_clean = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam,
                                 args.cg_iters, args.nystrom_m, device)
        W_oracle = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam,
                                  args.cg_iters, args.nystrom_m, device)

        def mw(W):
            return compute_miou(decode(W, Xv), vl)

        r = {'refs': {}, 'spectrum': {}, 'updates': {}, 'budget': {},
             'synthesis': []}
        r['refs'] = {'frozen': mw(W_clean), 'oracle': mw(W_oracle)}

        # ---- the NORMALIZED spectrum: S_hat = S/N, l_hat = l/N ----
        t_eig = tic()
        S = (Xd.t() @ Xd).float() / N
        eig, U = torch.linalg.eigh(S)                 # ascending, O(1) now
        sig = eig + args.lam / N                      # l_hat
        t_eig = toc(t_eig)
        gain = 1.0 / sig
        sp = r['spectrum']
        sp['eig_top8'] = [float(v) for v in eig.flip(0)[:8]]
        sp['eig_bottom8'] = [float(v) for v in eig[:8]]
        gq = torch.quantile(gain, torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0],
                                               device=device))
        sp['gain_quantiles'] = {'q50': float(gq[0].item()), 'q90': float(gq[1].item()),
                                'q99': float(gq[2].item()), 'q999': float(gq[3].item()),
                                'max': float(gq[4].item())}
        sp['participation_rank'] = float((sig.sum() ** 2 / (sig ** 2).sum()).item())
        sp['eigh_time_s'] = t_eig

        # ---- the T's (normalized: T_hat/N, T_oracle/N) ----
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}
        cls_idx_c = {c: (la[ci] == c).nonzero().squeeze(1) for c in classes}
        T_or = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            T_or[:, c] = Xp[cls_idx[c]].sum(dim=0)
        m0 = pl == 0
        if int(m0.sum().item()) > 0:
            T_or[:, 0] = Xp[m0].sum(dim=0)
        T_or = T_or / N
        mu_clean = {c: Xc[cls_idx_c[c]].mean(dim=0) for c in classes
                    if len(cls_idx_c[c]) > 0}

        # imperfect T per mean-k budget: oracle counts x random-k means
        T_hats = {}
        for k in mean_ks:
            T_h = torch.zeros(10000, NUM_CLASSES)
            for c in classes:
                idx = cls_idx[c]
                if len(idx) >= max(50, k):
                    torch.manual_seed(2)
                    T_h[:, c] = len(cls_idx[c]) * Xp[
                        idx[torch.randperm(len(idx))[:k]]].mean(dim=0)
            T_hats[str(k)] = T_h / N

        # spectral helpers (normalized frame)
        Uc = U.to(device).float()
        sig_d = sig.to(device)
        UtT_or = Uc.t() @ T_or.to(device).float()
        UtT_hats = {k: Uc.t() @ T_h.to(device).float() for k, T_h in T_hats.items()}

        def apply_filter(UtT, gains):
            return (Uc @ (gains.unsqueeze(1) * UtT)).cpu().float()

        def tcos(T_hat):
            coss = []
            for c in classes:
                if T_or[:, c].norm().item() < 1e-9:
                    continue
                coss.append(cos_sim(T_hat[:, c], T_or[:, c]))
            return float(np.mean(coss)) if coss else None

        def eval_pair_hats(make_gains):
            out = {}
            for k in mean_ks:
                W = apply_filter(UtT_hats[str(k)], make_gains())
                out[k] = {'w_cos': cos_sim(W, W_oracle), 'miou': mw(W)}
            W_or_w = apply_filter(UtT_or, make_gains())
            out['oracle'] = {'w_cos': cos_sim(W_or_w, W_oracle),
                             'miou': mw(W_or_w)}
            return out

        up = r['updates']
        # validation: the normalized ridge must reproduce W_oracle
        up['ridge_norm'] = eval_pair_hats(lambda: 1.0 / sig_d)
        up['_ridge_oracle_w_cos'] = up['ridge_norm']['oracle']['w_cos']

        # 9A-frac retest under the normalized spectrum
        up['9A_fractional'] = {}
        for beta in betas:
            up['9A_fractional'][str(beta)] = eval_pair_hats(
                lambda b=beta: sig_d.pow(-b))

        # 10-COMB: W_frozen + eta (W_beta - W_frozen)
        up['10_combo'] = {}
        Wf = W_clean
        for beta in combo_betas:
            W_beta = apply_filter(UtT_hats[str(mean_ks[-1])], sig_d.pow(-beta))
            up['10_combo'][str(beta)] = {}
            for eta in etas:
                W = Wf + eta * (W_beta - Wf)
                # oracle-retention arm: residual toward the oracle W at beta=1
                up['10_combo'][str(beta)][str(eta)] = {
                    'hat': {'w_cos': cos_sim(W, W_oracle), 'miou': mw(W)}}
                W_or_r = Wf + eta * (W_oracle - Wf)
                up['10_combo'][str(beta)][str(eta)]['oracle'] = {
                    'w_cos': cos_sim(W_or_r, W_oracle), 'miou': mw(W_or_r)}

        # 9B-clip retest (normalized gains bind now)
        up['9B_clipped'] = {}
        for gamma in gammas:
            up['9B_clipped'][str(gamma)] = eval_pair_hats(
                lambda g=gamma: torch.minimum(1.0 / sig_d,
                                              torch.full_like(sig_d, g)))

        # 9E-unstable removal retest (drop bottom p% of normalized gains)
        up['9E_unstable'] = {}
        for p in drop_ps:
            n_keep = int(round((1 - p / 100.0) * len(sig)))
            gains = torch.zeros_like(sig_d)
            gains[:n_keep] = (1.0 / sig_d)[:n_keep]
            up['9E_unstable'][str(p)] = eval_pair_hats(lambda g=gains: g)

        # ---- budget summary: the best 10-COMB per mean-k ----
        bd = r['budget']
        for k in mean_ks:
            best = None
            for beta in combo_betas:
                for eta in etas:
                    e = up['10_combo'][str(beta)][str(eta)]['hat']
                    if best is None or e['miou'] > best['miou']:
                        best = {'beta': beta, 'eta': eta, **e}
            bd[str(k)] = {'n_labels_total': k * len(classes),
                          'best_miou': best['miou'],
                          'best_w_cos': best['w_cos'],
                          'best_beta': best['beta'], 'best_eta': best['eta']}

        # ---- synthesis ----
        syn = r['synthesis']
        syn.append(f"COND {cond}: frozen {r['refs']['frozen']:.3f} / oracle "
                   f"{r['refs']['oracle']:.3f} | eigh {t_eig:.1f}s")
        syn.append(f"  spectrum (normalized): gain q50 {sp['gain_quantiles']['q50']:.3f} "
                   f"q99 {sp['gain_quantiles']['q99']:.3f} max "
                   f"{sp['gain_quantiles']['max']:.2f} | part-rank "
                   f"{sp['participation_rank']:.0f}")
        syn.append(f"  ridge(norm) oracle w_cos {up['_ridge_oracle_w_cos']:.3f} "
                   f"(validation: ~1.0 = the fix is exact)")
        syn.append(f"  9A frac (k={mean_ks[-1]}): " + " ".join(
            f"b{b}:{up['9A_fractional'][b][str(mean_ks[-1])]['miou']:.3f}"
            f"(w {up['9A_fractional'][b][str(mean_ks[-1])]['w_cos']:.3f})"
            for b in betas))
        syn.append(f"  10-COMB (k={mean_ks[-1]}): " + " ".join(
            f"b{b}:{max((up['10_combo'][b][e]['hat']['miou'] for e in etas), default=0):.3f}"
            for b in combo_betas))
        for k in mean_ks:
            e = bd[str(k)]
            syn.append(f"  budget k={k} ({e['n_labels_total']} labels): best "
                       f"miou {e['best_miou']:.3f} (w {e['best_w_cos']:.3f}, "
                       f"beta {e['best_beta']}, eta {e['best_eta']}) vs frozen "
                       f"{r['refs']['frozen']:.3f}")
        syn.append(f"  9B clip (k={mean_ks[-1]}): " + " ".join(
            f"g{g}:{up['9B_clipped'][g][str(mean_ks[-1])]['miou']:.3f}" for g in gammas))
        syn.append(f"  9E drop (k={mean_ks[-1]}): " + " ".join(
            f"p{p}:{up['9E_unstable'][p][str(mean_ks[-1])]['miou']:.3f}" for p in drop_ps))

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("Validation: ridge(norm) oracle w_cos ~1.0 confirms the normalization")
    print("  (S/N, T/N, l/N) leaves the ridge solution EXACTLY unchanged.")
    print("9A frac: the Iteration-9 retest under interpretable gains -- was the")
    print("  muted magnitude a normalization artifact?")
    print("10-COMB: W_frozen + eta(W_beta - W_frozen) -- the residual family with")
    print("  the FRACTIONAL direction. eta=0 reproduces frozen (never worse);")
    print("  the question is whether the corrected direction lifts mIoU past")
    print("  frozen under the imperfect T_hat.")
    print("budget: the method's real label cost (k*#classes total) and the best")
    print("  combined update at each cost.")
    print("9B/9E: the clip and unstable-removal now bind on the normalized gains.")
    print("If the combined update beats frozen at k=8-32 labels/class, it is the")
    print("  Pillar-3 mechanism. If not, the likely NEXT STEP is a constrained")
    print("  residual/prototype update on the frozen decoder -- the method must")
    print("  stay efficient and low-label-cost to be viable.")

if __name__ == "__main__":
    main()
