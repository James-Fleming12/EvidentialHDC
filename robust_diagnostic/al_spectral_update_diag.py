"""al_spectral_update_diag.py: Iteration-9 -- sensitivity-bounded probe updates
(eval-only, no plots).

Iteration 8's smoking gun: a T with t_cos 0.76-0.86 (objectively good) maps to
w_cos ~0.05-0.15, because the inverse covariance (S + lI)^-1 amplifies small
T errors along the low-variance directions (the ridge-relevant error was 4-6x).
This iteration does NOT try to estimate T better. It fixes the SAME imperfect
T_hat and changes only the DECODER PARAMETERIZATION, testing the 2x2 grid:

  update             T_oracle (ceiling)   T_hat (imperfect)
  ridge (baseline)   oracle (1.0)          current failure (~0.05)
  9A fractional      ?                     ?
  9B clipped ridge   ?                     ?
  9C frozen residual ?                     ?
  9D normalized res  ?                     ?
  9E unstable-removal ?                    ?

The verdict is 2-DIMENSIONAL: robustness gain (w_cos under T_hat) AND ceiling
retention (w_cos under T_oracle). A success is imperfect 0.70+ / oracle 0.90+;
a failure is imperfect 0.70 / oracle 0.40 (regularized away the signal). The
2x2 grid tested (T_oracle = ceiling, T_hat = imperfect): ridge (baseline),
9A fractional, 9B clipped, 9C frozen residual, 9D normalized residual,
9E unstable-subspace removal.

The spectrum is materialized once per condition (S = X^T X, 10k x 10k eigh on
GPU -- a diagnostic cost, not the deployed update), which also reports the
gain distribution g_j = 1/(sigma_j + l): the quantiles of the amplification
are the measured reason for the sensitivity.

 9A fractional ridge : W_beta = (S + lI)^-beta T, beta in {0,.25,.5,.75,1}
                       (beta=0 prototype-like, beta=1 ordinary ridge)
 9B clipped ridge    : g_j = min(1/(sigma_j + l), gamma), gamma sweep
 9C frozen residual  : W = W_frozen + eta (W_hat - W_frozen), eta sweep
                       (safety: eta=0 reproduces frozen exactly)
 9D normalized resid : same, but DeltaW normalized to ||W_frozen||_F
 9E unstable removal : drop the bottom p% eigen-directions (small sigma),
                       keep the stable subspace

All variants run with T_oracle (ceiling retention) and T_hat (robustness).
T_hat = oracle-count x random-32 class means (the best Iteration-8 estimator,
t_cos 0.76-0.86). The clean-mean T is kept as a second imperfect T for
robustness.

Usage:
  uv run python robust_diagnostic/al_spectral_update_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_spectral_update_covshift_ep10.json
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
    parser.add_argument("--gammas", type=str, default="4,8,16,32,64")
    parser.add_argument("--etas", type=str, default="0.05,0.1,0.2,0.4,0.8,1.0")
    parser.add_argument("--drop_ps", type=str, default="1,5,10,20,40",
                        help="9E: drop the bottom p% eigen-directions")
    parser.add_argument("--mean_k", type=int, default=32,
                        help="random points/class for the imperfect T_hat means")
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_spectral_update_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    betas = [float(x) for x in args.betas.split(',')]
    gammas = [float(x) for x in args.gammas.split(',')]
    etas = [float(x) for x in args.etas.split(',')]
    drop_ps = [float(x) for x in args.drop_ps.split(',')]

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

        r = {'refs': {}, 'spectrum': {}, 'updates': {}, 'synthesis': []}

        # frozen and oracle probes (the ordinary ridge, matrix-free)
        torch.manual_seed(SKETCH_SEED)
        Pm = (torch.rand(Xc.shape[1], args.nystrom_m, device=device) > 0.5).float() * 2 - 1
        Xc_d = Xc.to(device)
        XPc = Xc_d @ Pm
        Yc = onehot(la[ci], NUM_CLASSES).to(device)
        Shat_c = XPc.t() @ XPc + args.lam * torch.eye(args.nystrom_m, device=device)
        That_c = XPc.t() @ Yc
        W_clean = (Pm @ torch.linalg.solve(Shat_c, That_c)).cpu().float()
        Xd = Xp.to(device)
        P = (torch.rand(Xp.shape[1], args.nystrom_m, device=device) > 0.5).float() * 2 - 1
        XP = Xd @ P
        Y_or = onehot(pl, NUM_CLASSES).to(device)
        Shat = XP.t() @ XP + args.lam * torch.eye(args.nystrom_m, device=device)
        That = XP.t() @ Y_or
        x = (P @ torch.linalg.solve(Shat, That)).float()
        b = Xd.t() @ Y_or
        def A(v):
            return Xd.t() @ (Xd @ v)
        res = b - A(x)
        p = res.clone()
        rs_old = (res * res).sum(dim=0)
        for _ in range(args.cg_iters):
            Ap = A(p)
            ak = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
            x = x + ak.unsqueeze(0) * p
            res = res - ak.unsqueeze(0) * Ap
            rsn = (res * res).sum(dim=0)
            be = rsn / (rs_old + 1e-30)
            p = res + be.unsqueeze(0) * p
            rs_old = rsn
        W_oracle = x.cpu().float()

        def mw(W):
            return compute_miou(decode(W, Xv), vl)
        r['refs'] = {'frozen': mw(W_clean), 'oracle': mw(W_oracle)}

        # ---- the spectrum: S = X^T X, full eigh (diagnostic cost) ----
        S = (Xd.t() @ Xd).float()
        eig, U = torch.linalg.eigh(S)                  # ascending, U columns match
        sig = eig + args.lam
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
        sp['n_eig_gt_lambda'] = int((eig > args.lam).sum().item())
        sp['n_eig_gt_10lambda'] = int((eig > 10 * args.lam).sum().item())

        # ---- the T's ----
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}
        cls_idx_c = {c: (la[ci] == c).nonzero().squeeze(1) for c in classes}
        T_or = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            T_or[:, c] = Xp[cls_idx[c]].sum(dim=0)
        m0 = pl == 0
        if int(m0.sum().item()) > 0:
            T_or[:, 0] = Xp[m0].sum(dim=0)
        # imperfect T: oracle counts x random-32 class means (the best Iteration-8
        # estimator), plus the clean-mean T for robustness
        mu_hat = {}
        for c in classes:
            idx = cls_idx[c]
            if len(idx) >= max(50, args.mean_k):
                torch.manual_seed(2)
                mu_hat[c] = Xp[idx[torch.randperm(len(idx))[:args.mean_k]]].mean(dim=0)
        T_hat = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            if c in mu_hat:
                T_hat[:, c] = len(cls_idx[c]) * mu_hat[c]
        mu_clean = {c: Xc[cls_idx_c[c]].mean(dim=0) for c in classes
                    if len(cls_idx_c[c]) > 0}
        T_clean = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            if c in mu_clean:
                T_clean[:, c] = len(cls_idx[c]) * mu_clean[c]
        r['refs']['t_cos_hat'] = float(np.mean([
            cos_sim(T_hat[:, c], T_or[:, c]) for c in classes]))
        r['refs']['t_cos_clean'] = float(np.mean([
            cos_sim(T_clean[:, c], T_or[:, c]) for c in classes]))

        # helper: apply a spectral filter to T, evaluate on both T's
        Uc = U.to(device).float()
        UtT_or = Uc.t() @ T_or.to(device).float()
        UtT_hat = Uc.t() @ T_hat.to(device).float()
        UtT_clean = Uc.t() @ T_clean.to(device).float()
        sig_d = sig.to(device)

        def apply_filter(UtT, gains):
            return (Uc @ (gains.unsqueeze(1) * UtT)).cpu().float()

        def eval_pair(make_gains):
            out = {}
            for tname, UtT in [('oracle', UtT_or), ('hat', UtT_hat),
                               ('clean', UtT_clean)]:
                W = apply_filter(UtT, make_gains())
                out[tname] = {'w_cos': cos_sim(W, W_oracle), 'miou': mw(W)}
            return out

        up = r['updates']
        # baseline ridge: g = 1/sig
        up['ridge'] = eval_pair(lambda: 1.0 / sig_d)
        # 9A fractional: g = sig^-beta
        up['9A_fractional'] = {}
        for beta in betas:
            up['9A_fractional'][str(beta)] = eval_pair(
                lambda b=beta: sig_d.pow(-b))
        # 9B clipped: g = min(1/sig, gamma)
        up['9B_clipped'] = {}
        for gamma in gammas:
            up['9B_clipped'][str(gamma)] = eval_pair(
                lambda g=gamma: torch.minimum(1.0 / sig_d, torch.full_like(sig_d, g)))
        # 9C frozen residual: W = W_frozen + eta (W_hat - W_frozen)
        # W_hat = the ridge solution on T_hat (via the spectral filter)
        W_hat = apply_filter(UtT_hat, 1.0 / sig_d)
        up['9C_residual'] = {}
        for eta in etas:
            W = W_clean + eta * (W_hat - W_clean)
            up['9C_residual'][str(eta)] = {'oracle': None, 'hat': None, 'clean': None}
            up['9C_residual'][str(eta)]['hat'] = {'w_cos': cos_sim(W, W_oracle),
                                                  'miou': mw(W)}
            # ceiling: residual toward the ORACLE ridge solution
            W_oc = W_clean + eta * (W_oracle - W_clean)
            up['9C_residual'][str(eta)]['oracle'] = {'w_cos': cos_sim(W_oc, W_oracle),
                                                     'miou': mw(W_oc)}
        # 9D normalized residual: DeltaW normalized to ||W_frozen||_F
        up['9D_normalized'] = {}
        dW = W_hat - W_clean
        n_f = W_clean.norm().item()
        dW_n = dW / (dW.norm().item() + 1e-9) * n_f
        for eta in etas:
            W = W_clean + eta * dW_n
            up['9D_normalized'][str(eta)] = {'hat': {'w_cos': cos_sim(W, W_oracle),
                                                     'miou': mw(W)}}
            dWo = W_oracle - W_clean
            dWo_n = dWo / (dWo.norm().item() + 1e-9) * n_f
            W_oc = W_clean + eta * dWo_n
            up['9D_normalized'][str(eta)]['oracle'] = {'w_cos': cos_sim(W_oc, W_oracle),
                                                       'miou': mw(W_oc)}
        # 9E unstable-subspace removal: keep the top (100-p)% eigen-directions
        up['9E_unstable_removal'] = {}
        for p in drop_ps:
            n_keep = int(round((1 - p / 100.0) * len(sig)))
            gains = torch.zeros_like(sig_d)
            gains[:n_keep] = (1.0 / sig_d)[:n_keep]
            up['9E_unstable_removal'][str(p)] = eval_pair(lambda g=gains: g)

        # ---- synthesis ----
        syn = r['synthesis']
        syn.append(f"COND {cond}: frozen {r['refs']['frozen']:.3f} / oracle "
                   f"{r['refs']['oracle']:.3f} | t_cos_hat {r['refs']['t_cos_hat']:.3f} "
                   f"| t_cos_clean {r['refs']['t_cos_clean']:.3f}")
        syn.append(f"  spectrum: top eig {[round(v,0) for v in sp['eig_top8'][:3]]} "
                   f"| gain q50 {sp['gain_quantiles']['q50']:.1f} q99 "
                   f"{sp['gain_quantiles']['q99']:.1f} max {sp['gain_quantiles']['max']:.1f} "
                   f"| part-rank {sp['participation_rank']:.0f} | n_eig>l "
                   f"{sp['n_eig_gt_lambda']} n_eig>10l {sp['n_eig_gt_10lambda']}")
        syn.append(f"  ridge: hat w {up['ridge']['hat']['w_cos']:.3f} (m "
                   f"{up['ridge']['hat']['miou']:.3f}) | oracle w "
                   f"{up['ridge']['oracle']['w_cos']:.3f}")
        for b in betas:
            e = up['9A_fractional'][str(b)]
            syn.append(f"  9A beta={b}: hat w {e['hat']['w_cos']:.3f} (m "
                       f"{e['hat']['miou']:.3f}) | oracle w {e['oracle']['w_cos']:.3f}")
        for g in gammas:
            e = up['9B_clipped'][str(g)]
            syn.append(f"  9B gamma={g}: hat w {e['hat']['w_cos']:.3f} (m "
                       f"{e['hat']['miou']:.3f}) | oracle w {e['oracle']['w_cos']:.3f}")
        best_res = max(etas, key=lambda e: up['9C_residual'][str(e)]['hat']['w_cos'])
        e = up['9C_residual'][str(best_res)]
        syn.append(f"  9C residual eta={best_res}: hat w {e['hat']['w_cos']:.3f} "
                   f"(m {e['hat']['miou']:.3f}) | oracle-ret w "
                   f"{e['oracle']['w_cos']:.3f}")
        best_n = max(etas, key=lambda e: up['9D_normalized'][str(e)]['hat']['w_cos'])
        e = up['9D_normalized'][str(best_n)]
        syn.append(f"  9D normalized eta={best_n}: hat w {e['hat']['w_cos']:.3f} "
                   f"(m {e['hat']['miou']:.3f}) | oracle-ret w "
                   f"{e['oracle']['w_cos']:.3f}")
        best_e = max(drop_ps, key=lambda p: up['9E_unstable_removal'][str(p)]['hat']['w_cos'])
        e = up['9E_unstable_removal'][str(best_e)]
        syn.append(f"  9E drop {best_e}%: hat w {e['hat']['w_cos']:.3f} (m "
                   f"{e['hat']['miou']:.3f}) | oracle w {e['oracle']['w_cos']:.3f}")

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("The 2x2 verdict per update family: w_cos under T_hat (robustness) and")
    print("w_cos under T_oracle (ceiling retention).")
    print("  ridge: hat ~0.05 / oracle 1.00 (the baseline failure).")
    print("  9A fractional beta: beta<1 reduces the amplification; the question")
    print("     is whether hat improves WITHOUT oracle collapsing. beta=0 is the")
    print("     prototype-like endpoint, beta=1 the ridge.")
    print("  9B clipped gamma: the max tolerable gain -- the bias/variance knob.")
    print("  9C residual eta: W_frozen + eta(W_hat - W_frozen); eta=0 exactly")
    print("     reproduces frozen, so the curve starts at the frozen mIoU.")
    print("  9D normalized: same, but only the DIRECTION of the correction moves")
    print("     (magnitude set to ||W_frozen||_F).")
    print("  9E unstable removal: drop the bottom p% eigen-directions (small")
    print("     sigma = the amplified ones); keep the stable subspace.")
    print("Success = hat ~0.7+ AND oracle ~0.9+ (sensitivity down, ceiling kept).")
    print("Failure = hat up but oracle down (regularized away the signal).")
    print("spectrum: gain_quantiles show the amplification distribution -- the")
    print("  measured reason for the sensitivity; part-rank and n_eig thresholds")
    print("  show how many directions actually matter.")
    print("If ALL families fail the 2x2, the inverse covariance is the fundamental")
    print("  ceiling: the oracle advantage lives in low-variance directions that")
    print("  the label budget cannot estimate, and the path is a constrained")
    print("  residual/prototype update, not oracle-probe estimation.")

if __name__ == "__main__":
    main()
