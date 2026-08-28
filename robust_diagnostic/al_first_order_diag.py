"""al_first_order_diag.py: the decisive test -- is the missing-mass/covariance
problem an artifact of the RIDGE estimator, not a real requirement?

The measured facts:
  - Oracle U + 2-8 leverage labels close 34-76% of the gap (minimal-label).
  - The ridge's covariance (U^T X^T X U)^-1 amplifies label-error along
    low-variance directions (Iteration 6-8); 10-COMB beat frozen with a SMALL
    fractional step; partial whitening (beta=0.25) was MORE robust than full ridge.
  - No label-free U works; joint U,C refinement on few labels cannot recover U.
  - The residual R = W* - W0 is a decision-rule object, not a class-mean shift.

So: replace the Newton/ridge step with FIRST-ORDER / TRUST-REGION updates that
never estimate the pool covariance. The labels supply DIRECTION; the step size is
a controlled trust radius (and, in the TTA variant, a TTA-measured scale).

Methods tested per (b in budget, sweep over step size):
  oracle_ridge     reference: C = (U^T X^T X U + gI)^-1 G          [current best]
  oracle_first     C = eta * G        (G = U^T X^T (Y - X W0))      [no covariance]
  oracle_norm      C = rho * G/||G||  (trust-region, unit direction)
  full_grad        dW = eta * X^T (Y - P0), normalized              [no U at all]
  gradspan         U = orth(label gradients g_i), then first-order in span
                   (labels discover the adaptation subspace)
  boundary_pair    only update w_a - w_b for the queried confusion pairs
                   (the residual-as-boundary hypothesis, per-pair margin)
  tta_trust        oracle_norm with rho gated by a label-free instability
                   signal (conf_drop / mean_shift) -> zero-degradation

Read: if oracle_first/oracle_norm ~ oracle_ridge, the covariance was the
artifact and first-order updates are the escape. If gradspan ~ oracle-U, U
estimation is unnecessary. If tta_trust keeps the gain AND rejects healthy
conditions, it is the deployable mechanism.

Usage:
  uv run python robust_diagnostic/al_first_order_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_first_order_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                  labels=data["labels"], color_map=data["color_map"], learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"], sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_features(model, parser, device, num_frames=100):
    feats, lbls = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device); labels = batch[2].to(device).view(-1); mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol); z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(zf.cpu()); lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(lbls)


def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)


def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc); y[torch.arange(len(lbls)), lbls.long()] = 1; return y


def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    p = []
    for s in range(0, len(codes), chunk):
        p.append((codes[s:s + chunk].float() @ W).argmax(1))
    return torch.cat(p)


def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)


def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device); torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1], max(1, X.shape[0] - 1))
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P; Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device); That = XP.t() @ Yd
    x0 = P @ torch.linalg.solve(Shat, That)
    if X.shape[0] <= 8:
        return x0.float()
    x = x0; b = X.t() @ Yd
    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); d = (p * Ap).sum(0)
        if not torch.isfinite(d).all() or d.abs().max().item() < 1e-20:
            break
        a = rs / (d + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0); be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p; rs = rsn
        if not torch.isfinite(x).all():
            return x0.float()
    return x.float()


def lsq_residual(X_lab, Y_lab, W0, U, device, gamma=1e-6):
    Xd = X_lab.to(device).float(); Yd = Y_lab.to(device).float(); Ud = U.to(device)
    r = Ud.shape[1]; XU = Xd @ Ud
    A = XU.t() @ XU + gamma * torch.eye(r, device=device)
    b = XU.t() @ (Yd - Xd @ W0.to(device))
    return torch.linalg.solve(A, b).cpu()


def right_topk_svd(M, r):
    _, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    if not torch.isfinite(S).all():
        S = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
        Vh = torch.where(torch.isfinite(Vh), Vh, torch.zeros_like(Vh))
    return Vh[:r].t().float(), S.float()


def subspace_cos(U_hat, U_oracle, r):
    uh = U_hat[:, :r]; uo = U_oracle[:, :r]
    S = torch.linalg.svdvals((uh.t() @ uo).double())
    return float(S.mean().item())


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync(); return time.time()


def toc(t0):
    sync(); return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--pool_size", type=int, default=50000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=50000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--r_sweep", type=str, default="2,4")
    ap.add_argument("--budget_sweep", type=str, default="8,32")
    ap.add_argument("--step_sweep", type=str, default="0.05,0.2,0.8")
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    r_sweep = [int(x) for x in args.r_sweep.split(',')]
    budget_sweep = [int(x) for x in args.budget_sweep.split(',')]
    step_sweep = [float(x) for x in args.step_sweep.split(',')]
    rmax = max(r_sweep)

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    results = {'label': args.label, 'method': args.method_b, 'conds': {}}

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
        proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
        Xc = hdc_codes(fa[ci], proj, device).float()
        Xp = hdc_codes(pool, proj, device).float()
        Xv = hdc_codes(val, proj, device).float()
        del pool, val, f, l
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_oracle, _ = right_topk_svd(R.t(), rmax)   # R is d x C; transpose -> C x d rows

        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # label-free instability signal for the tta_trust gate
        # (conf_drop: frozen-probe mean softmax confidence on the pool vs a clean ref)
        pool_s = torch.softmax(Xp.float() @ W0.cpu(), dim=1)
        conf_pool = float(pool_s.max(dim=1).values.mean().item())
        clean_s = torch.softmax(Xc.float() @ W0.cpu(), dim=1)
        conf_clean = float(clean_s.max(dim=1).values.mean().item())
        conf_drop = max(conf_clean - conf_pool, 0.0)

        cond_res = {'refs': refs, 'gap': float(gap), 'conf_drop': conf_drop, 'budgets': {}}
        for b in budget_sweep:
            if b >= args.pool_size:
                continue
            # select b points by leverage-in-oracle-U (the known-good rule)
            lev = torch.norm(Xp.float() @ U_oracle[:, :2], p=2, dim=1)
            sel = torch.argsort(lev, descending=True)[:b].long()
            X_lab = Xp[sel]; Y_lab = onehot(pl[sel], NUM_CLASSES)
            P0_lab = torch.softmax(X_lab.float() @ W0.cpu(), dim=1)   # frozen soft targets

            b_res = {'methods': {}}
            for r in r_sweep:
                Ur = U_oracle[:, :r]
                # G = U^T X^T (Y - X W0) -- the low-rank label gradient
                resid = (Y_lab.float() - X_lab.float() @ W0.cpu())
                G = (X_lab.float() @ Ur).t() @ resid          # r x C
                Gn = G / (G.norm() + 1e-8)

                for s in step_sweep:
                    # oracle_ridge (reference)
                    C_r = lsq_residual(X_lab, Y_lab, W0, Ur, device)
                    W_r = W0.detach().cpu() + (Ur @ C_r)
                    d_r = mw(W_r, Xv, vl) - refs['frozen']
                    # oracle_first: C = s * G
                    W_f = W0.detach().cpu() + (Ur @ (s * G))
                    d_f = mw(W_f, Xv, vl) - refs['frozen']
                    # oracle_norm: C = s * G/||G||
                    W_n = W0.detach().cpu() + (Ur @ (s * Gn))
                    d_n = mw(W_n, Xv, vl) - refs['frozen']
                    b_res['methods'][f'r{r}_s{s}'] = {
                        'oracle_ridge_delta': float(d_r),
                        'oracle_ridge_gc': float(d_r / gap) if gap > 1e-9 else None,
                        'oracle_first_delta': float(d_f),
                        'oracle_first_gc': float(d_f / gap) if gap > 1e-9 else None,
                        'oracle_norm_delta': float(d_n),
                        'oracle_norm_gc': float(d_n / gap) if gap > 1e-9 else None,
                        '||G||': float(G.norm()),
                    }

                # gradspan: U = orth of the per-point label gradients
                grads = X_lab.float().t() @ (Y_lab.float() - P0_lab)   # d x C
                U_gs, _ = right_topk_svd(grads.t(), r)
                G_gs = (X_lab.float() @ U_gs).t() @ resid
                Gn_gs = G_gs / (G_gs.norm() + 1e-8)
                for s in step_sweep:
                    W_gs = W0.detach().cpu() + (U_gs @ (s * Gn_gs))
                    d_gs = mw(W_gs, Xv, vl) - refs['frozen']
                    b_res['methods'][f'gradspan_r{r}_s{s}'] = {
                        'delta': float(d_gs),
                        'gap_closed': float(d_gs / gap) if gap > 1e-9 else None,
                        'align_U_oracle': subspace_cos(U_gs, U_oracle, r),
                    }

                # full_grad: normalized full-space gradient (no U)
                Fg = (X_lab.float().t() @ (Y_lab.float() - P0_lab))
                Fgn = Fg / (Fg.norm() + 1e-8)
                for s in step_sweep:
                    W_fg = W0.detach().cpu() + (s * Fgn)
                    d_fg = mw(W_fg, Xv, vl) - refs['frozen']
                    b_res['methods'][f'full_grad_r{r}_s{s}'] = {
                        'delta': float(d_fg),
                        'gap_closed': float(d_fg / gap) if gap > 1e-9 else None,
                    }

            # boundary_pair: only move the confusion-pair boundaries w_a - w_b
            # for the queried points (true label a, frozen pred b), margin-driven.
            sm0 = torch.softmax(X_lab.float() @ W0.cpu(), dim=1)
            pred0 = sm0.argmax(dim=1)
            M_bp = torch.zeros_like(W0.detach().cpu())
            for i in range(len(sel)):
                a = int(pl[sel[i]].item()); b_p = int(pred0[i].item())
                if a == b_p or a == 0 or b_p == 0:
                    continue
                margin_err = sm0[i, b_p].item() - sm0[i, a].item()   # >0 if wrong pred more likely
                x = X_lab[i].float()
                M_bp[:, a] += margin_err * x
                M_bp[:, b_p] -= margin_err * x
            Mn = M_bp / (M_bp.norm() + 1e-8) if M_bp.norm() > 0 else M_bp
            for s in step_sweep:
                W_bp = W0.detach().cpu() + (s * Mn)
                d_bp = mw(W_bp, Xv, vl) - refs['frozen']
                b_res['methods'][f'boundary_pair_s{s}'] = {
                    'delta': float(d_bp),
                    'gap_closed': float(d_bp / gap) if gap > 1e-9 else None,
                    'n_pairs': int(((pred0 != pl[sel]) & (pred0 != 0) & (pl[sel] != 0)).sum().item()),
                }

            # tta_trust: oracle_norm with rho gated by conf_drop (0 -> no move)
            # rho_eff = step * g(conf_drop), g monotone; report raw (ungated) for contrast.
            for r in r_sweep:
                Ur = U_oracle[:, :r]
                resid = (Y_lab.float() - X_lab.float() @ W0.cpu())
                G = (X_lab.float() @ Ur).t() @ resid
                Gn = G / (G.norm() + 1e-8)
                for s in step_sweep:
                    g = min(1.0, conf_drop * 10.0)   # simple monotone gate
                    rho_eff = s * g
                    W_t = W0.detach().cpu() + (Ur @ (rho_eff * Gn))
                    d_t = mw(W_t, Xv, vl) - refs['frozen']
                    b_res['methods'][f'tta_trust_r{r}_s{s}'] = {
                        'delta': float(d_t),
                        'gap_closed': float(d_t / gap) if gap > 1e-9 else None,
                        'rho_eff': float(rho_eff),
                        'gated': float(g),
                    }
            cond_res['budgets'][str(b)] = b_res

        results['conds'][cond] = cond_res
        del Xc, Xp, Xv, W0, Ws, R, U_oracle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} conf_drop {conf_drop:.3f} ===")
        for b, br in cond_res['budgets'].items():
            for mk, mv in br['methods'].items():
                if 'oracle_ridge_delta' in mv:
                    print(f"  b={b:>2} {mk:18s} ridge {mv['oracle_ridge_gc']:+.2f} | first {mv['oracle_first_gc']:+.2f} | norm {mv['oracle_norm_gc']:+.2f}")
                elif 'gradspan' in mk:
                    print(f"  b={b:>2} {mk:18s} delta {mv['delta']:+.3f} gc {mv['gap_closed']:+.2f} alignU {mv['align_U_oracle']:.2f}")
                elif 'full_grad' in mk:
                    print(f"  b={b:>2} {mk:18s} delta {mv['delta']:+.3f} gc {mv['gap_closed']:+.2f}")
                elif 'boundary' in mk:
                    print(f"  b={b:>2} {mk:18s} delta {mv['delta']:+.3f} gc {mv['gap_closed']:+.2f} pairs {mv['n_pairs']}")
                elif 'tta_trust' in mk:
                    print(f"  b={b:>2} {mk:18s} delta {mv['delta']:+.3f} gc {mv['gap_closed']:+.2f} rho {mv['rho_eff']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("If oracle_first/oracle_norm ~ oracle_ridge: the covariance was the artifact.")
    print("If gradspan ~ oracle-U (align high, gc high): U estimation is unnecessary.")
    print("If tta_trust keeps the corrupted gain AND rejects healthy (rho~0): deployable.")
    print("boundary_pair: does updating only the confusion boundaries capture R?")


if __name__ == "__main__":
    main()
