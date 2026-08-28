"""al_trust_iter0_diag.py: Iteration 0 -- validate the three assumptions the
trust-region AL method (active_iterations_3.md) depends on, BEFORE building the
full pipeline.

The method (from the first-order/trust-region diagnostic):
  W1 = W0 + rho * U * G/||G||,   G = U^T X_lab^T (Y - X_lab W0),
  rho from a label-free TTA gauge (healthy -> 0, corrupted -> large),
  U = oracle (eval) or tangent-b8 (deployment, align 0.3-0.5).

Assumptions to validate:

A1. COARSE-U ROBUSTNESS: does the normalized trust-region step work with the
    tangent-b8 U, not just oracle U? Compare the gc-vs-rho curve for U in
    {oracle, tangent_b8, random}. If tangent ~ oracle >> random, the 0.3-0.5
    alignment is enough and U is deployable. If tangent ~ random, the method
    blocks on U and the next iteration targets U refinement.

A2. GATE VALIDITY: does a label-free gauge predict where a positive rho helps?
    For each condition compute the gauges (conf_drop, mean_shift_cos,
    r4_r1_disagree) AND the rho* that maximizes gc (oracle-U). Rank-correlate
    gauge vs (gc at large rho) across conditions: a valid gate has high gauge on
    conditions where rho helps and low gauge where it does not.

A3. ACCEPT/REJECT: is there a label-free validation score that separates good
    updates (positive gc) from bad ones (negative gc)? For each (U, rho) compute
    the candidate W1 and label-free score changes (d_conf_drop, d_disagree,
    d_margin). Correlate score change vs TRUE gc. If a monotone score separates
    sign(gc), the reject rule is viable.

Per condition (both extractors): the gc-vs-rho curves for all three U bases, the
gauges, and the accept/reject score-vs-gc correlations.

Usage:
  uv run python robust_diagnostic/al_trust_iter0_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_trust_iter0_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from scipy.stats import spearmanr
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


def tangent_U(Xp, pl, W0, sel, n_windows, lam, device):
    wins = torch.chunk(torch.randperm(len(sel)), n_windows)
    dW_stack = []
    for wi in wins:
        si = sel[wi]
        W_t = ridge_fit_soft(Xp[si], onehot(pl[si], NUM_CLASSES), lam, 8, 1000, device)
        dW_stack.append((W_t - W0).detach().cpu().t())
    D = torch.cat(dW_stack, dim=0)
    return D


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
    ap.add_argument("--r", type=int, default=2, help="single rank for the iter-0 validation")
    ap.add_argument("--b", type=int, default=8, help="label budget for direction + tangent-U")
    ap.add_argument("--n_windows", type=int, default=4)
    ap.add_argument("--rho_sweep", type=str, default="0.01,0.05,0.1,0.2,0.4,0.8")
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
    rho_sweep = [float(x) for x in args.rho_sweep.split(',')]
    r = args.r; b = args.b

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    results = {'label': args.label, 'method': args.method_b, 'r': r, 'b': b, 'conds': {}}

    # A2 gate correlation (across conditions)
    gate_rows = []

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
        U_oracle, _ = right_topk_svd(R.t(), r)

        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # ---- gauges (label-free) ----
        # mean_shift_cos: clean vs corrupted global feature mean cosine
        mean_clean = Xc.mean(dim=0); mean_pool = Xp.mean(dim=0)
        mean_shift_cos = float(F.cosine_similarity(mean_clean.unsqueeze(0), mean_pool.unsqueeze(0), dim=1).item())
        # conf_drop
        smc = torch.softmax(Xc.float() @ W0.cpu(), dim=1).max(dim=1).values.mean().item()
        smp = torch.softmax(Xp.float() @ W0.cpu(), dim=1).max(dim=1).values.mean().item()
        conf_drop = max(smc - smp, 0.0)
        # r4_r1_disagree: R4 (probe) vs R1 (prototype) disagreement on the pool
        protos = torch.zeros(NUM_CLASSES, Xp.shape[1])
        for c in range(1, NUM_CLASSES):
            m = (pl == c)
            if int(m.sum().item()) > 100:
                protos[c] = Xp[m].mean(dim=0)
        r4 = decode(W0, Xp)
        pn = F.normalize(Xp.float(), p=2, dim=1) @ F.normalize(protos.float(), p=2, dim=1).t()
        r1 = pn.argmax(dim=1)
        r4_r1_disagree = float((r4 != r1).float().mean().item())

        # ---- U bases ----
        # oracle, tangent-b8, random
        sel = torch.argsort(torch.norm(Xp.float() @ U_oracle, p=2, dim=1), descending=True)[:b].long()
        D_tan = tangent_U(Xp, pl, W0, sel, args.n_windows, args.lam, device)
        U_tan, _ = right_topk_svd(D_tan, r)
        torch.manual_seed(0)
        U_rand, _ = right_topk_svd(torch.randn(NUM_CLASSES, Xp.shape[1]), r)

        X_lab = Xp[sel]; Y_lab = onehot(pl[sel], NUM_CLASSES)
        resid = (Y_lab.float() - X_lab.float() @ W0.cpu())

        bases = {'oracle': U_oracle, 'tangent': U_tan, 'random': U_rand}
        curves = {}
        for uname, Ur in bases.items():
            align = subspace_cos(Ur, U_oracle, r)
            G = (X_lab.float() @ Ur).t() @ resid
            Gn = G / (G.norm() + 1e-8)
            gcs = {}
            for rho in rho_sweep:
                W1 = W0.detach().cpu() + (Ur @ (rho * Gn))
                d = mw(W1, Xv, vl) - refs['frozen']
                gcs[str(rho)] = {'delta': float(d),
                                 'gap_closed': float(d / gap) if gap > 1e-9 else None}
            curves[uname] = {'align_U_oracle': align, 'gc_vs_rho': gcs}
            # A3: accept/reject scores vs TRUE gc at each rho
            # score candidates: d_conf, d_disagree on the pool after W1
            for rho in rho_sweep:
                W1 = W0.detach().cpu() + (Ur @ (rho * Gn))
                sm1 = torch.softmax(Xp.float() @ W1, dim=1)
                conf1 = float(sm1.max(dim=1).values.mean().item())
                r4_1 = decode(W1, Xp)
                dis1 = float((r4_1 != r1).float().mean().item())
                curves[uname]['gc_vs_rho'][str(rho)]['d_conf'] = conf1 - smp
                curves[uname]['gc_vs_rho'][str(rho)]['d_disagree'] = dis1 - r4_r1_disagree
                curves[uname]['gc_vs_rho'][str(rho)]['score_comb'] = (conf1 - smp) - (dis1 - r4_r1_disagree)

        # best oracle gc (for A2 gate correlation)
        oracle_best_gc = max(v['gap_closed'] or -9 for v in curves['oracle']['gc_vs_rho'].values())
        gate_rows.append({'cond': cond, 'conf_drop': conf_drop, 'mean_shift_cos': mean_shift_cos,
                          'r4_r1_disagree': r4_r1_disagree, 'oracle_best_gc': oracle_best_gc,
                          'gap': float(gap), 'frozen': refs['frozen']})

        results['conds'][cond] = {
            'refs': refs, 'gap': float(gap),
            'gauges': {'conf_drop': conf_drop, 'mean_shift_cos': mean_shift_cos,
                       'r4_r1_disagree': r4_r1_disagree},
            'curves': curves,
        }
        del Xc, Xp, Xv, W0, Ws, R, U_oracle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    gauges: conf_drop {conf_drop:.3f} | mean_shift_cos {mean_shift_cos:.3f} | r4_r1_disagree {r4_r1_disagree:.3f}")
        for uname, cv in curves.items():
            gc = " ".join(f"{k}:{v['gap_closed']:+.2f}" for k, v in cv['gc_vs_rho'].items())
            print(f"    {uname:8s} alignU {cv['align_U_oracle']:.2f} | gc-vs-rho {gc}")
        print(f"    accept/reject: d_conf / d_disagree / comb vs gc (oracle U, rho 0.2):")
        for uname in ['oracle', 'tangent']:
            v = curves[uname]['gc_vs_rho']['0.2']
            print(f"      {uname:8s} gc {v['gap_closed']:+.2f} d_conf {v['d_conf']:+.4f} d_disagree {v['d_disagree']:+.4f} comb {v['score_comb']:+.4f}")

    # A2: gate rank-correlation across conditions
    gcs = [g['oracle_best_gc'] for g in gate_rows]
    for gname in ['conf_drop', 'mean_shift_cos', 'r4_r1_disagree']:
        vals = [g[gname] for g in gate_rows]
        try:
            rho_s, p_s = spearmanr(vals, gcs)
        except Exception:
            rho_s, p_s = None, None
        results[f'A2_gate_corr_{gname}'] = {'spearman': rho_s, 'p': p_s,
                                            'vals': vals, 'oracle_best_gc': gcs}
        print(f"\nA2 gate '{gname}' vs oracle_best_gc across {len(conds)} conds: spearman {rho_s}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A1 (coarse-U): does tangent gc-vs-rho track oracle, far above random?")
    print("  tangent ~ oracle >> random -> U deployable. tangent ~ random -> U is the blocker.")
    print("A2 (gate): is a gauge rank-correlated with oracle_best_gc across conditions?")
    print("  a valid gate has high gauge where rho helps and low where it does not.")
    print("A3 (accept/reject): does d_conf / d_disagree / comb separate sign(gc)?")
    print("  if the score is positive exactly when gc is positive, the reject rule is viable.")


if __name__ == "__main__":
    main()
