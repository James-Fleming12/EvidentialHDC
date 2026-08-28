"""al_trust_refine_diag.py: Iteration 0b -- sweep the fixes for the coarse-basis
sensitivity and the U-refinement variants, with explicit efficiency costs.

Iteration 0 (al_trust_iter0) showed the tangent-b8 U (align 0.3-0.4) FAILS in the
trust-region step: gc ~0 on fog/crosstalk, negative on healthy -- at the random-U
level. The oracle-U trust-region is large and monotone in rho, so the step form is
sound; the blocker is the coarse basis. This sweeps two families of fixes:

STEP-SIDE (make the step robust to a coarse U):
  oracle        reference: W = W0 + rho * U_or * G/||G|| (upper bound)
  tangent       baseline (iter0): W = W0 + rho * U_tan * G/||G||  [failed]
  A_grad        no U: direction = FULL label gradient, trust-region rho
                (the labels' own gradient; cheapest)
  A_fix         tangent-U span, direction = label gradient projected onto it
  A_hybrid      tangent-projected + gradient residual combined

U-REFINEMENT (improve U itself):
  U_avg         average M independent tangent draws: concatenate their
                provisional dW stacks, top-r SVD of the big stack
  U_windows     more provisional windows (16 vs 4) in the tangent construction
  U_sharpen     iterative: use the current coarse U's leverage to select the
                provisional-fit points, re-PCA, repeat T times

Efficiency units (all reported explicitly per method):
  R      = one provisional ridge fit on <=8 points (sketch m=min(n-1,1000), CG<=8)
  SVD    = one torch.linalg.svd on a (rows x 10000) matrix -- the dominant
           per-construction cost; cost scales with rows
  G      = one label-gradient computation (b x d matmul + d x C)
  DEC    = one full val decode (100k x 10000 @ W) -- the eval cost, shared by
           ALL methods (one per rho). DEC >> SVD >> R, so the method overhead is
           a small fraction of a single decode.

Usage:
  uv run python robust_diagnostic/al_trust_refine_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_trust_refine_<label>.json
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


def provisional_fit(Xp, pl, W0, idx, lam, device):
    """One provisional ridge fit on the points idx (<=8). Returns dW = W_t - W0
    as a C x d matrix (code-space rows), i.e. dW.t()."""
    W_t = ridge_fit_soft(Xp[idx], onehot(pl[idx], NUM_CLASSES), lam, 8, 1000, device)
    return (W_t - W0).detach().cpu().t()


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
    ap.add_argument("--b", type=int, default=8, help="label budget")
    ap.add_argument("--n_windows", type=int, default=4, help="tangent windows (baseline)")
    ap.add_argument("--n_avg_draws", type=int, default=8, help="U_avg: independent tangent draws")
    ap.add_argument("--n_sharpen_rounds", type=int, default=3, help="U_sharpen: refinement rounds")
    ap.add_argument("--rho_sweep", type=str, default="0.05,0.1,0.2,0.4,0.8")
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
    rho_sweep = [float(x) for x in args.rho_sweep.split(',')]
    rmax = max(r_sweep)
    b = args.b

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    results = {'label': args.label, 'method': args.method_b, 'b': b, 'conds': {},
               'efficiency': {}}

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
        U_oracle, _ = right_topk_svd(R.t(), rmax)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # --- the labeled set: leverage-in-oracle-U top-b ---
        lev = torch.norm(Xp.float() @ U_oracle[:, :2], p=2, dim=1)
        sel = torch.argsort(lev, descending=True)[:b].long()
        X_lab = Xp[sel]; Y_lab = onehot(pl[sel], NUM_CLASSES)
        P0_lab = torch.softmax(X_lab.float() @ W0.cpu(), dim=1)
        resid = (Y_lab.float() - X_lab.float() @ W0.cpu())
        # full label gradient (d x C) -- the labels' own direction
        G_full = X_lab.float().t() @ (Y_lab.float() - P0_lab)

        # --- build the U variants + efficiency ledger (cost in units R/SVD/G) ---
        Ubases = {}
        eff = {}

        # oracle + random references
        Ubases['oracle'] = U_oracle; eff['oracle'] = {'R': 0, 'SVD': 0, 'G': 0}
        torch.manual_seed(0)
        Ubases['random'], _ = right_topk_svd(torch.randn(NUM_CLASSES, Xp.shape[1]), rmax)
        eff['random'] = {'R': 0, 'SVD': 1, 'G': 0}

        # baseline tangent (n_windows)
        D_tan = torch.cat([provisional_fit(Xp, pl, W0, sel[wi], args.lam, device)
                           for wi in torch.chunk(torch.randperm(b), args.n_windows)], dim=0)
        U_tan, _ = right_topk_svd(D_tan, rmax)
        Ubases['tangent'] = U_tan
        eff['tangent'] = {'R': args.n_windows, 'SVD': 1, 'G': 0}

        # U_avg: M independent tangent draws -> big concatenated stack -> top-r SVD
        D_stack = [D_tan]
        for d_i in range(args.n_avg_draws - 1):
            torch.manual_seed(100 + d_i)
            wins = torch.chunk(torch.randperm(b), args.n_windows)
            D_stack.append(torch.cat([provisional_fit(Xp, pl, W0, sel[wi], args.lam, device)
                                      for wi in wins], dim=0))
        D_avg = torch.cat(D_stack, dim=0)
        U_avg, _ = right_topk_svd(D_avg, rmax)
        Ubases['U_avg'] = U_avg
        eff['U_avg'] = {'R': args.n_avg_draws * args.n_windows,
                        'SVD': 1, 'G': 0, 'SVD_rows': D_avg.shape[0]}

        # U_windows: more provisional windows in ONE construction
        nw_more = max(args.n_windows * 4, 16)
        D_more = torch.cat([provisional_fit(Xp, pl, W0, sel[wi], args.lam, device)
                            for wi in torch.chunk(torch.randperm(b), nw_more)], dim=0)
        U_w, _ = right_topk_svd(D_more, rmax)
        Ubases['U_windows'] = U_w
        eff['U_windows'] = {'R': nw_more, 'SVD': 1, 'SVD_rows': D_more.shape[0]}

        # U_sharpen: iterative -- use the current U's leverage to select provisional
        # fit points, re-PCA. Round 0 = tangent; each round re-selects b points.
        D_sh = [D_tan]
        U_cur = U_tan
        for rnd in range(args.n_sharpen_rounds):
            lev_c = torch.norm(Xp.float() @ U_cur[:, :2], p=2, dim=1)
            sel_c = torch.argsort(lev_c, descending=True)[:b].long()
            torch.manual_seed(200 + rnd)
            wins = torch.chunk(torch.randperm(b), args.n_windows)
            D_sh.append(torch.cat([provisional_fit(Xp, pl, W0, sel_c[wi], args.lam, device)
                                   for wi in wins], dim=0))
            U_cur, _ = right_topk_svd(torch.cat(D_sh, dim=0), rmax)
        Ubases['U_sharpen'] = U_cur
        eff['U_sharpen'] = {'R': args.n_windows * (args.n_sharpen_rounds + 1),
                            'SVD': args.n_sharpen_rounds + 1}

        # --- evaluate the trust-region step for each base ---
        curves = {}
        for uname, Ur in Ubases.items():
            align = subspace_cos(Ur, U_oracle, rmax)
            gcs = {}
            for r in r_sweep:
                Ur_r = Ur[:, :r]
                G_r = (X_lab.float() @ Ur_r).t() @ resid
                Gn_r = G_r / (G_r.norm() + 1e-8)
                for rho in rho_sweep:
                    W1 = W0.detach().cpu() + (Ur_r @ (rho * Gn_r))
                    d = mw(W1, Xv, vl) - refs['frozen']
                    gcs[f'r{r}_rho{rho}'] = {'delta': float(d),
                                             'gap_closed': float(d / gap) if gap > 1e-9 else None}
            curves[uname] = {'align_U_oracle': align, 'gc': gcs}
            # best gc over the whole sweep per rank
            for r in r_sweep:
                best = max((v['gap_closed'] or -9 for k, v in gcs.items() if k.startswith(f'r{r}_')),
                           default=None)
                curves[uname][f'best_gc_r{r}'] = best

        # A_fix and A_hybrid are STEP-side: they do not change U, they change the
        # direction used in the step. Evaluate separately (per rank).
        step_extra = {}
        for r in r_sweep:
            Ur = U_tan[:, :r]
            # A_fix: direction = label gradient projected onto the tangent span
            G_proj = Ur @ (Ur.t() @ G_full)
            Gp_n = G_proj / (G_proj.norm() + 1e-8)
            # A_hybrid: projected + residual component of the label gradient
            G_res = G_full - G_proj
            G_hy = G_proj + G_res / (G_res.norm() + 1e-8)
            Ghy_n = G_hy / (G_hy.norm() + 1e-8)
            # A_grad: full label gradient, no U
            Gg_n = G_full / (G_full.norm() + 1e-8)
            for name, Gn in [('A_grad', Gg_n), ('A_fix', Gp_n), ('A_hybrid', Ghy_n)]:
                gcs_s = {}
                for rho in rho_sweep:
                    W1 = W0.detach().cpu() + (rho * Gn)
                    d = mw(W1, Xv, vl) - refs['frozen']
                    gcs_s[f'r{r}_rho{rho}'] = {'delta': float(d),
                                               'gap_closed': float(d / gap) if gap > 1e-9 else None}
                best = max((v['gap_closed'] or -9 for v in gcs_s.values()), default=None)
                step_extra[f'{name}_r{r}'] = {'gc': gcs_s, 'best_gc': best}
            eff.setdefault('A_grad', {'R': 0, 'SVD': 0, 'G': 1})
            eff.setdefault('A_fix', {'R': args.n_windows, 'SVD': 1, 'G': 1})
            eff.setdefault('A_hybrid', {'R': args.n_windows, 'SVD': 1, 'G': 2})

        results['conds'][cond] = {'refs': refs, 'gap': float(gap), 'curves': curves,
                                  'step_extra': step_extra}
        del Xc, Xp, Xv, W0, Ws, R, U_oracle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        for uname, cv in curves.items():
            print(f"    {uname:10s} alignU {cv['align_U_oracle']:.2f} | best_gc r2 {cv.get('best_gc_r2')} r4 {cv.get('best_gc_r4')}")
        for name, sv in step_extra.items():
            print(f"    {name:10s} best_gc {sv['best_gc']:+.2f}")

    # efficiency ledger (fixed across conditions)
    results['efficiency'] = eff
    for name, e in eff.items():
        print(f"\nEFFICIENCY {name:10s}: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("STEP-SIDE: A_grad (labels' gradient, no U) / A_fix (project onto tangent) /")
    print("  A_hybrid (projected + residual). Does any fix beat the coarse tangent?")
    print("U-REFINEMENT: U_avg (average draws) / U_windows (more windows) / U_sharpen")
    print("  (iterative). Does a better U make the trust-region step work?")
    print("EFFICIENCY: units R (provisional fit) / SVD (svd of rows x 10000) / G (label grad).")
    print("  DEC (val decode) dominates all; method overhead is a small fraction.")


if __name__ == "__main__":
    main()
