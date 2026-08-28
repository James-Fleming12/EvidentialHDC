"""al_uest_joint_diag.py: joint U,C refinement on top of the tangent-U init.

The boundary/tangent diagnostic found tangent_b8 is the ONLY U estimator that
recovers the oracle residual (align 0.3-0.5), but its AL chain (leverage-in-U
then C-solve in a FIXED U) is ~0. The synthesis: instead of the separate
"discover U, then fit C" pipeline, jointly refine U and C on the labeled points:

    W_res = W0 + U C,   minimize ||X_lab (W0 + U C) - Y_lab||^2
    subject to U orthonormal (d x r).

Alternating scheme (the doc's proposal):
  for it in 1..T:
    C  <- solve_{C}  ||X_lab U C - R_lab||^2        (C-solve, exact least squares)
    U  <- U - lr * d/dU ||X_lab U C - R_lab||^2     (U-gradient)
    U  <- orthogonalize (QR)
where R_lab = Y_lab - X_lab W0.

The point: the same labels that discovered U via tangent-b8 (provisional tiny
fits) are now ALSO used to push U toward the oracle while fitting C, instead of
fixing U at its tangent value. Test: does the joint refinement (i) move U toward
U_oracle (align rising with iters), and (ii) turn the ~0 tangent AL chain into a
real gap-closed?

Per condition, per (b in budget, r in rank):
  - U_init = tangent-b8 construction (split b labels into n_windows provisional
    fits, stack dW, right SVD)  [the discovered U]
  - baseline: C-solve in FIXED U_init  -> delta/gc  (the current ~0 result)
  - joint: alternate C-solve / U-gradient / QR for T in {1, 5, 20} iters
    -> delta/gc, and align(U_final, U_oracle) per iters
  - reference: oracle-U C-solve at same (b, r) -> the ceiling at that budget
  - U-oracle-alignment of the REFINED U (does refinement pull toward oracle?)

Usage:
  uv run python robust_diagnostic/al_uest_joint_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_uest_joint_<label>.json
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


def tangent_U(Xp, pl, W0, sel, n_windows, lam, device):
    """The tangent-b8 construction: split the labeled points into tiny windows,
    fit a provisional ridge on each, stack their dW = W_t - W0, right-SVD."""
    wins = torch.chunk(torch.randperm(len(sel)), n_windows)
    dW_stack = []
    for wi in wins:
        si = sel[wi]
        W_t = ridge_fit_soft(Xp[si], onehot(pl[si], NUM_CLASSES), lam, 8, 1000, device)
        dW_stack.append((W_t - W0).detach().cpu().t())
    D = torch.cat(dW_stack, dim=0)
    return D


def joint_refine(X_lab, Y_lab, W0, U_init, lr, iters, device):
    """Alternate C-solve / U-gradient / QR. Returns (U, history)."""
    U = U_init.float().clone()
    R_lab = (Y_lab.float() - X_lab.float() @ W0.detach().cpu().float())   # n x C
    hist = []
    for _ in range(iters):
        # C-solve (exact least squares on the current U)
        XU = X_lab.float() @ U
        C = torch.linalg.solve(XU.t() @ XU + 1e-6 * torch.eye(U.shape[1]),
                               XU.t() @ R_lab)
        # U-gradient: d/dU ||X_lab U C - R_lab||^2 = 2 X_lab^T (X_lab U C - R_lab) C^T
        resid = X_lab.float() @ U @ C - R_lab
        grad = 2.0 * (X_lab.float().t() @ resid) @ C.t()
        U = U - lr * grad
        U, _ = torch.linalg.qr(U)
        hist.append({'C': C.detach().cpu(), 'U': U.detach().cpu().clone(),
                     'loss': float((resid ** 2).sum().item())})
    # final C-solve for the cleanest W_res
    C_f = lsq_residual(X_lab, Y_lab, W0, U, device)
    return U.detach().cpu(), C_f, hist


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
    ap.add_argument("--max_clean", type=int, default=50000,
                    help="cap on clean points for the frozen probe fit (50k=2GB GPU "
                         "copy; the clean fit saturates well before this)")
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--r_sweep", type=str, default="2,4")
    ap.add_argument("--budget_sweep", type=str, default="8,32")
    ap.add_argument("--n_windows", type=int, default=4, help="tangent-U provisional windows")
    ap.add_argument("--lr", type=float, default=1e-2, help="U-gradient learning rate")
    ap.add_argument("--joint_iters", type=str, default="1,5,20", help="alternation iterations to test")
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
    joint_iters = [int(x) for x in args.joint_iters.split(',')]
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
        # free the raw 128-d corrupted feature tensors (keep fa/la clean features,
        # which are reused across the condition loop, and the labels pl/vl)
        del pool, val, f, l
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_oracle, S_or = right_topk_svd(R.t(), rmax)   # R is d x C; transpose -> C x d rows

        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        cond_res = {'refs': refs, 'gap': float(gap), 'budgets': {}}
        for b in budget_sweep:
            if b < args.n_windows:
                continue
            # --- select b labeled points by leverage in the frozen-margin sense ---
            lev = torch.norm(Xp.float(), p=2, dim=1)
            sel = torch.argsort(lev, descending=True)[:b].long()

            # --- tangent-U init from the b labels (provisional tiny fits) ---
            D_tan = tangent_U(Xp, pl, W0, sel, args.n_windows, args.lam, device)

            b_res = {'U_align_init': {}, 'baseline': {}, 'joint': {}, 'oracle_ref': {}}
            for r in r_sweep:
                # tangent U at this rank
                U_tan, _ = right_topk_svd(D_tan, r)
                b_res['U_align_init'][str(r)] = subspace_cos(U_tan, U_oracle, r)

                # baseline: C-solve in FIXED U_tan (the current ~0 result)
                C_base = lsq_residual(Xp[sel], onehot(pl[sel], NUM_CLASSES), W0, U_tan, device)
                W_base = W0.detach().cpu() + (U_tan @ C_base)
                d_base = mw(W_base, Xv, vl) - refs['frozen']
                b_res['baseline'][f'r{r}_b{b}'] = {'delta': float(d_base),
                                                   'gap_closed': float(d_base / gap) if gap > 1e-9 else None,
                                                   'align_U': subspace_cos(U_tan, U_oracle, r)}

                # joint refinement from the tangent-U init
                for T in joint_iters:
                    U_j, C_j, hist = joint_refine(Xp[sel], onehot(pl[sel], NUM_CLASSES),
                                                  W0.detach().cpu(), U_tan, args.lr, T, device)
                    W_j = W0.detach().cpu() + (U_j @ C_j)
                    d_j = mw(W_j, Xv, vl) - refs['frozen']
                    b_res['joint'][f'r{r}_b{b}_T{T}'] = {
                        'delta': float(d_j),
                        'gap_closed': float(d_j / gap) if gap > 1e-9 else None,
                        'align_U_final': subspace_cos(U_j, U_oracle, r),
                        'loss_init': hist[0]['loss'] if hist else None,
                        'loss_final': hist[-1]['loss'] if hist else None,
                        'loss_history': [h['loss'] for h in hist],
                        'C_norm': float(C_j.norm()),
                    }
                # oracle reference at same (r, b): C-solve in oracle U
                U_or = U_oracle[:, :r]
                C_or = lsq_residual(Xp[sel], onehot(pl[sel], NUM_CLASSES), W0, U_or, device)
                W_or = W0.detach().cpu() + (U_or @ C_or)
                d_or = mw(W_or, Xv, vl) - refs['frozen']
                b_res['oracle_ref'][f'r{r}_b{b}'] = {'delta': float(d_or),
                                                     'gap_closed': float(d_or / gap) if gap > 1e-9 else None}
            cond_res['budgets'][str(b)] = b_res

        results['conds'][cond] = cond_res
        del Xc, Xp, Xv, W0, Ws, R, U_oracle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        for b, br in cond_res['budgets'].items():
            print(f"  b={b}  U_init_align={ {k: round(v,2) for k,v in br['U_align_init'].items()} }")
            for r in r_sweep:
                bl = br['baseline'][f'r{r}_b{b}']
                print(f"    r={r}  baseline delta {bl['delta']:+.3f} (gc {bl['gap_closed']:.2f})  alignU {bl['align_U']:.2f}")
                for T in joint_iters:
                    j = br['joint'][f'r{r}_b{b}_T{T}']
                    print(f"      joint T={T:2d}  delta {j['delta']:+.3f} (gc {j['gap_closed']:.2f})  alignU_final {j['align_U_final']:.2f}  loss {j['loss_init']:.1f}->{j['loss_final']:.1f}")
                or_ = br['oracle_ref'][f'r{r}_b{b}']
                print(f"      oracle-ref  delta {or_['delta']:+.3f} (gc {or_['gap_closed']:.2f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("U_init = tangent-U (provisional tiny fits). baseline = C-solve in FIXED U.")
    print("joint = alternate C-solve / U-gradient / QR for T iters, then C-solve.")
    print("oracle_ref = C-solve in oracle U at the same (r, b) -- the budget ceiling.")
    print("Read: does the joint refinement (i) move align_U_final toward oracle,")
    print("  (ii) raise delta/gc from baseline toward oracle_ref, (iii) lower the loss?")
    print("If align rises but gc stays ~0, U is fixed but the C leverage is wrong.")
    print("If both rise, the joint fit turns the discovered U into a real AL gain.")


if __name__ == "__main__":
    main()
