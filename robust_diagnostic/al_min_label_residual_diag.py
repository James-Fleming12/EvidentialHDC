"""al_min_label_residual_diag.py: how few TRUE labels can the low-rank residual
update W_res = W0 + U_r C use, and still give a meaningful update?

The established method (C30/C31) fits C on k=8 labels per class (56 total) with
oracle U_r (r=8), closing ~100% of the closeable gap. This diagnostic asks the
minimal-label question directly: with U_r ORACLE (to isolate the LABEL-count
bottleneck from the U-estimation bottleneck), sweep:

  - residual rank r in {2, 4, 8}  (the U-subspace dimension)
  - true-label budget b in {2, 4, 8, 16, 32, 56} (TOTAL points, not per class)
  - selection in {random, leverage_u, per_class}

per condition, and report gap-closed (mIoU - frozen)/(oracle - frozen) at each
(r, b) operating point.

The mechanism question: C is an r-dim vector fit from b points; the 8x8 system
needs >= r informative points to be well-posed, and each point's value is its
projection onto U (how much it moves the residual) AND its residual magnitude
(Y - XW0). The lever to make FEW points work is (i) lower r (the residual is
effectively rank 4-5, Iteration C20) and (ii) select the points with the highest
leverage in the U-subspace (N9: active querying in the residual directions),
which is NOT the same as farthest-point / confidence selection.

Also reports, per (r, b): cos(U^T x_selected, ...) style diagnostics and the
minimal b that first exceeds frozen at each r.

Usage:
  uv run python robust_diagnostic/al_min_label_residual_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_min_label_residual_<label>.json
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
    X = X.to(device); torch.manual_seed(SKETCH_SEED); m = min(m, X.shape[1])
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P; Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device); That = XP.t() @ Yd
    x = P @ torch.linalg.solve(Shat, That); b = X.t() @ Yd
    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); a = rs / ((p * Ap).sum(0) + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0); be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p; rs = rsn
    return x.float()


def lsq_residual(X_lab, Y_lab, W0, U, device):
    Xd = X_lab.to(device).float(); Yd = Y_lab.to(device).float(); Ud = U.to(device)
    r = Ud.shape[1]; XU = Xd @ Ud
    A = XU.t() @ XU + 1e-6 * torch.eye(r, device=device)
    b = XU.t() @ (Yd - Xd @ W0.to(device))
    return torch.linalg.solve(A, b).cpu()


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
    ap.add_argument("--max_clean", type=int, default=200000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--r_sweep", type=str, default="2,4,8")
    ap.add_argument("--budget_sweep", type=str, default="2,4,8,16,32,56")
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="covshift_ep10")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    r_sweep = [int(x) for x in args.r_sweep.split(',')]
    budget_sweep = [int(x) for x in args.budget_sweep.split(',')]

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

        W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_full, S, _ = torch.linalg.svd(R.double(), full_matrices=False)
        U_full = U_full.float(); S = S.float()

        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}

        r_dict = {}
        for r in r_sweep:
            Ur = U_full[:, :r]
            # reference: C fit from ALL pool labels in U_r -> the rank-r ceiling
            C_all = lsq_residual(Xp, onehot(pl, NUM_CLASSES), W0, Ur, device)
            W_all = W0.detach().cpu() + (Ur.cpu() @ C_all)
            ref_all = mw(W_all, Xv, vl)

            # leverage-in-U per point: ||x^T U_r||  (how much each point moves the residual)
            lev = torch.norm(Xp.float() @ Ur.cpu(), p=2, dim=1)
            b_dict = {}
            for b in budget_sweep:
                if b >= args.pool_size:
                    continue
                entries = {}
                # selection 1: random
                torch.manual_seed(7)
                sel_rnd = torch.randperm(len(Xp))[:b]
                # selection 2: leverage-in-U (N9: active query in the residual subspace)
                sel_lev = torch.argsort(lev, descending=True)[:b]
                # selection 3: per-class (balanced over the classes present)
                sel_pc = []
                for c in classes:
                    idx = cls_idx[c]
                    take = max(1, b // len(classes))
                    if len(idx) > 0:
                        sel_pc.append(idx[torch.randperm(len(idx))[:take]])
                if sel_pc:
                    sel_pc = torch.cat(sel_pc)[:b]
                    if len(sel_pc) < b:
                        rest = torch.randperm(len(Xp))[:b - len(sel_pc)]
                        sel_pc = torch.cat([sel_pc, rest])
                else:
                    sel_pc = sel_rnd

                for sname, sel in [('random', sel_rnd), ('leverage_u', sel_lev), ('per_class', sel_pc)]:
                    sel = sel.long()
                    C = lsq_residual(Xp[sel], onehot(pl[sel], NUM_CLASSES), W0, Ur, device)
                    W_res = W0.detach().cpu() + (Ur.cpu() @ C)
                    delta = mw(W_res, Xv, vl) - refs['frozen']
                    # quality of C: cos to the all-label C in the projected space
                    cos_c = float(F.cosine_similarity(C.flatten(), C_all.flatten(), dim=0).item()) if C.numel() > 0 else None
                    entries[sname] = {
                        'delta': float(delta),
                        'gap_closed': float(delta / gap) if gap > 1e-9 else None,
                        'cos_c': cos_c,
                        'n': int(len(sel)),
                    }
                b_dict[str(b)] = entries
            r_dict[str(r)] = {
                'ref_rank_r_ceiling_delta': float(ref_all - refs['frozen']),
                'gap_closed_rank_r_ceiling': float((ref_all - refs['frozen']) / gap) if gap > 1e-9 else None,
                'budgets': b_dict,
            }
        results['conds'][cond] = {'refs': refs, 'gap': float(gap), 'ranks': r_dict,
                                  'singular_values': S[:8].tolist()}
        del Xc, Xp, Xv, W0, Ws, R, U_full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        for r in r_sweep:
            rd = r_dict[str(r)]
            print(f"  r={r}  ceiling_delta {rd['ref_rank_r_ceiling_delta']:+.3f}")
            for b in budget_sweep:
                if str(b) not in r_dict[str(r)]['budgets']:
                    continue
                e = r_dict[str(r)]['budgets'][str(b)]
                print(f"    b={b:3d}  " + " ".join(
                    f"{s}:{v['delta']:+.3f}(gc {v['gap_closed']:.2f})" for s, v in e.items()))
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("For each condition, at each rank r, the label budget b in {2,4,8,16,32,56}:")
    print("  delta = mIoU(W0 + U_r C) - frozen; gap_closed = delta / (oracle - frozen).")
    print("  ceiling_delta = the rank-r UPPER BOUND (C fit from ALL pool labels).")
    print("Questions it answers:")
    print("  1. At what r does a couple of points (b=2-8) first EXCEED frozen?")
    print("  2. Is leverage_u (query in the residual subspace) better than random/per_class?")
    print("  3. Is the bottleneck the LABEL COUNT or the RANK (does lowering r to 2-4 help)?")
    print("  4. cos_c ~ 1 means the few-point C matches the all-label C direction.")


if __name__ == "__main__":
    main()
