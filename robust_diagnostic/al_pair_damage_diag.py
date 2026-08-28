"""al_pair_damage_diag.py: Iteration 0c -- clean->corrupted decision-conditioned U.

The Iteration-0b lesson: the missing half of U* is information of a DIFFERENT
kind -- it is a counterfactual (what classifier would be correct under the
corruption), and every U-estimator so far inferred U from the CORRUPTED
distribution alone. This diagnostic uses the clean/corrupted PAIRING (which we
uniquely have: KITTI-C is per-frame corruptions of seq-08, same scan geometry and
same labels), and conditions on DECISION DAMAGE.

The pairing: clean and corrupted scans are the SAME seq-08 scan (same projection
grid), so we pair features by matching the flattened valid-pixel grid position.
For each scan, both the clean and corrupted 128-d features are extracted at the
valid pixels; we encode both to the 10000-d code and pair by pixel index (they
are in the same flattened-mask order). Per paired pixel i:
    dx_i   = code_corr[i] - code_clean[i]     (corruption displacement, code space)
    dz_i   = dx_i^T W0                        (logit/decision damage)
    damage = (frozen clean correct) AND (frozen corrupted wrong)   [real label]
    loss_gain_i = CE(corr_pred, y) - CE(clean_pred, y)             [decision damage]

U constructions (all in the 10000-d code space, comparable to U_oracle):
  U_cross     : left singulars of  M_cross = sum_i dx_i dz_i^T
                ("which corruption displacements cause decision damage?")
  U_damage    : top-r of          sum_{damage} dx_i dx_i^T   (failure covariance)
  U_damage_w  : top-r of          sum_i loss_gain_i * dx_i dx_i^T
  U_dx_all    : top-r of          sum_i dx_i dx_i^T  (all pixels, weak control)
  oracle / tangent references.

The damage covariances (d x d) are accumulated only over a bounded sample of
paired pixels, then right-SVD'd. M_cross (d x C) accumulates cheaply over all
paired pixels.

Evaluation: align(U, U_oracle) for r in {2,4}, AND the trust-region step
W1 = W0 + rho * U * G/||G|| on a leverage-selected labeled set (re-extracted from
the corrupted pool), gc-vs-rho. The real test is whether a paired, damage-
conditioned U makes the trust-region step close gap on fog/crosstalk.

Gating for the U-predictor head:
  align > ~0.7  -> U* recoverable from the pairing; label-free U exists.
  align 0.3-0.7 -> learnable mapping (head / canonical adapter) is the route.
  align ~0      -> U* not in the corrupted side at all; only training the
                   extractor to expose U (canonical adapter) can work.

Usage:
  uv run python robust_diagnostic/al_pair_damage_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_pair_damage_<label>.json
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


def extract_clean(model, parser, device, num_frames=100):
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


def extract_paired_feats(model, clean_parser, corr_parser, device, num_frames, max_pixels):
    """Pair clean/corrupted 128-d features at the SAME (row, col) pixels of each
    scan's projection grid (KITTI-C preserves geometry, changes values). Yields
    per scan: (zc, zd, lbl, name_c, name_d, range_corr) where index k of zc/zd is
    the same (row, col). name_c/name_d are the scan identifiers; range_corr is the
    Pearson correlation of the RANGE channel at the paired pixels (a pairing
    sanity signal: aligned scans share geometry -> range_corr >> 0)."""
    with torch.no_grad():
        for i, (bc, bd) in enumerate(zip(clean_parser.get_train_set(), corr_parser.get_train_set())):
            if i >= num_frames:
                break
            out_c = model(bc[0].to(device)); z_c = out_c[2] if len(out_c) == 3 else out_c[1]
            out_d = model(bd[0].to(device)); z_d = out_d[2] if len(out_d) == 3 else out_d[1]
            mc = (bc[1].to(device) > 0); md = (bd[1].to(device) > 0)
            inter = mc & md
            n = int(inter.sum().item())
            if n < 100:
                continue
            if n > max_pixels:
                torch.manual_seed(i)
                keep = torch.randperm(n)[:max_pixels]
            else:
                keep = torch.arange(n)
            zc = z_c.permute(0, 2, 3, 1).reshape(-1, z_c.shape[1])[inter.view(-1)][keep]
            zd = z_d.permute(0, 2, 3, 1).reshape(-1, z_d.shape[1])[inter.view(-1)][keep]
            lbl = bc[2].to(device).view(-1)[inter.view(-1)][keep]
            # pairing sanity: range channel (input channel 0) at the SAME pixels
            rc = bc[0][0].view(-1)[inter.view(-1)][keep].float().cpu()
            rd = bd[0][0].view(-1)[inter.view(-1)][keep].float().cpu()
            rc_ = rc - rc.mean(); rd_ = rd - rd.mean()
            denom = (rc_.norm() * rd_.norm())
            range_corr = float((rc_ * rd_).sum().item() / (denom + 1e-8)) if denom > 0 else 0.0
            name_c = bc[5] if len(bc) > 5 else None
            name_d = bd[5] if len(bd) > 5 else None
            if (i % 20) == 0:
                print(f"    [paired] scan {i}: {len(zc)} pix, range_corr {range_corr:.3f}", flush=True)
            yield zc.cpu(), zd.cpu(), lbl.cpu(), name_c, name_d, range_corr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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


def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc); y[torch.arange(len(lbls)), lbls.long()] = 1; return y


def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    p = []
    for s in range(0, len(codes), chunk):
        p.append((codes[s:s+chunk].float() @ W).argmax(1))
    return torch.cat(p)


def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)


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
    ap.add_argument("--rho_sweep", type=str, default="0.05,0.1,0.2,0.4,0.8")
    ap.add_argument("--b", type=int, default=8, help="labeled points for the trust-region direction")
    ap.add_argument("--max_pixels", type=int, default=40000, help="per-scan paired pixels")
    ap.add_argument("--damage_sample", type=int, default=60000,
                    help="bounded paired-pixel sample for damage SVDs (60k = ~7GB "
                         "peak for the CC/CD/DX codes; enough for stable SVDs)")
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
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

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'conds': {}}

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        corr_parser = build_parser(cdir, DATA, ARCH)

        # corrupted pool for the probe/oracle/labeled set
        fd, ld = extract_clean(model, corr_parser, device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(fd))
        pool, pl = fd[perm[:args.pool_size]], ld[perm[:args.pool_size]]
        val, vl = fd[perm[-args.val_size:]], ld[perm[-args.val_size:]]
        mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
        Xc = torch.sign(fa[ci].to(device) @ proj).cpu().float()
        Xp = torch.sign(pool.to(device) @ proj).cpu().float()
        Xv = torch.sign(val.to(device) @ proj).cpu().float()
        del fd, ld, pool, val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_oracle, _ = right_topk_svd(R.t(), rmax)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        W0c = W0.detach().cpu()

        # ---- accumulate the paired damage statistics ----
        # Encode 128-d -> code PER SCAN for M_cross (codes discarded immediately);
        # keep only the cheap 128-d features for the bounded damage sample, encoded
        # ONCE at the end (avoids holding N x 10000 codes in memory).
        M_cross = torch.zeros(10000, NUM_CLASSES)     # sum dx dz^T  (d x C)
        sample_fc, sample_fd, sample_lbl = [], [], []
        n_total = 0
        n_name_match = 0
        n_scans = 0
        range_corrs = []
        for zc, zd, lbl, name_c, name_d, rcorr in extract_paired_feats(
                model, clean_parser, corr_parser, device, args.frames, args.max_pixels):
            n_scans += 1
            if name_c is not None and name_d is not None and name_c == name_d:
                n_name_match += 1
            range_corrs.append(rcorr)
            cc = torch.sign(zc.to(device) @ proj).cpu().float()
            cd = torch.sign(zd.to(device) @ proj).cpu().float()
            dx = cd - cc
            dz = dx.float() @ W0c                       # n x C
            M_cross += dx.float().t() @ dz
            del cc, cd, dx, dz
            # bounded sample: keep the CHEAP 128-d features (not codes)
            if n_total < args.damage_sample:
                take = min(len(zc), args.damage_sample - n_total)
                sample_fc.append(zc[:take]); sample_fd.append(zd[:take])
                sample_lbl.append(lbl[:take])
                n_total += take
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pairing_check = {
            'scans_paired': n_scans,
            'scan_names_match': n_name_match,
            'scan_names_aligned': bool(n_scans > 0 and n_name_match == n_scans),
            'mean_range_corr': float(np.mean(range_corrs)) if range_corrs else None,
            'min_range_corr': float(np.min(range_corrs)) if range_corrs else None,
        }
        print(f"    [pairing sanity] scans {n_scans} | names match {n_name_match}/{n_scans} "
              f"| mean range_corr {pairing_check['mean_range_corr']:.3f}", flush=True)

        # bounded sample: encode 128-d -> code ONCE (N x d at the end)
        FC = torch.cat(sample_fc, dim=0)
        FD = torch.cat(sample_fd, dim=0)
        LL = torch.cat(sample_lbl, dim=0)
        del sample_fc, sample_fd, sample_lbl
        N = len(FC)
        CC = torch.zeros(N, 10000)
        CD = torch.zeros(N, 10000)
        for s in range(0, N, 50000):
            e = min(s + 50000, N)
            CC[s:e] = torch.sign(FC[s:e].to(device) @ proj).cpu().float()
            CD[s:e] = torch.sign(FD[s:e].to(device) @ proj).cpu().float()
        del FC, FD
        DX = (CD - CC).float()
        pred_c = decode(W0, CC); pred_d = decode(W0, CD)
        correct_c = (pred_c == LL)
        correct_d = (pred_d == LL)
        damage = correct_c & ~correct_d                  # clean right, corr wrong
        # loss-gain weight (decision damage): CE increase clean->corr
        with torch.no_grad():
            logit_c = CC.float() @ W0c
            logit_d = CD.float() @ W0c
            ce_c = F.cross_entropy(logit_c, LL.long(), reduction='none')
            ce_d = F.cross_entropy(logit_d, LL.long(), reduction='none')
        loss_gain = (ce_d - ce_c).clamp(min=0).float()

        # ---- U constructions ----
        U_cross, _ = right_topk_svd(M_cross.t(), rmax)   # left singulars of d x C
        U_dx_all, _ = right_topk_svd(DX, rmax)
        U_damage, _ = right_topk_svd(DX[damage], rmax) if damage.sum() > 10 else (U_dx_all, None)
        DXw = DX * loss_gain.unsqueeze(1)
        U_damage_w, _ = right_topk_svd(DXw, rmax)

        # tangent reference from the corrupted pool (as in iter0)
        lev = torch.norm(Xp.float() @ U_oracle[:, :2], p=2, dim=1)
        sel = torch.argsort(lev, descending=True)[:args.b].long()
        wins = torch.chunk(torch.randperm(args.b), 4)
        D_tan = torch.cat([(ridge_fit_soft(Xp[sel[wi]], onehot(pl[sel[wi]], NUM_CLASSES),
                                           args.lam, 8, 1000, device) - W0).detach().cpu().t()
                           for wi in wins], dim=0)
        U_tan, _ = right_topk_svd(D_tan, rmax)

        Ubases = {'oracle': U_oracle, 'tangent': U_tan, 'U_cross': U_cross,
                  'U_dx_all': U_dx_all, 'U_damage': U_damage, 'U_damage_w': U_damage_w}

        # ---- evaluate: align + trust-region step ----
        X_lab = Xp[sel]; Y_lab = onehot(pl[sel], NUM_CLASSES)
        resid = (Y_lab.float() - X_lab.float() @ W0c)
        curves = {}
        for uname, Ur in Ubases.items():
            align = subspace_cos(Ur, U_oracle, rmax)
            gcs = {}
            for r in r_sweep:
                Ur_r = Ur[:, :r]
                G = (X_lab.float() @ Ur_r).t() @ resid
                Gn = G / (G.norm() + 1e-8)
                for rho in rho_sweep:
                    W1 = W0c + (Ur_r @ (rho * Gn))
                    d = mw(W1, Xv, vl) - refs['frozen']
                    gcs[f'r{r}_rho{rho}'] = {'delta': float(d),
                                             'gap_closed': float(d / gap) if gap > 1e-9 else None}
            curves[uname] = {'align_U_oracle': align, 'gc': gcs}
            for r in r_sweep:
                best = max((v['gap_closed'] or -9 for k, v in gcs.items() if k.startswith(f'r{r}_')),
                           default=None)
                curves[uname][f'best_gc_r{r}'] = best

        results['conds'][cond] = {'refs': refs, 'gap': float(gap), 'curves': curves,
                                  'damage_frac': float(damage.float().mean().item()),
                                  'n_damage_pix': int(damage.sum().item()),
                                  'pairing_check': pairing_check}
        del Xc, Xp, Xv, W0, Ws, R, U_oracle, CC, CD, DX
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    pairing: {pairing_check}")
        print(f"    damage_frac {results['conds'][cond]['damage_frac']:.3f} ({results['conds'][cond]['n_damage_pix']} pix)")
        for uname, cv in curves.items():
            print(f"    {uname:12s} alignU {cv['align_U_oracle']:.2f} | best r2 {cv.get('best_gc_r2'):+.2f} r4 {cv.get('best_gc_r4'):+.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("align > ~0.7 -> U* recoverable from the pairing (label-free U exists).")
    print("align 0.3-0.7 -> learnable mapping (head / canonical adapter) is the route.")
    print("align ~0 -> U* not in the corrupted side; only canonical-adapter training can work.")
    print("best_gc: does a paired damage-conditioned U make the trust-region step close gap?")


if __name__ == "__main__":
    main()
