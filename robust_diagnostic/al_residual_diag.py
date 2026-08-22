"""al_residual_diag.py: C20 -- the residual-compressibility diagnostic.

C19 showed the ball/spec + cov-shift hybrid fails because the two objectives
compete in ONE representation. The reframe (C19 analysis): cov-shift already
consumes most of the corruption residual (frozen ~= ceiling), so AL labels have
little left to buy; ball/spec leaves MORE residual but structures it so sparse
labels can correct it. The question C20 answers WITHOUT any training:

  Is the oracle residual R = W* - W0 (W* = oracle probe, W0 = frozen cov-shift
  probe) LOW-RANK? If yes, AL should estimate a small residual correction
  (W = W0 + U_r C, r << d) instead of a full 17 x 10k probe.

For each condition, per extractor:
  - cos(W0, W*), ||R||_F / ||W*||_F   (how much residual is there)
  - SVD of R: singular spectrum, effective rank, cumulative energy of top r
  - THE ORACLE RESIDUAL CURVE: mIoU(W0 + R_r) for r in {0,1,2,4,8,16,17,...}
    where R_r = U_r U_r^T R (the top-r left-singular projection of R). This is
    the CEILING of any low-rank residual AL method: how many directions are
    needed to recover the oracle gap.
  - Feature-space shift check: per-class mean shift (corrupted - clean) in the
    128-d space, SVD'd the same way -- does the corruption live in a small
    subspace of the FEATURE space too?

Usage:
  uv run python robust_diagnostic/al_residual_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_residual_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11
RS = [0, 1, 2, 4, 8, 16, 32, 64, 128]

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
            out = model(in_vol)
            z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(zf.cpu())
            lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(lbls)

def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)

def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc)
    y[torch.arange(len(lbls)), lbls.long()] = 1
    return y

def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    p = []
    for s in range(0, len(codes), chunk):
        p.append((codes[s:s + chunk].float() @ W).argmax(1))
    return torch.cat(p)

def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)

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
    rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p)
        a = rs / ((p * Ap).sum(0) + 1e-30)
        x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap
        rsn = (r * r).sum(0)
        be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p
        rs = rsn
    return x.float()

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def tic():
    sync()
    return time.time()

def toc(t0):
    sync()
    return time.time() - t0

def svd_metrics(R):
    """Singular spectrum diagnostics of a d x C residual matrix.
    Returns dict: singular values, effective rank, cumulative energy."""
    s = torch.linalg.svdvals(R.double()).cpu()
    s2 = s ** 2
    total = s2.sum().item()
    cum = torch.cumsum(s2, dim=0) / (total + 1e-30)
    part = (s2.sum() ** 2 / (s2 ** 2).sum()).item() if total > 0 else 0.0
    return {
        'singular_values': [float(v) for v in s],
        'effective_rank': float(part),
        'cum_energy': {str(r): float(cum[min(r, len(s)) - 1].item())
                       for r in RS if r >= 1 and r <= len(s)},
        'total_energy': float(total),
    }

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
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="med")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config))
    ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
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
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]

        proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
        Xc = hdc_codes(fa[ci], proj, device).float()
        Xp = hdc_codes(pool, proj, device).float()
        Xv = hdc_codes(val, proj, device).float()

        W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam,
                            args.cg_iters, args.nystrom_m, device)
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam,
                            args.cg_iters, args.nystrom_m, device)

        R = (Ws - W0).detach().cpu().float()
        r = {'refs': {}, 'residual': {}, 'curve': {}, 'feat_shift': {}}
        r['refs']['frozen'] = mw(W0, Xv, vl)
        r['refs']['oracle'] = mw(Ws, Xv, vl)
        r['residual']['w_cos'] = cos_sim(W0, Ws)
        r['residual']['rel_norm'] = float(R.norm() / (Ws.norm() + 1e-30))
        r['residual']['svd'] = svd_metrics(R)

        # THE oracle residual curve: mIoU(W0 + U_r U_r^T R)
        U, s, Vh = torch.linalg.svd(R.double(), full_matrices=False)
        U = U.float()
        Rr = {0: torch.zeros_like(R)}
        curve = {0: mw(W0, Xv, vl)}
        for rr in RS:
            if rr == 0 or rr > R.shape[1]:
                continue
            Ur = U[:, :rr]
            Rr[rr] = (Ur @ (Ur.t() @ R)).float()
            Wr = W0.detach().cpu() + Rr[rr]
            curve[rr] = mw(Wr, Xv, vl)
        r['curve'] = {str(k): float(v) for k, v in curve.items()}
        # also full-rank (should reproduce oracle)
        r['curve_full'] = mw(W0.detach().cpu() + R, Xv, vl)

        # feature-space shift check (128-d): per-class mean shift, SVD'd
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        clean_classes = sorted(set(la[ci].tolist()) & set(range(1, NUM_CLASSES)))
        cn_idx = {c: (la[ci] == c).nonzero().squeeze(1) for c in clean_classes}
        cp_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}
        # centered per-class means in the RAW 128-d space
        cn = fa[ci].float(); cp = pool.float()
        cn_c = cn - cn.mean(0, keepdim=True)
        cp_c = cp - cp.mean(0, keepdim=True)
        mu_cn = {c: cn_c[cn_idx[c]].mean(0) for c in classes if len(cn_idx[c]) > 0}
        mu_cp = {c: cp_c[cp_idx[c]].mean(0) for c in classes if len(cp_idx[c]) > 0}
        common = [c for c in classes if c in mu_cn and c in mu_cp]
        if common:
            shift = torch.stack([mu_cp[c] - mu_cn[c] for c in common])  # ncls x 128
            s2 = torch.linalg.svdvals(shift.double()).cpu() ** 2
            tot = s2.sum().item()
            cum = torch.cumsum(s2, dim=0) / (tot + 1e-30)
            r['feat_shift'] = {
                'n_classes': len(common),
                'classes': common,
                'effective_rank': float(s2.sum() ** 2 / (s2 ** 2).sum().item()) if tot > 0 else 0,
                'cum_energy': {str(rr): float(cum[min(rr, len(s2)) - 1].item())
                               for rr in RS if 1 <= rr <= len(s2)},
                'total_energy': float(tot),
                'shift_norms': {str(c): float(shift[i].norm().item())
                                for i, c in enumerate(common)},
            }

        results['conds'][cond] = r
        del Xc, Xp, Xv, W0, Ws, R, U, s, Vh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} "
              f"/ cos(W0,W*) {r['residual']['w_cos']:.3f} / ||R||/||W*|| {r['residual']['rel_norm']:.2f}")
        sv = r['residual']['svd']
        print(f"  residual SVD: eff-rank {sv['effective_rank']:.1f}, top-4 s "
              f"{[round(v,3) for v in sv['singular_values'][:4]]}, "
              f"cumE(r): " + " ".join(f"r{rr}:{v:.2f}" for rr, v in sv['cum_energy'].items() if rr in (1,2,4,8,16)))
        print(f"  oracle residual curve mIoU(W0+R_r): " + " ".join(
            f"r{rr}:{curve[rr]:.3f}" for rr in [0,1,2,4,8,16,32] if rr in curve))
        print(f"  full-rank residual mIoU {r['curve_full']:.3f} (should == oracle)")
        fs = r['feat_shift']
        if fs:
            print(f"  feat-shift (128-d): eff-rank {fs['effective_rank']:.1f}, cumE(r): "
                  + " ".join(f"r{rr}:{v:.2f}" for rr, v in fs['cum_energy'].items() if rr in (1,2,4,8,16)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("If the oracle residual curve climbs to near-oracle with r=4-8, AL")
    print("should estimate a LOW-RANK correction (W0 + U_r C), not a full probe.")
    print("If cum_energy(r=8) ~ 0.9+ and curve(r=8) ~ oracle, the residual is")
    print("compressible -> the C21 low-rank residual decoder is the route.")

if __name__ == "__main__":
    main()
