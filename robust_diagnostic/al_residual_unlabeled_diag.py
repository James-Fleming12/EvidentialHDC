"""al_residual_unlabeled_diag.py: does an UNLABELED basis make the low-rank
residual estimable?  Extension of al_residual_al_diag.py.

C21 showed: oracle basis U_r = SVD(R), R = W* - W0, IS estimable (oracle-basis
delta +0.05 to +0.14 at k=8), but est-basis U_r = SVD(R_sub), R_sub = W_sub - W0
with W_sub fit on the same k labels, collapses (the low-rank structure exists
but cannot be discovered from sparse labels via W_sub).

This tests whether U_r can be discovered WITHOUT labels, from unlabeled pool
structure:

  C. POOL-COVARIANCE basis: U_r = top-r eigenvectors of S = Xp^T Xp / N (the
     pool covariance in code space). No labels, just the pool geometry. The
     C21 oracle-basis ceiling is the bound; this is the deployable,
     unlabeled-basis version of the same low-rank correction.
  D. CODE-SHIFT basis: U_r = top-r left singular vectors of the per-class
     code-mean shift matrix M (17 x 10000, row c = mu_pool_c - mu_clean_c).
     Also unlabeled (needs only clean class means, no pool labels).

For each basis, per r in {1,2,4,8}, per k in {2,4,8}:
  C = (U^T X_lab^T X_lab U)^{-1} U^T X_lab^T (Y_lab - X_lab W0)
  W = W0 + U C, report delta vs frozen.

Usage:
  uv run python robust_diagnostic/al_residual_unlabeled_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_residual_unlabeled_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11
RS = [1, 2, 4, 8]
KS = [2, 4, 8]


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
    def A(v): return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); a = rs / ((p * Ap).sum(0) + 1e-30)
        x = x + a.unsqueeze(0) * p; r = r - a.unsqueeze(0) * Ap
        rsn = (r * r).sum(0); be = rsn / (rs + 1e-30); p = r + be.unsqueeze(0) * p; rs = rsn
    return x.float()


def lsq_residual(X_lab, Y_lab, W0, U, device):
    Xd = X_lab.to(device).float(); Yd = Y_lab.to(device).float(); U_d = U.to(device)
    r = U_d.shape[1]; XU = Xd @ U_d
    A = XU.t() @ XU + 1e-6 * torch.eye(r, device=device)
    b = XU.t() @ (Yd - Xd @ W0.to(device))
    C = torch.linalg.solve(A, b)
    return C.cpu()


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
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
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="med")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    results = {'label': args.label, 'method': args.method_b, 'rs': RS, 'ks': KS, 'conds': {}}
    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir): cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
        proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
        Xc = hdc_codes(fa[ci], proj, device).float()
        Xp = hdc_codes(pool, proj, device).float()
        Xv = hdc_codes(val, proj, device).float()
        Xd = Xp.to(device); N = Xp.shape[0]
        W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_full, _, _ = torch.linalg.svd(R.double(), full_matrices=False); U_full = U_full.float()
        # unlabeled bases: pool covariance top eigenvectors and code-shift SVD
        # pool covariance S = Xp^T Xp / N  (10k x 10k) -> top eigenvectors via eigh on double
        S = (Xd.double().t() @ Xd.double()) / N
        eigS, U_S = torch.linalg.eigh(S); U_S = U_S.float()  # ascending
        # top-r are the LARGEST eigenvalues -> last r columns
        # code-shift: per-class mean shift in code space
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}
        clean_classes = sorted(set(la[ci].tolist()) & set(range(1, NUM_CLASSES)))
        clean_idx = {c: (la[ci] == c).nonzero().squeeze(1) for c in clean_classes}
        # per-class code means
        mu_pool = {c: Xp[cls_idx[c]].mean(0) for c in classes if len(cls_idx[c])>0}
        mu_clean = {c: Xc[clean_idx[c]].mean(0) for c in classes if c in clean_idx and len(clean_idx[c])>0}
        common = [c for c in classes if c in mu_clean]
        if len(common) >= 2:
            M = torch.stack([mu_pool[c] - mu_clean[c] for c in common])  # ncls x 10000
            _, s_shift, Vh_shift = torch.linalg.svd(M.double(), full_matrices=False)
            U_shift = Vh_shift.t().float()  # 10000 x ncls, columns are shift directions
        else:
            U_shift = None
        r_cond = {'refs':{}, 'pool_cov':{}, 'code_shift':{}, 'oracle_ref':{}}
        r_cond['refs']['frozen'] = mw(W0, Xv, vl); r_cond['refs']['oracle'] = mw(Ws, Xv, vl)
        # oracle reference: how much does the TRUE residual with oracle U_r recover?
        for k in KS:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx) < max(50, k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx], NUM_CLASSES)
            # oracle U_r ceiling at this k
            for r in RS:
                U=U_full[:,:r]
                C=lsq_residual(X_lab, Y_lab, W0, U, device)
                W=W0.detach().cpu() + (U @ C)
                # store only r=4 and r=8 in refs for brevity, full grid in per-k
                pass
            # for brevity, store one r=4 oracle delta as the "oracle basis" number in refs
            U4=U_full[:,:4]; C4=lsq_residual(X_lab, Y_lab, W0, U4, device); W4=W0.detach().cpu()+(U4@C4)
            r_cond['refs'][f'oracle_basis_k{k}_r4']=mw(W4,Xv,vl)-r_cond['refs']['frozen']
        # main grids: pool-cov and code-shift bases, per k, per r
        for k in KS:
            lab_idx=[]
            for c in classes:
                idx=cls_idx[c]
                if len(idx) < max(50, k): continue
                torch.manual_seed(2); lab_idx.append(idx[torch.randperm(len(idx))[:k]])
            if not lab_idx: continue
            lab_idx=torch.cat(lab_idx); X_lab=Xp[lab_idx]; Y_lab=onehot(pl[lab_idx], NUM_CLASSES)
            n_labels=len(lab_idx)
            # pool covariance basis
            pc={'n_labels':n_labels}
            for r in RS:
                U=U_S[:,-r:].detach().cpu()  # top r eigenvectors (largest eigvals are at the end after eigh asc)
                C=lsq_residual(X_lab, Y_lab, W0, U, device)
                W=W0.detach().cpu() + (U @ C)
                pc[str(r)]={'miou':mw(W,Xv,vl),'delta':mw(W,Xv,vl)-r_cond['refs']['frozen']}
            r_cond['pool_cov'][str(k)]=pc
            # code-shift basis
            if U_shift is not None and U_shift.shape[1] >= 1:
                cs={'n_labels':n_labels}
                for r in RS:
                    rr=min(r, U_shift.shape[1])
                    U=U_shift[:,:rr]
                    C=lsq_residual(X_lab, Y_lab, W0, U, device)
                    W=W0.detach().cpu() + (U @ C)
                    cs[str(r)]={'miou':mw(W,Xv,vl),'delta':mw(W,Xv,vl)-r_cond['refs']['frozen']}
                r_cond['code_shift'][str(k)]=cs
        results['conds'][cond]=r_cond
        del Xc,Xp,Xv,W0,Ws,R,U_full,S,eigS,U_S
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r_cond['refs']['frozen']:.3f} / oracle {r_cond['refs']['oracle']:.3f}")
        for k in KS:
            if str(k) not in r_cond['pool_cov']: continue
            pc=r_cond['pool_cov'][str(k)]; cs=r_cond['code_shift'].get(str(k),{})
            print(f"  k={k}: pool-cov " + " ".join(f"r{r}:{pc[str(r)]['delta']:+.3f}" for r in RS) +
                  (" | code-shift " + " ".join(f"r{r}:{cs[str(r)]['delta']:+.3f}" for r in RS) if cs else ""))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w') as fh: json.dump(results,fh,indent=2,default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Pool-cov: does the UNLABELED top-r covariance capture the residual?")
    print("Code-shift: does the per-class mean shift subspace capture it?")
    print("Positive delta at k=2-4, r=4-8 => unlabeled basis is viable -> C21 fix.")

if __name__=="__main__": main()
