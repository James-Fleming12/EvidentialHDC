"""al_acq_sweep_diag.py: Iteration 1 -- does better label selection compensate for
the lack of U?

Experiment A from new_iters.md. The residual-subspace closure says U (the oracle
update direction) is not obtainable from 2-8 labels. The remaining question is
whether ACTIVE LEARNING can choose labels that are maximally useful to a classifier
WITHOUT needing to reconstruct the oracle residual.

Design: fix the DOWNSTREAM UPDATE to be the same for every acquisition rule (the
normalized first-order step W1 = W0 + rho * U * G/||G|| with G = U^T X_lab^T
(Y - X_lab W0), U = oracle -- the sound part from R5). Only the ACQUISITION rule
varies: which b pool points are labeled, then fed to the same step. If a
rule's gc at b=2/4/8 beats random, better selection compensates for the budget.

Acquisition rules (all label-free, computed on the frozen probe's pool predictions
BEFORE labels are queried):
  random            baseline
  margin            lowest |top-2 logit margin| (boundary points)
  entropy           highest prediction entropy
  tta_inst          highest augmentation-prediction variance (TTA instability)
  margin_tta        margin-ranked, then TTA-variance on the candidate set
  margin_div        margin-ranked, then farthest-point diversity in 128-d
  tta_div           tta-variance-ranked, then farthest-point diversity
  margin_tta_div    combined margin + TTA + diversity
  class_pair        per-(a,b) boundary budget: lowest-margin point per pair
  egl               expected gradient length (top-2 restricted)

All rules run on a CANDIDATE pool (label-free signals on the frozen probe). The
chosen b points are then labeled (oracle), and the SAME first-order step with
oracle U is applied. This isolates: with the update form fixed, does the SELECTION
matter?

Read: if margin/TTA/class_pair beats random at b=4-8 by a real margin, active
selection compensates for the label budget even without U. If they all tie, the
bottleneck is the update (few labels cannot drive even the sound first-order step),
and Experiment B (local corrections) is the next test.

Usage:
  uv run python robust_diagnostic/al_acq_sweep_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_acq_sweep_<label>.json
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
ACQ_RULES = ['random', 'margin', 'entropy', 'tta_inst', 'margin_tta',
             'margin_div', 'tta_div', 'margin_tta_div', 'class_pair', 'egl']


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
        p.append((codes[s:s+chunk].float() @ W).argmax(1))
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


def farthest_point(feats, cand_idx, b, device):
    """Farthest-point sampling in the 128-d feature space on a candidate set."""
    cf = F.normalize(feats[cand_idx].float(), p=2, dim=1).to(device)
    torch.manual_seed(3)
    sel = [int(torch.randint(len(cand_idx), (1,)).item())]
    dist = (cf - cf[sel[0]]).norm(dim=1)
    for _ in range(b - 1):
        nxt = int(dist.argmax().item())
        sel.append(nxt)
        d2 = (cf - cf[nxt]).norm(dim=1)
        dist = torch.minimum(dist, d2)
    return cand_idx[torch.tensor(sel)]


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
    ap.add_argument("--r", type=int, default=2, help="single rank for the first-order step")
    ap.add_argument("--budgets", type=str, default="2,4,8")
    ap.add_argument("--rho_sweep", type=str, default="0.05,0.2,0.8")
    ap.add_argument("--cand_frac", type=float, default=0.05,
                    help="fraction of the pool used as the acquisition candidate set")
    ap.add_argument("--tta_augs", type=int, default=5, help="augmentation draws for TTA instability")
    ap.add_argument("--rules", type=str, default=",".join(ACQ_RULES))
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
    budgets = [int(x) for x in args.budgets.split(',')]
    rho_sweep = [float(x) for x in args.rho_sweep.split(',')]
    rules = [x.strip() for x in args.rules.split(',') if x.strip()]
    r = args.r

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'r': r, 'budgets': budgets,
               'rules': rules, 'conds': {}}

    # ---- W0 + oracle U (fixed downstream basis) ----
    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        fd, ld = extract_clean(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(fd))
        pool_f, pl = fd[perm[:args.pool_size]], ld[perm[:args.pool_size]]
        val_f, vl = fd[perm[-args.val_size:]], ld[perm[-args.val_size:]]
        Xp = hdc_codes(pool_f, proj, device).float()
        Xv = hdc_codes(val_f, proj, device).float()
        del pool_f, val_f, fd, ld
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # oracle U + oracle decoder
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_or, _ = right_topk_svd(R.t(), r)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # ---- label-free acquisition signals on the pool ----
        sm = torch.softmax(Xp.float() @ W0c, dim=1)
        top2 = torch.topk(sm, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()      # low = boundary
        entropy = -(sm * (sm + 1e-12).log()).sum(dim=1)             # high = uncertain
        a_idx = top2.indices[:, 0]; b_idx = top2.indices[:, 1]

        # candidate set: top cand_frac by low margin (the boundary region)
        n_cand = max(int(args.cand_frac * len(Xp)), 8 * max(budgets))
        cand = torch.argsort(margin)[:n_cand]
        cand_margin = margin[cand]
        cand_entropy = entropy[cand]
        cand_sm = sm[cand]
        cand_a = a_idx[cand]; cand_b = b_idx[cand]

        # TTA-instability on the candidate set: variance of the frozen probe's
        # softmax over input augmentations (additive noise on the corrupted volume).
        # Re-extract candidate features with noise -> needs the raw 128-d features;
        # approximate with HDC-code perturbation (flip a fraction of bits).
        tta_var = torch.zeros(n_cand)
        if 'tta_inst' in rules or 'margin_tta' in rules or 'tta_div' in rules or 'margin_tta_div' in rules:
            Xcand = Xp[cand].float()
            draws = []
            for _ in range(args.tta_augs):
                torch.manual_seed(100 + _)
                flip = torch.rand_like(Xcand) < 0.02
                Xa = torch.where(flip, -Xcand, Xcand)
                sa = torch.softmax(Xa @ W0c, dim=1)
                draws.append(sa)
            draws = torch.stack(draws)                      # (K, n_cand, C)
            tta_var = draws.var(dim=0).mean(dim=1)          # mean over classes

        # EGL (top-2 restricted) on the candidate set
        egl = torch.zeros(n_cand)
        if 'egl' in rules:
            # EGL_ab = p_a||e_a-p|| + p_b||e_b-p||  (x fixed, ||x||=100 for sign codes)
            for i in range(n_cand):
                p = cand_sm[i]
                a = int(cand_a[i].item()); b = int(cand_b[i].item())
                ea = torch.zeros(NUM_CLASSES); ea[a] = 1.0
                eb = torch.zeros(NUM_CLASSES); eb[b] = 1.0
                egl[i] = p[a].item() * (ea - p).norm().item() + p[b].item() * (eb - p).norm().item()

        # ---- acquisition selectors (return POOL indices) ----
        cand_feats = pool_f[perm[:args.pool_size]]      # raw 128-d features (for diversity)

        def select(rule, b):
            if rule == 'random':
                torch.manual_seed(7)
                return cand[torch.randperm(n_cand)[:b]]
            if rule == 'margin':
                return cand[torch.argsort(cand_margin)[:b]]
            if rule == 'entropy':
                return cand[torch.argsort(cand_entropy, descending=True)[:b]]
            if rule == 'tta_inst':
                return cand[torch.argsort(tta_var, descending=True)[:b]]
            if rule == 'margin_tta':
                m = cand_margin / (cand_margin.max() + 1e-8)
                v = tta_var / (tta_var.max() + 1e-8)
                score = -m + v
                return cand[torch.argsort(score, descending=True)[:b]]
            if rule == 'margin_div':
                topM = torch.argsort(cand_margin)[:8 * b]
                return farthest_point(cand_feats, cand[topM], b, device)
            if rule == 'tta_div':
                topM = torch.argsort(tta_var, descending=True)[:8 * b]
                return farthest_point(cand_feats, cand[topM], b, device)
            if rule == 'margin_tta_div':
                m = cand_margin / (cand_margin.max() + 1e-8)
                v = tta_var / (tta_var.max() + 1e-8)
                score = -m + v
                topM = torch.argsort(score, descending=True)[:8 * b]
                return farthest_point(cand_feats, cand[topM], b, device)
            if rule == 'class_pair':
                # per-(a,b) budget: lowest-margin point of each top-2 pair
                pairs = cand_a * NUM_CLASSES + cand_b
                uniq, inv = torch.unique(pairs, return_inverse=True)
                sel = []
                order = torch.argsort(cand_margin)
                for u in range(len(uniq)):
                    members = order[inv[order] == u]
                    if len(members) > 0:
                        sel.append(members[0].item())
                sel = torch.tensor(sel)
                if len(sel) < b:
                    rest = torch.argsort(cand_margin)[:b]
                    sel = torch.cat([sel, rest[~torch.isin(rest, sel)]])[:b]
                return cand[sel[:b]]
            if rule == 'egl':
                return cand[torch.argsort(egl, descending=True)[:b]]
            torch.manual_seed(7)
            return cand[torch.randperm(n_cand)[:b]]

        # ---- evaluate each rule at each budget, SAME first-order step ----
        cond_res = {'refs': refs, 'gap': float(gap), 'rules': {}}
        for rule in rules:
            entry = {}
            for b in budgets:
                sel = select(rule, b).long()
                X_lab = Xp[sel]; Y_lab = onehot(pl[sel], NUM_CLASSES)
                resid = (Y_lab.float() - X_lab.float() @ W0c)
                G = (X_lab.float() @ U_or).t() @ resid       # r x C
                Gn = G / (G.norm() + 1e-8)
                gcs = {}
                for rho in rho_sweep:
                    W1 = W0c + (U_or @ (rho * Gn))
                    d = mw(W1, Xv, vl) - refs['frozen']
                    gcs[str(rho)] = {'delta': float(d),
                                     'gap_closed': float(d / gap) if gap > 1e-9 else None}
                best = max((v['gap_closed'] or -9 for v in gcs.values()), default=None)
                entry[str(b)] = {'gc': gcs, 'best_gc': best,
                                 'n_labels': int(len(sel))}
            cond_res['rules'][rule] = entry

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, U_or
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    oracle-U reference gc: " + " ".join(f"r{b}:{v['best_gc']:+.2f}" for b, v in
              sorted(cond_res['rules'].get('random', {}).items())) + " (random baseline)")
        for rule in rules:
            line = " ".join(f"b{b}:{v['best_gc']:+.2f}" for b, v in cond_res['rules'][rule].items())
            print(f"    {rule:15s} {line}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("With the DOWNSTREAM UPDATE fixed (first-order + oracle U), only the")
    print("acquisition varies. If margin/TTA/class_pair beats random at b=4-8 by a")
    print("real margin, active selection compensates for the label budget without U.")
    print("If they all tie, the bottleneck is the update (few labels cannot drive even")
    print("the sound step), and Experiment B (local corrections) is next.")


if __name__ == "__main__":
    main()
