"""al_propagation_method_diag.py: the METHOD decomposition -- which part of the
propagated-mean decoder fails, and what to add.

The improvement axes (Iteration 11) established: (a) A1 (true means x prop
counts) exceeds gcP, so the propagated MEANS lose ~0.36 gc at fixed counts;
(b) A1 still does NOT reach W_mean_oracle, so the COUNT REFERENCE also
contributes; (c) the update has no headroom; (d) mass-stratified anchors (B2)
is the one lever with real gain. This diagnostic goes deeper: for each anchor
SELECTION rule, it decomposes the mechanism into its measured pieces so we know
exactly what to add.

For each selection rule (random / mass-stratified B2 / loose-boost A4 / B2+A4),
the full 2x2 accounting (means x counts):

  gc_prop       M_prop x C_prop      (the method as-is)
  gc_oracle_M   M_star x C_prop      (TRUE means, prop counts -- A1: isolates
                                     the MEAN error at fixed counts)
  gc_oracle_C   M_prop x C_star      (prop means, ORACLE counts -- isolates the
                                     COUNT error at fixed means)
  gc_both       M_star x C_star      (the pool oracle, ~ W_mean_oracle)

  -> gc_oracle_M - gc_prop     = the MEAN error cost
  -> gc_oracle_C - gc_prop     = the COUNT error cost
  -> gc_both - gc_oracle_M     = the count-REFERENCE cost (C_prop vs C_star)
  This tells us which piece is the real gap under EACH selection rule.

Then the FIX arms (what to add):
  F1  count-reference correction: mass-corrected counts from the anchors'
      observed proportions (the Iteration-8 lever) -- does fixing the count
      reference close gc_prop -> gc_oracle_M?
  F2  per-class mean correction: for the classes with the largest whitened mean
      error (from the per-class breakdown), replace M_prop_c with the
      frozen-pseudo mean M_pseudo_c (a pool-stable prior) -- targeted, not
      global shrinkage (C3's global version hurt).
  F3  B2+A4 selection (mass + loose-boost) with the F1 count fix -- the
      combined candidate method.

Per-class diagnostics (per selection rule, per budget):
  - per-class propagated precision (as before)
  - per-class WHITENED mean error ||Sigma^-1 (M_prop_c - M_star_c)|| / ||
    Sigma^-1 M_star_c|| -- WHICH classes carry the mean error (the fix target)
  - per-class count error

Decisive reads:
  gc_oracle_M ~ gcP (mean error ~0) -> the means are fine; the gap is counts
  gc_oracle_C ~ gcP (count error ~0) -> the counts are fine; the gap is means
  F1 ~ gc_oracle_M                   -> the count fix closes the count-reference
                                       gap; use it
  F2 > gc_prop                       -> per-class mean correction helps; use it
  F3 (combined) best                 -> assemble the candidate method

Usage:
  uv run python robust_diagnostic/al_propagation_method_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_propagation_method_dglsspp.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11
LOOSE = {7, 13, 14}
RULES = ['random', 'mass', 'loose', 'mass_loose']


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


def solve_whitened(X, B, lam, iters, m, device):
    X = X.to(device); B = B.float().to(device)
    torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1], max(1, X.shape[0] - 1))
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    w0 = P @ torch.linalg.solve(Shat, P.t() @ B)
    if B.shape[0] <= 8:
        return w0.float()
    x = w0; b = B
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
            return w0.float()
    return x.float()


def class_means(X, y, nc):
    M = torch.zeros(nc, X.shape[1]); C = torch.zeros(nc)
    for c in range(nc):
        m = (y == c)
        if int(m.sum().item()) > 0:
            M[c] = X[m].mean(dim=0)
            C[c] = float(int(m.sum().item()))
    return M, C


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync(); return time.time()


def toc(t0):
    sync(); return time.time() - t0


def cos_(a, b):
    return float((a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--pool_size", type=int, default=20000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=30000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--b_anchors", type=str, default="2,4,8,16")
    ap.add_argument("--loose_mult", type=float, default=3.0)
    ap.add_argument("--count_alpha", type=float, default=1.0,
                    help="count-correction strength (F1): C_corr = (C_prop + a*C_anchor)/(1+a)")
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
    b_anchors = [int(x) for x in args.b_anchors.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_anchors': b_anchors,
               'loose_mult': args.loose_mult, 'count_alpha': args.count_alpha, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    lac = la[ci]
    W0 = ridge_fit_soft(Xc, onehot(lac, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
    M0, C0 = class_means(Xc, lac, NUM_CLASSES)
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
        del val_f, fd, ld
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        def gc(mi):
            return (mi - refs['frozen']) / gap if gap > 1e-9 else None

        M_star, C_star = class_means(Xp, pl, NUM_CLASSES)
        W_mean_oracle = solve_whitened(Xp, (M_star * C0.unsqueeze(1)).t().contiguous(),
                                       args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        gc_mean_oracle = gc(mw(W_mean_oracle, Xv, vl))

        pf = F.normalize(pool_f.float(), p=2, dim=1)

        Lp = Xp.float() @ W0c
        pred = Lp.argmax(1)
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        M_pseudo, _ = class_means(Xp, pred, NUM_CLASSES)

        cond_res = {'refs': refs, 'gap': float(gap),
                    'gc_mean_oracle': gc_mean_oracle, 'rules': {}}

        for rule in RULES:
            cond_res['rules'][rule] = {}
            for b in b_anchors:
                # ---- anchor selection per rule ----
                if rule == 'random':
                    torch.manual_seed(7)
                    anc = torch.cat([class_idx[c][torch.randperm(len(class_idx[c]))[:min(b, len(class_idx[c]))]]
                                     for c in range(1, NUM_CLASSES) if len(class_idx[c]) > 0])
                elif rule == 'mass':
                    torch.manual_seed(9)
                    mass = torch.tensor([float(len(class_idx[c])) for c in range(1, NUM_CLASSES)])
                    alloc = (mass / mass.sum() * b * (NUM_CLASSES - 1)).int().clamp(min=1)
                    pieces = []
                    for i, c in enumerate(range(1, NUM_CLASSES)):
                        idx = class_idx[c]
                        pieces.append(idx[torch.randperm(len(idx))[:int(min(alloc[i].item(), len(idx)))]])
                    anc = torch.cat(pieces)
                elif rule == 'loose':
                    torch.manual_seed(7)
                    pieces = []
                    for c in range(1, NUM_CLASSES):
                        idx = class_idx[c]
                        nb = int(b * (args.loose_mult if c in LOOSE else 1.0))
                        pieces.append(idx[torch.randperm(len(idx))[:min(nb, len(idx))]])
                    anc = torch.cat(pieces)
                elif rule == 'mass_loose':
                    torch.manual_seed(9)
                    mass = torch.tensor([float(len(class_idx[c])) for c in range(1, NUM_CLASSES)])
                    alloc = (mass / mass.sum() * b * (NUM_CLASSES - 1)).int().clamp(min=1)
                    pieces = []
                    for i, c in enumerate(range(1, NUM_CLASSES)):
                        idx = class_idx[c]
                        nb = int(alloc[i].item())
                        if c in LOOSE:
                            nb = int(nb * args.loose_mult)
                        nb = min(nb, len(idx))
                        pieces.append(idx[torch.randperm(len(idx))[:nb]])
                    anc = torch.cat(pieces)

                # ---- propagate ----
                anc_f = pf[anc]; anc_lab = pl[anc]
                nn = (pf @ anc_f.t()).argmax(1)
                prop_lab = anc_lab[nn]
                M_p, C_p = class_means(Xp, prop_lab, NUM_CLASSES)
                # anchor-observed proportions (for F1)
                anc_prop = torch.zeros(NUM_CLASSES)
                for c in range(1, NUM_CLASSES):
                    anc_prop[c] = float((anc_lab == c).float().mean().item())
                if anc_prop.sum().item() > 1e-9:
                    anc_prop = anc_prop / anc_prop.sum().item()

                def dec(M, C):
                    B = (M * C.unsqueeze(1)).t().contiguous()
                    W = solve_whitened(Xp, B, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
                    return gc(mw(W, Xv, vl))

                # ---- the 2x2 decomposition ----
                gc_prop = dec(M_p, C_p)
                gc_oracle_M = dec(M_star, C_p)      # A1
                gc_oracle_C = dec(M_p, C_star)
                gc_both = dec(M_star, C_star)

                # ---- F1 count-reference fix ----
                C_fix = C_p.clone()
                C_fix[1:] = (C_p[1:] + args.count_alpha * (anc_prop[1:] * C_p.sum().item())) / (1 + args.count_alpha)
                gc_F1 = dec(M_p, C_fix)
                # F1b: oracle-count-free, use the propagated counts only but
                # corrected toward a smooth prior (pool-uniform-ish)
                C_smooth = C_p.clone()
                C_smooth[1:] = (C_p[1:] + args.count_alpha * (C_p.sum().item() / (NUM_CLASSES - 1))) / (1 + args.count_alpha)
                gc_F1b = dec(M_p, C_smooth)

                # ---- F2 per-class mean correction (target the whitened-error
                #      classes, replace with the pseudo-mean) ----
                whit_err = {}
                for c in range(1, NUM_CLASSES):
                    if C_star[c] < 10:
                        continue
                    Bc = (M_p[c] - M_star[c]).unsqueeze(1)
                    Wd = solve_whitened(Xp, Bc, args.lam, args.cg_iters, args.nystrom_m, device).cpu()[:, 0]
                    Bs = M_star[c].unsqueeze(1)
                    Ws_c = solve_whitened(Xp, Bs, args.lam, args.cg_iters, args.nystrom_m, device).cpu()[:, 0]
                    whit_err[str(c)] = float(Wd.norm().item() / (Ws_c.norm().item() + 1e-12))
                M_f2 = M_p.clone()
                # replace the worst-4 classes by their pseudo-mean
                worst = sorted(whit_err.items(), key=lambda kv: -kv[1])[:4]
                for c, _ in worst:
                    M_f2[int(c)] = M_pseudo[int(c)]
                gc_F2 = dec(M_f2, C_p)

                # ---- F3 combined ----
                gc_F3 = dec(M_f2, C_fix)

                # ---- per-class precision + count error ----
                per_class = {}
                for c in range(1, NUM_CLASSES):
                    m = (pl == c)
                    if int(m.sum().item()) < 50:
                        continue
                    per_class[str(c)] = float((prop_lab[m] == pl[m]).float().mean().item())
                count_err = float((C_p[1:] - C_star[1:]).abs().sum().item() /
                                  (C_star[1:].sum().item() + 1e-12))

                cond_res['rules'][rule][str(b)] = {
                    'n_labels': int(len(anc)),
                    'gc_prop': gc_prop, 'gc_oracle_M': gc_oracle_M,
                    'gc_oracle_C': gc_oracle_C, 'gc_both': gc_both,
                    'gc_F1': gc_F1, 'gc_F1b': gc_F1b,
                    'gc_F2': gc_F2, 'gc_F3': gc_F3,
                    'mean_error_cost': (gc_oracle_M - gc_prop) if gc_prop is not None else None,
                    'count_error_cost': (gc_oracle_C - gc_prop) if gc_prop is not None else None,
                    'count_ref_cost': (gc_both - gc_oracle_M) if gc_oracle_M is not None else None,
                    'per_class_prec': per_class, 'count_err': count_err,
                    'worst_whitened_classes': [int(c) for c, _ in worst]}

                print(f"  [{rule:9s}] b{b} (n={int(len(anc))}): gcP {gc_prop:+.2f} "
                      f"orM {gc_oracle_M:+.2f} orC {gc_oracle_C:+.2f} both {gc_both:+.2f} | "
                      f"F1 {gc_F1:+.2f} F1b {gc_F1b:+.2f} F2 {gc_F2:+.2f} F3 {gc_F3:+.2f}")

        results['conds'][cond] = cond_res
        del Ws, M_star, pool_f, pf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    W_mean_oracle gc {gc_mean_oracle:+.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("THE 2x2 DECOMPOSITION (per rule per budget):")
    print("  gc_oracle_M - gc_prop = MEAN error cost (true means, prop counts)")
    print("  gc_oracle_C - gc_prop = COUNT error cost (prop means, oracle counts)")
    print("  gc_both - gc_oracle_M = count-REFERENCE cost (C_prop vs C_star)")
    print("  -> whichever cost dominates tells us WHAT TO ADD.")
    print("F1  count-reference fix (toward anchor proportions)  ~ gc_oracle_M -> use it")
    print("F2  per-class mean fix (worst whitened-error classes -> pseudo-mean)")
    print("F3  F1+F2 combined (the candidate method)")
    print("per_class_prec / count_err / worst_whitened_classes = the fix targets")


if __name__ == "__main__":
    main()
