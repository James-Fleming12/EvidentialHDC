"""al_propagation_geometry_diag.py: what FEATURE-SPACE property predicts the
per-class mean error of the propagated-mean decoder?

The method decomposition (Iteration 12) found the propagated MEANS are the
dominant cost on crosstalk (+0.42 gc), and the fix arms (F2/F3) failed. The key
unresolved question: is the per-class mean error driven by ASSIGNMENT
CONTAMINATION (fixable by a better label rule) or by the INTRINSIC GEOMETRY of
the feature space (needs a different estimator, e.g. shrinkage or mean-shift)?

The decisive control -- the CORRECT-ASSIGNMENT-ONLY mean. For each class:
    M_prop_all[c]  = mean over ALL points the propagation assigns to c (current)
    M_correct[c]   = mean over the points assigned to c that are CORRECTLY
                     assigned (true class = c)
    M_star[c]      = oracle mean
  If whitened(M_correct - M_star) ~ 0 but whitened(M_prop_all - M_star) is
  large  -> the error is CONTAMINATION (wrong assignments pull the mean):
             a better assignment rule fixes it.
  If whitened(M_correct - M_star) is ALSO large -> even CORRECT labels give a
             bad mean: INTRINSIC geometry / code-space saturation: a better
             rule cannot fix it (need a different mean estimator).

The geometry properties measured per class (to find what tracks the correct-
label mean error):
    intra_cos   mean cosine of class-c points to their class centroid (how
                fat/tight the class is in the 128-d space)
    inter_cos   mean cosine of class-c points to the NEAREST OTHER class
                centroid (class separation)
    mass        number of pool points in the class
    mode_frac   k-means (K_c=4 within the class) dominant-cluster fraction
                (mono-modal if ~1.0, multi-modal if << 1)
    code_intra  intra-class cosine in the 10000-d CODE space (the saturation
                reference -- if the code space saturates, the code-space
                aggregation may cap the mean quality regardless of labels)

Also reported:
    the contamination CONFUSION: for each class, which TRUE classes the wrong
    assignments come from (does the contamination come from a specific
    confusable neighbor, which a pair-aware rule could fix?)
    the contamination DISTANCE: are the wrongly-assigned points near the class
    core (harmless) or far outliers (damaging)?

Decisive reads:
  M_correct ~ M_star everywhere          -> contamination-driven: better rules
                                            (pair-aware, weighted) will help.
  M_correct far from M_star on the same  -> geometry-driven: no rule fixes it;
  classes that have high M_prop error       need shrinkage / mean-shift / a
                                            non-mean estimator.
  Which geometry property correlates with the correct-label mean error tells
  us WHICH estimator to build (shrinkage for high variance, mode-split for
  multi-modal, code-space fix for saturation).

Usage:
  uv run python robust_diagnostic/al_propagation_geometry_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_propagation_geometry_dglsspp.json
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


def whitened_err(X, lam, iters, m, device, v, ref):
    """||Sigma^-1 (v - ref)|| / ||Sigma^-1 ref|| for single code-space vectors."""
    Bv = (v - ref).unsqueeze(1)
    Br = ref.unsqueeze(1)
    Wv = solve_whitened(X, Bv, lam, iters, m, device).cpu()[:, 0]
    Wr = solve_whitened(X, Br, lam, iters, m, device).cpu()[:, 0]
    return float(Wv.norm().item() / (Wr.norm().item() + 1e-12))


def cos_(a, b):
    return float((a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12))


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
    ap.add_argument("--pool_size", type=int, default=20000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=30000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--b_anchors", type=str, default="8")
    ap.add_argument("--loose_mult", type=float, default=3.0)
    ap.add_argument("--k_modes", type=int, default=4)
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
               'loose_mult': args.loose_mult, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    lac = la[ci]
    W0 = ridge_fit_soft(Xc, onehot(lac, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
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

        M_star, C_star = class_means(Xp, pl, NUM_CLASSES)

        pf = F.normalize(pool_f.float(), p=2, dim=1)
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}

        cond_res = {'refs': refs, 'gap': float(gap), 'budgets': {}}

        for b in b_anchors:
            # mass+loose selection (the validated selection from the method diag)
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
            anc_f = pf[anc]; anc_lab = pl[anc]
            nn = (pf @ anc_f.t()).argmax(1)
            prop_lab = anc_lab[nn]

            # per-class: full propagated mean, correct-only mean, geometry
            per_class = {}
            for c in range(1, NUM_CLASSES):
                if C_star[c] < 50:
                    continue
                asg = prop_lab == c
                n_asg = int(asg.sum().item())
                if n_asg == 0:
                    continue
                M_prop_all = Xp[asg].float().mean(dim=0)
                correct = asg & (pl == c)
                n_corr = int(correct.sum().item())
                M_correct = Xp[correct].float().mean(dim=0) if n_corr > 0 else None
                # geometry (128-d)
                idx_c = class_idx[c]
                ctr = pf[idx_c].mean(dim=0)
                ctr = ctr / (ctr.norm() + 1e-12)
                intra = float((pf[idx_c] * ctr).sum(dim=1).mean().item())
                # nearest other-class centroid
                best_other = -1.0; best_cls = -1
                for d in range(1, NUM_CLASSES):
                    if d == c or C_star[d] < 50:
                        continue
                    ctr_d = pf[class_idx[d]].mean(dim=0)
                    ctr_d = ctr_d / (ctr_d.norm() + 1e-12)
                    s = float((ctr * ctr_d).item())
                    if s > best_other:
                        best_other = s; best_cls = d
                # code-space intra (saturation reference) -- reported via
                # Werr (already code-space); the 128-d intra_cos is the
                # informative geometry
                pc = {
                    'n': int(C_star[c].item()),
                    'assign_prec': float((prop_lab[idx_c] == c).float().mean().item()),
                    'n_asg': n_asg, 'n_correct': n_corr,
                    'Werr_all': whitened_err(Xp, args.lam, args.cg_iters, args.nystrom_m, device,
                                             M_prop_all, M_star[c]),
                    'Werr_correct': whitened_err(Xp, args.lam, args.cg_iters, args.nystrom_m, device,
                                                 M_correct, M_star[c]) if M_correct is not None else None,
                    'intra_cos': intra, 'nearest_other': best_cls,
                    'inter_cos': best_other,
                }
                # contamination confusion: which true classes the wrong
                # assignments come from
                wrong = asg & (pl != c)
                if int(wrong.sum().item()) > 0:
                    src = torch.bincount(pl[wrong], minlength=NUM_CLASSES)[1:]
                    top = torch.argsort(src, descending=True)[:3]
                    pc['contam_src'] = [int(t + 1) for t in top]
                    pc['contam_frac'] = float(int(wrong.sum().item()) / max(n_asg, 1))
                    # contamination distance: mean 128-d cos of wrong points to
                    # class centroid vs correct points
                    wf = pf[wrong]; cf = pf[correct] if n_corr > 0 else None
                    pc['contam_centroid_cos'] = float((wf * ctr).sum(dim=1).mean().item())
                    if cf is not None:
                        pc['correct_centroid_cos'] = float((cf * ctr).sum(dim=1).mean().item())
                per_class[str(c)] = pc

            # the decisive read: correlation across classes of
            #   assign_prec vs Werr_all   (is error driven by contamination?)
            #   Werr_correct vs Werr_all  (does correct-only fix the error?)
            precs = [v['assign_prec'] for v in per_class.values()]
            werr_a = [v['Werr_all'] for v in per_class.values()]
            werr_c = [v['Werr_correct'] for v in per_class.values() if v['Werr_correct'] is not None]
            corr_pw = float(torch.corrcoef(torch.stack([torch.tensor(precs, dtype=torch.float),
                                                        torch.tensor(werr_a, dtype=torch.float)]))[0, 1].item()) \
                if len(precs) > 1 else None
            mean_werr_a = sum(werr_a) / len(werr_a) if werr_a else None
            mean_werr_c = sum(werr_c) / len(werr_c) if werr_c else None
            cond_res['budgets'][str(b)] = {
                'per_class': per_class,
                'corr_prec_Werr': corr_pw,
                'mean_Werr_all': mean_werr_a,
                'mean_Werr_correct_only': mean_werr_c,
                'n_labels': int(len(anc))}
            print(f"  b{b} (n={int(len(anc))}): mean Werr_all {mean_werr_a:.2f} "
                  f"Werr_correct {mean_werr_c:.2f} | corr(prec, Werr) {corr_pw:+.2f}")

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, M_star, pool_f, pf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("THE DECISIVE CONTROL (per class):")
    print("  Werr_all     = whitened error of the full propagated mean (current)")
    print("  Werr_correct = whitened error of the CORRECT-ONLY mean")
    print("  -> if Werr_correct ~ 0 while Werr_all large: the error is")
    print("     CONTAMINATION (wrong assignments) -> a better rule fixes it.")
    print("  -> if Werr_correct is ALSO large: the error is INTRINSIC GEOMETRY")
    print("     / code-space saturation -> a rule cannot fix it; need a")
    print("     different mean estimator (shrinkage, mode-split, 128-d).")
    print("  mean_Werr_all vs mean_Werr_correct_only across classes = the split.")
    print("  corr(assign_prec, Werr_all): if NEGATIVE, low-precision classes have")
    print("     high error (contamination); if ~0, precision is not the driver.")
    print("Geometry (per class): intra_cos (fat/tight), inter_cos + nearest_other")
    print("  (separation), n (mass). The property that tracks Werr_correct tells")
    print("  us WHICH estimator to build.")
    print("Contamination: contam_src (which true classes the wrong assignments")
    print("  come from -- a pair-aware rule could fix a specific confusable),")
    print("  contam_centroid_cos vs correct_centroid_cos (are wrong points near")
    print("  the core, harmless, or far outliers, damaging?).")


if __name__ == "__main__":
    main()
