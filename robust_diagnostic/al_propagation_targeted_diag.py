"""al_propagation_targeted_diag.py: the two NEVER-TESTED levers on the
PROPAGATED-MEAN decoder (the strongest few-label mechanism, +0.26/+0.45 gc on
DGLSS++ fog/crosstalk at ~112-125 labels).

Everything else in the arc has been measured or closed: the mean estimator axes
(A1-A5), the update axes (C1-C3), and the selection axes B1/B2/B3 (improve
diag). The decision-correction family is bounded below propagation (8B branch).
Two levers remain untested:

A. TARGETED ANCHOR ACQUISITION (the P3 idea). P3 (8B branch) showed the frozen
   error mass is concentrated in the driveable_surface/sidewalk/terrain pairs
   (11-13, 11-14): 40% of ALL crosstalk frozen errors in the top-2 pairs. Prior
   selection only tried random (gcP), high-margin (B3, negative), confidence
   (B1, negative/weak), and mass-stratified (B2). NEVER tried: spending labels
   AT the decision boundary where the frozen decoder is wrong.
     A1 p3_margin: per-class LOWEST-margin anchors (label the frontier points;
        label-free, deployable). The complement of B3 (highest margin, negative).
     A4 p3_prior: class budget concentrated on {11,13,14} (the P3 pairs) at fixed
        total budget (a fixed prior, deployable without labels).
     A2 err_alloc: class budget proportional to per-class FROZEN-ERROR mass on
        the pool (the allocation CEILING; needs pool labels -> oracle arm).
     A3 both: err_alloc budget + within-class margin selection (ceiling of
        targeted acquisition).
   Baselines: random gcP (current method), mass-stratified B2.

B. COMPOSITION: the two positive mechanisms STACKED, sharing the same labeled
   set (the anchors). Stage 1: the propagated-mean decoder W_r (the +0.26/+0.45
   gc). Stage 2: the feature-conditioned calibration (the 8B +0.12 gc) with the
   target re-measured against W_r at the anchors: r = Y - softmax(X @ W_r),
   applied to the propagated logits Lv_r = Xv @ W_r. Also reports the
   calibration-ALONE arm (8B reproduction, target vs W0) for the additivity
   comparison, and shuffled-label nulls for both.

Decisive reads:
  A1/A4 > gcP by >= +0.1   -> acquisition IS the lever; spend labels at the
                              decision boundary (A4 deployable, A1 deployable)
  A2/A3 > gcP but A1/A4 ~ gcP -> the gain needs the label-error oracle (no
                              deployable rule; P3's prior is not enough)
  gc_comp > gcP + gc_cal_alone  -> composition compounds (the two mechanisms
                              are complementary, not overlapping)
  gc_comp ~ gcP           -> calibration does not add on top of propagation
  gc_comp ~ gc_comp_shuf  -> the composition gain is noise

DGLSS++ only (fog/crosstalk). Usage:
  uv run python robust_diagnostic/al_propagation_targeted_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_propagation_targeted_dglsspp.json
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
P3_CLASSES = {11, 13, 14}          # driveable_surface, sidewalk, terrain


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


def featurize(X, W0c, proto_clean, args, knn_ref=None, knn_lab=None):
    """The 8B label-free feature set (frozen decoder W0c)."""
    L = X.float() @ W0c
    sm = torch.softmax(L, dim=1)
    top2 = torch.topk(L, 2, dim=1)
    margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
    ent = -(sm * (sm + 1e-12).log()).sum(1)
    conf = sm.max(1).values
    p1 = top2.values[:, 0]; p2 = top2.values[:, 1]
    pred = L.argmax(1)
    draws = []
    for _ in range(args.tta_augs):
        torch.manual_seed(100 + _)
        flip = torch.rand_like(X) < 0.02
        draws.append(torch.softmax(torch.where(flip, -X, X) @ W0c, dim=1))
    draws = torch.stack(draws)
    tta_var = draws.var(dim=0).mean(1)
    tta_agree = (draws.argmax(2) == pred.unsqueeze(0)).float().mean(0)
    Xn = F.normalize(X.float(), p=2, dim=1)
    psim = Xn @ proto_clean.t()
    proto_pred = psim.argmax(1)
    proto_dist = 1.0 - psim.gather(1, pred.unsqueeze(1)).squeeze(1)
    proto_dis = (proto_pred != pred).float()
    if knn_ref is not None:
        sub = torch.randperm(len(knn_ref))[:args.knn_sub]
        ref = F.normalize(knn_ref[sub].float(), p=2, dim=1)
        ref_lab = knn_lab[sub]
        dk = torch.zeros(len(X)); ld_ = torch.zeros(len(X))
        for s in range(0, len(X), 5000):
            sim = Xn[s:s+5000] @ ref.t()
            tk = torch.topk(sim, args.knn, dim=1)
            dk[s:s+5000] = 1.0 - tk.values.mean(1)
            rl = ref_lab[tk.indices]
            ld_[s:s+5000] = (rl != pred[s:s+5000].unsqueeze(1)).float().mean(1)
    else:
        dk = torch.zeros(len(X)); ld_ = torch.zeros(len(X))
    tta_pred = draws.mean(0).argmax(1)
    clf_div = ((proto_pred != pred).float() + (tta_pred != pred).float())
    Fm = torch.stack([margin, ent, conf, p1, p2, tta_var, tta_agree,
                      proto_dist, proto_dis, dk, ld_, clf_div], dim=1)
    return Fm, pred, sm, L


def calibrate(lab, Xp, pl, Wt, Fp_s, Fv_s, Lv_base):
    """Fit per-class correction of r = Y - softmax(X @ Wt) at the anchored
    points, apply to Lv_base (val logits). Returns corrected logits."""
    Y_lab = onehot(pl[lab], NUM_CLASSES).float()
    r_lab = Y_lab - torch.softmax(Xp[lab].float() @ Wt, dim=1)
    Lv_c = Lv_base.clone()
    for c in range(1, NUM_CLASSES):
        A = torch.cat([Fp_s[lab], torch.ones(len(lab), 1)], dim=1)
        sol = torch.linalg.lstsq(A.double(), r_lab[:, c].double().unsqueeze(1)).solution
        Av = torch.cat([Fv_s, torch.ones(len(Fv_s), 1)], dim=1)
        Lv_c[:, c] += (Av.double() @ sol).squeeze(1)
    return Lv_c


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
    ap.add_argument("--tta_augs", type=int, default=5)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--knn_sub", type=int, default=10000)
    ap.add_argument("--b_anchors", type=str, default="2,8")
    ap.add_argument("--p3_mult", type=float, default=3.0, help="per-class anchor mult for the P3 classes (A4)")
    ap.add_argument("--p3_classes", type=str, default="11,13,14")
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
    p3_classes = {int(x) for x in args.p3_classes.split(',')}

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_anchors': b_anchors,
               'p3_classes': sorted(p3_classes), 'p3_mult': args.p3_mult, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    lac = la[ci]
    W0 = ridge_fit_soft(Xc, onehot(lac, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
    _, C0 = class_means(Xc, lac, NUM_CLASSES)
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # clean prototypes for the feature set (8B setup)
    torch.manual_seed(1)
    pci = torch.randperm(len(fa))[:50000]
    Xc_full = hdc_codes(fa[pci], proj, device).float()
    proto_clean = torch.zeros(NUM_CLASSES, Xc_full.shape[1])
    for c in range(1, NUM_CLASSES):
        m = (la[pci] == c)
        if int(m.sum().item()) > 50:
            proto_clean[c] = Xc_full[m].mean(dim=0)
    proto_clean = F.normalize(proto_clean.float(), p=2, dim=1)
    del Xc_full
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

        # rich features (8B), once per cond
        Fp, pred_p, sm_p, Lp = featurize(Xp, W0c, proto_clean, args, knn_ref=Xp, knn_lab=pl)
        Fv, pred_v, sm_v, Lv = featurize(Xv, W0c, proto_clean, args, knn_ref=Xp, knn_lab=pl)
        Fp_cat = torch.cat([Fp, onehot(pred_p, NUM_CLASSES)], dim=1)
        Fv_cat = torch.cat([Fv, onehot(pred_v, NUM_CLASSES)], dim=1)
        mu = Fp_cat.mean(0); sd = Fp_cat.std(0).clamp(min=1e-8)
        Fp_s = (Fp_cat - mu) / sd
        Fv_s = (Fv_cat - mu) / sd

        margin = Fp[:, 0]
        pf = F.normalize(pool_f.float(), p=2, dim=1)
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        present = [c for c in range(1, NUM_CLASSES) if len(class_idx[c]) > 0]
        err_per_class = torch.zeros(NUM_CLASSES)
        for c in present:
            err_per_class[c] = float((pred_p[class_idx[c]] != c).float().sum().item())

        def propagate(anchors):
            anc_f = pf[anchors]; anc_lab = pl[anchors]
            nn = (pf @ anc_f.t()).argmax(1)
            return anc_lab[nn]

        def decoder(M, C):
            B = (M * C.unsqueeze(1)).t().contiguous()
            W = solve_whitened(Xp, B, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
            return gc(mw(W, Xv, vl))

        cond_res = {'refs': refs, 'gap': float(gap), 'gc_mean_oracle': gc_mean_oracle,
                    'present_classes': present, 'budgets': {}}

        for b in b_anchors:
            total_b = b * len(present)
            # random (the current method / gcP)
            torch.manual_seed(7)
            anc_rand = torch.cat([class_idx[c][torch.randperm(len(class_idx[c]))[:min(b, len(class_idx[c]))]]
                                  for c in present])
            # mass-stratified (B2 reference)
            torch.manual_seed(9)
            mass = torch.tensor([float(len(class_idx[c])) for c in present])
            alloc_mass = (mass / mass.sum() * total_b).int().clamp(min=1)
            anc_mass = torch.cat([class_idx[c][torch.randperm(len(class_idx[c]))[:int(min(alloc_mass[i].item(), len(class_idx[c])))]]
                                  for i, c in enumerate(present)])
            # A1 p3_margin: per-class LOWEST-margin anchors (the decision frontier)
            anc_margin = torch.cat([class_idx[c][torch.argsort(margin[class_idx[c]])[:min(b, len(class_idx[c]))]]
                                    for c in present])
            # A4 p3_prior: p3_mult budget on the P3 classes at fixed total
            torch.manual_seed(7)
            anc_p3 = []
            for c in present:
                nb = int(b * (args.p3_mult if c in p3_classes else 1.0))
                anc_p3.append(class_idx[c][torch.randperm(len(class_idx[c]))[:min(nb, len(class_idx[c]))]])
            anc_p3 = torch.cat(anc_p3)
            # A2 err_alloc: budget proportional to frozen-error mass (oracle allocation)
            torch.manual_seed(9)
            errv = err_per_class[present].clamp(min=1e-6)
            alloc_err = (errv / errv.sum() * total_b).int().clamp(min=1)
            anc_err = torch.cat([class_idx[c][torch.randperm(len(class_idx[c]))[:int(min(alloc_err[i].item(), len(class_idx[c])))]]
                                 for i, c in enumerate(present)])
            # A3 both: err_alloc budget + within-class margin selection
            anc_both = torch.cat([class_idx[c][torch.argsort(margin[class_idx[c]])[:int(min(alloc_err[i].item(), len(class_idx[c])))]]
                                  for i, c in enumerate(present)])

            prop_rand = propagate(anc_rand)
            M_r, C_r = class_means(Xp, prop_rand, NUM_CLASSES)
            gcP = decoder(M_r, C_r)
            gcB2 = decoder(*class_means(Xp, propagate(anc_mass), NUM_CLASSES))
            gcA1 = decoder(*class_means(Xp, propagate(anc_margin), NUM_CLASSES))
            gcA4 = decoder(*class_means(Xp, propagate(anc_p3), NUM_CLASSES))
            gcA2 = decoder(*class_means(Xp, propagate(anc_err), NUM_CLASSES))
            gcA3 = decoder(*class_means(Xp, propagate(anc_both), NUM_CLASSES))

            # ---- PART B: composition on the random anchors (shared labeled set) ----
            W_r = solve_whitened(Xp, (M_r * C_r.unsqueeze(1)).t().contiguous(),
                                 args.lam, args.cg_iters, args.nystrom_m, device).cpu()
            Lv_r = Xv.float() @ W_r
            # calibration ALONE (8B reproduction: target vs W0, base Lv)
            Lv_cal = calibrate(anc_rand, Xp, pl, W0c, Fp_s, Fv_s, Lv)
            gc_cal_alone = gc(compute_miou(Lv_cal.argmax(1), vl))
            torch.manual_seed(9)
            r_lab0 = onehot(pl[anc_rand], NUM_CLASSES).float() - \
                torch.softmax(Xp[anc_rand].float() @ W0c, dim=1)
            r_shuf0 = r_lab0[torch.randperm(len(r_lab0))]
            Lv_sh0 = Lv.clone()
            for c in range(1, NUM_CLASSES):
                A = torch.cat([Fp_s[anc_rand], torch.ones(len(anc_rand), 1)], dim=1)
                sol = torch.linalg.lstsq(A.double(), r_shuf0[:, c].double().unsqueeze(1)).solution
                Av = torch.cat([Fv_s, torch.ones(len(Fv_s), 1)], dim=1)
                Lv_sh0[:, c] += (Av.double() @ sol).squeeze(1)
            gc_cal_alone_shuf = gc(compute_miou(Lv_sh0.argmax(1), vl))
            # COMPOSITION: target vs W_r, base Lv_r
            Lv_comp = calibrate(anc_rand, Xp, pl, W_r, Fp_s, Fv_s, Lv_r)
            gc_comp = gc(compute_miou(Lv_comp.argmax(1), vl))
            torch.manual_seed(11)
            r_labR = onehot(pl[anc_rand], NUM_CLASSES).float() - \
                torch.softmax(Xp[anc_rand].float() @ W_r, dim=1)
            r_shufR = r_labR[torch.randperm(len(r_labR))]
            Lv_shR = Lv_r.clone()
            for c in range(1, NUM_CLASSES):
                A = torch.cat([Fp_s[anc_rand], torch.ones(len(anc_rand), 1)], dim=1)
                sol = torch.linalg.lstsq(A.double(), r_shufR[:, c].double().unsqueeze(1)).solution
                Av = torch.cat([Fv_s, torch.ones(len(Fv_s), 1)], dim=1)
                Lv_shR[:, c] += (Av.double() @ sol).squeeze(1)
            gc_comp_shuf = gc(compute_miou(Lv_shR.argmax(1), vl))

            cond_res['budgets'][str(b)] = {
                'n_labels': {'rand': int(len(anc_rand)), 'mass': int(len(anc_mass)),
                             'margin': int(len(anc_margin)), 'p3': int(len(anc_p3)),
                             'err': int(len(anc_err)), 'both': int(len(anc_both))},
                'gcP': gcP, 'gcB2_mass': gcB2,
                'A1_p3_margin': gcA1, 'A4_p3_prior': gcA4,
                'A2_err_alloc': gcA2, 'A3_both': gcA3,
                'gc_cal_alone': gc_cal_alone, 'gc_cal_alone_shuf': gc_cal_alone_shuf,
                'gc_comp': gc_comp, 'gc_comp_shuf': gc_comp_shuf}
            print(f"  b{b}: gcP {gcP:+.2f} | B2mass {gcB2:+.2f} | "
                  f"A1 margin {gcA1:+.2f} A4 p3prior {gcA4:+.2f} "
                  f"A2 err {gcA2:+.2f} A3 both {gcA3:+.2f}")
            print(f"      cal_alone {gc_cal_alone:+.2f} (shuf {gc_cal_alone_shuf:+.2f}) | "
                  f"comp {gc_comp:+.2f} (shuf {gc_comp_shuf:+.2f})")

        results['conds'][cond] = cond_res
        del Ws, M_star, pool_f, Fp, Fv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    W_mean_oracle gc {gc_mean_oracle:+.2f} | present {present}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. TARGETED ACQUISITION (vs gcP random):")
    print("   A1 p3_margin (lowest-margin anchors, label-free) and A4 p3_prior")
    print("      (budget on 11/13/14, fixed prior) are the DEPLOYABLE rules.")
    print("   A2 err_alloc (budget ~ frozen-error mass) and A3 both are the")
    print("      ORACLE ceilings (need pool labels). If A1/A4 ~ gcP but A2/A3")
    print("      >> gcP, the gain needs the label-error oracle (not deployable).")
    print("B. COMPOSITION (propagation + calibration, shared labels):")
    print("   gc_comp vs gcP (adds?) and vs gcP + gc_cal_alone (compounds?) and")
    print("   vs gc_comp_shuf (noise?).")

if __name__ == "__main__":
    main()
