"""al_feature_correction_diag.py: the 8B branch -- is the oracle decision
correction delta_z* predictable from a RICH label-free feature set, and can a
few labels calibrate the resulting decision-correction model? (DGLSS++ only,
fog/crosstalk)

The covariance-structure gate (al_cov_structure_diag.py) found delta_z* is
predictable from just {margin, entropy, conf} (mean R^2 0.52-0.54, gc +0.10).
This extends that with a rich feature set and tests whether the correction is
LABEL-CALIBRATABLE (the actual AL mechanism) vs oracle-only.

PART 1 -- ORACLE CEILING with the RICH feature set.
  Features (all label-free): margin, entropy, confidence, p1, p2, TTA variance,
  TTA agreement, prototype distance, prototype disagreement, density (kNN in
  the code space vs a pool subsample), local disagreement (kNN pseudo-label),
  classifier disagreement (frozen vs prototype vs TTA-mean), pseudo-class onehot.
  Target: delta_z*_c = z*_c - z0_c (the oracle logit correction per class).
  - Linear model fit on the FULL pool -> per-class R^2, mean R^2, and the
    classification gain of the corrected decoder (fit pool, apply val).
  - A nonlinear (RFF) version -> is linear enough?
  - Feature ablation: single-feature R^2 and leave-one-out -- WHICH features
    carry the predictability (what to collect).

PART 2 -- FEW-LABEL CALIBRATION (the actual AL mechanism).
  The oracle target delta_z* needs full pool labels. The label-estimable
  analogue at a labeled point is r = Y - softmax(z0) (the residual). Fit the
  same feature model on b labels/class (1/2/4/8/16) to predict r, apply on val.
  - gc vs b, vs the oracle ceiling gc, vs frozen.
  - Null control: shuffled labels at the same budget (is the few-label gain real?).
  If the few-label curve approaches the oracle ceiling, the correction is
  LABEL-CALIBRATABLE -- a real mechanism. If it collapses, the features predict
  but labels cannot calibrate (structure is label-free; the target is not).

PART 3 -- the FIXED D1 floor diagnostic.
  The prior D1 (per-pair oracle flips near the boundary) was saturated because
  nearly every val point flips frozen->oracle. FIX: count flips restricted to
  the FROZEN-ERROR set (where the frozen probe is actually wrong) -- the true
  decision floors where labels should be spent.

PART 4 -- TTA as an ACQUISITION signal (the 8C branch).
  corr(TTA instability, |delta_z*|) and corr(TTA instability, P(frozen wrong)).
  If TTA instability predicts WHERE the correction is needed (not the
  direction), it is a label-free acquisition signal for the calibration.

Decisive reads:
  P1 R^2 / gc grow past the 3-feature +0.10  -> a rich feature set carries the
     correction (the 8B mechanism is real).
  P2 few-label gc ~ oracle ceiling gc        -> LABEL-CALIBRATABLE (real method).
     few-label gc ~ 0                        -> features predict but labels
     cannot calibrate (target not estimable from few labels).
  P2 shuffled ~ few-label                    -> the gain is noise.
  P3 the corrected floor counts are informative (not saturated).
  P4 corr(TTA, |delta_z*|) > 0               -> TTA is an acquisition signal.

Usage:
  uv run python robust_diagnostic/al_feature_correction_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_feature_correction_dglsspp.json
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
    ap.add_argument("--b_labels", type=str, default="1,2,4,8,16")
    ap.add_argument("--n_rff", type=int, default=64)
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
    b_labels = [int(x) for x in args.b_labels.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_labels': b_labels, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    lac = la[ci]
    W0 = ridge_fit_soft(Xc, onehot(lac, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
    M0, C0 = class_means(Xc, lac, NUM_CLASSES)
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # clean prototypes for proto_dist
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
        Ws_c = Ws.detach().cpu()
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        def gc(mi):
            return (mi - refs['frozen']) / gap if gap > 1e-9 else None

        def featurize(X, W0c, proto_clean, knn_ref=None, knn_lab=None):
            """Label-free features on a set of code-space points."""
            L = X.float() @ W0c
            sm = torch.softmax(L, dim=1)
            top2 = torch.topk(L, 2, dim=1)
            margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
            ent = -(sm * (sm + 1e-12).log()).sum(1)
            conf = sm.max(1).values
            p1 = top2.values[:, 0]; p2 = top2.values[:, 1]
            pred = L.argmax(1)
            # TTA var + agreement
            draws = []
            for _ in range(args.tta_augs):
                torch.manual_seed(100 + _)
                flip = torch.rand_like(X) < 0.02
                draws.append(torch.softmax(torch.where(flip, -X, X) @ W0c, dim=1))
            draws = torch.stack(draws)
            tta_var = draws.var(dim=0).mean(1)
            tta_agree = (draws.argmax(2) == pred.unsqueeze(0)).float().mean(0)
            # proto dist / disagree
            Xn = F.normalize(X.float(), p=2, dim=1)
            psim = Xn @ proto_clean.t()
            proto_pred = psim.argmax(1)
            proto_dist = 1.0 - psim.gather(1, pred.unsqueeze(1)).squeeze(1)
            proto_dis = (proto_pred != pred).float()
            # density + local disagree (kNN vs pool subsample)
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
            # classifier disagreement (frozen vs proto vs tta-mean)
            tta_pred = draws.mean(0).argmax(1)
            clf_div = ((proto_pred != pred).float() + (tta_pred != pred).float())
            Fm = torch.stack([margin, ent, conf, p1, p2, tta_var, tta_agree,
                              proto_dist, proto_dis, dk, ld_, clf_div], dim=1)
            return Fm, pred, sm, L

        Fp, pred_p, sm_p, Lp = featurize(Xp, W0c, proto_clean, knn_ref=Xp, knn_lab=pl)
        Fv, pred_v, sm_v, Lv = featurize(Xv, W0c, proto_clean, knn_ref=Xp, knn_lab=pl)
        # add pseudo-class onehot
        Fp_cat = torch.cat([Fp, onehot(pred_p, NUM_CLASSES)], dim=1)
        Fv_cat = torch.cat([Fv, onehot(pred_v, NUM_CLASSES)], dim=1)
        mu = Fp_cat.mean(0); sd = Fp_cat.std(0).clamp(min=1e-8)
        Fp_s = (Fp_cat - mu) / sd
        Fv_s = (Fv_cat - mu) / sd

        # oracle target: delta_z*
        dZ_pool = Xp.float() @ Ws_c - Lp
        dZ_val = Xv.float() @ Ws_c - Lv

        # ---- PART 1: ORACLE CEILING ----
        r2 = {}
        for c in range(1, NUM_CLASSES):
            y = dZ_pool[:, c]
            A = torch.cat([Fp_s, torch.ones(len(Fp_s), 1)], dim=1)
            sol = torch.linalg.lstsq(A.double(), y.double().unsqueeze(1)).solution
            pd = (A.double() @ sol).squeeze(1)
            ssr = ((y.double() - pd) ** 2).sum().item()
            sst = ((y.double() - y.double().mean()) ** 2).sum().item()
            r2[str(c)] = 1.0 - ssr / (sst + 1e-12)
        mean_r2 = sum(v for v in r2.values()) / len(r2) if r2 else None
        # corrected decoder (oracle target, rich features)
        Lv_c = Lv.clone()
        for c in range(1, NUM_CLASSES):
            y = dZ_pool[:, c]
            A = torch.cat([Fp_s, torch.ones(len(Fp_s), 1)], dim=1)
            sol = torch.linalg.lstsq(A.double(), y.double().unsqueeze(1)).solution
            Av = torch.cat([Fv_s, torch.ones(len(Fv_s), 1)], dim=1)
            Lv_c[:, c] += (Av.double() @ sol).squeeze(1)
        gc_rich = gc(compute_miou(Lv_c.argmax(1), vl))
        # RFF nonlinear
        torch.manual_seed(3)
        Wrff = torch.randn(Fp_s.shape[1], args.n_rff)
        fp_r = torch.cos(Fp_s @ Wrff) / (args.n_rff ** 0.5)
        fv_r = torch.cos(Fv_s @ Wrff) / (args.n_rff ** 0.5)
        Lv_r = Lv.clone()
        for c in range(1, NUM_CLASSES):
            y = dZ_pool[:, c]
            A = torch.cat([fp_r, torch.ones(len(fp_r), 1)], dim=1)
            sol = torch.linalg.lstsq(A.double(), y.double().unsqueeze(1)).solution
            Av = torch.cat([fv_r, torch.ones(len(fv_r), 1)], dim=1)
            Lv_r[:, c] += (Av.double() @ sol).squeeze(1)
        gc_rff = gc(compute_miou(Lv_r.argmax(1), vl))
        # feature ablation: single-feature R^2 (on the meaningful classes)
        abla = {}
        for j, name in enumerate(['margin','entropy','conf','p1','p2','tta_var','tta_agree',
                                  'proto_dist','proto_dis','density','local_dis','clf_div']):
            r2j = []
            for c in [11, 13, 14, 15, 16, 7]:
                y = dZ_pool[:, c]
                A = torch.cat([Fp_s[:, j:j+1], torch.ones(len(Fp_s), 1)], dim=1)
                sol = torch.linalg.lstsq(A.double(), y.double().unsqueeze(1)).solution
                pd = (A.double() @ sol).squeeze(1)
                ssr = ((y.double() - pd) ** 2).sum().item()
                sst = ((y.double() - y.double().mean()) ** 2).sum().item()
                r2j.append(1.0 - ssr / (sst + 1e-12))
            abla[name] = sum(r2j) / len(r2j) if r2j else None

        # ---- PART 2: FEW-LABEL CALIBRATION ----
        # label-estimable target at a labeled point: r = Y - softmax(z0)
        # (the residual; no oracle logits needed)
        p2 = {}
        for b in b_labels:
            torch.manual_seed(7)
            lab_idx = []
            for c in range(1, NUM_CLASSES):
                idx = torch.nonzero(pl == c).squeeze(1)
                if len(idx) == 0:
                    continue
                lab_idx.append(idx[torch.randperm(len(idx))[:min(b, len(idx))]])
            lab_idx = torch.cat(lab_idx)
            Y_lab = onehot(pl[lab_idx], NUM_CLASSES).float()
            p_lab = torch.softmax(Xp[lab_idx].float() @ W0c, dim=1)
            r_lab = Y_lab - p_lab                       # the residual target
            Lv_cal = Lv.clone()
            for c in range(1, NUM_CLASSES):
                A = torch.cat([Fp_s[lab_idx], torch.ones(len(lab_idx), 1)], dim=1)
                sol = torch.linalg.lstsq(A.double(), r_lab[:, c].double().unsqueeze(1)).solution
                Av = torch.cat([Fv_s, torch.ones(len(Fv_s), 1)], dim=1)
                Lv_cal[:, c] += (Av.double() @ sol).squeeze(1)
            gc_cal = gc(compute_miou(Lv_cal.argmax(1), vl))
            # null: shuffled labels
            torch.manual_seed(9)
            r_shuf = r_lab[torch.randperm(len(r_lab))]
            Lv_sh = Lv.clone()
            for c in range(1, NUM_CLASSES):
                A = torch.cat([Fp_s[lab_idx], torch.ones(len(lab_idx), 1)], dim=1)
                sol = torch.linalg.lstsq(A.double(), r_shuf[:, c].double().unsqueeze(1)).solution
                Av = torch.cat([Fv_s, torch.ones(len(Fv_s), 1)], dim=1)
                Lv_sh[:, c] += (Av.double() @ sol).squeeze(1)
            gc_shuf = gc(compute_miou(Lv_sh.argmax(1), vl))
            p2[str(b)] = {'gc_cal': gc_cal, 'gc_shuf': gc_shuf, 'n': int(len(lab_idx))}

        # ---- PART 3: FIXED D1 FLOOR (restricted to the frozen-error set) ----
        pred_or = (Xv.float() @ Ws_c).argmax(1)
        pred_0 = Lv.argmax(1)
        frozen_err = pred_0 != vl                        # where the probe is wrong
        top_pairs = {}
        for a in range(1, NUM_CLASSES):
            for b in range(a + 1, NUM_CLASSES):
                if C0[a] < 20 or C0[b] < 20:
                    continue
                pair = ((pred_0 == a) & (pred_or == b)) | ((pred_0 == b) & (pred_or == a))
                top_pairs[f"{a}-{b}"] = {'flips_in_err': int((pair & frozen_err).sum().item()),
                                         'err_total': int(frozen_err.sum().item())}
        top_pairs = dict(sorted(top_pairs.items(), key=lambda kv: -kv[1]['flips_in_err'])[:6])

        # ---- PART 4: TTA as acquisition ----
        # TTA instability (variance over bit-flip augmentations), chunked
        tta_var_v = torch.zeros(len(Xv))
        for s in range(0, len(Xv), 5000):
            Xch = Xv[s:s+5000]
            draws = []
            for _ in range(args.tta_augs):
                torch.manual_seed(100 + _)
                flip = torch.rand_like(Xch) < 0.02
                draws.append(torch.softmax(torch.where(flip, -Xch, Xch) @ W0c, dim=1))
            tta_var_v[s:s+5000] = torch.stack(draws).var(dim=0).mean(1)
        dz_norm = torch.norm(dZ_val, dim=1)
        fe = frozen_err.float()
        corr_tta_dz = float(torch.corrcoef(torch.stack([tta_var_v, dz_norm]))[0, 1].item()) \
            if len(tta_var_v) > 1 else None
        corr_tta_err = float(torch.corrcoef(torch.stack([tta_var_v, fe]))[0, 1].item()) \
            if len(tta_var_v) > 1 else None

        cond_res = {'refs': refs, 'gap': float(gap),
                    'P1': {'mean_r2': mean_r2, 'per_class_r2': r2,
                           'gc_rich': gc_rich, 'gc_rff': gc_rff, 'ablation': abla},
                    'P2': p2, 'P3_floors': top_pairs,
                    'P4': {'corr_tta_dz': corr_tta_dz, 'corr_tta_err': corr_tta_err}}
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, pool_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    P1 mean_r2 {mean_r2:.3f} | gc_rich {gc_rich:+.2f} gc_rff {gc_rff:+.2f}")
        print("    P1 ablation: " + " ".join(f"{k}:{v:.2f}" for k, v in abla.items()))
        print("    P2 few-label: " + " ".join(f"b{k}:cal{v['gc_cal']:+.2f}shuf{v['gc_shuf']:+.2f}"
                                               for k, v in p2.items()))
        print("    P3 floors: " + " ".join(f"{k}:{v['flips_in_err']}/{v['err_total']}"
                                            for k, v in top_pairs.items()))
        print(f"    P4 corr(tta,|dz*|) {corr_tta_dz:+.2f} corr(tta,err) {corr_tta_err:+.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("P1 (oracle ceiling, rich features): mean_r2 and gc_rich vs the 3-feature")
    print("   +0.10. gc_rff (nonlinear) vs linear. Ablation shows WHICH features")
    print("   carry the predictability (what to collect).")
    print("P2 (few-label calibration): gc_cal vs gc_rich (oracle) and gc_shuf")
    print("   (null). If gc_cal ~ gc_rich at modest b, the correction is")
    print("   LABEL-CALIBRATABLE (a real mechanism). If gc_cal ~ 0, the features")
    print("   predict but labels cannot calibrate. If gc_cal ~ gc_shuf, it is noise.")
    print("P3 (fixed floors): flips_in_err / err_total per pair -- the true decision")
    print("   floors (where labels should be spent), now restricted to the frozen-")
    print("   error set (the prior D1 was saturated).")
    print("P4 (TTA acquisition): corr(TTA instability, |delta_z*|) and corr(TTA, ")
    print("   frozen error). If > 0, TTA tells WHERE the correction is needed.")


if __name__ == "__main__":
    main()
