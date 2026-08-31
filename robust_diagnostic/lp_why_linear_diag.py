"""lp_why_linear_diag.py: WHY does the HDC linear classifier consistently beat
the HDC prototype classifier on every condition?
(docs/lin_probe_training/validation.md)

Same frozen DGLSS++ features, same clean-only fit, two decoder families
(prototype = mean binarized code / cosine, linear = ridge probe on the codes).
This script measures the properties of the FEATURE SPACE and the CORRUPTIONS
that explain the persistent gap, per condition and severity:

  P1 feature-space isotropy (128-d): participation ratio + top-5 variance
     fraction. An anisotropic space dominates every random projection -> the
     codes saturate -> the prototype means collapse. (The Phase 8 'isotropy'
     mechanism.)
  P2 code diversity: dead-coordinate fraction + mean pairwise code cosine.
     If the codes are near-constant the prototype centroids coincide.
  P3 prototype centroid separation: off-diagonal mean cosine of the class
     prototype means in the 10000-d code space (clean and per condition).
  P4 per-class mean shift clean->condition (1 - cos) and within-class
     dispersion (mean cosine to own-class mean). Does the corruption move the
     class centroids (proto collapse) or just widen the classes?
  P5 the clean reference: does the linear-vs-proto gap already exist on CLEAN
     (a feature-space property) or only under corruption?
  P6 gap decomposition + disagreement: per-class linear-minus-proto IoU, and
     where the two decoders disagree, P(linear right | disagree) vs P(proto
     right | disagree) -- the direct measure of 'the structure the prototype
     throws away' (the linear probe recovers it, the centroid cannot).

Decisive reads:
  P5 gap ~ 0 on clean, grows under corruption        -> a corruption-collapse
     mechanism (mean shift / code saturation), the README story
  P5 gap already present on clean                    -> a static space property
     (anisotropy / centroid coincidence), not corruption-specific
  P6 P(linear right | disagree) >> P(proto right)    -> the probe recovers
     within-class structure the mean throws away; direction for a cheaper
     decoder that keeps the mIoU
  P2/P3 collapse together under fog/crosstalk         -> the encoding saturates
     on the destroyer conditions (target for representation / encoding fixes)

Usage:
  uv run python robust_diagnostic/lp_why_linear_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --out robust_diagnostic/logs/lp_why_linear_dglsspp.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
CONDS_ALL = ["fog", "crosstalk", "snow", "wet_ground", "incomplete_echo",
             "beam_missing", "motion_blur", "cross_sensor"]


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                  labels=data["labels"], color_map=data["color_map"], learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"], sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_full(model, parser, device, num_frames=100):
    feats, preds, lbls = [], [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device); labels = batch[2].to(device).view(-1); mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol)
            pred = out[0]
            z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            pf = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])[mask]
            feats.append(zf.cpu()); preds.append(pf.cpu()); lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(preds), torch.cat(lbls)


def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)


def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc); y[torch.arange(len(lbls)), lbls.long()] = 1; return y


def ridge_fit_exact(X, Y, lam, device, chunk=50000):
    d = X.shape[1]; nc = Y.shape[1]
    S = torch.zeros(d, d, device=device); T = torch.zeros(d, nc, device=device)
    for s in range(0, len(X), chunk):
        Xc = X[s:s + chunk].to(device); Yc = Y[s:s + chunk].to(device)
        S += Xc.t() @ Xc; T += Xc.t() @ Yc
    A = S.double() + lam * torch.eye(d, dtype=torch.float64, device=device)
    return torch.linalg.solve(A, T.double()).float()


def build_prototypes(codes, lbls, nc=NUM_CLASSES):
    protos = torch.zeros(nc, codes.shape[1]); counts = torch.zeros(nc)
    for c in range(nc):
        m = lbls == c
        if int(m.sum().item()) > 0:
            protos[c] = codes[m].float().mean(dim=0)
            counts[c] = float(int(m.sum().item()))
    return F.normalize(protos, p=2, dim=1), counts


def per_class_iou(preds, lbls, nc=NUM_CLASSES):
    present = set(lbls.tolist())
    ious = {}
    for c in range(1, nc):
        if c not in present:
            continue
        tp = int(((preds == c) & (lbls == c)).sum().item())
        fp = int(((preds == c) & (lbls != c)).sum().item())
        fn = int(((preds != c) & (lbls == c)).sum().item())
        denom = tp + fp + fn
        ious[str(c)] = tp / denom if denom > 0 else 0.0
    return ious


def isotropy(z):
    zc = z - z.mean(0)
    S = (zc.t() @ zc) / len(zc)
    ev = torch.linalg.eigvalsh(S.double()).clamp(min=0)
    s = ev.sum().item()
    pr = (s * s) / ((ev ** 2).sum().item()) if s > 1e-12 else None
    top5 = ev[-5:].sum().item() / s if s > 1e-12 else None
    return pr, top5


def dead_coords(codes):
    if len(codes) < 2:
        return None
    const = (codes > 0).all(0) | (codes < 0).all(0)
    return float(const.float().mean().item())


def code_div(codes, seed=1, sub=5000):
    n = min(sub, len(codes))
    torch.manual_seed(seed)
    idx = torch.randperm(len(codes))[:n]
    nz = F.normalize(codes[idx].float(), dim=1)
    return float((nz @ nz.t()).mean().item())


def offdiag_mean_cos(means):
    mn = F.normalize(means.float(), dim=1)
    G = mn @ mn.t()
    mask = ~torch.eye(G.shape[0], dtype=torch.bool)
    return float(G[mask].mean().item())


def class_mean_shift(clean_means, cond_means, classes):
    shifts = {}
    for c in classes:
        a = F.normalize(clean_means[c].float().unsqueeze(0), dim=1)
        b = F.normalize(cond_means[c].float().unsqueeze(0), dim=1)
        shifts[str(c)] = float((1 - (a @ b.t())).item())
    return shifts


def class_dispersion(codes, lbls, means, classes, sub=10000):
    disp = {}
    for c in classes:
        m = lbls == c
        if int(m.sum().item()) == 0:
            continue
        pts = codes[m][:sub].float()
        nz = F.normalize(pts, dim=1)
        mc = F.normalize(means[c].float().unsqueeze(0), dim=1)
        disp[str(c)] = float((nz @ mc.t()).mean().item())
    return disp


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
    ap.add_argument("--fit_clean", type=int, default=30000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--proj_dim", type=int, default=10000)
    ap.add_argument("--geo_sub", type=int, default=20000, help="subsample for the geometry diagnostics")
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--sevs", type=str, default="light,moderate,heavy")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    sevs = [s.strip() for s in args.sevs.split(',') if s.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    cf, cp, cl = extract_full(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=cf.shape[1], dim_out=args.proj_dim, device=device)
    results = {'label': args.label, 'method': args.method_b, 'sevs': sevs,
               'fit_clean': args.fit_clean, 'val_size': args.val_size, 'lam': args.lam,
               'proj_dim': args.proj_dim, 'geo_sub': args.geo_sub, 'conds': {}}

    cf_fit = cf[:args.fit_clean]; cl_fit = cl[:args.fit_clean]
    Xc = hdc_codes(cf_fit, proj, device).float()
    W = ridge_fit_exact(Xc, onehot(cl_fit, NUM_CLASSES), args.lam, device).detach().cpu()
    protos, _ = build_prototypes(Xc, cl_fit)
    protos = protos.float()
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    present_classes = sorted(set(cl_fit.tolist()) & set(range(1, NUM_CLASSES)))
    clean_means = torch.zeros(NUM_CLASSES, cf.shape[1])
    for c in present_classes:
        m = cl_fit == c
        if int(m.sum().item()) > 0:
            clean_means[c] = cf_fit[m].float().mean(dim=0)

    def geo(feats, lbls):
        """Feature-space / code-space geometry on a subsample."""
        n = min(args.geo_sub, len(feats))
        torch.manual_seed(3)
        idx = torch.randperm(len(feats))[:n]
        z = feats[idx]
        lv = lbls[idx]
        pr, top5 = isotropy(z)
        Xg = hdc_codes(z, proj, device).float()
        cond_means = torch.zeros(NUM_CLASSES, Xg.shape[1])
        for c in present_classes:
            m = lv == c
            if int(m.sum().item()) > 0:
                cond_means[c] = Xg[m].float().mean(dim=0)
        return {'participation_ratio': pr, 'top5_var_frac': top5,
                'dead_coords': dead_coords(Xg), 'code_pair_cos': code_div(Xg),
                'proto_pair_cos': offdiag_mean_cos(cond_means),
                'mean_shift': class_mean_shift(clean_means, cond_means, present_classes),
                'dispersion': class_dispersion(Xg, lv, cond_means, present_classes)}

    def eval_set(feats, preds, lbls, name):
        Xv = hdc_codes(feats[:args.val_size], proj, device).float()
        lv = lbls[:args.val_size]
        lin_p = (Xv @ W).argmax(1)
        pro_p = (Xv @ protos.t()).argmax(1)
        r = {'mIoU_linear': compute_miou(lin_p, lv),
             'mIoU_proto': compute_miou(pro_p, lv),
             'gap_linear_minus_proto': compute_miou(lin_p, lv) - compute_miou(pro_p, lv)}
        r['per_class_linear'] = per_class_iou(lin_p, lv)
        r['per_class_proto'] = per_class_iou(pro_p, lv)
        r['per_class_gap'] = {k: r['per_class_linear'].get(k, 0.0) - r['per_class_proto'].get(k, 0.0)
                              for k in r['per_class_linear']}
        dis = lin_p != pro_p
        n_dis = int(dis.sum().item())
        lin_right = lin_p == lv; pro_right = pro_p == lv
        r['n_disagree'] = n_dis
        if n_dis > 0:
            r['P_linear_right_given_disagree'] = float((dis & lin_right).sum().item()) / n_dis
            r['P_proto_right_given_disagree'] = float((dis & pro_right).sum().item()) / n_dis
            r['P_both_wrong_given_disagree'] = float(((dis & ~lin_right) & ~pro_right).sum().item()) / n_dis
        r['n'] = int(len(lv))
        del Xv
        print(f"  {name}: linear {r['mIoU_linear']:.3f} | proto {r['mIoU_proto']:.3f} | "
              f"gap {r['gap_linear_minus_proto']:+.3f} | disagree {n_dis}")
        return r

    t0 = tic()
    clean_res = eval_set(cf, cp, cl, "clean")
    clean_res['geo'] = geo(cf, cl)
    results['clean'] = clean_res

    for cond in conds:
        cond_res = {'sevs': {}}
        for sev in sevs:
            cdir = os.path.join(args.kittic_dir, cond, sev)
            if not os.path.exists(cdir):
                print(f"  [{cond}/{sev}] dir missing, skipped")
                continue
            fd, pd, ld = extract_full(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            r = eval_set(fd, pd, ld, f"{cond}/{sev}")
            r['geo'] = geo(fd, ld)
            cond_res['sevs'][sev] = r
            del fd, pd, ld
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ev = [v for v in cond_res['sevs'].values()]
        if ev:
            for k in ('mIoU_linear', 'mIoU_proto', 'gap_linear_minus_proto'):
                cond_res[k + '_3sev_mean'] = float(sum(v[k] for v in ev) / len(ev))
        results['conds'][cond] = cond_res

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out} ({toc(t0):.0f}s)")
    print("\n=== WHY-LINEAR READ ===")
    print("P5 clean gap: does linear beat proto on CLEAN (static space property)")
    print("   or does the gap only appear under corruption (collapse mechanism)?")
    print("P6 disagreement: where they differ, is the linear probe right more")
    print("   often? (the structure the prototype throws away)")
    print("P1/P2/P3/P4: isotropy, code diversity, centroid separation, mean shift")
    print("   and dispersion -- does the corruption collapse the prototypes?")


if __name__ == "__main__":
    main()
