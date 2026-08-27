"""probe_linear_prop_diag.py: what FEATURE-SPACE PROPERTIES does the R4 linear
classifier need for ZERO-SHOT transfer, and which extractor property predicts it?

Diagnostic 10's rep-robustness probe measured the CEILING interaction, but the
comparison of interest (hyper best on KITTI-C, worst on NuScenes-C) is the
ZERO-SHOT line. Zero-shot uses the CLEAN-fit W0 decoded on corrupted features --
there is no "fit on corrupted data", so the question is: does the clean-fit
boundary survive the clean->corrupted feature shift?

From Diagnostic 10, class_shift (clean->corr per-class cosine) already tracks the
zero-shot pattern: cov's input-IN keeps features stable on KITTI-C fog/crosstalk
(low shift, high zs); hyper shifts a lot on fog and NuScenes-C (high shift, low
zs). This probe measures the linear-classifier-relevant properties AND the
zero-shot transfer directly:

  1. PRE-SIGN MARGIN : fraction of codes with |pre-sign activation| < 0.5.
     Codes near the binarization boundary flip sign under the corruption-induced
     shift -> the clean-fit W0 boundary is fragile there.
  2. FISHER SCATTER RATIO / WITHIN-CLASS VAR / EFFRANK : corrupted code geometry.
  3. CLASS_SHIFT : clean->corr per-class cosine (the zero-shot transfer driver).
  4. MARGIN SWEEP : frozen (W0) accuracy after zeroing pre-sign activations below
     a threshold -- directly shows how much of the ZERO-SHOT signal sits at
     fragile sign-flip boundaries.
  5. frozen (W0 zero-shot) + ceiling (W* re-fit) for the correlation.

Whichever property correlates with FROZEN (the zero-shot) across extractors is
what to optimize in the feature extractor.

Usage:
  uv run python robust_diagnostic/probe_linear_prop_diag.py \
    --max_frames 200 --out robust_diagnostic/logs/probe_linear_prop.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, build_nuscenes_parser, hdc_codes, onehot, ridge_fit_exact,
    ConfAccum, NUM_CLASSES)
from robust_diagnostic.probe_rep_robustness_diag import (
    feature_stream, subsample_pairs)

MARGIN_THRESH = 0.5  # |pre-sign| < 0.5 => near the binarization boundary


def pre_sign_margins(feats, proj, device, chunk=100000):
    """Fraction of codes with |pre-sign activation| < MARGIN_THRESH, computed on a
    subsample. The R4 classifier threshold is sign(z.P); points near zero flip
    easily."""
    feats = feats.to(device).float()
    n_small = 0; n_tot = 0
    for s in range(0, len(feats), chunk):
        e = min(s + chunk, len(feats))
        pre = feats[s:e] @ proj
        n_small += int((pre.abs() < MARGIN_THRESH).sum().item())
        n_tot += pre.numel()
    return n_small / max(n_tot, 1)


def fisher_ratio(codes, lbls, num_classes=NUM_CLASSES):
    """tr(S_between)/tr(S_within) on the binarized code. Higher = more linearly
    separable. S_between = sum_c N_c (mu_c - mu)(mu_c - mu)^T, S_within = sum over
    classes of within-class scatter."""
    codes = codes.float()
    mu = codes.mean(dim=0)
    sb = torch.zeros(codes.shape[1], device=codes.device)
    sw = torch.zeros(codes.shape[1], device=codes.device)
    for c in range(1, num_classes):
        m = lbls == c
        if m.sum() == 0:
            continue
        x = codes[m]
        muc = x.mean(dim=0)
        sb += m.sum() * ((muc - mu) ** 2)
        sw += ((x - muc) ** 2).sum(dim=0)
    sw = sw.clamp(min=1e-8)
    return float((sb.sum() / sw.sum()).item())


def within_class_var(codes, lbls, num_classes=NUM_CLASSES):
    """Mean per-class code variance (tighter = better for a linear probe)."""
    vars_ = []
    for c in range(1, num_classes):
        m = lbls == c
        if m.sum() > 1:
            vars_.append(float(codes[m].var(dim=0).mean().item()))
    return float(np.mean(vars_)) if vars_ else float('nan')


def effrank(codes, device, K=50):
    """Participation-ratio effective rank on a subsample (coarse)."""
    torch.manual_seed(7)
    idx = torch.randperm(len(codes))[:min(len(codes), 80000)]
    Xs = codes[idx].to(device).float()
    S = Xs.t() @ Xs
    evals = torch.linalg.eigvalsh(S).flip(0)
    evals = evals[:K].clamp(min=1e-10)
    return float((evals.sum() ** 2 / (evals ** 2).sum()).item())


def margin_sweep_frozen(feats, lbls, W0, proj, device, tau, chunk=100000):
    """Frozen R4 accuracy after zeroing pre-sign activations below tau*scale.
    Takes the RAW 128-d features; computes pre-sign = feats @ proj, zeros the
    fragile (small-|pre-sign|) coordinates, re-binarizes, and decodes with the
    CLEAN-fit W0. A large drop = the zero-shot boundary sits at sign-flip
    coordinates that the corruption-induced shift flips."""
    feats = feats.to(device).float()
    pre = feats @ proj                      # (N, 10000) pre-sign activations
    scale = pre.abs().mean()
    thr = tau * scale
    z = pre.clone()
    z[pre.abs() < thr] = 0.0
    codes_z = torch.sign(z).float()
    acc = ConfAccum()
    for s in range(0, len(codes_z), chunk):
        e = min(s + chunk, len(codes_z))
        acc.update((codes_z[s:e] @ W0.to(device)).argmax(1).cpu(), lbls[s:e])
    return acc.miou()


def class_shift(clean_means, corr_means, num_classes=NUM_CLASSES):
    """Mean clean->corrupted per-class cosine similarity (1 = no shift). This is
    the zero-shot-relevant property: the clean-fit W0 transfers to the corrupted
    features only if the per-class means don't move much."""
    sims = []
    for c in range(1, num_classes):
        if clean_means[c].norm() > 0 and corr_means[c].norm() > 0:
            sims.append(F.cosine_similarity(clean_means[c], corr_means[c], dim=0).item())
    return float(np.mean(sims)) if sims else float('nan')


def class_means(feats, lbls, num_classes=NUM_CLASSES):
    means = torch.zeros(num_classes, feats.shape[1])
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        m = lbls == c
        if m.sum() > 0:
            means[c] += feats[m].sum(dim=0); counts[c] += m.sum()
    for c in range(num_classes):
        if counts[c] > 0:
            means[c] /= counts[c]
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--nusc_dir", type=str, default="/mnt/alpha/jmfleming/nuscenes_kitti")
    ap.add_argument("--nusc_c_dir", type=str, default="/mnt/bravo/jmfleming/nuscenes_c_kitti")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--nusc_labels", type=str, default="config/labels/nuscenes_new.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--clean_fit_n", type=int, default=100000)
    ap.add_argument("--pool_cap", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--extractors", type=str,
                    default="hyper_kitti:baseline:logs/kitti_pretrain,"
                            "dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp,"
                            "cov_kitti:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan,"
                            "hyper_nusc:baseline:logs/nusc_pretrain,"
                            "dgl_nusc:supcon_vib_dglsspp:robust_diagnostic/logs/nusc_dglsspp_21ep,"
                            "cov_nusc:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/nusc_covshift_21ep")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    nusc_data = yaml.safe_load(open(args.nusc_labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.gen_trainers import GenTrainer
    from modules.oracle_core import get_hdc_projection
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    results = {'label': 'linear_prop', 'extractors': {}}

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        # dataset keyed by a "_kitti"/"_nusc" label suffix; default to KITTI so a
        # bare label (e.g. "geoid") is NOT silently treated as NuScenes (a cross-
        # domain clean/W0 mismatch that produced bogus class_shift/frozen/ceiling).
        if lab.endswith('_nusc'):
            dset = 'nusc'
        else:
            dset = 'kitti'
        clean_parser = (build_parser(args.kitti_dir, DATA, ARCH)
                        if dset == 'kitti'
                        else build_nuscenes_parser(args.nusc_dir, nusc_data, ARCH))
        results['extractors'][lab] = {'method': method, 'dataset': dset, 'conds': {}}

        # clean reference: W0 + properties
        t0 = time.time()
        cf, cl = [], []
        for zf, l in feature_stream(model, clean_parser, device, args.max_frames):
            cf.append(zf); cl.append(l)
        cf = torch.cat(cf); cl = torch.cat(cl)
        cf_s, cl_s = subsample_pairs(cf, cl, cap=args.clean_fit_n)
        Xc = hdc_codes(cf_s, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl_s, NUM_CLASSES), args.lam, device).to(device)
        clean_ef = effrank(Xc, device)
        clean_means = class_means(cf_s, cl_s)
        print(f"  clean {dset}: {len(cf)} pts, W0 done, effrank={clean_ef:.1f} ({time.time()-t0:.0f}s)")
        del cf, cl, cf_s, cl_s, Xc
        torch.cuda.empty_cache()

        for cond in conds:
            if dset == 'kitti':
                cdir = os.path.join(args.kittic_dir, cond, 'heavy')
                if not os.path.exists(cdir):
                    cdir = os.path.join(args.kittic_dir, cond, 'moderate')
                parser = build_parser(cdir, DATA, ARCH)
            else:
                parser = build_nuscenes_parser(os.path.join(args.nusc_c_dir, cond, 'heavy'),
                                               nusc_data, ARCH)
            t1 = time.time()
            pf, pl = [], []
            for zf, l in feature_stream(model, parser, device, args.max_frames):
                pf.append(zf); pl.append(l)
            pf = torch.cat(pf); pl = torch.cat(pl)
            pf_s, pl_s = subsample_pairs(pf, pl, cap=args.pool_cap)
            Xp = hdc_codes(pf_s, proj, device).float()

            margin_frac = pre_sign_margins(pf_s, proj, device)
            fisher = fisher_ratio(Xp, pl_s)
            wcv = within_class_var(Xp, pl_s)
            ef = effrank(Xp, device)
            # ceiling W* on the corrupted pool, decoded on the same pool points
            Ws = ridge_fit_exact(Xp, onehot(pl_s, NUM_CLASSES), args.lam, device).to(device)
            acc = ConfAccum()
            for s in range(0, len(Xp), 100000):
                e = min(s + 100000, len(Xp))
                acc.update((Xp[s:e].to(device) @ Ws).argmax(1).cpu(), pl_s[s:e])
            ceiling = acc.miou()
            # frozen (W0) for reference
            acc0 = ConfAccum()
            for s in range(0, len(Xp), 100000):
                e = min(s + 100000, len(Xp))
                acc0.update((Xp[s:e].to(device) @ W0).argmax(1).cpu(), pl_s[s:e])
            frozen = acc0.miou()
            # margin sweep: how much frozen (zero-shot) signal is fragile
            sweeps = {}
            for tau in (0.5, 1.0, 2.0):
                sweeps[f'tau{tau}'] = margin_sweep_frozen(
                    pf_s, pl_s, W0, proj, device, tau)
            corr_means = class_means(pf_s, pl_s)
            shift = class_shift(clean_means, corr_means)

            entry = {'pre_sign_margin_lt05_frac': margin_frac,
                     'fisher_ratio': fisher,
                     'within_class_var': wcv,
                     'effrank': ef,
                     'class_shift_clean_to_corr': shift,
                     'ceiling': ceiling, 'frozen': frozen,
                     'margin_sweep': sweeps}
            results['extractors'][lab]['conds'][cond] = entry
            print(f"  [{cond}] pre-sign<0.5={margin_frac:.3f} fisher={fisher:.3f} "
                  f"class_shift={shift:.3f} | frozen={frozen:.3f} "
                  f"ceiling={ceiling:.3f} | margin_sweep {sweeps} ({time.time()-t1:.0f}s)")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
