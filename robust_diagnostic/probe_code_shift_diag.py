"""probe_code_shift_diag.py: what actually separates the fog/crosstalk WINNER
(cov-shift, input-IN) from the equal-raw-shift LOSER (GeoID) -- is it the CODE
space, not the raw feature space?

The GeoID-loss extractor (Diagnostic-12 follow-up) achieved the same RAW 128-d
class_shift as cov-shift (fog 0.907 vs 0.933, crosstalk 0.921 vs 0.980) but did
NOT get the zero-shot (frozen 0.098/0.093 vs cov 0.344/0.579) or ceiling
(0.363/0.276 vs cov 0.582/0.698). Both are "feature-stable" under corruption,
yet only cov's clean-fit W0 transfers. This probe tests the hypothesis that the
relevant stability is in the BINARIZED 10000-d CODE space (where W0 actually
operates), not the raw 128-d space that class_shift was measured on.

Per (extractor, condition), measured in BOTH raw 128-d and binarized 10000-d:
  1. class_shift_raw   : clean->corr per-class mean cosine on 128-d features
  2. class_shift_code  : clean->corr per-class mean cosine on the binarized code
     (the space W0 lives in). If cov's code-shift is much lower than geoid's,
     THAT is the real lever.
  3. fisher_raw / fisher_code : between/within scatter in each space.
  4. W0_align_cos : cosine between the corrupted-code class means and the
     clean-fit W0 columns (does W0 point at the corrupted classes?).
  5. Also an ablation: applying cov-style input-IN (channels {0,4}) at EVAL to
     the GeoID model -- if that rescues it, the mechanism is the input transform
     at eval (not the learned weights).

This tells us whether to optimize the CODE-space shift (e.g. a loss on the
binarized code), the raw shift (already achieved by GeoID, insufficient), or the
W0-to-code alignment.

Usage:
  uv run python robust_diagnostic/probe_code_shift_diag.py \
    --max_frames 200 --out robust_diagnostic/logs/probe_code_shift.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, hdc_codes, onehot, ridge_fit_exact, ConfAccum, NUM_CLASSES)
from robust_diagnostic.probe_linear_prop_diag import (
    class_shift, class_means, fisher_ratio)
from robust_diagnostic.probe_rep_robustness_diag import feature_stream, subsample_pairs
from robust_diagnostic.probe_gated_inputin_diag import apply_input_in_channels


def code_means(codes, lbls, num_classes=NUM_CLASSES):
    """Per-class mean of the binarized code, L2-normalized (prototypes)."""
    means = torch.zeros(num_classes, codes.shape[1])
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        m = lbls == c
        if m.sum() > 0:
            means[c] += codes[m].sum(dim=0); counts[c] += m.sum()
    for c in range(num_classes):
        if counts[c] > 0:
            means[c] = F.normalize(means[c], p=2, dim=0)
    return means


def class_shift_code(clean_means, corr_means, num_classes=NUM_CLASSES):
    sims = []
    for c in range(1, num_classes):
        if clean_means[c].norm() > 0 and corr_means[c].norm() > 0:
            sims.append(F.cosine_similarity(clean_means[c], corr_means[c], dim=0).item())
    return float(np.mean(sims)) if sims else float('nan')


def w0_align_cos(corr_code_means, W0, num_classes=NUM_CLASSES):
    """Mean cosine between each class's corrupted-code prototype and its W0 column
    (the clean-fit decision direction). High = W0 still points at the corrupted
    classes."""
    sims = []
    for c in range(1, num_classes):
        if corr_code_means[c].norm() > 0:
            sims.append(F.cosine_similarity(corr_code_means[c], W0[:, c], dim=0).item())
    return float(np.mean(sims)) if sims else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--clean_fit_n", type=int, default=100000)
    ap.add_argument("--pool_cap", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--extractors", type=str,
                    default="hyper_kitti:baseline:logs/kitti_pretrain,"
                            "cov_kitti:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan,"
                            "geoid:supcon_vib_geoid:robust_diagnostic/logs/geoid_full/supcon_vib_geoid")
    ap.add_argument("--test_inputin_eval", type=int, default=1,
                    help="1 = also measure the GeoID model WITH cov-style input-IN applied at eval")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.gen_trainers import GenTrainer
    from modules.oracle_core import get_hdc_projection
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    results = {'label': 'code_shift', 'extractors': {}}

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        results['extractors'][lab] = {'method': method, 'conds': {}}

        # clean reference
        t0 = time.time()
        cf, cl = [], []
        for zf, l in feature_stream(model, clean_parser, device, args.max_frames):
            cf.append(zf); cl.append(l)
        cf = torch.cat(cf); cl = torch.cat(cl)
        cf_s, cl_s = subsample_pairs(cf, cl, cap=args.clean_fit_n)
        Xc = hdc_codes(cf_s, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl_s, NUM_CLASSES), args.lam, device).to(device)
        raw_clean_means = class_means(cf_s, cl_s)
        code_clean_means = code_means(Xc, cl_s)
        print(f"  clean done, W0 fit ({time.time()-t0:.0f}s)")
        del cf, cl, cf_s, cl_s, Xc
        torch.cuda.empty_cache()

        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            parser = build_parser(cdir, DATA, ARCH)
            t1 = time.time()

            # corrupted raw features
            pf, pl = [], []
            for zf, l in feature_stream(model, parser, device, args.max_frames):
                pf.append(zf); pl.append(l)
            pf = torch.cat(pf); pl = torch.cat(pl)
            pf_s, pl_s = subsample_pairs(pf, pl, cap=args.pool_cap)
            Xp = hdc_codes(pf_s, proj, device).float()

            raw_corr_means = class_means(pf_s, pl_s)
            code_corr_means = code_means(Xp, pl_s)
            entry = {
                'class_shift_raw': class_shift(raw_clean_means, raw_corr_means),
                'class_shift_code': class_shift_code(code_clean_means, code_corr_means),
                'fisher_code': fisher_ratio(Xp, pl_s),
                'w0_align_cos': w0_align_cos(code_corr_means, W0),
                'frozen': None, 'ceiling': None,
            }
            # frozen + ceiling
            Ws = ridge_fit_exact(Xp, onehot(pl_s, NUM_CLASSES), args.lam, device).to(device)
            acc0 = ConfAccum(); accs = ConfAccum()
            for s in range(0, len(Xp), 100000):
                e = min(s + 100000, len(Xp))
                acc0.update((Xp[s:e].to(device) @ W0).argmax(1).cpu(), pl_s[s:e])
                accs.update((Xp[s:e].to(device) @ Ws).argmax(1).cpu(), pl_s[s:e])
            entry['frozen'] = acc0.miou(); entry['ceiling'] = accs.miou()
            del pf, pl, pf_s, pl_s, Xp
            torch.cuda.empty_cache()

            # ablation: GeoID model with cov-style input-IN at eval
            if args.test_inputin_eval and lab == 'geoid':
                def stream_in(parser_, max_frames):
                    out = []
                    model.eval()
                    with torch.no_grad():
                        for i, batch in enumerate(parser_.get_train_set()):
                            if max_frames > 0 and i >= max_frames: break
                            in_vol = batch[0].to(device)
                            labels = batch[2].to(device).view(-1)
                            mask = (batch[1].to(device) > 0).view(-1)
                            iv = apply_input_in_channels(model, in_vol)
                            ot = model(iv)
                            z8 = ot[2] if len(ot) == 3 else ot[1]
                            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
                            out.append((zf.cpu(), labels[mask].cpu()))
                    return out
                gpf, gpl = [], []
                for zf, l in stream_in(parser, args.max_frames):
                    gpf.append(zf); gpl.append(l)
                gpf = torch.cat(gpf); gpl = torch.cat(gpl)
                gpf_s, gpl_s = subsample_pairs(gpf, gpl, cap=args.pool_cap)
                gXp = hdc_codes(gpf_s, proj, device).float()
                gacc0 = ConfAccum()
                for s in range(0, len(gXp), 100000):
                    e = min(s + 100000, len(gXp))
                    gacc0.update((gXp[s:e].to(device) @ W0).argmax(1).cpu(), gpl_s[s:e])
                entry['geoid_inputin_eval_frozen'] = gacc0.miou()
                del gpf, gpl, gpf_s, gpl_s, gXp
                torch.cuda.empty_cache()

            results['extractors'][lab]['conds'][cond] = entry
            print(f"  [{cond}] raw_shift={entry['class_shift_raw']:.3f} "
                  f"code_shift={entry['class_shift_code']:.3f} "
                  f"w0_align={entry['w0_align_cos']:.3f} "
                  f"frozen={entry['frozen']:.3f} ceiling={entry['ceiling']:.3f} "
                  f"({time.time()-t1:.0f}s)")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
