"""probe_rep_robustness_diag.py: WHY is the plain-supervised HyperLiDAR extractor
(`baseline`) MORE robust than DGLSS++/cov-shift on KITTI-C but LESS on NuScenes-C?

README: on KITTI-C hyper's R4 ceiling (53.9) beats dglsspp (50.7) and cov-shift
(46.7); on NuScenes-C hyper (55.1) loses to dglsspp (63.5) and cov-shift (61.0).
Same plain extractor, opposite outcome on the two benchmarks. This probe measures
the network dynamics + feature-space invariance that could explain it.

Per (extractor, dataset, condition), with a CLEAN reference:
  1. FEATURE-SPACE SHIFT  : clean->corrupted per-class mean cosine (invariance:
     does the class geometry survive?). Lower shift = more invariant features.
  2. PER-CLASS SEPARABILITY : mean pairwise cosine distance of class means on
     clean vs corrupted (structure retention; collapsed = 0.10-ish vs 0.7 clean).
  3. BN RUNNING-STAT MISMATCH at conv_1.bn (the Diagnostic-7 detector) -- network
     dynamics: how far frozen stats drift under the corruption.
  4. CODE VARIANCE + EFFECTIVE RANK (participation ratio) -- is the code more
     compressed/directional under corruption?
  5. INPUT CHANNEL STATS (range/remission variance, D3) -- the input-side trigger.

Runs the extractors on their HOME dataset (kitti extractors on KITTI-C, nusc
extractors on NuScenes-C), so the cross-extractor comparison is in-domain.

Usage:
  uv run python robust_diagnostic/probe_rep_robustness_diag.py \
    --max_frames 200 --out robust_diagnostic/logs/probe_rep_robustness.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy, signal, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F

def _signal_handler(signum, frame):
    """Log exactly what killed us: signal number + a stack trace of where we were."""
    with open('logs/probe_rep_robustness_signal.txt', 'a') as fh:
        fh.write(f"\n[{time.strftime('%H:%M:%S')}] CAUGHT signal {signum} "
                 f"({signal.Signals(signum).name if signum in signal.Signals else '?'})\n")
        traceback.print_stack(frame, file=fh)
        fh.write("(handler returning -> default action: process will terminate)\n")
    sys.exit(128 + signum)

for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    try:
        signal.signal(_s, _signal_handler)
    except Exception:
        pass
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, build_nuscenes_parser, stream_frames, hdc_codes, NUM_CLASSES)
from robust_diagnostic.probe_fog_collapse_layer_diag import LayerStats
from robust_diagnostic.probe_covshift_mechanism_diag import (
    class_means_feats, input_stats_stream, topk_evals)

STAGES = ['conv1', 'conv2', 'conv3', 'layer1', 'layer2', 'layer3', 'layer4', 'conv_1', 'conv_2']


def per_class_sep(means):
    """Mean pairwise cosine distance between per-class mean vectors (1=separated, 0=merged)."""
    keys = [c for c in range(1, NUM_CLASSES) if means[c].norm() > 0]
    if len(keys) < 2:
        return float('nan')
    vecs = F.normalize(torch.stack([means[c] for c in keys]), p=2, dim=1)
    sim = vecs @ vecs.t()
    iu = torch.triu_indices(len(keys), len(keys), offset=1)
    return float(1.0 - sim[iu[0], iu[1]].mean())


def class_shift(clean_means, corr_means):
    """Mean clean->corrupted per-class cosine similarity (1 = no shift)."""
    sims = []
    for c in range(1, NUM_CLASSES):
        if clean_means[c].norm() > 0 and corr_means[c].norm() > 0:
            sims.append(F.cosine_similarity(clean_means[c], corr_means[c], dim=0).item())
    return float(np.mean(sims)) if sims else float('nan')


def feature_stream(model, parser, device, max_frames=0):
    """Stream (zf, labels) per frame, features kept on CPU for geometry."""
    out = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if max_frames > 0 and i >= max_frames:
                break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            out_tuple = model(in_vol)
            z8 = out_tuple[2] if len(out_tuple) == 3 else out_tuple[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            out.append((zf.cpu(), labels[mask].cpu()))
    return out


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
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow")
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
    results = {'label': 'rep_robustness', 'extractors': {}}

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        dset = 'kitti' if lab.endswith('_kitti') else 'nusc'
        if dset == 'kitti':
            clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        else:
            clean_parser = build_nuscenes_parser(args.nusc_dir, nusc_data, ARCH)
        results['extractors'][lab] = {'method': method, 'dataset': dset, 'conds': {}}

        # clean reference: feature geometry + BN mismatch
        t0 = time.time()
        cf, cl = [], []
        for zf, l in feature_stream(model, clean_parser, device, args.max_frames):
            cf.append(zf); cl.append(l)
        cf = torch.cat(cf); cl = torch.cat(cl)
        clean_means = class_means_feats(cf, cl)
        clean_sep = per_class_sep(clean_means)
        Xc = hdc_codes(cf, proj, device).float()
        clean_code_var = float(torch.var(Xc).item())
        clean_rank = topk_evals(Xc, K=50, device=device)
        effrank_clean = clean_rank[0].item() / (clean_rank ** 2).sum().item() * clean_rank.sum().item()
        print(f"  clean {dset}: {len(cf)} pts, sep={clean_sep:.3f}, "
              f"code_var={clean_code_var:.4f}, effrank={effrank_clean:.1f} ({time.time()-t0:.0f}s)")
        del cf, cl, Xc
        torch.cuda.empty_cache()

        # clean BN mismatch baseline at conv_1.bn
        stats = LayerStats().attach(model)
        for batch in clean_parser.get_train_set():
            with torch.no_grad():
                model(batch[0].to(device))
        stats.detach()
        clean_bn = stats.bn.get('conv_1.bn', {'mm': torch.zeros(1), 'n': 0})
        clean_bn_mm = float(clean_bn['mm'].mean() / max(clean_bn['n'], 1))
        print(f"  clean conv_1.bn mismatch: {clean_bn_mm:.3f}")

        for cond in conds:
            if dset == 'kitti':
                cdir = os.path.join(args.kittic_dir, cond, 'heavy')
                if not os.path.exists(cdir):
                    cdir = os.path.join(args.kittic_dir, cond, 'moderate')
                parser = build_parser(cdir, DATA, ARCH)
            else:
                cdir = os.path.join(args.nusc_c_dir, cond, 'heavy')
                parser = build_nuscenes_parser(cdir, nusc_data, ARCH)
            t1 = time.time()

            # corrupted feature geometry
            pf, pl = [], []
            for zf, l in feature_stream(model, parser, device, args.max_frames):
                pf.append(zf); pl.append(l)
            pf = torch.cat(pf); pl = torch.cat(pl)
            corr_means = class_means_feats(pf, pl)
            corr_sep = per_class_sep(corr_means)
            shift = class_shift(clean_means, corr_means)
            Xp = hdc_codes(pf, proj, device).float()
            corr_code_var = float(torch.var(Xp).item())
            corr_rank = topk_evals(Xp, K=50, device=device)
            effrank_corr = corr_rank[0].item() / (corr_rank ** 2).sum().item() * corr_rank.sum().item()
            del pf, pl, Xp
            torch.cuda.empty_cache()

            # corrupted BN mismatch at conv_1.bn
            stats = LayerStats().attach(model)
            for batch in parser.get_train_set():
                with torch.no_grad():
                    model(batch[0].to(device))
            stats.detach()
            cb = stats.bn.get('conv_1.bn', {'mm': torch.zeros(1), 'n': 0})
            corr_bn_mm = float(cb['mm'].mean() / max(cb['n'], 1))

            # input channel stats
            instats = input_stats_stream(model, parser, device, max_frames=args.max_frames)

            entry = {
                'class_shift_clean_to_corr': shift,
                'sep_clean': clean_sep, 'sep_corr': corr_sep,
                'sep_retention': corr_sep / clean_sep if clean_sep > 0 else float('nan'),
                'bn_mismatch_conv1_clean': clean_bn_mm, 'bn_mismatch_conv1_corr': corr_bn_mm,
                'code_var_clean': clean_code_var, 'code_var_corr': corr_code_var,
                'effrank_clean': effrank_clean, 'effrank_corr': effrank_corr,
                'input_stats_corr': instats,
            }
            results['extractors'][lab]['conds'][cond] = entry
            print(f"  [{cond}] shift={shift:.3f} sep {clean_sep:.2f}->{corr_sep:.2f} "
                  f"(ret {entry['sep_retention']:.2f}) bn_mm {clean_bn_mm:.2f}->{corr_bn_mm:.2f} "
                  f"code_var {clean_code_var:.3f}->{corr_code_var:.3f} "
                  f"effrank {effrank_clean:.1f}->{effrank_corr:.1f} ({time.time()-t1:.0f}s)")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
