"""probe_fog_collapse_diag.py: is KITTI-C fog so different from NuScenes-C fog that
DGLSS++'s feature space COLLAPSES on one but not the other?

The story to validate: DGLSS++ zero-shot is ~0.097 on KITTI-C fog (collapsed) but
~0.4 on NuScenes-C fog (in-domain W0, NOT collapsed), because the two fog
generators produce different input statistics (KITTI-C fog: remission var ~0;
NuScenes-C fog: range var ~5000, remission kept).

Direct collapse test per (extractor, dataset, condition) -- a SINGLE streaming
pass over the corrupted frames:

  1. per-class nearest-mean recall in the RAW 128-d features (sep_feat) and the
     binarized 10000-d code (sep_code), using clean prototypes (from clean
     frames of the SAME dataset). If the feature space collapses, recall -> the
     ~1/17 random baseline; if it survives, recall stays high.
  2. frozen R4 mIoU with an IN-DOMAIN W0 (KITTI-clean W0 for KITTI-C, nuScenes-
     clean W0 for NuScenes-C) -- the corrected zero-shot that motivated this.
  3. code/feat variance + effective rank (collapse = variance/rank drop).

Run on DGLSS++ (the extractor claimed to collapse) plus cov-shift for contrast,
on fog + crosstalk across KITTI-C and NuScenes-C. MAX_FRAMES keeps it quick.

Usage:
  uv run python robust_diagnostic/probe_fog_collapse_diag.py \
    --max_frames 200 --out robust_diagnostic/logs/probe_fog_collapse.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, build_nuscenes_parser, stream_frames, reservoir_collect,
    hdc_codes, onehot, ridge_fit_exact, build_prototypes, NUM_CLASSES)

CLASS_NAMES = ['unlabeled', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
               'terrain', 'manmade', 'vegetation']

def class_means_feats(feats, lbls, nc=NUM_CLASSES):
    means = torch.zeros(nc, feats.shape[1]); counts = torch.zeros(nc)
    for c in range(nc):
        m = lbls == c
        if m.sum() > 0:
            means[c] += feats[m].sum(dim=0); counts[c] += m.sum()
    for c in range(nc):
        if counts[c] > 0:
            means[c] /= counts[c]
    return F.normalize(means, p=2, dim=1)

def nearest_mean_recall_stream(model, parser, proj, device, refs_code, refs_feat,
                               max_frames=0, chunk=100000):
    """Per-class nearest-mean recall (code + raw feature) on the corrupted stream.
    refs_code: (K, 10000) normalized code prototypes; refs_feat: (K, 128)."""
    rc = torch.zeros(NUM_CLASSES); rf = torch.zeros(NUM_CLASSES)
    sc = torch.zeros(NUM_CLASSES); sf = torch.zeros(NUM_CLASSES)
    refs_code_d = F.normalize(refs_code.float(), p=2, dim=1).to(device)
    refs_feat_d = F.normalize(refs_feat.float(), p=2, dim=1).to(device)
    lbl = torch.arange(NUM_CLASSES).to(device)
    for zf, labels, fi in stream_frames(model, parser, device, max_frames):
        n = len(zf)
        if n == 0:
            continue
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            codes = torch.sign(zf[s:e].to(device) @ proj).float()
            # code nearest-mean
            simc = F.normalize(codes, p=2, dim=1) @ refs_code_d.t()
            pc = lbl[simc.argmax(1)]
            # raw-feature nearest-mean
            zfn = F.normalize(zf[s:e].to(device).float(), p=2, dim=1)
            simf = zfn @ refs_feat_d.t()
            pf = lbl[simf.argmax(1)]
            l = labels[s:e].to(device)
            for c in range(1, NUM_CLASSES):
                lc = (l == c)
                if lc.any():
                    sc[c] += lc.sum(); rc[c] += (pc[lc] == c).sum()
                    sf[c] += lc.sum(); rf[c] += (pf[lc] == c).sum()
            del codes
        del zf, labels
    return ({CLASS_NAMES[c]: float(rc[c] / sc[c].clamp(min=1)) for c in range(NUM_CLASSES)},
            {CLASS_NAMES[c]: float(rf[c] / sf[c].clamp(min=1)) for c in range(NUM_CLASSES)})

def frozen_w0_fit(model, parser, proj, device, clean_fit_n, seed=7):
    cf, cl, _ = reservoir_collect(stream_frames(model, parser, device, 0), clean_fit_n, seed)
    Xc = hdc_codes(cf, proj, device).float()
    W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), 1e-3, device)
    protos_code = build_prototypes(Xc, cl, device=device).cpu()
    protos_feat = class_means_feats(cf, cl)
    return W0, protos_code, protos_feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--nusc_dir", type=str, default="/mnt/alpha/jmfleming/nuscenes_kitti")
    ap.add_argument("--nusc_c_dir", type=str, default="/mnt/bravo/jmfleming/nuscenes_c_kitti")
    ap.add_argument("--nusc_labels", type=str, default="config/labels/nuscenes_c.yaml")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--clean_fit_n", type=int, default=100000)
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--extractors", type=str,
                    default="dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp,"
                            "cov_kitti:supcon_vib_dglsspp_inputin_in_chan:"
                            "robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.oracle_core import get_hdc_projection
    from modules.gen_trainers import GenTrainer
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    nusc_data = yaml.safe_load(open(args.nusc_labels))
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    results = {'label': 'fog_collapse', 'extractors': {}}

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        results['extractors'][lab] = {'method': method, 'datasets': {}}
        for dset in ('kitti', 'nusc'):
            # clean W0 + prototypes IN-DOMAIN
            if dset == 'kitti':
                clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
            else:
                clean_parser = build_nuscenes_parser(args.nusc_dir, nusc_data, ARCH)
            t0 = time.time()
            W0, protos_code, protos_feat = frozen_w0_fit(model, clean_parser, proj, device,
                                                         args.clean_fit_n)
            print(f"  [{dset}] in-domain W0 + clean protos done ({time.time()-t0:.0f}s)")
            results['extractors'][lab]['datasets'][dset] = {}
            for cond in conds:
                if dset == 'kitti':
                    cdir = os.path.join(args.kittic_dir, cond, 'heavy')
                    if not os.path.exists(cdir):
                        cdir = os.path.join(args.kittic_dir, cond, 'moderate')
                    parser = build_parser(cdir, DATA, ARCH)
                else:
                    parser = build_nuscenes_parser(os.path.join(args.nusc_c_dir, cond, 'heavy'),
                                                   nusc_data, ARCH)
                print(f"  [{dset}/{cond}] collapse metrics...")
                t1 = time.time()
                rec_code, rec_feat = nearest_mean_recall_stream(
                    model, parser, proj, device, protos_code, protos_feat,
                    max_frames=args.max_frames)
                # frozen R4 mIoU with the in-domain W0 (reuse stream + a ConfMatrix)
                from robust_diagnostic.al_full_dataset_diag import ConfMatrix
                cm = ConfMatrix()
                W0d = W0.to(device)
                for zf, labels, fi in stream_frames(model, parser, device, args.max_frames):
                    n = len(zf)
                    for s in range(0, n, 100000):
                        e = min(s + 100000, n)
                        codes = torch.sign(zf[s:e].to(device) @ proj).float()
                        cm.update((codes @ W0d).argmax(1).cpu(), labels[s:e])
                    del zf, labels
                miou = cm.miou()
                rec_code_m = {CLASS_NAMES[c]: rec_code[CLASS_NAMES[c]] for c in range(1, NUM_CLASSES)}
                rec_feat_m = {CLASS_NAMES[c]: rec_feat[CLASS_NAMES[c]] for c in range(1, NUM_CLASSES)}
                mean_rec_code = float(np.mean([v for v in rec_code_m.values() if v > 0]))
                mean_rec_feat = float(np.mean([v for v in rec_feat_m.values() if v > 0]))
                results['extractors'][lab]['datasets'][dset][cond] = {
                    'frozen_R4_miou': miou,
                    'mean_recall_code': mean_rec_code,
                    'mean_recall_feat': mean_rec_feat,
                    'recall_code': rec_code, 'recall_feat': rec_feat,
                    'n_scan_time_s': round(time.time() - t1, 1),
                }
                print(f"    frozen R4={miou:.3f} code recall={mean_rec_code:.3f} "
                      f"feat recall={mean_rec_feat:.3f} ({time.time()-t1:.0f}s)")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
