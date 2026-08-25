"""probe_nusc_c_w0source_diag.py: validate the corrected NuScenes-C story with
ONE consistent run -- zero-shot W0 fit on nuScenes-clean (in-domain) instead of
KITTI-clean (64-beam, cross-domain), and the ceiling W* fit in-domain on the
corrupted pool.

Why this exists: the README's corrected NuScenes-C table currently mixes two
sources -- zero-shot from the mechanism probe's `frozen_W0_alt`, ceiling from
the authoritative al-diag. This run reproduces BOTH in one harness, using the
EXACT `eval_target_condition` machinery that produced al_nuscenes_c*.json, so
the frozen/ceiling/gap/AL story is self-consistent and directly comparable.

Per extractor (NuScenes-trained pair), per NuScenes-C condition (heavy):
  * zero-shot: W0 fit on nuScenes-clean (reservoir 200k, seed 7) -- in-domain
  * ceiling:   W* fit on the corrupted pool (reservoir 400k, seed 42)
  * W_res     : the 56+500 random-bank low-rank residual (oracle U r=8)
  * val       : every point of every frame, pool points excluded
  * also reports the KITTI-clean-W0 frozen for reference (the contaminated
    baseline that motivated this run)

Usage:
  uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py \
    --out robust_diagnostic/logs/probe_nusc_c_w0source_ep10.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, build_nuscenes_parser, stream_frames, reservoir_collect,
    hdc_codes, onehot, ridge_fit_exact, build_prototypes,
    eval_target_condition, NUM_CLASSES, CONDS_ALL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--nusc_dir", type=str, default="/mnt/alpha/jmfleming/nuscenes_kitti",
                    help="pristine nuScenes in KITTI format (nuScenes-clean W0 source)")
    ap.add_argument("--nusc_c_dir", type=str, default="/mnt/bravo/jmfleming/nuscenes_c_kitti")
    ap.add_argument("--nusc_c_sev", type=str, default="heavy",
                    help="NuScenes-C severity (heavy/moderate/light) for the 3-severity "
                         "average matching GeoID's reporting")
    ap.add_argument("--nusc_labels", type=str, default="config/labels/nuscenes_c.yaml")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--bank_k", type=int, default=8)
    ap.add_argument("--bank_extra", type=int, default=500)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--no_wres", type=int, default=1,
                    help="skip the W_res oracle-U SVD (slow); only frozen/ceiling "
                         "are needed for the W0-source probe")
    ap.add_argument("--proj_dim", type=int, default=10000,
                    help="HDC projection dimension (default 10000; 2000 to verify "
                         "the code-2000 peak on NuScenes-C)")
    ap.add_argument("--extractors", type=str,
                    default="cov_nusc:supcon_vib_dglsspp_inputin_in_chan:"
                            "robust_diagnostic/logs/nusc_covshift_21ep,"
                            "dgl_nusc:supcon_vib_dglsspp:"
                            "robust_diagnostic/logs/nusc_dglsspp_21ep")
    ap.add_argument("--skip_existing", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.oracle_core import get_hdc_projection
    from modules.gen_trainers import GenTrainer
    proj = get_hdc_projection(dim_in=128, dim_out=args.proj_dim, device=device)
    nusc_data = yaml.safe_load(open(args.nusc_labels))
    conds = CONDS_ALL

    results = {'label': 'nusc_w0source', 'extractors': {}}
    if args.skip_existing and os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
            print(f"  [resume] loaded {args.out}")
        except Exception as e:
            print(f"  [resume] WARNING: {e}; starting fresh")

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        if args.skip_existing and lab in results.get('extractors', {}):
            print(f"  [{lab}] already in {args.out}, skipping")
            continue
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model

        # ---- W0 fit on nuScenes-clean (IN-DOMAIN zero-shot) ----
        nusc_clean_parser = build_nuscenes_parser(args.nusc_dir, nusc_data, ARCH)
        t0 = time.time()
        print(f"=== [{lab}] in-domain W0 fit (nuScenes-clean reservoir {args.clean_fit_n}) ===")
        cf, cl, ck = reservoir_collect(stream_frames(model, nusc_clean_parser, device, args.max_frames),
                                       args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0_nusc = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
        protos_clean = build_prototypes(Xc, cl, device=device)
        print(f"  W0_nusc done ({len(cf)} pts, {time.time()-t0:.0f}s)")

        # ---- also the contaminated KITTI-clean W0 for the reference row ----
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        cfk, clk, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                        args.clean_fit_n, 7)
        Xck = hdc_codes(cfk, proj, device).float()
        W0_kitti = ridge_fit_exact(Xck, onehot(clk, NUM_CLASSES), args.lam, device)

        entry = {'method': method, 'conds': {}, 'W0_nusc_n': len(cf)}
        results['extractors'][lab] = entry
        for cond in conds:
            cdir = os.path.join(args.nusc_c_dir, cond, args.nusc_c_sev)
            parser = build_nuscenes_parser(cdir, nusc_data, ARCH)
            print(f"\n=== [{lab}] {cond} ===")
            r = eval_target_condition(model, parser, proj, device, W0_nusc, protos_clean,
                                      args, label=lab, cond_name=cond, bal=None)
            # reference: frozen with the KITTI-clean W0 (contaminated baseline)
            from collections import defaultdict
            pf, pl, pk = reservoir_collect(stream_frames(model, parser, device, args.max_frames),
                                           args.pool_cap, 42)
            ex_by_frame = defaultdict(list)
            for f, i in pk.tolist():
                ex_by_frame[f].append(i)
            ex_by_frame = {f: torch.tensor(sorted(s), dtype=torch.long) for f, s in ex_by_frame.items()}
            from robust_diagnostic.al_full_dataset_diag import stream_decode_full
            accs = stream_decode_full(model, parser, proj, device,
                                      {'frozen_kittiW0': {'type': 'w', 'W': W0_kitti.detach().cpu()}},
                                      exclude=ex_by_frame, max_frames=args.max_frames)
            r['frozen_kittiW0'] = accs['frozen_kittiW0'].miou()
            r['delta_zs_in_vs_cross'] = r['linear_frozen'] - r['frozen_kittiW0']
            entry['conds'][cond] = r
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
            print(f"  [checkpoint] {lab}/{cond} saved to {args.out}")
        print(f"\n[checkpoint] extractor {lab} done")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
