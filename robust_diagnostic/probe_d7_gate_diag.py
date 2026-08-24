"""probe_d7_gate_diag.py: CORRECTED D7 gate test -- build a FRESH model with
input_in=False and load the checkpoint (the Iteration-0 mechanism probe's
in-place `model.input_in = False` toggle was inert, giving 0.000 deltas).

Question (cov_full_scale.md, improvement path 1): if cov-shift's per-scan
input-IN is gated OFF at eval, do the healthy / cross-domain ceilings recover
toward DGLSS++ while keeping the fog/crosstalk rescue? The gate-off model is a
train/eval mismatch (the checkpoint was trained WITH input-IN), so this tests
INFERENCE-TIME gating, NOT the counterfactual model.

Per cov extractor (cov_kitti, cov_nusc), per condition, full harness:
  * build ResNet with input_in=False, load the checkpoint (strict=False)
  * frozen W0 (clean fit) + ceiling W* (pool fit) decoded on the FULL val
  * reports the input_in=False ceiling/frozen; compare against the
    authoritative input_in=True numbers (al_full_dataset_ep10.json / 
    al_nuscenes_c.json) in the log.

Usage:
  uv run python robust_diagnostic/probe_d7_gate_diag.py \
    --out robust_diagnostic/logs/probe_d7_gate_ep10.json
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
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--nusc_c_dir", type=str, default="/mnt/bravo/jmfleming/nuscenes_c_kitti")
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
    ap.add_argument("--extractors", type=str,
                    default="cov_kitti:supcon_vib_dglsspp_inputin_in_chan:"
                            "robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/"
                            "supcon_vib_dglsspp_inputin_in_chan,"
                            "cov_nusc:supcon_vib_dglsspp_inputin_in_chan:"
                            "robust_diagnostic/logs/nusc_covshift_21ep")
    ap.add_argument("--skip_existing", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.oracle_core import get_hdc_projection
    from modules.gen_trainers import GenTrainer
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    nusc_data = yaml.safe_load(open(args.nusc_labels))

    results = {'label': 'd7_gate', 'extractors': {}}
    if args.skip_existing and os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
            print(f"  [resume] loaded {args.out}")
        except Exception:
            pass

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) GATE-OFF (input_in=False) ===\n{'='*80}")
        if args.skip_existing and lab in results.get('extractors', {}):
            print(f"  [{lab}] already in {args.out}, skipping")
            continue
        arch = _copy.deepcopy(ARCH)
        # Build a FRESH model with input_in DISABLED, then load the checkpoint.
        # CRITICAL: the method name determines input_in (GenTrainer overwrites
        # twobranch from INPUT_NORM_VARIANTS). 'supcon_vib_dglsspp_inputin_in_chan'
        # FORCES input_in=True (gen_trainers.py:290-301). Use
        # 'supcon_vib_dglsspp_instancenorm' instead: same architecture (internal
        # InstanceNorm, norm='in', same param count) but NOT in INPUT_NORM_VARIANTS,
        # so input_in stays False. Loading the inputin_in_chan checkpoint into it
        # (strict=False) gives the cov-shift model with ONLY the per-scan input-IN
        # disabled -- the correct inference-time gating test.
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path,
                             method='supcon_vib_dglsspp_instancenorm')
        model = trainer.model
        print(f"  model input_in={getattr(model, 'input_in', 'n/a')} "
              f"(expect False for the gate-off build)")

        entry = {'method': method, 'conds': {}}
        results['extractors'][lab] = entry
        # clean fit ONCE per extractor (same W0 for every condition)
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        t0 = time.time()
        cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                      args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
        protos_clean = build_prototypes(Xc, cl, device=device)
        print(f"  clean fit done ({time.time()-t0:.0f}s)")
        for dataset, conds in (('kittic', CONDS_ALL), ('nuscenes_c', CONDS_ALL)):
            for cond in conds:
                if dataset == 'kittic':
                    cdir = os.path.join(args.kittic_dir, cond, 'heavy')
                    if not os.path.exists(cdir):
                        cdir = os.path.join(args.kittic_dir, cond, 'moderate')
                    parser = build_parser(cdir, DATA, ARCH)
                else:
                    parser = build_nuscenes_parser(os.path.join(args.nusc_c_dir, cond, 'heavy'),
                                                   nusc_data, ARCH)
                t0 = time.time()
                r = eval_target_condition(model, parser, proj, device, W0, protos_clean,
                                          args, label=lab, cond_name=cond, bal=None)
                entry['conds'][f'{dataset}:{cond}'] = r
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                with open(args.out, 'w') as fh:
                    json.dump(results, fh, indent=2, default=float)
                print(f"  [{dataset}/{cond}] gate-off frozen={r['linear_frozen']:.3f} "
                      f"ceiling={r['linear_ceiling']:.3f} ({time.time()-t0:.0f}s) "
                      f"[checkpoint saved]")
        print(f"[checkpoint] {lab} done")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
