"""eval_stoch_gate_diag.py: gate eval for a CONDITIONAL input-IN (stochastic)
trained checkpoint. Measures frozen mIoU with the per-scan input-IN ON vs OFF on
the SAME trained model, on fog/crosstalk (rescue must stay when ON) and
snow/wet_ground (capacity must recover when OFF).

Unlike the failed eval-only gate probe (whose weights were trained with input-IN
always on), this checkpoint was trained with input_in_prob so the network
learned BOTH normalized and raw inputs. The OFF eval is now a mode the weights
support, not a train/eval mismatch.

The model is built with input_in=True (the stochastic method). For the OFF
eval we set model.input_in = False before decode -- valid because the training
exposed the network to raw inputs with probability (1 - input_in_prob).

Usage:
  uv run python robust_diagnostic/eval_stoch_gate_diag.py \
    --ckpt_dir robust_diagnostic/logs/micro_stoch_stoch/supcon_vib_dglsspp_inputin_in_chan_stoch \
    --method supcon_vib_dglsspp_inputin_in_chan_stoch \
    --out robust_diagnostic/logs/micro_stoch_gate_stoch.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, build_prototypes, ConfAccum, NUM_CLASSES)

CONDS = ['fog', 'crosstalk', 'snow', 'wet_ground']

def decode_frozen(model, parser, proj, device, W, max_frames=0, chunk=100000):
    """Frozen decode of the full val stream with the given W."""
    cm = ConfAccum()
    for zf, labels, fi in stream_frames(model, parser, device, max_frames):
        n = len(zf)
        if n == 0:
            continue
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            codes = torch.sign(zf[s:e].to(device) @ proj).float()
            preds = (codes @ W).argmax(1).cpu()
            cm.update(preds, labels[s:e])
        del zf, labels
    return cm.miou(), cm.n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--method", type=str, required=True)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--clean_fit_n", type=int, default=50000)
    ap.add_argument("--conds", type=str, default=",".join(CONDS))
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.oracle_core import get_hdc_projection
    from modules.gen_trainers import GenTrainer
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    arch = _copy.deepcopy(ARCH)
    trainer = GenTrainer(arch, DATA, args.kitti_dir, args.ckpt_dir,
                         path=args.ckpt_dir, method=args.method)
    model = trainer.model
    print(f"  model input_in_prob={getattr(model, 'input_in_prob', 'n/a')} "
          f"(expect a float for a conditional checkpoint)")
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                  args.clean_fit_n, 7)
    Xc = hdc_codes(cf, proj, device).float()
    W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), 1e-3, device).to(device)

    results = {'label': 'stoch_gate', 'method': args.method,
               'input_in_prob': getattr(model, 'input_in_prob', None), 'conds': {}}
    for cond in conds:
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        parser = build_parser(cdir, DATA, ARCH)
        entry = {}
        # ON eval (as trained with input_in_prob, eval path -> always-on)
        t0 = time.time()
        miou_on, n_on = decode_frozen(model, parser, proj, device, W0, args.max_frames)
        entry['on'] = {'frozen': miou_on, 'n': n_on}
        # OFF eval (valid because training exposed raw inputs)
        model.input_in = False
        miou_off, n_off = decode_frozen(model, parser, proj, device, W0, args.max_frames)
        model.input_in = True
        entry['off'] = {'frozen': miou_off, 'n': n_off}
        entry['delta_off_minus_on'] = miou_off - miou_on
        results['conds'][cond] = entry
        print(f"  {cond:12s} ON={miou_on:.3f} OFF={miou_off:.3f} "
              f"delta={miou_off-miou_on:+.3f} (n={n_on},{n_off}) ({time.time()-t0:.0f}s)")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
