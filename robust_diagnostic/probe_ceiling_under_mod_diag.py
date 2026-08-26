"""probe_ceiling_under_mod_diag.py: do input normalization / BN-stat changes
create a MORE ROBUST FEATURE SPACE (higher recoverable ceiling), or do they
merely fit the zero-shot classifier better?

The earlier BN/input probes (Diagnostics 8, 8b, 9) always fit W* on the FROZEN
features, so they answered "does the modification fit the zero-shot W0 better"
(NEGATIVE for BN, small for gated input-IN) but NOT "does the modification raise
the labeled ceiling". This probe refits BOTH W0 (clean) and W* (corrupted pool)
on the MODIFIED features, so the ceiling-under-modification is directly compared
to the frozen ceiling.

Modes tested (each is a feature-space modification applied at eval AND at the
pool/clean fit):
  none         : baseline (no modification) -> the current frozen ceiling
  inputin_on   : per-scan input-IN on channels {0,4} on EVERY scan
  inputin_gate : input-IN on channels {0,4} ONLY when bn_mismatch_conv_1 > tau
  bn_reanchor  : re-estimate late-BN running stats on the corrupted stream
                 (statistic substitution), apply to clean+pool+val forwards

Per mode, per condition:
  W0_mod = ridge fit on MODIFIED clean features
  W*_mod = ridge fit on MODIFIED corrupted pool
  decode MODIFIED val with W0_mod  -> zero-shot under modification
  decode MODIFIED val with W*_mod  -> CEILING under modification   <-- the answer
  (also decode with the frozen W* for the mismatch reference)

Decisive comparison: ceiling_under_mod vs the none-mode ceiling. If a mode's
ceiling > the frozen ceiling, the modification creates a more robust feature
space (real headroom gain for TTA/AL); if equal, it only re-fits the classifier.

Usage:
  uv run python robust_diagnostic/probe_ceiling_under_mod_diag.py \
    --max_frames 200 --out robust_diagnostic/logs/probe_ceiling_under_mod.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, ConfAccum, NUM_CLASSES)
from robust_diagnostic.probe_gated_inputin_diag import (
    GateDetector, apply_input_in_channels)
from robust_diagnostic.probe_bn_reanchor_diag import (
    BNStatAccum, substitute_bn_stats, restore_bn_stats, snapshot_bn)

GATE_BN = 'conv_1.bn'


def forward_mod(model, in_vol, mode, tau):
    """Apply the modification to the input volume, then run the model forward.
    Returns z8 features (B, C, H, W)."""
    if mode in ('inputin_on', 'inputin_gate'):
        if mode == 'inputin_gate':
            # need the detector on the RAW input first
            raise RuntimeError("gated mode must use extract_features_mod")
        in_vol = apply_input_in_channels(model, in_vol)
    return model(in_vol)


def extract_features_mod(model, parser, proj, device, mode, tau=0.0, max_frames=0):
    """Streaming feature extraction (zf, labels, frame_idx) with the modification
    applied per frame. mode in {none, inputin_on, inputin_gate, bn_reanchor}.
    For bn_reanchor, the BN stats must be substituted BEFORE calling (the caller
    sets them from the corrupted stream)."""
    model.eval()
    out = []
    det = GateDetector(GATE_BN).attach(model) if mode == 'inputin_gate' else None
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if max_frames > 0 and i >= max_frames:
                break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            gate = False
            if mode == 'inputin_gate':
                det.signal = []
                model(in_vol)                      # pass 1: detector on raw input
                gate = (det.signal[-1] if det.signal else 0.0) > tau
            if mode in ('inputin_on', 'inputin_gate'):
                if gate or mode == 'inputin_on':
                    in_vol = apply_input_in_channels(model, in_vol)
            out_tuple = model(in_vol)
            z8 = out_tuple[2] if len(out_tuple) == 3 else out_tuple[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            out.append((zf.cpu(), labels[mask].cpu(), i))
    if det is not None:
        det.detach()
    return out


def fit_and_decode(feats_stream, val_stream, proj, device, lam, n_classes=NUM_CLASSES,
                   clean_n=100000, pool_n=200000, seed_clean=7, seed_pool=42):
    """Given feature streams (each a list of (zf, labels, idx) tuples), reservoir-
    sample clean/pool, fit W0 and W*, and decode the val stream. Returns
    (frozen, ceiling, n_val)."""
    def streamer(stream, cap, seed):
        return reservoir_collect(iter(stream), cap, seed)
    clean_feats, clean_lbls, _ = streamer(feats_stream['clean'], clean_n, seed_clean)
    pool_feats, pool_lbls, _ = streamer(feats_stream['pool'], pool_n, seed_pool)
    Xc = hdc_codes(clean_feats, proj, device).float()
    Xp = hdc_codes(pool_feats, proj, device).float()
    W0 = ridge_fit_exact(Xc, onehot(clean_lbls, n_classes), lam, device).to(device)
    Ws = ridge_fit_exact(Xp, onehot(pool_lbls, n_classes), lam, device).to(device)
    acc0 = ConfAccum(); accs = ConfAccum()
    for zf, labels, _ in val_stream:
        n = len(zf)
        for s in range(0, n, 100000):
            e = min(s + 100000, n)
            codes = torch.sign(zf[s:e].to(device) @ proj).float()
            acc0.update((codes @ W0).argmax(1).cpu(), labels[s:e])
            accs.update((codes @ Ws).argmax(1).cpu(), labels[s:e])
    return acc0.miou(), accs.miou(), acc0.n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--clean_fit_n", type=int, default=100000)
    ap.add_argument("--pool_cap", type=int, default=200000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--modes", type=str,
                    default="none,inputin_on,inputin_gate,bn_reanchor")
    ap.add_argument("--extractors", type=str,
                    default="dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.oracle_core import get_hdc_projection
    from modules.gen_trainers import GenTrainer
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    results = {'label': 'ceiling_under_mod', 'modes': modes, 'extractors': {}}

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)

        # calibrate the gate tau on clean (bn_mismatch_conv_1)
        det = GateDetector(GATE_BN).attach(model)
        for batch in clean_parser.get_train_set():
            with torch.no_grad():
                model(batch[0].to(device))
        det.detach()
        s = torch.tensor(det.signal)
        tau = float(s.mean() + 3.0 * s.std())
        print(f"  clean bn_mismatch_conv_1 mean={s.mean():.3f} sd={s.std():.3f} -> tau={tau:.3f}")

        results['extractors'][lab] = {'method': method, 'tau': tau, 'conds': {}}
        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            parser = build_parser(cdir, DATA, ARCH)
            print(f"\n=== [{cond}] ===")

            # per-mode: extract clean / pool / val features under the modification,
            # then fit W0+W* on the MODIFIED features and decode.
            # bn_reanchor: re-estimate late-BN stats on the corrupted stream first,
            # then apply to ALL forwards.
            bn_snapshot = None
            if 'bn_reanchor' in modes:
                # re-estimate BN stats on the corrupted stream (frozen mode)
                accum = BNStatAccum(scope='late').attach(model)
                for zf, labels, fi in stream_frames(model, parser, device, args.max_frames):
                    pass
                accum.detach()
                bn_snapshot = snapshot_bn(model)
                substitute_bn_stats(model, accum.stats())

            for mode in modes:
                if mode == 'bn_reanchor' and bn_snapshot is None:
                    continue
                t0 = time.time()
                streams = {
                    'clean': extract_features_mod(model, clean_parser, proj, device, mode, tau, args.max_frames),
                    'pool':  extract_features_mod(model, parser, proj, device, mode, tau, args.max_frames),
                    'val':   extract_features_mod(model, parser, proj, device, mode, tau, args.max_frames),
                }
                frozen, ceiling, n_val = fit_and_decode(
                    streams, streams['val'], proj, device, args.lam,
                    clean_n=args.clean_fit_n, pool_n=args.pool_cap)
                results['extractors'][lab]['conds'].setdefault(cond, {})[mode] = {
                    'frozen': frozen, 'ceiling': ceiling, 'n_val': n_val,
                }
                print(f"  [{mode:12s}] frozen={frozen:.4f} ceiling={ceiling:.4f} "
                      f"({time.time()-t0:.0f}s)")
            # restore frozen BN for the next condition
            if bn_snapshot is not None:
                restore_bn_stats(model, bn_snapshot)
            torch.cuda.empty_cache()

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
