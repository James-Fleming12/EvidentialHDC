"""probe_ingate_stats_diag.py: the SELF-DETECTING input-IN gate (micro, eval-only).

Motivation (cov_full_scale.md, improvement path 1): gating OFF the cov-shift
per-scan input-IN at inference recovers healthy-condition ceilings, but a gate
keyed on "corruption type" is impractical (we do not know the type in advance).
This tests a STATISTICS-threshold gate that needs NO corruption-type knowledge:

  for each scan, compute the per-scan mean/var of channels {0,4} (range,
  remission) over valid points. Compare to the CLEAN reference (the parser's
  fixed img_means/img_stds). Engage the per-scan input-IN only when the scan's
  statistics deviate from clean by more than a threshold tau:
      engage iff  |var_scan / var_clean - 1| > tau   (on range and/or remission)

  - clean / healthy scans  -> stats near reference  -> input-IN SKIPPED
    (healthy capacity preserved: the D1 clean gap cov 0.520 vs dgl 0.640)
  - genuinely shifted scans -> stats far from reference -> input-IN ENGAGED
    (fog/crosstalk rescue kept)

This is a clean-vs-corrupted discriminator, not a corruption-type classifier.
The same per-scan statistics are computed in D3 of the mechanism probe, so the
gate signal already exists at inference.

Implementation (eval-only, no retrain): the gate-off model is built as
'supcon_vib_dglsspp_instancenorm' (same arch, input_in=False; verified in the D7
probe). To apply the GATE (selective input-IN per scan), we re-implement the
per-scan normalization in the eval loop for scans whose stats exceed tau, and
leave the others untouched -- matching what a deployment-time gate would do.

Outputs per (extractor, dataset, condition): frozen + ceiling mIoU for
  always_on  (current cov-shift behavior, reference)
  always_off (gate fully disabled)
  gate_tau   (per-scan stats threshold, tau in {0.1, 0.5, 1.0, 2.0})
on fog/crosstalk (rescue must stay) and snow/wet_ground (capacity must recover).

Usage:
  uv run python robust_diagnostic/probe_ingate_stats_diag.py \
    --max_frames 200 --out robust_diagnostic/logs/probe_ingate_stats.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, build_nuscenes_parser, hdc_codes, onehot,
    ridge_fit_exact, build_prototypes, ConfAccum, NUM_CLASSES)

# channels the input-IN normalizes: 0 = range, 4 = remission
GATE_CH = (0, 4)

def per_scan_stats(in_vol):
    """Per-scan mean/var of channels {0,4} over valid points (batched)."""
    valid = (in_vol[:, 0:1, :, :] > 0).float()
    mu = {}; var = {}
    for c in GATE_CH:
        x = in_vol[:, c:c+1, :, :] * valid
        denom = valid.sum(dim=(2, 3)).clamp(min=1)
        m = (x.sum(dim=(2, 3)) / denom)                     # (B,1)
        v = ((x - m.unsqueeze(-1).unsqueeze(-1)).pow(2) * valid).sum(dim=(2, 3)) / denom
        mu[c] = m.squeeze(1); var[c] = v.squeeze(1)
    return mu, var

def gate_apply(in_vol, tau, clean_mu, clean_var):
    """Apply per-scan input-IN only to scans whose {range,remission} stats deviate
    from the clean reference by more than tau. Returns the (possibly normalized)
    input. This mirrors the deployment-time gate: engage iff the scan looks
    shifted vs clean, no corruption-type knowledge."""
    mu, var = per_scan_stats(in_vol)
    # deviate if range var OR remission var is far from clean (relative diff)
    dev = torch.zeros(len(in_vol), dtype=torch.bool, device=in_vol.device)
    for c in GATE_CH:
        rel = (var[c] / max(clean_var[c], 1e-12)).clamp(min=1e-3, max=1e3)
        dev |= (rel - 1.0).abs() > tau
    if not dev.any():
        return in_vol
    x = in_vol.clone()
    valid = (in_vol[:, 0:1, :, :] > 0).float()
    for c in GATE_CH:
        sub = x[:, c:c+1, :, :]
        xv = sub * valid
        denom = valid.sum(dim=(2, 3)).clamp(min=1)
        m = xv.sum(dim=(2, 3), keepdim=True) / denom
        v = ((xv - m).pow(2) * valid).sum(dim=(2, 3), keepdim=True) / denom
        std = v.clamp(min=1e-6).sqrt()
        normed = ((xv - m) / std) * valid
        # only overwrite the engaged scans
        x[dev, c:c+1, :, :] = torch.where(valid[dev].bool(),
                                          normed[dev], x[dev, c:c+1, :, :])
    return x

def stream_forward_gated(model, parser, proj, device, W, tau, clean_mu, clean_var,
                         max_frames=0, chunk=100000, exclude=None):
    """Stream all frames, apply the gate to the input (per-scan), decode with W."""
    cm = ConfAccum()
    for i, batch in enumerate(parser.get_train_set()):
        if max_frames > 0 and i >= max_frames:
            break
        in_vol = batch[0].to(device)
        mask = (batch[1].to(device) > 0).view(-1)
        if tau is not None:
            in_vol = gate_apply(in_vol, tau, clean_mu, clean_var)
        with torch.no_grad():
            out = model(in_vol)
            z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            labels = batch[2].to(device).view(-1)[mask].cpu()
        n = len(zf)
        if n == 0:
            continue
        skip = None
        if exclude and i in exclude:
            ex = exclude[i]
            pos = torch.searchsorted(ex, torch.arange(n))
            skip = (pos < len(ex)) & (ex[pos.clamp(max=len(ex)-1)] == torch.arange(n))
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            codes = torch.sign(zf[s:e].to(device) @ proj).float()
            preds = (codes @ W).argmax(1).cpu()
            lbls = labels[s:e]
            if skip is not None:
                m = ~skip[s:e]
                preds, lbls = preds[m], lbls[m]
            cm.update(preds, lbls)
        del zf, labels
    return cm.miou(), cm.n

def clean_ref_stats(model, parser, device, frames=100):
    """The clean reference stats: mean/var of {0,4} over clean scans."""
    mu_acc = {c: 0.0 for c in GATE_CH}; var_acc = {c: 0.0 for c in GATE_CH}; n = 0
    for i, batch in enumerate(parser.get_train_set()):
        if i >= frames:
            break
        in_vol = batch[0]
        mu, var = per_scan_stats(in_vol)
        for c in GATE_CH:
            mu_acc[c] += mu[c].mean().item()
            var_acc[c] += var[c].mean().item()
        n += len(in_vol)
    return {c: mu_acc[c]/max(1, n) for c in GATE_CH}, {c: var_acc[c]/max(1, n) for c in GATE_CH}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--clean_fit_n", type=int, default=50000)
    ap.add_argument("--taus", type=str, default="0.1,0.5,1.0,2.0")
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--extractors", type=str,
                    default="cov_kitti:supcon_vib_dglsspp_inputin_in_chan:"
                            "robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/"
                            "supcon_vib_dglsspp_inputin_in_chan")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.oracle_core import get_hdc_projection
    from modules.gen_trainers import GenTrainer
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    taus = [float(t) for t in args.taus.split(',') if t.strip()]
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    results = {'label': 'ingate_stats', 'extractors': {}}
    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        # gate-off model (input_in=False, same arch) -- the gate RE-ADDS per-scan
        # normalization selectively in the eval loop.
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path,
                             method='supcon_vib_dglsspp_instancenorm')
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        print(f"  model input_in={getattr(model, 'input_in', 'n/a')} (expect False; gate is in the loop)")

        # clean W0 + clean prototypes
        from robust_diagnostic.al_full_dataset_diag import stream_frames, reservoir_collect
        cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                      args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), 1e-3, device).to(device)

        # clean reference stats for the gate
        clean_mu, clean_var = clean_ref_stats(model, clean_parser, device)
        print(f"  clean ref stats: range mu={clean_mu[0]:.3f} var={clean_var[0]:.3f} "
              f"| remission mu={clean_mu[4]:.3f} var={clean_var[4]:.6f}")

        results['extractors'][lab] = {'method': method, 'conds': {}}
        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            parser = build_parser(cdir, DATA, ARCH)
            entry = {}
            for tau in [None] + taus:   # None = always-off, taus = gated, plus always-on ref
                key = 'always_off' if tau is None else f'tau_{tau}'
                t0 = time.time()
                miou, n = stream_forward_gated(model, parser, proj, device, W0, tau,
                                               clean_mu, clean_var,
                                               max_frames=args.max_frames)
                entry[key] = {'frozen': miou, 'n': n}
                print(f"  {cond:12s} {key:12s} frozen={miou:.3f} n={n} ({time.time()-t0:.0f}s)")
            results['extractors'][lab]['conds'][cond] = entry
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
