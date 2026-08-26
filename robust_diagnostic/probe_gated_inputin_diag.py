"""probe_gated_inputin_diag.py: test-time adaptation for KITTI-C fog/crosstalk
by GATING a per-scan input re-anchor on the BN-mismatch detector.

The mechanism chain (inter_cov.md):
  - D3: KITTI-C fog erases the remission channel (var -> 1.3e-5) at the input.
  - Diagnostic 8b: the classes are already merged at the BN *input* (pre-BN
    separability collapses 0.70 -> 0.10), so no late-network fix works; the
    recoverable gap is INPUT-bound.
  - Diagnostic 7: per-scan bn_mismatch_conv_1.bn separates fog/crosstalk from
    clean at AUROC 1.000 (label-free).
  - The ONLY thing that rescues fog/crosstalk is re-anchoring the input (cov-
    shift's per-scan input-IN on channels {0,4}), but always-on it costs 0.12
    clean capacity (D1).

This probe tests the TTA: on a PLAIN DGLSS++ extractor (the default), per scan
  gate = bn_mismatch_conv_1 > tau  (detector fires on collapsed scans)
  if gate: apply per-scan input-IN to channels {0,4} before the forward
  else:    raw input (as trained)
so fog/crosstalk get the input re-anchor they need, healthy scans stay raw.
We compare, per condition:
  raw_frozen   : W0 decode, no input-IN anywhere (current DGLSS++ zero-shot)
  gated        : W0 decode with detector-gated input-IN
  always_on    : W0 decode with input-IN on EVERY scan (the cov-shift trade)
  ceiling      : W* on the corrupted pool, no input-IN (labeled oracle bound)

Run on dgl_kitti (supcon_vib_dglsspp), fog + crosstalk + a healthy cond (snow)
to confirm the gate only helps the collapsed ones.

Usage:
  uv run python robust_diagnostic/probe_gated_inputin_diag.py \
    --max_frames 200 --out robust_diagnostic/logs/probe_gated_inputin.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, ConfAccum, NUM_CLASSES)

GATE_BN = 'conv_1.bn'  # the Diagnostic-7 detector (AUROC 1.000)


class GateDetector:
    """Per-scan BN-mismatch signal at a named BN input, computed live during a
    streaming pass (the Diagnostic-7 detector)."""

    def __init__(self, bn_name=GATE_BN):
        self.bn_name = bn_name
        self.signal = []           # per-frame scalar
        self._hooks = []
        self._names = {}

    def _hook(self, module, *args):
        x = args[0][0].float()
        if module.num_batches_tracked <= 0:
            return
        mu = x.mean(dim=(2, 3))
        sig = module.running_var.sqrt().to(x.device) + module.eps
        mm = (mu - module.running_mean.to(x.device)).abs() / sig
        self.signal.append(float(mm.mean().item()))

    def attach(self, model):
        self._names = {id(m): name for name, m in model.named_modules()}
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d) and name == self.bn_name:
                self._hooks.append(m.register_forward_hook(self._hook))
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def apply_input_in_channels(model, in_vol):
    """Per-scan input-IN on channels {0,4} (the cov-shift rescue), applied OUTSIDE
    the model forward (the model is plain DGLSS++, input_in=False). Mirrors
    ResNet._input_instancenorm with norm_channels=(0,4)."""
    x = in_vol.clone()
    sub = x[:, (0, 4)]
    valid = (x[:, 0:1, :, :] > 0).float()
    xv = sub * valid
    denom = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    mu = xv.sum(dim=(2, 3), keepdim=True) / denom
    var = ((xv - mu).pow(2) * valid).sum(dim=(2, 3), keepdim=True) / denom
    std = var.clamp(min=1e-6).sqrt()
    x[:, (0, 4)] = ((xv - mu) / std) * valid
    return x


def decode_w_gated(model, parser, proj, device, W, tau, max_frames=0,
                   mode='gated'):
    """Streaming W decode. mode: 'gated' = input-IN only when detector fires;
    'raw' = never; 'always_on' = always."""
    acc = ConfAccum()
    det = GateDetector().attach(model)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if max_frames > 0 and i >= max_frames:
                break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            # compute the detector on the RAW input (first forward, features discarded)
            if mode == 'gated':
                det.signal = []
                model(in_vol)
                s = det.signal[-1] if det.signal else 0.0
                gate = s > tau
            else:
                gate = (mode == 'always_on')
            # apply input-IN to {0,4} if gated, then extract features
            if gate:
                in_vol = apply_input_in_channels(model, in_vol)
            out = model(in_vol)
            z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            n = len(zf)
            for s in range(0, n, 100000):
                e = min(s + 100000, n)
                codes = torch.sign(zf[s:e].to(device) @ proj).float()
                acc.update((codes @ W.to(device)).argmax(1).cpu(), labels[s:e].cpu())
            del zf
    det.detach()
    return acc.miou(), acc.n


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
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow")
    ap.add_argument("--tau", type=float, default=None,
                    help="gate threshold on bn_mismatch_conv_1 (default: auto-calibrate from clean, mean+3sd)")
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
    results = {'label': 'gated_inputin', 'tau': args.tau, 'extractors': {}}

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)

        # W0 on clean
        t0 = time.time()
        cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                      args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device).to(device)
        print(f"  clean W0 done ({len(cf)} pts, {time.time()-t0:.0f}s)")
        del cf, cl, Xc
        torch.cuda.empty_cache()

        # calibrate the gate threshold on the CLEAN stream (bn_mismatch_conv_1)
        if args.tau is None:
            det = GateDetector().attach(model)
            for batch in clean_parser.get_train_set():
                with torch.no_grad():
                    model(batch[0].to(device))
            det.detach()
            s = torch.tensor(det.signal)
            tau = float(s.mean() + 3.0 * s.std())
            print(f"  clean bn_mismatch_conv_1: mean={s.mean():.3f} sd={s.std():.3f} "
                  f"-> tau={tau:.3f} (mean+3sd)")
        else:
            tau = args.tau

        results['extractors'][lab] = {'method': method, 'tau': tau, 'conds': {}}
        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            parser = build_parser(cdir, DATA, ARCH)
            t0 = time.time()

            # labeled ceiling W* on the corrupted pool
            pf, pl, _ = reservoir_collect(stream_frames(model, parser, device, args.max_frames),
                                          args.pool_cap, 42)
            Xp = hdc_codes(pf, proj, device).float()
            Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device).to(device)
            del pf, pl, Xp
            torch.cuda.empty_cache()

            miou_raw, n_val = decode_w_gated(model, parser, proj, device, W0, tau, args.max_frames, 'raw')
            miou_gated, _ = decode_w_gated(model, parser, proj, device, W0, tau, args.max_frames, 'gated')
            miou_on, _ = decode_w_gated(model, parser, proj, device, W0, tau, args.max_frames, 'always_on')
            miou_ceil, _ = decode_w_gated(model, parser, proj, device, Ws, tau, args.max_frames, 'raw')

            entry = {'n_val': n_val,
                     'raw_frozen': miou_raw, 'gated': miou_gated, 'always_on': miou_on,
                     'ceiling': miou_ceil,
                     'gated_delta': miou_gated - miou_raw,
                     'always_on_delta': miou_on - miou_raw,
                     'gap_to_ceiling_raw': miou_ceil - miou_raw,
                     'gap_to_ceiling_gated': miou_ceil - miou_gated}
            results['extractors'][lab]['conds'][cond] = entry
            print(f"  [{cond}] raw={miou_raw:.3f} gated={miou_gated:.3f} "
                  f"(+{miou_gated-miou_raw:+.3f}) always_on={miou_on:.3f} "
                  f"(+{miou_on-miou_raw:+.3f}) ceiling={miou_ceil:.3f} "
                  f"({time.time()-t0:.0f}s)")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
