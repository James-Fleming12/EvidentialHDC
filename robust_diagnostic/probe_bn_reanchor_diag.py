"""probe_bn_reanchor_diag.py: how much does a BatchNorm running-stat re-anchor
recover on the collapsed conditions, relative to the labeled ceiling?

Diagnostic 5 showed the KITTI-C fog/crosstalk collapse is a frozen-BatchNorm
running-stat mismatch that peaks at the late bottleneck (conv_1.bn / conv_2.bn,
4.6-5.3x clean mismatch), NOT input saturation. The geometry dies because the
frozen running stats (calibrated on clean range/remission) are out of
calibration by the time the signal reaches the 640->256 fusion.

This probe answers: if we re-estimate the late BN running_mean/running_var from
the CORRUPTED stream (statistic substitution -- label-free, closed-form), how
much of the frozen->ceiling gap closes? Decisive comparison per condition:

  frozen      : W0 (fit on clean) decoded with FROZEN BN  = the baseline zero-shot
  bn_recal    : W0 decoded with re-estimated (substituted) BN stats
  ceiling     : W* (fit on corrupted pool) decoded with frozen BN = the labeled bound
  bn_recal_W* : W* decoded with re-estimated BN stats (does recal help the ceiling too?)

Scope control (--bn_scope): which BN modules to re-anchor
  bottleneck : conv_1.bn, conv_2.bn (the 4.6-5.3x mismatch peak)
  late       : layer3.*, layer4.* + conv_1.bn + conv_2.bn
  all        : every BatchNorm

Statistic substitution is the classic TENT/BN-Adapt-style re-estimation:
running_mean = mean of the corrupted-stream activations at that BN's input,
running_var = var thereof, over the stream (or a capped number of frames).

Runtime: ~4-5 streaming passes per condition x 2 conds, 200-500 frames each
(~10-30s/pass). Well under an hour.

Usage:
  uv run python robust_diagnostic/probe_bn_reanchor_diag.py \
    --max_frames 500 --out robust_diagnostic/logs/probe_bn_reanchor.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, build_prototypes, stream_decode_full, ConfAccum, NUM_CLASSES)

SCOPE_MATCH = {
    'bottleneck': ('conv_1.bn', 'conv_2.bn'),
    'late': ('layer3.', 'layer4.', 'conv_1.bn', 'conv_2.bn'),
    'all': ('',),  # prefix '' matches everything
}


class BNStatAccum:
    """Accumulate per-channel mean/var of a BN's INPUT activations over a stream.
    Same hook-signature handling as the layer probe (*args, last-arg-safe)."""

    def __init__(self, scope='late'):
        self.scope = scope
        self.means = {}   # name -> (sum, n)  (running sum of per-channel mean over frames)
        self.vars = {}    # name -> (sum, n)
        self.counts = {}  # name -> frames seen
        self._hooks = []
        self._names = {}

    def _hook(self, module, *args):
        inp = args[0]
        x = inp[0].float()
        name = self._names.get(id(module), None)
        if name is None or not self._in_scope(name):
            return
        # per-channel mean/var over (B, H, W) -- BN input at eval is (B, C, H, W)
        mu = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)
        c = self.counts.setdefault(name, 0)
        self.counts[name] = c + 1
        if name not in self.means:
            self.means[name] = mu.detach().cpu()
            self.vars[name] = var.detach().cpu()
        else:
            self.means[name] = (self.means[name] * c + mu.detach().cpu()) / (c + 1)
            self.vars[name] = (self.vars[name] * c + var.detach().cpu()) / (c + 1)

    def _in_scope(self, name):
        prefixes = SCOPE_MATCH[self.scope]
        return any(name.startswith(p) for p in prefixes)

    def attach(self, model):
        self._names = {id(m): name for name, m in model.named_modules()}
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                self._hooks.append(m.register_forward_hook(self._hook))
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def stats(self):
        return {name: (self.means[name], self.vars[name]) for name in self.means}


def substitute_bn_stats(model, stats):
    """Set running_mean/running_var on the target BN modules from re-estimated stats."""
    applied = []
    for name, m in model.named_modules():
        if name in stats and isinstance(m, torch.nn.BatchNorm2d):
            mu, var = stats[name]
            m.running_mean.copy_(mu.float().to(m.running_mean.device))
            m.running_var.copy_(var.float().clamp(min=1e-6).to(m.running_var.device))
            applied.append(name)
    return applied


def restore_bn_stats(model, snapshot):
    """Restore the ORIGINAL running stats captured in `snapshot` (name -> (mean,var))."""
    for name, m in model.named_modules():
        if name in snapshot and isinstance(m, torch.nn.BatchNorm2d):
            mu, var = snapshot[name]
            m.running_mean.copy_(mu)
            m.running_var.copy_(var)


def snapshot_bn(model):
    return {name: (m.running_mean.detach().clone(), m.running_var.detach().clone())
            for name, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)}


def decode_w(model, parser, proj, device, W, max_frames=0):
    acc = ConfAccum()
    for zf, labels, fi in stream_frames(model, parser, device, max_frames):
        n = len(zf)
        for s in range(0, n, 100000):
            e = min(s + 100000, n)
            codes = torch.sign(zf[s:e].to(device) @ proj).float()
            acc.update((codes @ W.to(device)).argmax(1).cpu(), labels[s:e])
        del zf, labels
    return acc.miou(), acc.n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=500)
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--bn_scope", type=str, default="late",
                    choices=['bottleneck', 'late', 'all'])
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
    results = {'label': 'bn_reanchor', 'bn_scope': args.bn_scope, 'extractors': {}}

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        bn_snapshot = snapshot_bn(model)

        # W0 on clean (frozen BN)
        t0 = time.time()
        cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                      args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device).to(device)
        print(f"  clean W0 done ({len(cf)} pts, {time.time()-t0:.0f}s)")
        del cf, cl, Xc
        torch.cuda.empty_cache()

        results['extractors'][lab] = {'method': method, 'conds': {}}
        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            parser = build_parser(cdir, DATA, ARCH)
            t0 = time.time()

            # 1) labeled ceiling W* on the corrupted pool (frozen BN)
            pf, pl, _ = reservoir_collect(stream_frames(model, parser, device, args.max_frames),
                                          args.pool_cap, 42)
            Xp = hdc_codes(pf, proj, device).float()
            Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device).to(device)
            print(f"  [{cond}] pool {len(pf)} pts, W* done ({time.time()-t0:.0f}s)")
            del pf, pl, Xp
            torch.cuda.empty_cache()

            # 2) frozen baseline: W0 with frozen BN
            miou_frozen, n_frozen = decode_w(model, parser, proj, device, W0, args.max_frames)

            # 3) statistic substitution: re-estimate BN stats on the corrupted stream
            accum = BNStatAccum(scope=args.bn_scope).attach(model)
            for zf, labels, fi in stream_frames(model, parser, device, args.max_frames):
                pass
            accum.detach()
            stats = accum.stats()
            print(f"  [{cond}] re-estimated BN stats for {len(stats)} modules "
                  f"(scope={args.bn_scope}, {accum.counts} frames)")

            # 3a) W0 decoded with re-estimated BN
            applied = substitute_bn_stats(model, stats)
            miou_recal, n_recal = decode_w(model, parser, proj, device, W0, args.max_frames)
            # 3b) W* decoded with re-estimated BN (does recal help the ceiling too?)
            miou_recal_Ws, _ = decode_w(model, parser, proj, device, Ws, args.max_frames)
            restore_bn_stats(model, bn_snapshot)

            # 4) ceiling baseline: W* with frozen BN
            miou_ceiling, n_ceiling = decode_w(model, parser, proj, device, Ws, args.max_frames)

            gap_frozen_ceiling = miou_ceiling - miou_frozen
            gap_recal = miou_ceiling - miou_recal
            frac = (miou_recal - miou_frozen) / gap_frozen_ceiling if gap_frozen_ceiling > 0 else float('nan')
            entry = {
                'n_val': n_frozen,
                'frozen': miou_frozen, 'ceiling': miou_ceiling,
                'frozen_to_ceiling_gap': gap_frozen_ceiling,
                'bn_recal': miou_recal, 'bn_recal_gap_to_ceiling': gap_recal,
                'bn_recal_frac_of_gap': frac,
                'bn_recal_Ws': miou_recal_Ws,
                'bn_modules_reanchored': applied,
                'bn_recal_delta': miou_recal - miou_frozen,
            }
            results['extractors'][lab]['conds'][cond] = entry
            print(f"  [{cond}] frozen={miou_frozen:.3f} bn_recal={miou_recal:.3f} "
                  f"(+{miou_recal-miou_frozen:+.3f}) ceiling={miou_ceiling:.3f} "
                  f"| bn_recal closes {frac:.0%} of frozen->ceiling gap "
                  f"({time.time()-t0:.0f}s)")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
