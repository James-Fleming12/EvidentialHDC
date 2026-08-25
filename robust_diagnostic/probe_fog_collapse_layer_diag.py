"""probe_fog_collapse_layer_diag.py: WHERE within DGLSS++ does the KITTI-C
fog/crosstalk collapse originate?

D3 showed the trigger is input-statistics collapse (remission var -> ~0, range
var -> 4.0 on KITTI-C fog). The fog-collapse probe showed the FINAL geometry
collapses (nearest-mean recall 0.143, frozen R4 0.068). But the internal path is
unknown: does the first block saturate/zero (input problem), do frozen BatchNorm
running-stats mismatch (BN-stat problem), or does the collapse build gradually
across blocks (late-stage / decoder-side)?

This probe registers forward hooks on each ResNet-34 stage (conv1, conv2, conv3,
layer1-4, conv_1, conv_2) and, per stage, per condition (clean vs KITTI-C
fog/crosstalk):

  1. activation mean / variance per channel (collapse = variance drop),
  2. % units saturated (|x| > 5) or dead (near-zero variance),
  3. BatchNorm running-stat mismatch: |E[x] - mu_running| / sigma_running at the
     BN input of each conv/basic block. Under clean inputs this is ~O(1); a
     collapsed input channel should push early BN inputs far from their running
     stats.

The decisive split:
  - first block already saturates/zeros  -> fix is at the INPUT (re-anchor
    range/remission before the trunk)
  - collapse builds gradually            -> a late per-scan re-normalization (or
    data-dependent gate) can rescue it without touching the input branch

Per-frame signals are also recorded so a second question is answered in the same
run: does any per-scan layer statistic cleanly SEPARATE the collapsed streams
(fog/crosstalk, "needs normalization") from clean/healthy? Reported as AUROC per
candidate signal (Diagnostic 7).

Run on DGLSS++ (dgl_kitti, the extractor that collapses). Default 200 frames /
fog,crosstalk. Fast: 1 extractor x 3 streams (clean + 2 conds), capped frames.

Usage:
  uv run python robust_diagnostic/probe_fog_collapse_layer_diag.py \
    --max_frames 200 --out robust_diagnostic/logs/probe_fog_collapse_layer.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from robust_diagnostic.al_full_dataset_diag import build_parser, stream_frames

# (module attribute path on the ResNet_34, short label) in forward order
STAGES = [('conv1', 'conv1'), ('conv2', 'conv2'), ('conv3', 'conv3'),
          ('layer1', 'layer1'), ('layer2', 'layer2'), ('layer3', 'layer3'),
          ('layer4', 'layer4'), ('conv_1', 'conv_1'), ('conv_2', 'conv_2')]


class LayerStats:
    """Accumulate per-stage activation + BN-mismatch stats over a frame stream.

    Stage hooks capture the post-activation feature map (1, C, H, W); we keep
    running sums of per-channel mean and variance over the spatial dims, plus
    the fraction of |x| > 5 (saturated) and channels with var < 1e-6 (dead).
    BatchNorm hooks capture |E[x] - mu_running| / sigma_running at each BN input.
    """

    def __init__(self):
        self.stage = {}
        self.bn = {}
        self.signals = {}          # candidate -> list of per-frame scalars
        self._hooks = []

    def _stage_hook(self, label):
        def hook(module, inp, out):
            x = out
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.detach().float()
            C = x.shape[1]
            mu = x.mean(dim=(0, 2, 3)).cpu()
            var = x.var(dim=(0, 2, 3), unbiased=False).cpu()
            sat = (x.abs() > 5.0).float().mean().item()
            dead = (var < 1e-6).float().mean().item()
            s = self.stage.setdefault(label, {'mm': torch.zeros(C), 'vv': torch.zeros(C),
                                              'sat': 0.0, 'dead': 0.0, 'n': 0, 'C': C})
            s['mm'] += mu
            s['vv'] += var
            s['sat'] += sat
            s['dead'] += dead
            s['n'] += 1
            # per-frame candidate signals for the detector AUROC (Diagnostic 7)
            self.signals.setdefault(f'{label}_mean_act', []).append(float(mu.mean()))
            self.signals.setdefault(f'{label}_mean_var', []).append(float(var.mean()))
        return hook

    def _bn_hook(self, module, inp):
        x = inp[0].float()
        if module.num_batches_tracked <= 0:
            return
        mu = x.mean(dim=(2, 3))
        sig = module.running_var.sqrt().to(x.device) + module.eps
        mm = (mu - module.running_mean.to(x.device)).abs() / sig
        mm = mm.mean(dim=0).cpu()
        key = self._bn_names.get(id(module), 'bn?')
        b = self.bn.setdefault(key, {'mm': torch.zeros(mm.shape), 'n': 0, 'C': mm.shape[0]})
        b['mm'] += mm
        b['n'] += 1
        # per-frame BN-mismatch signal (the detector candidate that probes the
        # frozen-stats hypothesis directly)
        self.signals.setdefault(f'bn_mismatch_{key}', []).append(float(mm.mean()))

    def attach(self, model):
        self._bn_names = {id(m): name for name, m in model.named_modules()}
        for path, label in STAGES:
            m = model
            for part in path.split('.'):
                m = getattr(m, part)
            self._hooks.append(m.register_forward_hook(self._stage_hook(label)))
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                self._hooks.append(m.register_forward_hook(self._bn_hook))
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def report(self):
        out = {}
        for label, s in self.stage.items():
            n = max(s['n'], 1)
            out[label] = {
                'mean_act': float(s['mm'].mean() / n),
                'mean_var': float(s['vv'].mean() / n),
                'min_var': float(s['vv'].min() / n),
                'sat_frac': s['sat'] / n,
                'dead_frac': s['dead'] / n,
                'n_frames': s['n'],
            }
        out['bn_mismatch'] = {}
        for name, b in sorted(self.bn.items()):
            out['bn_mismatch'][name] = {
                'mean_mismatch': float(b['mm'].mean() / max(b['n'], 1)),
                'n': b['n']}
        out['signals'] = {k: list(v) for k, v in self.signals.items()}
        return out


def auroc(pos, neg):
    """Area under the ROC curve for a signal separating pos (collapsed) vs neg (clean)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return None
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # handle ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = ranks[i:j + 1].mean()
        i = j + 1
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--extractors", type=str,
                    default="dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.gen_trainers import GenTrainer
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    results = {'label': 'fog_collapse_layer', 'max_frames': args.max_frames, 'extractors': {}}

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model.to(device).eval()
        results['extractors'][lab] = {'method': method, 'streams': {}}

        streams = [('clean', args.kitti_dir)] + \
                  [(c, os.path.join(args.kittic_dir, c, 'heavy')) for c in conds]
        stream_data = {}
        for stream_name, cdir in streams:
            if stream_name != 'clean' and not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, stream_name, 'moderate')
                if not os.path.exists(cdir):
                    print(f"  [skip] {stream_name} not found at {cdir}")
                    continue
            parser = build_parser(cdir, DATA, ARCH)
            stats = LayerStats().attach(model)
            t0 = time.time()
            n_frames = 0
            for zf, labels, fi in stream_frames(model, parser, device, args.max_frames):
                n_frames += 1
            stats.detach()
            rep = stats.report()
            print(f"  [{stream_name}] {n_frames} frames in {time.time()-t0:.0f}s "
                  f"(conv1 mean_act={rep['conv1']['mean_act']:.3f} var={rep['conv1']['mean_var']:.3e})")
            stream_data[stream_name] = rep
            results['extractors'][lab]['streams'][stream_name] = rep

        # Diagnostic 7: per-scan detector AUROC, clean vs each corrupted stream
        clean_sig = stream_data.get('clean', {}).get('signals', {})
        if clean_sig:
            cands = sorted(clean_sig.keys())
            detector = {}
            for cond in conds:
                if cond not in stream_data:
                    continue
                pos = stream_data[cond].get('signals', {})
                detector[cond] = {}
                for cand in cands:
                    if cand in pos:
                        au = auroc(pos[cand], clean_sig[cand])
                        if au is not None:
                            detector[cond][cand] = round(au, 4)
            # best per condition
            for cond, row in detector.items():
                if row:
                    best = max(row, key=row.get)
                    print(f"  [AUROC {cond}] best detector={best} {row[best]:.3f}")
            results['extractors'][lab]['detector_auroc'] = detector
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
