"""probe_bn_labeled_diag.py: with LABELS, does updating the BN layer itself
(fitting the per-channel affine gamma/beta so corrupted per-class activations
map to their clean counterparts) recover the fog/crosstalk collapse?

The label-free BN re-anchor (probe_bn_reanchor_diag.py) was NEGATIVE: re-estimating
running_mean/var from the corrupted stream hurt both conditions. But that probe
only moved the running statistics -- it did NOT touch the BN affine (gamma, beta).
This probe tests the labeled ORACLE: with labels, fit each late BN's affine per
channel (least squares over classes) so the corrupted post-BN per-class means
match the clean post-BN per-class means, then decode with W0.

If even the labeled affine fit does not recover, the BN layer is definitively not
the lever -- because an affine (scale+shift) per channel is the maximal expressivity
a BN layer has. That is the decisive "is BN a valid direction" test.

The probe also measures per-class SEPARABILITY of the pre-BN activations on clean
vs corrupted: mean pairwise cosine distance between per-class mean vectors at each
scoped BN. If the corrupted classes are already merged in the BN INPUT (separability
~ 0), no affine can split them -> the info is gone before BN, and the direction is
dead regardless of labels.

Steps per condition (dgl_kitti, fog+crosstalk):
  W0 on clean (frozen BN)                                    -> frozen baseline
  labeled affine fit (per scoped BN, per channel, LSQ over classes):
      minimize sum_c ( gamma * pre_corr[c,ch] + beta - post_clean[c,ch] )^2
      target = clean post-BN per-class mean; source = corrupted pre-BN per-class mean
  decode W0 with: affine-only, affine + re-estimated running stats
  W* on corrupted pool (frozen BN)                           -> labeled ceiling
  report pre-BN class separability clean vs corrupted per scoped BN

Runtime: ~4-5 streaming passes x 2 conds x 500 frames (~15-25 min).

Usage:
  uv run python robust_diagnostic/probe_bn_labeled_diag.py \
    --max_frames 500 --out robust_diagnostic/logs/probe_bn_labeled.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, ConfAccum, NUM_CLASSES)

SCOPE_MATCH = {
    'bottleneck': ('conv_1.bn', 'conv_2.bn'),
    'late': ('layer3.', 'layer4.', 'conv_1.bn', 'conv_2.bn'),
    'all': ('',),
}


class PerClassBNAccum:
    """Accumulate per-class per-channel means of a BN's input or output
    activations over a stream, using the projected label grid (64x2048) aligned
    to the BN's spatial resolution. Set `.labels` to the current frame's label
    grid (batch[2]) before each forward; the hooks downscale it (nearest) to the
    BN's spatial size and accumulate class means over valid pixels."""

    def __init__(self, scope='late', capture='input'):
        self.scope = scope
        self.capture = capture  # 'input' or 'output'
        self.labels = None      # set per frame: (1, 64, 2048) long grid
        self.sums = {}          # name -> (class, ch) sums tensor
        self.counts = {}        # name -> class counts
        self._hooks = []
        self._names = {}

    def _in_scope(self, name):
        return any(name.startswith(p) for p in SCOPE_MATCH[self.scope])

    def _hook(self, module, *args):
        x = (args[0] if self.capture == 'input' else args[-1])
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.detach().float()          # (1, C, h, w)
        C, h, w = x.shape[1], x.shape[2], x.shape[3]
        name = self._names.get(id(module))
        if name is None or not self._in_scope(name):
            return
        if self.labels is None:
            return
        # downscale the label grid to (1,1,h,w); self.labels is (1, 64, 2048)
        lg = F.interpolate(self.labels.float().unsqueeze(0),
                           size=(h, w), mode='nearest').long()[0, 0]
        valid = lg > 0
        if not valid.any():
            return
        x0 = x[0]                       # (C, h, w)
        if name not in self.sums:
            self.sums[name] = torch.zeros(NUM_CLASSES, C)
            self.counts[name] = torch.zeros(NUM_CLASSES)
        sums = self.sums[name].to(x0.device)
        counts = self.counts[name].to(x0.device)
        flat_x = x0.permute(1, 2, 0).reshape(-1, C)   # (h*w, C)
        flat_l = lg.reshape(-1)                        # (h*w,)
        for c in range(1, NUM_CLASSES):
            m = (flat_l == c) & valid.reshape(-1)
            if m.any():
                sums[c] += flat_x[m].sum(dim=0)
                counts[c] += m.sum().float()
        self.sums[name] = sums.cpu()
        self.counts[name] = counts.cpu()

    def attach(self, model):
        self._names = {id(m): name for name, m in model.named_modules()}
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d) and self._in_scope(name):
                hook = m.register_forward_hook(self._hook)
                self._hooks.append((name, hook))
        return self

    def detach(self):
        for name, h in self._hooks:
            h.remove()
        self._hooks = []

    def class_means(self):
        """Return {name: (class -> (C,) mean vector, C)} over classes with counts."""
        out = {}
        for name, s in self.sums.items():
            cnt = self.counts[name]
            means = {}
            for c in range(1, NUM_CLASSES):
                if cnt[c] > 0:
                    means[c] = (s[c] / cnt[c]).float()
            out[name] = (means, s.shape[1])
        return out


def separability(class_means):
    """Mean pairwise cosine distance between per-class mean vectors (0 = merged)."""
    if len(class_means) < 2:
        return float('nan')
    keys = list(class_means.keys())
    vecs = torch.stack([F.normalize(class_means[c], p=2, dim=0) for c in keys])
    sims = vecs @ vecs.t()
    n = len(keys)
    iu = torch.triu_indices(n, n, offset=1)
    return float(1.0 - sims[iu[0], iu[1]].mean())


def fit_affine(pre_means, post_targets, channels):
    """Per-channel affine (gamma, beta) minimizing
    sum_c ( gamma*pre[c,ch] + beta - target[c,ch] )^2 over classes present in both."""
    classes = sorted(set(pre_means) & set(post_targets))
    gamma = torch.ones(channels)
    beta = torch.zeros(channels)
    if len(classes) < 2:
        return gamma, beta
    for ch in range(channels):
        x = torch.stack([pre_means[c][ch] for c in classes])
        y = torch.stack([post_targets[c][ch] for c in classes])
        # normal equations for [gamma, beta]
        X = torch.stack([x, torch.ones_like(x)], dim=1)   # (n, 2)
        A = X.t() @ X
        b = X.t() @ y
        try:
            sol = torch.linalg.solve(A + 1e-6 * torch.eye(2), b)
            gamma[ch] = sol[0]
            beta[ch] = sol[1]
        except Exception:
            pass
    return gamma, beta


def substitute_affine(model, affines):
    """Set BN weight/bias from the fitted {name: (gamma, beta)}."""
    applied = []
    for name, m in model.named_modules():
        if name in affines and isinstance(m, torch.nn.BatchNorm2d):
            g, b = affines[name]
            m.weight.copy_(g.float().to(m.weight.device))
            m.bias.copy_(b.float().to(m.bias.device))
            applied.append(name)
    return applied


def snapshot_affine(model):
    return {name: (m.weight.detach().clone(), m.bias.detach().clone())
            for name, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)}


def restore_affine(model, snap):
    for name, m in model.named_modules():
        if name in snap and isinstance(m, torch.nn.BatchNorm2d):
            m.weight.copy_(snap[name][0])
            m.bias.copy_(snap[name][1])


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
    results = {'label': 'bn_labeled', 'bn_scope': args.bn_scope, 'extractors': {}}

    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        bn_snapshot = snapshot_affine(model)

        # W0 on clean (frozen BN + affine)
        t0 = time.time()
        cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                      args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device).to(device)
        print(f"  clean W0 done ({len(cf)} pts, {time.time()-t0:.0f}s)")
        del cf, cl, Xc
        torch.cuda.empty_cache()

        # clean per-class post-BN means (the labeled target) on the CLEAN stream
        clean_out = PerClassBNAccum(scope=args.bn_scope, capture='output').attach(model)
        clean_sep = {}
        for i, batch in enumerate(clean_parser.get_train_set()):
            if args.max_frames > 0 and i >= args.max_frames:
                break
            clean_out.labels = batch[2].to(device)
            with torch.no_grad():
                model(batch[0].to(device))
            if i % 100 == 0:
                print(f"  [clean post-BN] frame {i}...")
        clean_out.detach()
        clean_means = clean_out.class_means()
        for name, (means, C) in clean_means.items():
            clean_sep[name] = separability(means)
        print(f"  clean post-BN per-class means done ({time.time()-t0:.0f}s, "
              f"{len(clean_means)} scoped BNs)")
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

            # 2) frozen baseline
            miou_frozen, n_val = decode_w(model, parser, proj, device, W0, args.max_frames)

            # 3) corrupted pre-BN per-class means (labeled source)
            corr_in = PerClassBNAccum(scope=args.bn_scope, capture='input').attach(model)
            corr_sep = {}
            for i, batch in enumerate(parser.get_train_set()):
                if args.max_frames > 0 and i >= args.max_frames:
                    break
                corr_in.labels = batch[2].to(device)
                with torch.no_grad():
                    model(batch[0].to(device))
                if i % 100 == 0:
                    print(f"  [{cond} pre-BN] frame {i}...")
            corr_in.detach()
            corr_means = corr_in.class_means()
            for name, (means, C) in corr_means.items():
                corr_sep[name] = separability(means)
            torch.cuda.empty_cache()

            # 4) fit labeled affine: map corrupted pre-BN means to clean post-BN means
            affines = {}
            for name, (cmeans, C) in corr_means.items():
                if name not in clean_means:
                    continue
                clean_cm, _ = clean_means[name]
                affines[name] = fit_affine(cmeans, clean_cm, C)

            # 4a) affine-only (keep running stats frozen)
            applied = substitute_affine(model, affines)
            miou_affine, _ = decode_w(model, parser, proj, device, W0, args.max_frames)
            restore_affine(model, bn_snapshot)

            # 4b) affine + re-estimated running stats
            from robust_diagnostic.probe_bn_reanchor_diag import (
                BNStatAccum, substitute_bn_stats, restore_bn_stats, snapshot_bn)
            accum = BNStatAccum(scope=args.bn_scope).attach(model)
            for zf, labels, fi in stream_frames(model, parser, device, args.max_frames):
                pass
            accum.detach()
            bn_snap2 = snapshot_bn(model)
            substitute_bn_stats(model, accum.stats())
            substitute_affine(model, affines)
            miou_affine_stats, _ = decode_w(model, parser, proj, device, W0, args.max_frames)
            restore_bn_stats(model, bn_snap2)
            restore_affine(model, bn_snapshot)

            # 5) labeled ceiling with frozen BN
            miou_ceiling, _ = decode_w(model, parser, proj, device, Ws, args.max_frames)

            entry = {
                'n_val': n_val,
                'frozen': miou_frozen, 'ceiling': miou_ceiling,
                'bn_labeled_affine': miou_affine,
                'bn_labeled_affine_delta': miou_affine - miou_frozen,
                'bn_labeled_affine_stats': miou_affine_stats,
                'bn_labeled_affine_stats_delta': miou_affine_stats - miou_frozen,
                'pre_bn_sep_clean': clean_sep, 'pre_bn_sep_corr': corr_sep,
                'affine_modules': applied,
            }
            results['extractors'][lab]['conds'][cond] = entry
            frac = (miou_affine - miou_frozen) / (miou_ceiling - miou_frozen) if miou_ceiling > miou_frozen else float('nan')
            print(f"  [{cond}] frozen={miou_frozen:.3f} labeled_affine={miou_affine:.3f} "
                  f"(+{miou_affine-miou_frozen:+.3f}) +stats={miou_affine_stats:.3f} "
                  f"ceiling={miou_ceiling:.3f} | affine closes {frac:.0%} of gap "
                  f"({time.time()-t0:.0f}s)")
            # pre-BN separability summary
            for name in sorted(set(clean_sep) & set(corr_sep)):
                print(f"      sep[{name}] clean={clean_sep[name]:.3f} corr={corr_sep[name]:.3f}")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
        print(f"[checkpoint] {lab} saved")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
