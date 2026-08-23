"""probe_nusc_c_dglsspp_vs_covshift_diag.py: why does the DGLSS++ ceiling beat
cov-shift on NuScenes -> NuScenes-C, when on KITTI -> KITTI-C cov-shift wins the
ceiling by 2-3x?

The paper numbers show the flip at the CEILING only:
  KITTI-C    (KITTI-trained): cov-shift ceil 0.369 fog vs dglsspp 0.170 (2.2x)
  NuScenes-C (NuScenes-trn):  cov-shift ceil 0.404 fog vs dglsspp 0.526 (dgl wins)
while cov-shift keeps the zero-shot edge on NuScenes-C (20.3 vs 13.3 mean).

This diagnostic measures the MECHANISM per extractor, per condition, in one
streaming pass (same full harness: 200k clean fit, 400k pool, spectral-exact
ridge, pool excluded from val):

  * recoverable residual  r = ||W* - W0||_F / ||W0||_F   (how much structure the
    labeled ceiling can add; the "compression" test -- cov-shift caps the
    ceiling if its residual is systematically smaller)
  * per-class ceiling structure: W0 IoU, W* IoU, delta, and the corrupted-pool
    per-class support (does one extractor's pool carry more recoverable classes?)
  * code-space nearest-mean separability (clean vs pool prototypes) per class
  * AL-gauge signals on the code: mean_shift_cos, conf_drop (does cov-shift's
    normalization change the shift geometry on NuScenes-C vs KITTI-C?)

Run on the NuScenes-trained pair over all NuScenes-C heavy conditions, plus the
KITTI-trained pair over KITTI-C (fog/crosstalk/wet_ground) as the contrast where
cov-shift wins the ceiling.

Usage:
  uv run python robust_diagnostic/probe_nusc_c_dglsspp_vs_covshift_diag.py \
    --out robust_diagnostic/logs/probe_nusc_c_flip_ep10.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, build_prototypes, knn_predict, NUM_CLASSES, CONDS_ALL)

CLASS_NAMES = ['unlabeled', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
               'terrain', 'manmade', 'vegetation']

class ConfMatrix:
    """Streaming per-class confusion (rows = pred, cols = true) -> per-class IoU."""
    def __init__(self, nc=NUM_CLASSES):
        self.C = torch.zeros(nc, nc); self.n = 0
    def update(self, preds, lbls):
        p = preds.long(); l = lbls.long()
        self.C += torch.bincount(p * NUM_CLASSES + l, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
        self.n += len(p)
    def per_class_iou(self):
        tp = torch.diag(self.C); fp = self.C.sum(1) - tp; fn = self.C.sum(0) - tp
        d = tp + fp + fn
        return (tp / d.clamp(min=1e-9)).tolist()
    def miou(self):
        ious = self.per_class_iou()
        present = self.C.sum(0)[1:] > 0
        vals = [ious[c] for c in range(1, NUM_CLASSES) if present[c - 1]]
        return float(np.mean(vals)) if vals else 0.0

def class_means_feats(feats, lbls, num_classes=NUM_CLASSES):
    means = torch.zeros(num_classes, feats.shape[1])
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        m = lbls == c
        if m.sum() > 0:
            means[c] += feats[m].sum(dim=0); counts[c] += m.sum()
    for c in range(num_classes):
        if counts[c] > 0:
            means[c] /= counts[c]
    return F.normalize(means, p=2, dim=1)

def stream_per_class(model, parser, proj, device, decoders, exclude=None,
                     max_frames=0, chunk=100000, nn_prep=None):
    """Stream full val; per-class IoU per ridge decoder + per-class nearest-mean
    recall per nn_prep entry (kind code/feat). Returns (per_class, miou)."""
    pc = {name: ConfMatrix() for name in decoders}
    nn_tp = {n: torch.zeros(NUM_CLASSES) for n in (nn_prep or {})}
    nn_sup = {n: torch.zeros(NUM_CLASSES) for n in (nn_prep or {})}
    prep = {}
    for name, dec in decoders.items():
        if dec['type'] == 'w':
            prep[name] = ('w', dec['W'].to(device))
        else:
            prep[name] = ('proto', dec['protos'].to(device), dec['proto_lbls'].to(device))
    nn_d = {}
    for name, cfg in (nn_prep or {}).items():
        refs = F.normalize(cfg['refs'].float(), p=2, dim=1).to(device)
        nn_d[name] = (cfg['kind'], refs, cfg['lbls'].to(device))
    for zf, labels, fi in stream_frames(model, parser, device, max_frames):
        n = len(zf)
        if n == 0:
            continue
        skip = None
        if exclude and fi in exclude:
            ex = exclude[fi]
            pos = torch.searchsorted(ex, torch.arange(n))
            skip = (pos < len(ex)) & (ex[pos.clamp(max=len(ex) - 1)] == torch.arange(n))
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            codes = torch.sign(zf[s:e].to(device) @ proj).float()
            for name, p in prep.items():
                if p[0] == 'w':
                    preds = (codes @ p[1]).argmax(1).cpu()
                else:
                    sims = F.normalize(codes, p=2, dim=1) @ p[1].t()
                    preds = p[2][sims.argmax(1)].cpu()
                lbls = labels[s:e]
                if skip is not None:
                    m = ~skip[s:e]
                    preds, lbls = preds[m], lbls[m]
                pc[name].update(preds, lbls)
            for name, (kind, refs, rlbls) in nn_d.items():
                if kind == 'code':
                    sims = F.normalize(codes, p=2, dim=1) @ refs.t()
                else:
                    sims = F.normalize(zf[s:e].to(device).float(), p=2, dim=1) @ refs.t()
                preds = rlbls[sims.argmax(1)].cpu()
                lbls = labels[s:e]
                if skip is not None:
                    m = ~skip[s:e]
                    preds, lbls = preds[m], lbls[m]
                for c in range(1, NUM_CLASSES):
                    lc = (lbls == c)
                    if lc.any():
                        nn_sup[name][c] += lc.sum()
                        nn_tp[name][c] += (preds[lc] == c).sum()
            del codes
        del zf, labels
    nn = {n: {'recall': nn_tp[n] / nn_sup[n].clamp(min=1), 'support': nn_sup[n]}
          for n in (nn_prep or {})}
    return pc, nn

def run_extractor_condition(model, parser, proj, device, W0, protos_clean,
                            feat_means_clean, args, label, cond_name):
    from collections import defaultdict
    t0 = time.time()
    print(f"\n=== [{label}] {cond_name} ===")
    pf, pl, pk = reservoir_collect(stream_frames(model, parser, device, args.max_frames),
                                   args.pool_cap, 42)
    Xp = hdc_codes(pf, proj, device).float()
    Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device)
    protos_pool = build_prototypes(Xp, pl, device=device)
    feat_means_pool = class_means_feats(pf, pl)
    resid = (Ws - W0).detach().cpu().float()
    r_norm = float(torch.norm(resid) / torch.norm(W0.detach().cpu().float()))
    # gauge: code-mean shift + confidence drop between clean and corrupted pool
    W0d = W0.to(device)
    with torch.no_grad():
        mean_c = F.normalize(protos_clean.mean(0, keepdim=True).float(), p=2, dim=1)
        mean_p = F.normalize(protos_pool.mean(0, keepdim=True).float(), p=2, dim=1)
        mean_shift_cos = float((mean_c * mean_p).sum().item())
        logits_p = Xp.to(device).float() @ W0d
        conf_p = torch.softmax(logits_p, dim=1).max(1).values.mean().item()
        # clean conf reference on the clean reservoir (approx: use clean protos sim)
        conf_c = float(F.normalize(protos_clean, p=2, dim=1).mean(0).norm().item())
    pool_counts = torch.bincount(pl.long(), minlength=NUM_CLASSES)

    from collections import defaultdict as _dd
    ex_by_frame = _dd(list)
    for f, i in pk.tolist():
        ex_by_frame[f].append(i)
    ex_by_frame = {f: torch.tensor(sorted(s), dtype=torch.long) for f, s in ex_by_frame.items()}
    decoders = {
        'frozen': {'type': 'w', 'W': W0.detach().cpu()},
        'ceiling': {'type': 'w', 'W': Ws.detach().cpu()},
        'proto_clean': {'type': 'proto', 'protos': protos_clean.cpu(),
                        'proto_lbls': torch.arange(NUM_CLASSES)},
        'proto_pool': {'type': 'proto', 'protos': protos_pool.cpu(),
                       'proto_lbls': torch.arange(NUM_CLASSES)},
    }
    nn_prep = {
        'sep_code_clean': {'kind': 'code', 'refs': protos_clean.cpu(),
                           'lbls': torch.arange(NUM_CLASSES)},
        'sep_code_pool': {'kind': 'code', 'refs': protos_pool.cpu(),
                          'lbls': torch.arange(NUM_CLASSES)},
        'sep_feat_clean': {'kind': 'feat', 'refs': feat_means_clean,
                           'lbls': torch.arange(NUM_CLASSES)},
        'sep_feat_pool': {'kind': 'feat', 'refs': feat_means_pool,
                          'lbls': torch.arange(NUM_CLASSES)},
    }
    print(f"  resid ||W*-W0||/||W0||={r_norm:.3f} shift_cos={mean_shift_cos:.3f} "
          f"conf_p={conf_p:.3f} conf_c={conf_c:.3f} ({time.time()-t0:.0f}s)")
    pc, nn = stream_per_class(model, parser, proj, device, decoders,
                              exclude=ex_by_frame, max_frames=args.max_frames,
                              nn_prep=nn_prep)
    n_val = pc['frozen'].n
    out = {
        'n_pool': len(pf), 'n_val': n_val,
        'resid_rel': r_norm, 'mean_shift_cos': mean_shift_cos,
        'conf_pool': conf_p, 'conf_clean_ref': conf_c,
        'frozen': pc['frozen'].miou(), 'ceiling': pc['ceiling'].miou(),
        'gap': pc['ceiling'].miou() - pc['frozen'].miou(),
        'proto_clean': pc['proto_clean'].miou(), 'proto_pool': pc['proto_pool'].miou(),
        'pool_support': {CLASS_NAMES[c]: int(pool_counts[c]) for c in range(NUM_CLASSES)},
        'per_class_frozen': {CLASS_NAMES[c]: float(pc['frozen'].per_class_iou()[c])
                             for c in range(NUM_CLASSES)},
        'per_class_ceiling': {CLASS_NAMES[c]: float(pc['ceiling'].per_class_iou()[c])
                              for c in range(NUM_CLASSES)},
        'per_class_gap': {CLASS_NAMES[c]:
                          float(pc['ceiling'].per_class_iou()[c] - pc['frozen'].per_class_iou()[c])
                          for c in range(NUM_CLASSES)},
    }
    for k in nn_prep:
        out[k] = {'recall': {CLASS_NAMES[c]: float(nn[k]['recall'][c])
                             for c in range(NUM_CLASSES)},
                  'support': {CLASS_NAMES[c]: int(nn[k]['support'][c])
                              for c in range(NUM_CLASSES)}}
    print(f"  frozen {out['frozen']:.3f} / ceiling {out['ceiling']:.3f} "
          f"(gap {out['gap']:+.3f}) | proto {out['proto_clean']:.3f}/{out['proto_pool']:.3f} "
          f"| n_val {n_val} ({time.time()-t0:.0f}s)")
    del pf, pl, pk, Xp, Ws, resid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--nusc_c_dir", type=str,
                    default="/mnt/bravo/jmfleming/nuscenes_c_kitti")
    ap.add_argument("--nusc_labels", type=str, default="config/labels/nuscenes_c.yaml")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.oracle_core import get_hdc_projection
    from modules.gen_trainers import GenTrainer
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)

    # extractors: label:method:path:dataset(kitti|nuscenes_c)
    specs = [
        ("cov_kitti", "supcon_vib_dglsspp_inputin_in_chan",
         "robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan",
         "kittic", ["fog", "crosstalk", "wet_ground"]),
        ("dgl_kitti", "supcon_vib_dglsspp",
         "robust_diagnostic/logs/supcon_vib_dglsspp", "kittic",
         ["fog", "crosstalk", "wet_ground"]),
        ("cov_nusc", "supcon_vib_dglsspp_inputin_in_chan",
         "robust_diagnostic/logs/nusc_covshift_21ep", "nuscenes_c", CONDS_ALL),
        ("dgl_nusc", "supcon_vib_dglsspp",
         "robust_diagnostic/logs/nusc_dglsspp_21ep", "nuscenes_c", CONDS_ALL),
    ]
    results = {'label': 'nusc_dglsspp_flip', 'extractors': {}}
    for lab, method, path, dataset, conds in specs:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}, {dataset}) ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        t0 = time.time()
        print(f"=== [{lab}] clean fit (KITTI seq-08 reservoir {args.clean_fit_n}) ===")
        cf, cl, ck = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                       args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
        protos_clean = build_prototypes(Xc, cl, device=device)
        feat_means_clean = class_means_feats(cf, cl)
        print(f"  clean: {len(cf)} pts done ({time.time()-t0:.0f}s)")
        results['extractors'][lab] = {'method': method, 'dataset': dataset, 'conds': {}}
        for cond in conds:
            if dataset == 'kittic':
                cdir = os.path.join(args.kittic_dir, cond, 'heavy')
                if not os.path.exists(cdir):
                    cdir = os.path.join(args.kittic_dir, cond, 'moderate')
                parser = build_parser(cdir, DATA, ARCH)
            else:
                nusc_data = yaml.safe_load(open(args.nusc_labels))
                from robust_diagnostic.al_full_dataset_diag import build_nuscenes_parser
                parser = build_nuscenes_parser(os.path.join(args.nusc_c_dir, cond, 'heavy'),
                                               nusc_data, ARCH)
            r = run_extractor_condition(model, parser, proj, device, W0, protos_clean,
                                        feat_means_clean, args, label=lab, cond_name=cond)
            results['extractors'][lab]['conds'][cond] = r
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\n[checkpoint] {lab} done, saved to {args.out}")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
