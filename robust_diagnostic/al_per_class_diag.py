"""al_per_class_diag.py: per-class mIoU breakdown on the FULL dataset (paper-
ready harness), for the 17-class setting.

Motivation: the full-scale mIoU (fog 0.322 frozen / 0.369 ceiling) is much lower
than the 100-frame harness suggested, and the evaluated-class frequency map is
~infx imbalanced (6 of 16 evaluated classes have ~0 support: barrier,
construction_vehicle, traffic_cone, trailer, bus, bicycle). This diagnostic
measures, per class, on every point of every frame of seq 08:

  1. pre-condition (clean KITTI) per-class IoU and support, frozen W0,
  2. post-condition (each KITTI-C condition) per-class IoU, frozen W0 and
     ceiling W* (the labeled bound),
  3. the per-class IoU DELTA (post - pre) and the per-class gap
     (ceiling - frozen), to see whether minority classes are weaker AND
     whether they are the ones with the recoverable headroom,
  4. WHY: the confusion structure (top-3 classes a minority's true points are
     predicted as, and vice versa) + per-class pool support + per-class
     recall, so a new loss term can target the right failure mode.

Reuses the full-dataset streaming harness (reservoir pool, spectral-exact
ridge, ConfAccum-style accumulation) from al_full_dataset_diag.py.

Usage:
  uv run python robust_diagnostic/al_per_class_diag.py \
    --path_b <ckpt> --method_b <method> --label perclass_ep10 \
    --out robust_diagnostic/logs/al_per_class_ep10.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, build_prototypes, stream_decode_full, NUM_CLASSES, CONDS_ALL)

CLASS_NAMES = ['unlabeled', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
               'terrain', 'manmade', 'vegetation']

class ConfMatrix:
    """Streaming 17x17 confusion matrix (rows = predicted, cols = true)."""
    def __init__(self, nc=NUM_CLASSES):
        self.C = torch.zeros(nc, nc)
        self.n = 0
    def update(self, preds, lbls):
        p = preds.long(); l = lbls.long()
        idx = p * NUM_CLASSES + l
        self.C += torch.bincount(idx, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
        self.n += len(p)
    def per_class_iou(self):
        tp = torch.diag(self.C)
        fp = self.C.sum(1) - tp
        fn = self.C.sum(0) - tp
        d = tp + fp + fn
        return (tp / d.clamp(min=1e-9)).tolist()
    def recall(self):
        return (torch.diag(self.C) / self.C.sum(0).clamp(min=1e-9)).tolist()
    def precision(self):
        return (torch.diag(self.C) / self.C.sum(1).clamp(min=1e-9)).tolist()
    def support(self):
        return self.C.sum(0).tolist()
    def miou(self):
        ious = []
        for c in range(1, NUM_CLASSES):
            if self.C[:, c].sum() == 0:
                continue
            tp, fp, fn = self.C[c, c], self.C[c].sum() - self.C[c, c], self.C[:, c].sum() - self.C[c, c]
            d = tp + fp + fn
            ious.append(float(tp / d) if d > 0 else 0.0)
        return float(np.mean(ious)) if ious else 0.0

def stream_confusions(model, parser, proj, device, decoders, exclude=None,
                      max_frames=0, chunk=100000):
    """Like stream_decode_full but accumulates full confusion matrices."""
    accs = {name: ConfMatrix() for name in decoders}
    prep = {}
    for name, dec in decoders.items():
        if dec['type'] == 'w':
            prep[name] = ('w', dec['W'].to(device))
        else:
            prep[name] = ('proto', dec['protos'].to(device), dec['proto_lbls'].to(device))
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
                accs[name].update(preds, lbls)
            del codes
        del zf, labels
    return accs

def top_confusions(conf, cls, k=3):
    """Top-k classes a class's TRUE points are predicted as (confusion rows)."""
    true_col = conf[:, cls]
    idx = torch.argsort(true_col, descending=True)
    return [(int(c), float(true_col[c])) for c in idx[:k] if int(c) != cls]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = ALL frames of seq 08")
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="perclass_ep10")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = __import__('modules.gen_trainers', fromlist=['GenTrainer']).GenTrainer(
        ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    proj = __import__('modules.oracle_core', fromlist=['get_hdc_projection']).get_hdc_projection(
        dim_in=128, dim_out=10000, device=device)

    # ---- clean W0 + clean prototypes (reservoir, seed 7) ----
    t0 = time.time()
    print(f"=== clean fit (reservoir {args.clean_fit_n}) ===")
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    cf, cl, ck = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                   args.clean_fit_n, 7)
    Xc = hdc_codes(cf, proj, device).float()
    W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
    protos_clean = build_prototypes(Xc, cl, device=device)
    print(f"  clean: {len(cf)} points ({time.time()-t0:.0f}s)")
    del cf, cl, ck, Xc
    torch.cuda.empty_cache()

    results = {'label': args.label, 'method': args.method_b,
               'class_names': CLASS_NAMES, 'conds': {}}

    # ---- PRE-CONDITION: clean KITTI per-class (frozen W0) ----
    print("\n=== clean (pre-condition) ===")
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    accs = stream_confusions(model, clean_parser, proj, device,
                             {'frozen': {'type': 'w', 'W': W0.detach().cpu()}},
                             max_frames=args.max_frames)
    results['clean'] = {
        'support': accs['frozen'].support(),
        'iou': accs['frozen'].per_class_iou(),
        'recall': accs['frozen'].recall(),
        'precision': accs['frozen'].precision(),
        'miou': accs['frozen'].miou(), 'n_val': accs['frozen'].n,
    }
    print(f"  clean mIoU {results['clean']['miou']:.3f} (n={accs['frozen'].n})")
    for c in range(1, NUM_CLASSES):
        print(f"  {c:2d} {CLASS_NAMES[c]:<22s} sup {results['clean']['support'][c]:>10.0f} "
              f"IoU {results['clean']['iou'][c]:.3f}")

    for cond in conds:
        t0 = time.time()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        cparser = build_parser(cdir, DATA, ARCH)

        # pass 1: pool -> ceiling W* + pool protos
        print(f"\n=== {cond} (pass 1: pool) ===")
        pf, pl, pk = reservoir_collect(stream_frames(model, cparser, device, args.max_frames),
                                       args.pool_cap, 42)
        Xp = hdc_codes(pf, proj, device).float()
        Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device)
        protos_pool = build_prototypes(Xp, pl, device=device)
        pool_support = {int(c): int((pl == c).sum()) for c in range(NUM_CLASSES)}
        print(f"  pool {len(pf)} points, ceiling W* done ({time.time()-t0:.0f}s)")

        # pass 2: full-dataset confusion, pool excluded
        from collections import defaultdict
        ex_by_frame = defaultdict(list)
        for f, i in pk.tolist():
            ex_by_frame[f].append(i)
        ex_by_frame = {f: torch.tensor(sorted(s), dtype=torch.long) for f, s in ex_by_frame.items()}
        decoders = {
            'frozen': {'type': 'w', 'W': W0.detach().cpu()},
            'ceiling': {'type': 'w', 'W': Ws.detach().cpu()},
            'proto_frozen': {'type': 'proto', 'protos': protos_clean.cpu(),
                             'proto_lbls': torch.arange(NUM_CLASSES)},
        }
        accs = stream_confusions(model, cparser, proj, device, decoders,
                                 exclude=ex_by_frame, max_frames=args.max_frames)
        cfrz, cceil = accs['frozen'].C, accs['ceiling'].C

        r = {
            'n_pool': len(pf), 'n_val': accs['frozen'].n,
            'pool_support': pool_support,
            'val_support': accs['frozen'].support(),
            'frozen_iou': accs['frozen'].per_class_iou(),
            'ceiling_iou': accs['ceiling'].per_class_iou(),
            'frozen_recall': accs['frozen'].recall(),
            'ceiling_recall': accs['ceiling'].recall(),
            'gap': [accs['ceiling'].per_class_iou()[c] - accs['frozen'].per_class_iou()[c]
                    for c in range(NUM_CLASSES)],
            'miou_frozen': accs['frozen'].miou(),
            'miou_ceiling': accs['ceiling'].miou(),
            'confusions': {str(c): top_confusions(cfrz, c) for c in range(1, NUM_CLASSES)
                           if cfrz[:, c].sum() > 0},
        }
        results['conds'][cond] = r
        print(f"  mIoU frozen {r['miou_frozen']:.3f} / ceiling {r['miou_ceiling']:.3f} "
              f"(n={r['n_val']}, {time.time()-t0:.0f}s)")
        print(f"  {'cls':<3s} {'name':<22s} {'sup':>8s} {'pre':>5s} {'post':>5s} {'ceil':>5s} "
              f"{'gap':>6s}  top confusion")
        for c in range(1, NUM_CLASSES):
            pre = results['clean']['iou'][c]
            post = r['frozen_iou'][c]
            ceil = r['ceiling_iou'][c]
            conf = r['confusions'].get(str(c), [])
            conf_s = ' '.join(f'{CLASS_NAMES[cc]}({v:.0f})' for cc, v in conf[:2])
            print(f"  {c:<3d} {CLASS_NAMES[c]:<22s} {r['val_support'][c]:>8.0f} "
                  f"{pre:>5.2f} {post:>5.2f} {ceil:>5.2f} {ceil-post:>+6.2f}  {conf_s}")
        del pf, pl, pk, Xp, Ws, accs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
