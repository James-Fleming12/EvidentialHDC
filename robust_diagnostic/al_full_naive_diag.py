"""al_full_naive_diag.py: FULL-DATASET confirmation that the findings that closed
the label-free TTA / naive-AL routes still hold at full KITTI-C scale (every
point of every frame of seq 08, ~300M points/condition), on the paper-ready
harness.

One streaming pass per condition computes, on the FULL val set:

  R4 ridge decoders
    linear_frozen          : W0, clean-fit (200k reservoir, seed 7)
    linear_ceiling         : W*, corrupted-pool fit (400k reservoir, seed 42)
    linear_selftrain       : NAIVE TTA -- ridge refit on the corrupted pool with
                             the frozen probe's pseudo-labels, NO gate
    linear_selftrain_conf  : NAIVE TTA -- same but gated to the top-50%
                             confidence pseudo-labels
    linear_randbank        : NAIVE AL  -- ridge on the 56+500 random bank with
                             TRUE labels, no oracle low-rank
    linear_W_res_pseudo    : memory-bank AL (current method), oracle U r=8,
                             pseudo-labels on the 500 random bank points
    linear_W_res_true      : memory-bank AL, oracle U r=8, true bank labels
  R1 prototype decoders
    proto_frozen / proto_ceiling  (class-mean cosine, clean / pool)

Per-class separability in BOTH feature spaces (one extra accumulation in the
same decode pass):
    code-kNN  : nearest class-mean in the binarized 10000-d HDC code
    feat-kNN  : nearest class-mean in the raw 128-d network features
  for clean prototypes and corrupted-pool prototypes, evaluated on the FULL
  corrupted stream. Plus a clean-reference separability from a held-out clean
  reservoir half (prototypes from half A, kNN accuracy on half B), so "the
  minority weakness is/is not in the feature space itself" is measured, not
  assumed.

This is the scale check for:
  1. naive pseudo-label TTA cannot beat the frozen decode (Iteration 9-12
     closure, previously 100-frame),
  2. naive random AL (plain ridge on random true labels) is weak vs the
     memory-bank W_res,
  3. minority classes are/aren't separable in the network code vs the HDC code
     (the "not the probe fit" claim).

Usage:
  uv run python robust_diagnostic/al_full_naive_diag.py \
    --path_b <ckpt_dir> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label full_naive_ep10 --out robust_diagnostic/logs/al_full_naive_ep10.json

Output JSON: extractors[<label>].conds[<cond>] = {
  linear_frozen / linear_ceiling / linear_selftrain / linear_selftrain_conf /
  linear_randbank / linear_W_res_pseudo / linear_W_res_true (+_delta each) /
  proto_frozen / proto_ceiling / n_pool / n_val,
  sep_code_clean / sep_code_pool / sep_feat_clean / sep_feat_pool :
     {cls: recall} + {cls: support},
  sep_code_clean_ref / sep_feat_clean_ref: clean-reservoir held-out kNN recall
}
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, build_prototypes, knn_predict, ConfAccum,
    NUM_CLASSES, CONDS_ALL)

CLASS_NAMES = ['unlabeled', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
               'terrain', 'manmade', 'vegetation']

def class_means_feats(feats, lbls, num_classes=NUM_CLASSES):
    """Per-class mean of the RAW 128-d features, L2-normalized (network-space
    prototypes / nearest-mean rule in the feature space)."""
    means = torch.zeros(num_classes, feats.shape[1])
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        m = lbls == c
        if m.sum() > 0:
            means[c] += feats[m].sum(dim=0)
            counts[c] += m.sum()
    for c in range(num_classes):
        if counts[c] > 0:
            means[c] /= counts[c]
    return F.normalize(means, p=2, dim=1)

def bank_indices(pl, args):
    """The README 56+500 random bank (seeds 2/3): k=8 true per class + 500 random."""
    classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
    cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}
    lab_idx = []
    for c in classes:
        idx = cls_idx[c]
        if len(idx) < max(50, args.bank_k):
            continue
        torch.manual_seed(2)
        lab_idx.append(idx[torch.randperm(len(idx))[:args.bank_k]])
    lab_idx = torch.cat(lab_idx) if lab_idx else torch.tensor([], dtype=torch.long)
    avail = torch.arange(len(pl))
    mask = torch.ones(len(pl), dtype=torch.bool); mask[lab_idx] = False
    torch.manual_seed(3)
    extra = avail[mask][torch.randperm(len(avail[mask]))[:args.bank_extra]]
    return torch.cat([lab_idx, extra]), lab_idx, extra

def stream_decode_full_extra(model, parser, proj, device, decoders, nn_prep,
                             exclude=None, max_frames=0, chunk=100000):
    """Like stream_decode_full but ALSO accumulates per-class nearest-mean
    separability (code and feature space) in the same pass.
    decoders: {name: {'type':'w'|'proto'|'w_bias', ...}} (mIoU via ConfAccum).
    nn_prep:  {name: {'kind':'code'|'feat', 'refs': (K,d), 'lbls': (K,)}} ->
              per-class recall of nearest-mean to `refs`. Returns (accs,
              {name: {'recall': tensor(nc), 'support': tensor(nc)}})."""
    accs = {name: ConfAccum() for name in decoders}
    nn_tp = {n: torch.zeros(NUM_CLASSES) for n in nn_prep}
    nn_sup = {n: torch.zeros(NUM_CLASSES) for n in nn_prep}
    prep = {}
    for name, dec in decoders.items():
        if dec['type'] == 'w':
            prep[name] = ('w', dec['W'].to(device))
        elif dec['type'] == 'w_bias':
            prep[name] = ('w_bias', dec['W'].to(device), dec['bias'].to(device))
        else:
            prep[name] = ('proto', dec['protos'].to(device), dec['proto_lbls'].to(device))
    nn_prep_d = {}
    for name, cfg in nn_prep.items():
        refs = F.normalize(cfg['refs'].float(), p=2, dim=1).to(device)
        nn_prep_d[name] = (cfg['kind'], refs, cfg['lbls'].to(device))
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
                elif p[0] == 'w_bias':
                    preds = (codes @ p[1] + p[2]).argmax(1).cpu()
                else:
                    sims = F.normalize(codes, p=2, dim=1) @ p[1].t()
                    preds = p[2][sims.argmax(1)].cpu()
                lbls = labels[s:e]
                if skip is not None:
                    m = ~skip[s:e]
                    preds, lbls = preds[m], lbls[m]
                accs[name].update(preds, lbls)
            for name, (kind, refs, rlbls) in nn_prep_d.items():
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
    nn = {name: {'recall': nn_tp[name].clone(),
                 'support': nn_sup[name].clone()} for name in nn_prep}
    return accs, nn

def eval_condition(model, parser, proj, device, W0, protos_clean, feat_means_clean,
                   cf_halfA, cl_halfA, cf_halfB, cl_halfB, args, label, cond_name):
    t0 = time.time()
    print(f"\n=== [{label}] {cond_name} (pass 1: pool) ===")
    pf, pl, pk = reservoir_collect(stream_frames(model, parser, device, args.max_frames),
                                   args.pool_cap, 42)
    print(f"  pool: {len(pf)} points")
    Xp = hdc_codes(pf, proj, device).float()
    Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device)
    protos_pool = build_prototypes(Xp, pl, device=device)
    feat_means_pool = class_means_feats(pf, pl)

    # --- naive TTA: pseudo-label refit on the corrupted pool (no gate) ---
    Xp_d = Xp.to(device)
    with torch.no_grad():
        logits = Xp_d @ W0.to(device)
        yhat = logits.argmax(1)
        conf = torch.softmax(logits, dim=1).max(1).values
    W_selftrain = ridge_fit_exact(Xp, onehot(yhat.cpu(), NUM_CLASSES), args.lam, device)
    gate = conf >= torch.quantile(conf, 0.5)
    print(f"  naive self-train: {len(yhat)} pseudo-labeled pool pts "
          f"({gate.sum().item()} top-50% conf gated)")
    W_selftrain_conf = ridge_fit_exact(Xp[gate], onehot(yhat[gate].cpu(), NUM_CLASSES),
                                       args.lam, device)

    # --- naive AL: plain ridge on the 56+500 random bank, true labels ---
    bank_idx, lab_idx, extra = bank_indices(pl, args)
    W_randbank = ridge_fit_exact(Xp[bank_idx], onehot(pl[bank_idx], NUM_CLASSES),
                                 args.lam, device)
    print(f"  naive AL randbank: {len(bank_idx)} true-label bank pts "
          f"({len(lab_idx)} + {len(extra)})")

    # --- memory-bank AL (current method): W_res with oracle U r=8 ---
    R = (Ws - W0).detach().cpu().float()
    U8 = torch.linalg.svd(R.double(), full_matrices=False)[0][:, :args.r].float()
    extra_pred = knn_predict(pf[extra], pf[lab_idx], pl[lab_idx], k=1, device=device)
    X_lab = torch.cat([Xp[lab_idx], Xp[extra]], dim=0)
    Y_pseudo = torch.cat([onehot(pl[lab_idx], NUM_CLASSES),
                          onehot(extra_pred, NUM_CLASSES)], dim=0)
    Y_true = torch.cat([onehot(pl[lab_idx], NUM_CLASSES),
                        onehot(pl[extra], NUM_CLASSES)], dim=0)
    XU = X_lab.to(device).float() @ U8.to(device)
    A = XU.t() @ XU + 1e-6 * torch.eye(args.r, device=device)
    W0d = W0.to(device)
    Cp = torch.linalg.solve(A, XU.t() @ (Y_pseudo.to(device).float()
                                         - X_lab.to(device).float() @ W0d)).cpu()
    Ct = torch.linalg.solve(A, XU.t() @ (Y_true.to(device).float()
                                         - X_lab.to(device).float() @ W0d)).cpu()
    W_res_pseudo = W0.detach().cpu() + (U8.cpu() @ Cp)
    W_res_true = W0.detach().cpu() + (U8.cpu() @ Ct)
    print(f"  memory-bank W_res (oracle U r={args.r}) done ({time.time()-t0:.0f}s)")

    # --- pass 2: one FULL decode, ridge decoders + code/feat nearest-mean ---
    from collections import defaultdict
    ex_by_frame = defaultdict(list)
    for f, i in pk.tolist():
        ex_by_frame[f].append(i)
    ex_by_frame = {f: torch.tensor(sorted(s), dtype=torch.long) for f, s in ex_by_frame.items()}
    decoders = {
        'linear_frozen': {'type': 'w', 'W': W0.detach().cpu()},
        'linear_ceiling': {'type': 'w', 'W': Ws.detach().cpu()},
        'linear_selftrain': {'type': 'w', 'W': W_selftrain.detach().cpu()},
        'linear_selftrain_conf': {'type': 'w', 'W': W_selftrain_conf.detach().cpu()},
        'linear_randbank': {'type': 'w', 'W': W_randbank.detach().cpu()},
        'linear_W_res_pseudo': {'type': 'w', 'W': W_res_pseudo},
        'linear_W_res_true': {'type': 'w', 'W': W_res_true},
        'proto_frozen': {'type': 'proto', 'protos': protos_clean.cpu(),
                         'proto_lbls': torch.arange(NUM_CLASSES)},
        'proto_ceiling': {'type': 'proto', 'protos': protos_pool.cpu(),
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
    print(f"  pass 2: full decode + separability over ALL frames...")
    accs, nn = stream_decode_full_extra(model, parser, proj, device, decoders,
                                        nn_prep, exclude=ex_by_frame,
                                        max_frames=args.max_frames)
    n_val = accs['linear_frozen'].n
    m = {k: accs[k].miou() for k in decoders}
    out = {
        'n_pool': len(pf), 'n_val': n_val,
        'linear_frozen': m['linear_frozen'], 'linear_ceiling': m['linear_ceiling'],
        'linear_gap': m['linear_ceiling'] - m['linear_frozen'],
        'linear_selftrain': m['linear_selftrain'],
        'linear_selftrain_delta': m['linear_selftrain'] - m['linear_frozen'],
        'linear_selftrain_conf': m['linear_selftrain_conf'],
        'linear_selftrain_conf_delta': m['linear_selftrain_conf'] - m['linear_frozen'],
        'linear_randbank': m['linear_randbank'],
        'linear_randbank_delta': m['linear_randbank'] - m['linear_frozen'],
        'linear_W_res_pseudo': m['linear_W_res_pseudo'],
        'linear_W_res_pseudo_delta': m['linear_W_res_pseudo'] - m['linear_frozen'],
        'linear_W_res_true': m['linear_W_res_true'],
        'linear_W_res_true_delta': m['linear_W_res_true'] - m['linear_frozen'],
        'proto_frozen': m['proto_frozen'], 'proto_ceiling': m['proto_ceiling'],
        'proto_gap': m['proto_ceiling'] - m['proto_frozen'],
        'bank_n': len(bank_idx),
    }
    for k in nn_prep:
        out[k] = {'recall': {CLASS_NAMES[c]: float(nn[k]['recall'][c])
                             for c in range(NUM_CLASSES)},
                  'support': {CLASS_NAMES[c]: int(nn[k]['support'][c])
                              for c in range(NUM_CLASSES)}}
    # clean-reference separability: held-out clean reservoir halves
    XhA = hdc_codes(cf_halfA, proj, device).float()
    XhB = hdc_codes(cf_halfB, proj, device).float()
    protosA_code = build_prototypes(XhA, cl_halfA, device=device).cpu()
    featA = class_means_feats(cf_halfA, cl_halfA)
    out['sep_code_clean_ref'] = nearest_mean_recall(
        XhB, protosA_code, cl_halfB, kind='code')
    out['sep_feat_clean_ref'] = nearest_mean_recall(
        cf_halfB, featA, cl_halfB, kind='feat')
    print(f"  [R4] frozen {m['linear_frozen']:.3f} / ceiling {m['linear_ceiling']:.3f} "
          f"| selftrain {m['linear_selftrain']:.3f} ({m['linear_selftrain']-m['linear_frozen']:+.3f}) "
          f"conf {m['linear_selftrain_conf']:.3f} ({m['linear_selftrain_conf']-m['linear_frozen']:+.3f})")
    print(f"  [AL] randbank {m['linear_randbank']:.3f} ({m['linear_randbank']-m['linear_frozen']:+.3f}) "
          f"| W_res pseudo {m['linear_W_res_pseudo']:.3f} "
          f"({m['linear_W_res_pseudo']-m['linear_frozen']:+.3f}) "
          f"true {m['linear_W_res_true']:.3f} ({m['linear_W_res_true']-m['linear_frozen']:+.3f})")
    print(f"  [R1] frozen {m['proto_frozen']:.3f} / ceiling {m['proto_ceiling']:.3f} | "
          f"n_val {n_val} ({time.time()-t0:.0f}s)")
    del pf, pl, pk, Xp, Ws, R, U8, X_lab
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

def nearest_mean_recall(X, refs, lbls, kind='code', chunk=200000):
    """Per-class recall of nearest-mean to `refs` over X (used for the clean
    held-out reference, in-memory)."""
    recall = torch.zeros(NUM_CLASSES); support = torch.zeros(NUM_CLASSES)
    refs_n = F.normalize(refs.float(), p=2, dim=1)
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        Xc = X[s:e].float()
        Xn = F.normalize(Xc, p=2, dim=1)
        preds = (Xn @ refs_n.t()).argmax(1)
        l = lbls[s:e]
        for c in range(1, NUM_CLASSES):
            lc = (l == c)
            if lc.any():
                support[c] += lc.sum()
                recall[c] += (preds[lc] == c).sum()
    return {'recall': {CLASS_NAMES[c]: float(recall[c]) for c in range(NUM_CLASSES)},
            'support': {CLASS_NAMES[c]: int(support[c]) for c in range(NUM_CLASSES)}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = ALL frames of seq 08")
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--bank_k", type=int, default=8)
    ap.add_argument("--bank_extra", type=int, default=500)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--extractors", type=str,
                    default="cov_ep10:supcon_vib_dglsspp_inputin_in_chan:"
                            "robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/"
                            "supcon_vib_dglsspp_inputin_in_chan")
    ap.add_argument("--label", type=str, default="full_naive_ep10")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    extractors = [tuple(e.strip().split(':')) for e in args.extractors.split(',') if e.strip()]
    from modules.oracle_core import get_hdc_projection
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)

    results = {'label': args.label, 'max_frames': args.max_frames,
               'clean_fit_n': args.clean_fit_n, 'pool_cap': args.pool_cap,
               'extractors': {lab: {'method': method, 'conds': {}}
                              for lab, method, _ in extractors}}
    for lab, method, path in extractors:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}, {path}) ===\n{'='*80}")
        from modules.gen_trainers import GenTrainer
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)

        t0 = time.time()
        print(f"=== [{lab}] clean fit (all clean frames, reservoir {args.clean_fit_n}) ===")
        cf, cl, ck = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                       args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
        protos_clean = build_prototypes(Xc, cl, device=device)
        feat_means_clean = class_means_feats(cf, cl)
        print(f"  clean: {len(cf)} points, W0 + clean protos done ({time.time()-t0:.0f}s)")

        # held-out clean reservoir halves for the clean-reference separability
        half = args.clean_fit_n // 2
        torch.manual_seed(5)
        perm = torch.randperm(len(cf))
        cf_halfA, cl_halfA = cf[perm[:half]], cl[perm[:half]]
        cf_halfB, cl_halfB = cf[perm[half:2 * half]], cl[perm[half:2 * half]]

        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            cparser = build_parser(cdir, DATA, ARCH)
            r = eval_condition(model, cparser, proj, device, W0, protos_clean,
                               feat_means_clean, cf_halfA, cl_halfA, cf_halfB, cl_halfB,
                               args, label=lab, cond_name=cond)
            results['extractors'][lab]['conds'][cond] = r
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\n[checkpoint] extractor {lab} done, saved to {args.out}")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
