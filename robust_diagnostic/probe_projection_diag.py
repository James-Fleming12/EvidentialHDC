"""probe_projection_diag.py: HDC projection variants for linear separability
and mIoU (eval-only on the frozen cov-shift ep10 extractor).

The HDC code is sign(z R) with a FIXED random +-1 projection R (the free
parameter nobody has touched). This diagnostic tests whether any projection
variant improves the R4 linear-probe mIoU, the minority classes, or the code
statistics, and whether a label-free proxy can SELECT the best projection per
condition (dynamic projection selection).

Design (bounded harness, projection-comparison only):
  * clean: reservoir 200k -> per-variant W0 fit + whitening/LDA stats
  * per condition: reservoir pool 400k (seed 42) + val 400k (seed 43)
  * per variant: codes = sign(z R_eff), W0 fit (clean), W* fit (pool), decode
    val with both -> frozen/ceiling mIoU + per-class minority IoU + code stats
  * features are extracted ONCE per condition (held as 128-d); only the codes
    are rebuilt per variant, so the model forward is shared.

Variants:
  bern       : baseline (current get_hdc_projection, +-1 p=0.5)
  gauss      : N(0,1) entries
  sparse_k1  : +-1, ONE nonzero per row (each code bit = one feature's sign)
  sparse_k8  : +-1/sqrt(8), 8 nonzeros per row (sparse random projection)
  ternary    : {-1,0,+1} with p=(1/4,1/2,1/4)
  zca        : total ZCA whitening of the 128-d features, then bern
  within_whn : within-class whitening S_w^{-1/2} then bern (amplifies the
               low-variance / minority directions = "blow up patterns")
  rotated    : random orthonormal feature rotation Q, then bern (spreads the
               dominant shared mean, kills dead coordinates)
  dim5k      : bern at 5k dims (efficiency check)
  dim20k     : bern at 20k dims (capacity check)
  concat2    : two independent 5k bern projections concatenated (10k total)

Dynamic selection: per condition, report which variant a label-free proxy
(highest val-code Hamming distance / lowest dead-frac) would pick vs the
oracle-best mIoU variant, to test whether a proxy can select.

Usage:
  uv run python robust_diagnostic/probe_projection_diag.py \
    --path_b <ckpt> --method_b <method> --label proj_ep10 \
    --out robust_diagnostic/logs/probe_projection_ep10.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, NUM_CLASSES, CONDS_ALL)
from robust_diagnostic.al_per_class_diag import ConfMatrix

MINORITY = [2, 6, 7, 10, 12]   # bicycle, motorcycle, pedestrian, truck, other_flat

def rng_projector(dim_in, dim_out, kind, seed, device):
    """Build one projection R (dim_in x dim_out) on cpu, seed 42 per variant."""
    g = torch.Generator().manual_seed(seed)
    if kind == 'bern':
        return ((torch.rand(dim_in, dim_out, generator=g) > 0.5).float() * 2 - 1)
    if kind == 'gauss':
        return torch.randn(dim_in, dim_out, generator=g)
    if kind == 'sparse_k1':
        idx = torch.randint(0, dim_in, (dim_out,), generator=g)
        R = torch.zeros(dim_in, dim_out)
        R[idx, torch.arange(dim_out)] = torch.where(
            torch.rand(dim_out, generator=g) > 0.5, 1.0, -1.0)
        return R
    if kind == 'sparse_k8':
        k = 8
        R = torch.zeros(dim_in, dim_out)
        for c in range(dim_out):
            rows = torch.randperm(dim_in, generator=g)[:k]
            R[rows, c] = torch.where(torch.rand(k, generator=g) > 0.5, 1.0, -1.0) / np.sqrt(k)
        return R
    if kind == 'ternary':
        u = torch.rand(dim_in, dim_out, generator=g)
        return ((u > 0.75).float() - (u < 0.25).float())   # {-1, 0, +1}
    raise ValueError(kind)

def code_stats(codes, lbls):
    """Code-space statistics on a sample: dead-frac, hamming, per-class
    separability ratio (mean intra-class cos / mean inter-class cos)."""
    n = len(codes)
    frac_pos = (codes > 0).float().mean(dim=0)
    dead = float(((frac_pos < 0.05) | (frac_pos > 0.95)).float().mean().item())
    torch.manual_seed(7)
    a = codes[torch.randperm(n)[:20000]]
    b = codes[torch.randperm(n)[:20000]]
    hamm = float((1.0 - (a == b).float().mean(dim=1)).mean().item())
    # per-class separability on a 20k subsample
    idx = torch.randperm(n)[:20000]
    cs, ls = codes[idx], lbls[idx]
    cs_n = F.normalize(cs.float(), p=2, dim=1)
    intra, inter, cnt = [], [], 0
    for c in range(1, NUM_CLASSES):
        m = ls == c
        if m.sum() < 100:
            continue
        cc = cs_n[m]
        sim = cc @ cc.T
        intra.append(float(sim[~torch.eye(len(cc), dtype=torch.bool)].mean()))
        other = cs_n[~m][:2000]
        if len(other):
            inter.append(float((cc[:2000] @ other.T).mean()))
        cnt += 1
    sep = (float(np.mean(intra)) if intra else 0.0) - (float(np.mean(inter)) if inter else 0.0)
    return {'dead_frac': dead, 'hamming': hamm, 'sep': sep}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = ALL frames of seq 08")
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--val_cap", type=int, default=400000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--variants", type=str,
                    default="bern,gauss,sparse_k1,sparse_k8,ternary,zca,within_whn,rotated,dim5k,dim20k,concat2")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="proj_ep10")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    trainer = __import__('modules.gen_trainers', fromlist=['GenTrainer']).GenTrainer(
        ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model

    # ---- clean features (reservoir, seed 7): W0 fits + whitening stats ----
    t0 = time.time()
    print(f"=== clean reservoir ({args.clean_fit_n}) ===")
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                  args.clean_fit_n, 7)
    cf, cl = cf.to(device), cl.long()
    print(f"  clean {len(cf)} pts ({time.time()-t0:.0f}s)")

    # whitening transforms on clean (for the zca / within_whn / rotated variants)
    transforms = {}
    zc = cf - cf.mean(0)
    cov = (zc.T @ zc) / len(cf)
    evals, evecs = torch.linalg.eigh(cov.double())
    transforms['zca'] = (evecs / (evals + 1e-4).sqrt().unsqueeze(0)) @ evecs.T   # C^-1/2
    Sw = torch.zeros(128, 128, device=device, dtype=torch.float64)
    for c in range(1, NUM_CLASSES):
        m = cl == c
        if m.sum() < 100:
            continue
        dc = cf[m].double() - cf[m].double().mean(0)
        Sw += (dc.T @ dc) / len(cf)
    ev_w, evc_w = torch.linalg.eigh(Sw)
    transforms['within_whn'] = (evc_w / (ev_w + 1e-4).sqrt().unsqueeze(0)) @ evc_w.T
    g = torch.Generator().manual_seed(5)
    Q = torch.linalg.qr(torch.randn(128, 128, generator=g).double())[0]
    transforms['rotated'] = Q
    transforms = {k: v.float() for k, v in transforms.items()}
    del zc, cov, Sw
    torch.cuda.empty_cache()

    results = {'label': args.label, 'variants': variants, 'minority_classes': MINORITY,
               'conds': {}}
    for cond in conds:
        t0 = time.time()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        cparser = build_parser(cdir, DATA, ARCH)

        # ---- pool + val reservoirs: ONE stream pass, split a combined reservoir ----
        print(f"\n=== {cond} (extract once: pool + val) ===", flush=True)
        cfv, clv, _ = reservoir_collect(
            stream_frames(model, cparser, device, args.max_frames, progress=cond),
            args.pool_cap + args.val_cap, 42)
        pf, pl = cfv[:args.pool_cap].to(device), clv[:args.pool_cap].to(device).long()
        vf, vl = cfv[args.pool_cap:].to(device), clv[args.pool_cap:].to(device).long()
        print(f"  pool {len(pf)} / val {len(vf)} pts ({time.time()-t0:.0f}s)", flush=True)

        cond_res = {}
        best = {'miou': -1.0, 'variant': None}
        proxy = {'hamming': -1.0, 'variant': None}
        for v in variants:
            t1 = time.time()
            # build the effective projection for this variant
            if v in ('zca', 'within_whn', 'rotated'):
                base = rng_projector(128, 10000, 'bern', 42, device)
                R_eff = (transforms[v] @ base).to(device)
            elif v == 'dim5k':
                R_eff = rng_projector(128, 5000, 'bern', 42, device)
            elif v == 'dim20k':
                R_eff = rng_projector(128, 20000, 'bern', 42, device)
            elif v == 'concat2':
                R_eff = torch.cat([rng_projector(128, 5000, 'bern', 42, device),
                                   rng_projector(128, 5000, 'bern', 43, device)], dim=1)
            else:
                R_eff = rng_projector(128, 10000, v, 42, device)
            R_eff = R_eff.to(device)

            Xc = torch.sign(cf @ R_eff).float()
            Xp = torch.sign(pf @ R_eff).float()
            Xv = torch.sign(vf @ R_eff).float()
            W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
            Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device)

            cm_f = ConfMatrix(); cm_c = ConfMatrix()
            for s in range(0, len(Xv), 100000):
                e = min(s + 100000, len(Xv))
                cm_f.update((Xv[s:e] @ W0).argmax(1).cpu(), vl[s:e].cpu())
                cm_c.update((Xv[s:e] @ Ws).argmax(1).cpu(), vl[s:e].cpu())
            frozen, ceiling = cm_f.miou(), cm_c.miou()
            pc_f = cm_f.per_class_iou()
            stats = code_stats(Xv, vl)

            entry = {
                'frozen': frozen, 'ceiling': ceiling,
                'gap': ceiling - frozen,
                'minority_iou': {str(c): pc_f[c] for c in MINORITY},
                **stats, 'dim': R_eff.shape[1],
            }
            cond_res[v] = entry
            if ceiling > best['miou']:
                best = {'miou': ceiling, 'variant': v}
            if stats['hamming'] > proxy['hamming']:
                proxy = {'hamming': stats['hamming'], 'variant': v}
            print(f"  {v:<12s} dim {R_eff.shape[1]:>5d} frozen {frozen:.3f} ceiling "
                  f"{ceiling:.3f} (gap {ceiling-frozen:+.3f}) | dead {stats['dead_frac']:.3f} "
                  f"hamm {stats['hamming']:.3f} sep {stats['sep']:+.3f} | "
                  f"min(maj) {min(pc_f[c] for c in MINORITY):.3f} "
                  f"({time.time()-t1:.0f}s)", flush=True)
            del Xc, Xp, Xv, W0, Ws, cm_f, cm_c
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cond_res['_best_ceiling'] = best
        cond_res['_proxy_hamming'] = proxy
        results['conds'][cond] = cond_res
        print(f"  == {cond}: oracle-best {best['variant']} ({best['miou']:.3f}), "
              f"hamming-proxy pick {proxy['variant']} ({proxy['hamming']:.3f})", flush=True)
        del pf, pl, vf, vl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
