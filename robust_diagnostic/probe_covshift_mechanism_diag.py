"""probe_covshift_mechanism_diag.py: Iteration-0 mechanism diagnostics for the
cov-shift healthy/cross-domain ceiling loss (docs/cov_shift/cov_full_scale.md).

Runs the eval-only diagnostics (1-7, 9-10) from Iteration 0 in ONE process per
extractor, on the FULL harness (deep-copied ARCH, 200k clean fit, 400k pool,
spectral-exact ridge, pool excluded from val):

  D1  CLEAN BASELINE DECOMPOSITION: frozen + ceiling on the CLEAN val stream
      (clean pool, seed 42), so each corrupted-condition deficit splits into
      clean-inherited gap + corruption-interaction:
        interaction = (dgl_corr - cov_corr) - (dgl_clean - cov_clean)
  D2  PER-CLASS recoverable map: per-class frozen / ceiling / gap on each cond.
  D3  INPUT-STATISTICS CALIBRATION: per-scan mean/std of channels {0,4}
      (range, remission) over valid points, clean vs each corrupted condition.
  D4  RESIDUAL + CONDITIONING: ||W*-W0||/||W0||, S = X^T X eigenvalue spectrum
      (top-k, participation-ratio effective rank), and a ridge lambda sweep
      {1e-4,1e-3,1e-2} on the pool fit (val decode at each lambda).
  D5  CODE-VS-RAW separability + BINARIZATION health (bit balance, pre-sign
      margin = |x.p| distribution near 0).
  D6  VARIANCE / effective rank of the code and raw features per condition.
  D7  NORMALIZATION-LEVER ABLATION (eval-only): re-decode with model.input_in
      disabled (gate-off) and with model.scale_only toggled, vs the baseline
      frozen/ceiling. Tests inference-time gating, NOT the counterfactual model.
  D9  W0-SOURCE CONTROL: refit W0 on nuScenes-clean val frames and re-measure
      frozen (KITTI-clean-fit W0 vs nuScenes-clean-fit W0) for the NuScenes-C
      extractors. Ceiling W* is unaffected (fit in-domain).
  D10 R1-vs-R4 headroom decomposition: proto_ceiling (R1 geometry) vs
      linear_ceiling (R4) per condition.

Diagnostic 8 (training ablations) is NOT eval-only and is a separate step.

Extractors (mirror the flip probe):
  cov_kitti / dgl_kitti : KITTI-trained pair on KITTI-C (all 8 conds)
  cov_nusc  / dgl_nusc  : NuScenes-trained pair on NuScenes-C (all 8 heavy)

Usage:
  uv run python robust_diagnostic/probe_covshift_mechanism_diag.py \
    --out robust_diagnostic/logs/probe_covshift_mechanism_ep10.json
"""
import os, sys, time, argparse, json, yaml
from collections import defaultdict
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

def input_stats_stream(model, parser, device, max_frames=0, report=200):
    """Per-scan mean/std of channels {0,4} (range, remission) over valid points,
    plus the total code/feature stats hooks. Yields nothing; accumulates."""
    ch = (0, 4)
    mu_sum = torch.zeros(len(ch)); var_sum = torch.zeros(len(ch)); n_scans = 0
    for i, batch in enumerate(parser.get_train_set()):
        if max_frames > 0 and i >= max_frames:
            break
        if i % report == 0:
            print(f"  [input stats] frame {i}...", flush=True)
        in_vol = batch[0]
        valid = (in_vol[:, 0:1, :, :] > 0).float()
        for j, c in enumerate(ch):
            x = in_vol[:, c:c + 1, :, :] * valid
            denom = valid.sum(dim=(2, 3)).clamp(min=1)
            m = (x.sum(dim=(2, 3)) / denom)
            var = ((x - m.unsqueeze(-1).unsqueeze(-1)).pow(2) * valid).sum(dim=(2, 3)) / denom
            mu_sum[j] += m.mean().item(); var_sum[j] += var.mean().item()
        n_scans += len(in_vol)
    return {'n_scans': n_scans,
            'mean_ch': {str(c): float(mu_sum[j] / max(1, n_scans)) for j, c in enumerate(ch)},
            'var_ch': {str(c): float(var_sum[j] / max(1, n_scans)) for j, c in enumerate(ch)}}

def stream_decode_mech(model, parser, proj, device, decoders, exclude=None,
                       max_frames=0, chunk=100000):
    """Stream full val; per-class IoU per decoder. decoders: {name: W(10000,K)}."""
    pc = {name: ConfMatrix() for name in decoders}
    prep = {name: ('w', dec.to(device)) for name, dec in decoders.items()}
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
                preds = (codes @ p[1]).argmax(1).cpu()
                lbls = labels[s:e]
                if skip is not None:
                    m = ~skip[s:e]
                    preds, lbls = preds[m], lbls[m]
                pc[name].update(preds, lbls)
            del codes
        del zf, labels
    return pc

def topk_evals_effrank(X, K=50, iters=3, seed=42, device='cuda', subsample=50000):
    """Randomized top-K eigenvalues of S = X^T X (matrix-free) + participation
    ratio effective rank computed EXACTLY on a subsample's S_sub (eigh of the
    d x d matrix on ~50k points; the full X X^T is far too large)."""
    d = X.shape[1]; n = len(X)
    g = torch.Generator().manual_seed(seed)
    Omega = torch.randn(d, K, generator=g)
    def apply_S(v):
        return X.t() @ (X @ v)
    Y = apply_S(Omega)
    for _ in range(iters):
        Y = apply_S(Y)
    Q, _ = torch.linalg.qr(Y.float())
    T = Q.t() @ apply_S(Q)
    evals, _ = torch.linalg.eigh(T)   # ascending
    evals = evals.flip(0).float()
    # exact effective rank on a subsample of the code (S_sub is d x d)
    torch.manual_seed(seed)
    idx = torch.randperm(n)[:min(n, subsample)]
    Xs = X[idx].to(device).float()
    Ss = Xs.t() @ Xs
    e_all = torch.linalg.eigvalsh(Ss.double()).flip(0)
    e_all = e_all.clamp(min=1e-12)
    pr = float((e_all.sum() ** 2) / (e_all.pow(2).sum()))
    return {'topk': evals[:K].tolist(), 'effrank_pr': pr,
            'tr_ratio_top50': float(e_all[:50].sum() / e_all.sum())}

def run_condition(model, parser, proj, device, W0, protos_clean, feat_means_clean,
                  args, label, cond_name, clean_pool=None, w0_alt=None):
    from collections import defaultdict
    t0 = time.time()
    print(f"\n=== [{label}] {cond_name} ===")
    pf, pl, pk = reservoir_collect(stream_frames(model, parser, device, args.max_frames),
                                   args.pool_cap, 42)
    Xp = hdc_codes(pf, proj, device).float()
    pool_counts = torch.bincount(pl.long(), minlength=NUM_CLASSES)

    # D3: input statistics calibration (cheap, one extra stream pass)
    t3 = time.time()
    instats = input_stats_stream(model, parser, device, args.max_frames)
    print(f"  [D3] input stats {instats} ({time.time()-t3:.0f}s)")

    # D4: residual + conditioning + lambda sweep
    Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device)
    resid = (Ws - W0).detach().cpu().float()
    r_norm = float(torch.norm(resid) / torch.norm(W0.detach().cpu().float()))
    spec = topk_evals_effrank(Xp, K=50, device=device)
    lam_sweep = {}
    for lam in (1e-4, 1e-3, 1e-2):
        Wl = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), lam, device)
        lam_sweep[str(lam)] = {'W': Wl.detach().cpu()}

    protos_pool = build_prototypes(Xp, pl, device=device)
    feat_means_pool = class_means_feats(pf, pl)

    # D6/D5: variance + bit balance + sign margin on the pool code
    code_var = torch.var(Xp.float(), dim=0)
    bit_balance = Xp.float().mean(0)  # +1 fraction = (1+mean)/2
    pre_sign = (pf.to(device).float() @ proj).abs().clamp(min=1e-12)
    margin_frac = float((pre_sign < 0.5).float().mean().item())
    feat_var = torch.var(pf.float(), dim=0)

    ex_by_frame = defaultdict(list)
    for f, i in pk.tolist():
        ex_by_frame[f].append(i)
    ex_by_frame = {f: torch.tensor(sorted(s), dtype=torch.long) for f, s in ex_by_frame.items()}

    decoders = {'frozen': W0.detach().cpu(), 'ceiling': Ws.detach().cpu()}
    decoders.update({f'ceil_lam{lam}': specW for lam, specW in lam_sweep.items()})
    if w0_alt is not None:
        decoders['frozen_W0_alt'] = w0_alt.detach().cpu()
    if args.gate_off and getattr(model, 'input_in', False):
        model.input_in = False
        decoders['ceiling_gate_off'] = Ws.detach().cpu()
        decoders['frozen_gate_off'] = W0.detach().cpu()
    pc = stream_decode_mech(model, parser, proj, device, decoders,
                            exclude=ex_by_frame, max_frames=args.max_frames)
    if args.gate_off:
        model.input_in = True   # restore for later conditions

    n_val = pc['frozen'].n
    m = {k: pc[k].miou() for k in decoders}
    out = {
        'n_pool': len(pf), 'n_val': n_val,
        'input_stats': instats,
        'resid_rel': r_norm,
        'spec': {'topk': spec['topk'][:10], 'effrank_pr': spec['effrank_pr']},
        'code_var': float(code_var.mean().item()),
        'code_var_std': float(code_var.std().item()),
        'bit_balance_frac_pos': float((bit_balance > 0).float().mean().item()),
        'sign_margin_frac_lt05': margin_frac,
        'feat_var': float(feat_var.mean().item()),
        'frozen': m['frozen'], 'ceiling': m['ceiling'],
        'gap': m['ceiling'] - m['frozen'],
        'ceil_lam': {k: m[k] for k in lam_sweep},
        'per_class_frozen': {CLASS_NAMES[c]: float(pc['frozen'].per_class_iou()[c])
                             for c in range(NUM_CLASSES)},
        'per_class_ceiling': {CLASS_NAMES[c]: float(pc['ceiling'].per_class_iou()[c])
                              for c in range(NUM_CLASSES)},
        'per_class_gap': {CLASS_NAMES[c]:
                          float(pc['ceiling'].per_class_iou()[c] - pc['frozen'].per_class_iou()[c])
                          for c in range(NUM_CLASSES)},
        'pool_support': {CLASS_NAMES[c]: int(pool_counts[c]) for c in range(NUM_CLASSES)},
    }
    if w0_alt is not None:
        out['frozen_W0_alt'] = m['frozen_W0_alt']
    if args.gate_off and 'ceiling_gate_off' in m:
        out['ceiling_gate_off'] = m['ceiling_gate_off']
        out['frozen_gate_off'] = m['frozen_gate_off']
        out['gate_delta_ceiling'] = m['ceiling_gate_off'] - m['ceiling']
    print(f"  [R4] frozen {out['frozen']:.3f} / ceiling {out['ceiling']:.3f} "
          f"(gap {out['gap']:+.3f}) resid {r_norm:.3f} effrank {spec['effrank_pr']} "
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
    ap.add_argument("--nusc_c_dir", type=str, default="/mnt/bravo/jmfleming/nuscenes_c_kitti")
    ap.add_argument("--nusc_dir", type=str, default="/mnt/alpha/jmfleming/nuscenes_kitti")
    ap.add_argument("--nusc_labels", type=str, default="config/labels/nuscenes_c.yaml")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--gate_off", type=int, default=1,
                    help="1 = also decode with model.input_in disabled (D7 gate test)")
    ap.add_argument("--extractors", type=str, default="all",
                    help="comma list of cov_kitti,dgl_kitti,cov_nusc,dgl_nusc (default all)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    import copy as _copy
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.oracle_core import get_hdc_projection
    from modules.gen_trainers import GenTrainer
    from robust_diagnostic.al_full_dataset_diag import build_nuscenes_parser
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)

    specs = {
        'cov_kitti': ('supcon_vib_dglsspp_inputin_in_chan',
                      'robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan',
                      'kittic', CONDS_ALL, False),
        'dgl_kitti': ('supcon_vib_dglsspp',
                      'robust_diagnostic/logs/supcon_vib_dglsspp',
                      'kittic', CONDS_ALL, False),
        'cov_nusc': ('supcon_vib_dglsspp_inputin_in_chan',
                     'robust_diagnostic/logs/nusc_covshift_21ep',
                     'nuscenes_c', CONDS_ALL, True),
        'dgl_nusc': ('supcon_vib_dglsspp',
                     'robust_diagnostic/logs/nusc_dglsspp_21ep',
                     'nuscenes_c', CONDS_ALL, True),
    }
    want = [s.strip() for s in args.extractors.split(',') if s.strip()]
    if want == ['all']:
        want = list(specs)
    nusc_data = yaml.safe_load(open(args.nusc_labels))
    # D9: nuScenes-clean parser for the W0-source control
    nusc_clean_parser = build_nuscenes_parser(args.nusc_dir, nusc_data, ARCH)

    results = {'label': 'covshift_mechanism', 'extractors': {}}
    for lab in want:
        method, path, dataset, conds, is_nusc = specs[lab]
        print(f"\n{'='*80}\n=== extractor {lab} ({method}, {dataset}) ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        print(f"=== [{lab}] clean fit (KITTI seq-08 reservoir {args.clean_fit_n}) ===")
        cf, cl, ck = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                       args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
        protos_clean = build_prototypes(Xc, cl, device=device)
        feat_means_clean = class_means_feats(cf, cl)

        # D1 clean baseline: frozen + ceiling on the CLEAN val stream
        clean_pool, clean_lbl, clean_key = reservoir_collect(
            stream_frames(model, clean_parser, device, args.max_frames), args.pool_cap, 42)
        Xclean = hdc_codes(clean_pool, proj, device).float()
        Wclean = ridge_fit_exact(Xclean, onehot(clean_lbl, NUM_CLASSES), args.lam, device)
        # D1 clean ceiling decoded on a clean held-out portion (reservoir exclusions)
        clean_ex = defaultdict(list)
        for f, i in clean_key.tolist():
            clean_ex[f].append(i)
        clean_ex = {f: torch.tensor(sorted(s), dtype=torch.long) for f, s in clean_ex.items()}
        pcc = stream_decode_mech(model, clean_parser, proj, device,
                                 {'frozen': W0.detach().cpu(), 'ceiling': Wclean.detach().cpu()},
                                 exclude=clean_ex, max_frames=args.max_frames)
        clean_res = {'frozen': pcc['frozen'].miou(), 'ceiling': pcc['ceiling'].miou(),
                     'gap': pcc['ceiling'].miou() - pcc['frozen'].miou(),
                     'n_val': pcc['frozen'].n}

        # D9: W0-source control -- fit W0 on nuScenes-clean val frames
        w0_alt = None
        if is_nusc:
            print(f"=== [{lab}] D9 W0-source control (nuScenes-clean fit) ===")
            nc_pool, nc_lbl, _ = reservoir_collect(
                stream_frames(model, nusc_clean_parser, device, args.max_frames),
                args.clean_fit_n, 7)
            Xnc = hdc_codes(nc_pool, proj, device).float()
            w0_alt = ridge_fit_exact(Xnc, onehot(nc_lbl, NUM_CLASSES), args.lam, device)
            print(f"  W0_nusc done ({len(nc_pool)} pts)")

        results['extractors'][lab] = {'method': method, 'dataset': dataset,
                                      'clean': clean_res, 'conds': {}}
        for cond in conds:
            if dataset == 'kittic':
                cdir = os.path.join(args.kittic_dir, cond, 'heavy')
                if not os.path.exists(cdir):
                    cdir = os.path.join(args.kittic_dir, cond, 'moderate')
                parser = build_parser(cdir, DATA, ARCH)
            else:
                parser = build_nuscenes_parser(os.path.join(args.nusc_c_dir, cond, 'heavy'),
                                               nusc_data, ARCH)
            r = run_condition(model, parser, proj, device, W0, protos_clean,
                              feat_means_clean, args, label=lab, cond_name=cond,
                              w0_alt=w0_alt)
            results['extractors'][lab]['conds'][cond] = r
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\n[checkpoint] {lab} done, saved to {args.out}")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
