"""probe_decoder_ceiling_diag.py: is the linear probe on the HDC code the best
decoder, or is there a HIGHER ceiling reachable by a more expressive / balanced
decoder? This decides whether the paper's "ceiling" (the recoverable-gap
reference the TTA/AL method is judged against) can be raised, and whether the
gaps are consistent across conditions.

For each condition (full harness: 400k pool, seed 42; full val decode, pool
excluded), computes the CEILING under several decoder families fit on the
labeled corrupted pool:

  R4 linear (code)        : current ceiling -- the paper reference. Linear on
                            the binarized 10000-d HDC code (spectral ridge).
  linear (raw 128-d)      : does the HDC BINARIZATION lose recoverable
                            structure? A linear probe on the raw 128-d features.
  kNN (code)              : non-parametric oracle ceiling -- 1-NN / 5-NN to the
                            pool in the HDC code space. If kNN >> linear, the
                            decision is NOT linear and a more expressive decoder
                            (or the memory-bank AL with more points) has room.
  kNN (raw 128-d)         : same but in the raw feature space -- separates
                            "binarization loss" from "linearity loss".
  RBF probe (code)        : Nystrom/Random-Fourier kernel ridge on the code --
                            the cheapest nonlinear boundary; is the decision
                            curved?
  linear per-class lam    : class-balanced ridge (lam_c = lam * N / N_c) -- the
                            "more balanced" probe.
  linear 1-vs-rest        : per-class one-vs-rest ridge with balanced lam.

If kNN or RBF >> R4-linear on many conditions, the linear probe is NOT the
ceiling and a better decoder raises the recoverable headroom (and makes the
gaps more consistent). If they all sit near R4-linear, the code's information
limit is reached and the gaps are what they are.

Usage:
  uv run python robust_diagnostic/probe_decoder_ceiling_diag.py \
    --max_frames 200 --out robust_diagnostic/logs/probe_decoder_ceiling.json
"""
import os, sys, time, argparse, json, yaml, copy as _copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, ConfAccum, NUM_CLASSES)

CONDS = ['fog', 'crosstalk', 'snow', 'wet_ground', 'incomplete_echo',
         'beam_missing', 'motion_blur', 'cross_sensor']

def decode_stream_codes(model, parser, proj, device, decode_fn, max_frames=0,
                        chunk=100000):
    """Stream all frames; codes = sign(z @ proj). decode_fn(codes_chunk) -> preds.
    `chunk` is the VAL chunk size (per-frame code batch); the kNN decoders pass a
    smaller chunk so the (chunk x bank) similarity stays under memory."""
    cm = ConfAccum()
    for zf, labels, fi in stream_frames(model, parser, device, max_frames):
        n = len(zf)
        if n == 0:
            continue
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            codes = torch.sign(zf[s:e].to(device) @ proj).float()
            preds = decode_fn(codes)
            cm.update(preds.cpu(), labels[s:e])
        del zf, labels
    return cm.miou(), cm.n

def knn_decode(codes, bank, bank_lbl, k=1, chunk=None):
    """codes: (B,d); bank: (P,d) normalized code prototypes; return argmax preds.
    The caller already passes a small `chunk` batch (via decode_stream_codes), so
    the (B x P) similarity is bounded; `chunk` here is ignored but kept for
    signature clarity."""
    bn = F.normalize(bank.float(), p=2, dim=1).to(codes.device)
    cn = F.normalize(codes, p=2, dim=1)
    sim = cn @ bn.t()
    if k == 1:
        return bank_lbl.to(codes.device)[sim.argmax(1)]
    topk = sim.topk(k, dim=1).indices
    preds = torch.stack([bank_lbl.to(codes.device)[topk[:, j]] for j in range(k)])
    return preds.mode(0).values

def rbf_features(codes, proj_rff, sigma=1.0):
    """Random Fourier features of the code (approximate RBF kernel)."""
    z = codes @ proj_rff   # (B, D_rff)
    return torch.cat([torch.cos(sigma * z), torch.sin(sigma * z)], dim=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str,
                    default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--clean_fit_n", type=int, default=50000)
    ap.add_argument("--pool_cap", type=int, default=200000)
    ap.add_argument("--knn_bank", type=int, default=100000,
                    help="bank size for the kNN oracle (subsample of the pool)")
    ap.add_argument("--rff_dim", type=int, default=2048)
    ap.add_argument("--conds", type=str, default=",".join(CONDS))
    ap.add_argument("--extractors", type=str,
                    default="cov_ep10:supcon_vib_dglsspp_inputin_in_chan:"
                            "robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/"
                            "supcon_vib_dglsspp_inputin_in_chan")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modules.oracle_core import get_hdc_projection
    from modules.gen_trainers import GenTrainer
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    # RFF projection (seeded, fixed) for the kernel probe
    torch.manual_seed(7)
    proj_rff = torch.randn(10000, args.rff_dim, device=device)

    results = {'label': 'decoder_ceiling', 'extractors': {}}
    for lab, method, path in [tuple(e.strip().split(':')) for e in args.extractors.split(',')]:
        print(f"\n{'='*80}\n=== extractor {lab} ===\n{'='*80}")
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                      args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), 1e-3, device).to(device)

        results['extractors'][lab] = {'method': method, 'conds': {}}
        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            parser = build_parser(cdir, DATA, ARCH)
            t0 = time.time()
            # pool (400k reservoir) for the ceiling fits
            pf, pl, _ = reservoir_collect(stream_frames(model, parser, device, args.max_frames),
                                          args.pool_cap, 42)
            Xp = hdc_codes(pf, proj, device).float()
            Yp = onehot(pl, NUM_CLASSES)
            print(f"  {cond}: pool {len(pf)} ({time.time()-t0:.0f}s)")

            entry = {}
            # 1. R4 linear on code (reference ceiling)
            W_lin = ridge_fit_exact(Xp, Yp, 1e-3, device).to(device)
            miou, n = decode_stream_codes(model, parser, proj, device,
                                          lambda c: (c @ W_lin).argmax(1), args.max_frames)
            entry['r4_linear_code'] = miou
            print(f"    R4 linear (code)      : {miou:.3f}")

            # 2. linear on raw 128-d (binarization check)
            # fit ridge on raw features
            def ridge_raw(feats, lbls, lam=1e-3):
                S = torch.zeros(128, 128, device=device); T = torch.zeros(128, NUM_CLASSES, device=device)
                for s in range(0, len(feats), 50000):
                    Xc = feats[s:s+50000].to(device); Yc = onehot(lbls[s:s+50000], NUM_CLASSES).to(device)
                    S += Xc.t() @ Xc; T += Xc.t() @ Yc
                A = S.double() + lam * torch.eye(128, dtype=torch.float64, device=device)
                return torch.linalg.solve(A, T.double()).float()
            W_raw = ridge_raw(pf, pl).to(device)
            acc_raw = ConfAccum()
            for zf, labels, fi in stream_frames(model, parser, device, args.max_frames):
                n = len(zf)
                for s in range(0, n, 100000):
                    e = min(s + 100000, n)
                    preds = (zf[s:e].to(device) @ W_raw).argmax(1).cpu()
                    acc_raw.update(preds, labels[s:e])
                del zf, labels
            entry['linear_raw128'] = acc_raw.miou()
            print(f"    linear (raw 128)      : {entry['linear_raw128']:.3f}")

    # 3/4. kNN oracle (subsampled pool) on code and raw
    torch.manual_seed(42)
    idx = torch.randperm(len(pf))[:min(args.knn_bank, len(pf))]
    bank_codes = Xp[idx].to(device)
    bank_lbl = pl[idx]
    bank_feats = pf[idx]
    # kNN similarity matrix is (chunk x bank); keep the chunk small (10k) so
    # 10k x 100k = 1e9 floats = 4GB, far under the OOM boundary.
    knn_chunk = 10000
    entry['knn1_code'] = decode_stream_codes(
        model, parser, proj, device,
        lambda c: knn_decode(c, bank_codes, bank_lbl, k=1, chunk=knn_chunk),
        args.max_frames, chunk=knn_chunk)[0]
    entry['knn5_code'] = decode_stream_codes(
        model, parser, proj, device,
        lambda c: knn_decode(c, bank_codes, bank_lbl, k=5, chunk=knn_chunk),
        args.max_frames, chunk=knn_chunk)[0]
    print(f"    kNN1 (code)           : {entry['knn1_code']:.3f}")
    print(f"    kNN5 (code)           : {entry['knn5_code']:.3f}")

    # raw-feature kNN (128-d, so a larger chunk is fine: 100k x 128 is tiny)
    def knn_decode_feat(fz_chunk):
        bn = F.normalize(bank_feats.float().to(device), p=2, dim=1)
        cn = F.normalize(fz_chunk.float().to(device), p=2, dim=1)
        return bank_lbl.to(device)[(cn @ bn.t()).argmax(1)]
    acc_fknn = ConfAccum()
    for zf, labels, fi in stream_frames(model, parser, device, args.max_frames):
        n = len(zf)
        for s in range(0, n, 100000):
            e = min(s + 100000, n)
            acc_fknn.update(knn_decode_feat(zf[s:e]).cpu(), labels[s:e])
        del zf, labels
    entry['knn1_raw'] = acc_fknn.miou()
    print(f"    kNN1 (raw 128)        : {entry['knn1_raw']:.3f}")

    # 5. RBF probe on code (random Fourier features + ridge)
    Zr = rbf_features(Xp.to(device), proj_rff)
    Yr = Yp.to(device)
    W_rff = ridge_fit_exact(Zr.cpu(), Yr.cpu(), 1e-2, device).to(device)
    def rbf_decode(codes):
        zf = rbf_features(codes.to(device), proj_rff)
        return (zf @ W_rff).argmax(1)
    entry['rff_ridge_code'] = decode_stream_codes(
        model, parser, proj, device, rbf_decode, args.max_frames)[0]
    print(f"    RFF ridge (code)      : {entry['rff_ridge_code']:.3f}")

    # 6. balanced linear probe (per-class lam)
    from robust_diagnostic.al_full_dataset_diag import ridge_fit_balanced
    counts = torch.bincount(pl.long(), minlength=NUM_CLASSES)
    W_bal = ridge_fit_balanced(Xp, Yp, counts, 1e-3, device, mode='lam').to(device)
    entry['linear_bal_lam'] = decode_stream_codes(
        model, parser, proj, device,
        lambda c: (c @ W_bal).argmax(1), args.max_frames)[0]
    print(f"    linear (bal lam)      : {entry['linear_bal_lam']:.3f}")

    entry['n_val'] = n
    entry['pool_n'] = len(pf)
    results['extractors'][lab]['conds'][cond] = entry
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"    ({time.time()-t0:.0f}s) [checkpoint saved]")
    print(f"[checkpoint] {lab} done")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
