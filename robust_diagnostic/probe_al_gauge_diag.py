"""probe_al_gauge_diag.py: can a NAIVE LABEL-FREE signal predict whether a
condition is worth active learning? (eval-only, frozen cov-shift ep10.)

The AL-closeable gap (ceiling - frozen) at full scale is small on most
conditions (fog +0.047, crosstalk +0.014, snow +0.013, wet_ground +0.122) --
so spending labels is only justified where the gap is real. This diagnostic
tests whether any signal computable WITHOUT labels on the corrupted condition
tracks the measured gap, i.e. a deployable "should we do AL here?" gauge.

Label-free signals (all computed from the corrupted features + the clean-fit
probe only, no corrupted labels):
  norm_ratio     : mean ||z|| corrupted / clean (norm inflation)
  mean_shift_cos : cosine between the global clean and corrupted feature means
  conf_drop      : clean-fit probe's mean max-softmax confidence drop
                   (corrupted vs clean)
  dead_frac      : HDC code dead-coordinate fraction on the corrupted pool
  hamming        : mean pairwise code Hamming distance on the corrupted pool
  r4_r1_disagree : disagreement rate between the R4 linear probe and the R1
                   prototype decode on the corrupted pool (both frozen/clean-fit)

The reference gap (ceiling - frozen) is measured with labels on a bounded val
reservoir. Spearman rho(gap, signal) across the 8 conditions tells whether a
signal is a usable gauge; the best single signal is reported.

Usage:
  uv run python robust_diagnostic/probe_al_gauge_diag.py \
    --path_b <ckpt> --method_b <method> --label gauge_ep10 \
    --out robust_diagnostic/logs/probe_al_gauge_ep10.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, build_prototypes, NUM_CLASSES, CONDS_ALL)
from robust_diagnostic.al_per_class_diag import ConfMatrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = ALL frames of seq 08")
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--val_cap", type=int, default=200000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="gauge_ep10")
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

    # ---- clean: W0 + prototypes + clean reference stats ----
    t0 = time.time()
    print(f"=== clean (reservoir {args.clean_fit_n}) ===")
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                  args.clean_fit_n, 7)
    cf, cl = cf.to(device), cl.long()
    Xc = torch.sign(cf @ proj).float()
    W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
    protos_clean = build_prototypes(Xc, cl, device=device).cpu()
    norm_clean = float(cf.norm(dim=1).mean())
    conf_clean = float(F.softmax(Xc @ W0, dim=1).max(1).values.mean())
    mean_clean = cf.mean(0)
    print(f"  clean {len(cf)} pts, norm {norm_clean:.3f}, conf {conf_clean:.3f} "
          f"({time.time()-t0:.0f}s)")
    del cf, cl, Xc
    torch.cuda.empty_cache()

    rows = {}
    for cond in conds:
        t0 = time.time()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        cparser = build_parser(cdir, DATA, ARCH)

        # pool reservoir (seed 42) -- signals need NO corrupted labels
        pf, pl, _ = reservoir_collect(stream_frames(model, cparser, device, args.max_frames),
                                      args.pool_cap, 42)
        vf, vl, _ = reservoir_collect(stream_frames(model, cparser, device, args.max_frames),
                                      args.val_cap, 43)
        pf, pl = pf.to(device), pl.long()
        vf, vl = vf.to(device), vl.long()
        Xp = torch.sign(pf @ proj).float()
        Xv = torch.sign(vf @ proj).float()

        # --- label-free signals (no corrupted labels used) ---
        norm_ratio = float(pf.norm(dim=1).mean() / norm_clean)
        mean_shift_cos = float(F.cosine_similarity(mean_clean.unsqueeze(0),
                                                   pf.mean(0).unsqueeze(0)).item())
        # simpler robust label-free signals, all on the corrupted pool:
        conf_pool = float(F.softmax(Xp @ W0, dim=1).max(1).values.mean())
        conf_drop = conf_clean - conf_pool
        frac_pos = (Xp > 0).float().mean(0)
        dead_frac = float(((frac_pos < 0.05) | (frac_pos > 0.95)).float().mean().item())
        torch.manual_seed(7)
        a = Xp[torch.randperm(len(Xp))[:20000]]
        b = Xp[torch.randperm(len(Xp))[:20000]]
        hamming = float((1.0 - (a == b).float().mean(1)).mean().item())
        # R4 (linear) vs R1 (prototype) disagreement on the corrupted pool
        r4 = (Xp @ W0).argmax(1)
        sims = F.normalize(Xp, p=2, dim=1) @ protos_clean.to(device).t()
        r1 = torch.arange(NUM_CLASSES, device=device)[sims.argmax(1)]
        r4_r1_disagree = float((r4 != r1).float().mean().item())

        # --- reference gap (WITH labels, bounded val) ---
        Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device)
        cm_f, cm_c = ConfMatrix(), ConfMatrix()
        for s in range(0, len(Xv), 100000):
            e = min(s + 100000, len(Xv))
            cm_f.update((Xv[s:e] @ W0).argmax(1).cpu(), vl[s:e].cpu())
            cm_c.update((Xv[s:e] @ Ws).argmax(1).cpu(), vl[s:e].cpu())
        frozen, ceiling = cm_f.miou(), cm_c.miou()
        gap = ceiling - frozen

        rows[cond] = {
            'norm_ratio': norm_ratio, 'mean_shift_cos': mean_shift_cos,
            'conf_drop': conf_drop,
            'dead_frac': dead_frac, 'hamming': hamming,
            'r4_r1_disagree': r4_r1_disagree,
            'frozen': frozen, 'ceiling': ceiling, 'gap': gap,
            'n_pool': len(pf), 'n_val': len(vf),
        }
        print(f"\n=== {cond} (gap {gap:+.3f}) ===")
        print(f"  norm_ratio {norm_ratio:.3f} | mean_shift_cos {mean_shift_cos:.3f} | "
              f"conf_drop {conf_drop:+.3f} | dead {dead_frac:.3f} | hamm {hamming:.3f} | "
              f"r4r1_disagree {r4_r1_disagree:.3f}")
        del pf, pl, vf, vl, Xp, Xv, Ws, cm_f, cm_c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Spearman rho(gap, signal) across conditions ---
    sig_names = ['norm_ratio', 'mean_shift_cos', 'conf_drop', 'dead_frac',
                 'hamming', 'r4_r1_disagree']
    rhos = {}
    gaps = np.array([rows[c]['gap'] for c in conds])
    for s in sig_names:
        vals = np.array([rows[c][s] for c in conds])
        rho = spearmanr(gaps, vals).statistic
        rhos[s] = float(rho)
        print(f"  Spearman rho(gap, {s:<16s}) = {rho:+.3f}")
    best = max(rhos, key=lambda k: abs(rhos[k]))
    print(f"\nBest single gauge signal: {best} (|rho| = {abs(rhos[best]):.3f})")
    rows['_spearman'] = rhos
    rows['_best_signal'] = best

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(rows, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
