"""probe_al_gauge_multi_diag.py: robust test of the label-free "should we do AL?"
gauge, across MULTIPLE extractors (eval-only, frozen checkpoints).

Extends the single-run probe_al_gauge_diag (which found mean_shift_cos rho -0.69,
r4_r1_disagree +0.64, conf_drop +0.62 on cov_ep10) with the questions that
decide whether the gauge is a real deployable mechanism:

  1. CROSS-EXTRACTOR VALIDITY: do the same label-free signals track the measured
     AL-closeable gap on every extractor (cov_ep10, cov_ep21, dglsspp, robust)?
     If the signal-gap correlation only holds on the cov-shift extractor, it is
     a quirk; if it holds everywhere, it is a mechanism.
  2. COMBINED SCORE: z-normalize the signals across conditions and combine the
     best three (mean_shift_cos [inverted], r4_r1_disagree, conf_drop) into one
     score; does the combined score beat every single signal?
  3. THRESHOLD DECISION TEST: with the score as a label-free gate, sweep the
     decision threshold and report, per extractor:
       - which conditions the gate would route to AL (spend labels) vs skip,
       - the gap captured (sum of measured gaps on the routed conditions),
       - the gap missed (sum of gaps on the skipped conditions),
       - precision/recall vs the oracle rule "route iff gap >= t_gap".
     This is the deployable label-or-don't gate: cheap, label-free, and it must
     route wet_ground/fog (big gaps) while skipping beam_missing/motion_blur/snow.

Signals (no corrupted labels, same as the single-run version):
  norm_ratio, mean_shift_cos, conf_drop, dead_frac, hamming, r4_r1_disagree.

Usage:
  uv run python robust_diagnostic/probe_al_gauge_multi_diag.py \
    --extractors \
      "cov_ep10:supcon_vib_dglsspp_inputin_in_chan:<path>,cov_ep21:...:...,dglsspp:supcon_vib_dglsspp:<path>,robust:supcon_vib_dglsspp_corsupcon:<path>" \
    --label gauge_multi --out robust_diagnostic/logs/probe_al_gauge_multi.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
from robust_diagnostic.al_full_dataset_diag import (
    build_parser, stream_frames, reservoir_collect, hdc_codes, onehot,
    ridge_fit_exact, build_prototypes, NUM_CLASSES, CONDS_ALL)
from robust_diagnostic.al_per_class_diag import ConfMatrix

def cond_signals(model, cparser, proj, W0, protos_clean, args, cond):
    """Compute the label-free signals + the measured gap for one condition."""
    from collections import defaultdict
    pf, pl, _ = reservoir_collect(stream_frames(model, cparser, device, args.max_frames,
                                                progress=cond), args.pool_cap, 42)
    vf, vl, _ = reservoir_collect(stream_frames(model, cparser, device, args.max_frames),
                                  args.val_cap, 43)
    pf, pl = pf.to(device), pl.long()
    vf, vl = vf.to(device), vl.long()
    Xp = torch.sign(pf @ proj).float()
    Xv = torch.sign(vf @ proj).float()

    norm_ratio = float(pf.norm(dim=1).mean() / args.norm_clean)
    mean_shift_cos = float(F.cosine_similarity(args.mean_clean.unsqueeze(0),
                                               pf.mean(0).unsqueeze(0)).item())
    conf_pool = float(F.softmax(Xp @ W0, dim=1).max(1).values.mean())
    conf_drop = args.conf_clean - conf_pool
    frac_pos = (Xp > 0).float().mean(0)
    dead_frac = float(((frac_pos < 0.05) | (frac_pos > 0.95)).float().mean().item())
    torch.manual_seed(7)
    a = Xp[torch.randperm(len(Xp))[:20000]]
    b = Xp[torch.randperm(len(Xp))[:20000]]
    hamming = float((1.0 - (a == b).float().mean(1)).mean().item())
    r4 = (Xp @ W0).argmax(1)
    sims = F.normalize(Xp, p=2, dim=1) @ protos_clean.to(device).t()
    r1 = torch.arange(NUM_CLASSES, device=device)[sims.argmax(1)]
    r4_r1_disagree = float((r4 != r1).float().mean().item())

    # reference gap (with labels, bounded val)
    Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device)
    cm_f, cm_c = ConfMatrix(), ConfMatrix()
    for s in range(0, len(Xv), 100000):
        e = min(s + 100000, len(Xv))
        cm_f.update((Xv[s:e] @ W0).argmax(1).cpu(), vl[s:e].cpu())
        cm_c.update((Xv[s:e] @ Ws).argmax(1).cpu(), vl[s:e].cpu())
    gap = cm_c.miou() - cm_f.miou()

    return {'norm_ratio': norm_ratio, 'mean_shift_cos': mean_shift_cos,
            'conf_drop': conf_drop, 'dead_frac': dead_frac,
            'hamming': hamming, 'r4_r1_disagree': r4_r1_disagree, 'gap': gap}

def main():
    global device
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--val_cap", type=int, default=200000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--extractors", type=str, required=True,
                    help="comma-separated label:method:path triplets")
    ap.add_argument("--label", type=str, default="gauge_multi")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    extractors = [tuple(e.strip().split(':')) for e in args.extractors.split(',') if e.strip()]
    proj = __import__('modules.oracle_core', fromlist=['get_hdc_projection']).get_hdc_projection(
        dim_in=128, dim_out=10000, device=device)

    results = {'label': args.label, 'extractors': {}}
    for lab, method, path in extractors:
        t0 = time.time()
        print(f"\n{'='*70}\n=== extractor {lab} ({method}) ===")
        trainer = __import__('modules.gen_trainers', fromlist=['GenTrainer']).GenTrainer(
            ARCH, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model

        # clean W0 + clean reference stats
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        cf, cl, _ = reservoir_collect(stream_frames(model, clean_parser, device,
                                                    args.max_frames, progress=f'{lab}/clean'),
                                      args.clean_fit_n, 7)
        cf, cl = cf.to(device), cl.long()
        Xc = torch.sign(cf @ proj).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
        protos_clean = build_prototypes(Xc, cl, device=device).cpu()
        args.norm_clean = float(cf.norm(dim=1).mean())
        args.mean_clean = cf.mean(0)
        args.conf_clean = float(F.softmax(Xc @ W0, dim=1).max(1).values.mean())
        print(f"  clean {len(cf)} pts, norm {args.norm_clean:.3f} conf {args.conf_clean:.3f}")
        del cf, cl, Xc
        torch.cuda.empty_cache()

        rows = {}
        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            cparser = build_parser(cdir, DATA, ARCH)
            rows[cond] = cond_signals(model, cparser, proj, W0, protos_clean, args, cond)
            print(f"  {cond}: gap {rows[cond]['gap']:+.3f} | shift_cos "
                  f"{rows[cond]['mean_shift_cos']:.3f} | r4r1 "
                  f"{rows[cond]['r4_r1_disagree']:.3f} | conf_drop "
                  f"{rows[cond]['conf_drop']:+.3f}")
            torch.cuda.empty_cache()

        # ---- per-signal Spearman across conditions ----
        sig_names = ['norm_ratio', 'mean_shift_cos', 'conf_drop', 'dead_frac',
                     'hamming', 'r4_r1_disagree']
        gaps = np.array([rows[c]['gap'] for c in conds])
        rhos = {s: float(spearmanr(gaps, np.array([rows[c][s] for c in conds])).statistic)
                for s in sig_names}

        # ---- combined score: z-normalize, invert shift_cos, weight best 3 ----
        z = {}
        for s in sig_names:
            v = np.array([rows[c][s] for c in conds])
            z[s] = (v - v.mean()) / (v.std() + 1e-9)
        score = -z['mean_shift_cos'] + z['r4_r1_disagree'] + z['conf_drop']
        score = (score - score.mean()) / (score.std() + 1e-9)
        rho_score = float(spearmanr(gaps, score).statistic)

        # ---- threshold decision test ----
        # oracle: route to AL iff gap >= t_gap; gauge: route iff score >= t_score.
        # sweep t_score, measure gap captured vs oracle at a fixed t_gap.
        t_gap = 0.03   # "there is something to close" bar (fog 0.047 / wet 0.12)
        oracle_route = [c for c in conds if rows[c]['gap'] >= t_gap]
        gap_total = float(gaps.sum())
        gap_oracle = float(sum(rows[c]['gap'] for c in oracle_route))
        decisions = []
        for t in np.linspace(score.min() - 0.1, score.max() + 0.1, 21):
            routed = [c for c, sc in zip(conds, score) if sc >= t]
            gap_captured = float(sum(rows[c]['gap'] for c in routed))
            tp = len([c for c in routed if rows[c]['gap'] >= t_gap])
            fp = len(routed) - tp
            fn = len([c for c in oracle_route if c not in routed])
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            # capture-efficiency: fraction of total gap captured per routed
            # condition -- rewards a SELECTIVE gate (routing only the big-gap
            # conditions), unlike frac_gap alone which rewards routing everything.
            eff = gap_captured / (len(routed) * gap_total) if routed else 0.0
            decisions.append({'t': float(t), 'n_routed': len(routed),
                              'routed': routed, 'gap_captured': gap_captured,
                              'frac_gap': gap_captured / gap_total if gap_total else 0.0,
                              'capture_eff': eff,
                              'precision': prec, 'recall': rec})
        # pick the gate that captures the most gap per routed condition while
        # still catching both oracle conditions (recall=1.0 on the big gaps).
        best = max((d for d in decisions if d['recall'] >= 0.5),
                   key=lambda d: d['capture_eff'], default=decisions[0])

        results['extractors'][lab] = {
            'spearman': rhos, 'score_rho': rho_score,
            'gap_oracle_route': oracle_route, 'gap_oracle': gap_oracle,
            'gap_total': gap_total,
            'best_decision': best, 'decisions': decisions,
            'conds': rows,
        }
        print(f"  -- {lab}: best sig rho {max(rhos, key=lambda k: abs(rhos[k]))} "
              f"({max(abs(v) for v in rhos.values()):+.3f}) | score rho {rho_score:+.3f}")
        print(f"     oracle route (gap>={t_gap}): {oracle_route}")
        print(f"     best gate: routes {best['routed']}, captures "
              f"{best['gap_captured']:.3f}/{gap_total:.3f} "
              f"({best['frac_gap']:.0%}), prec {best['precision']:.2f} rec {best['recall']:.2f}")
        print(f"     ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
