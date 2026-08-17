"""ttacollapse_diag.py: per-class diagnosis of the dircons medium TTA collapse.

Iteration-18 finding: the dircons decoupling raised the labeled ceiling (car crosstalk
oracle 0.280 -> 0.430, ts 0.190 -> 0.272) but the crosstalk label-free naive-EMA gap
collapsed (+0.52 -> +0.14), with traffic-sign naive 0.203 -> 0.071 and building
0.257 -> 0.072. This script tests which mechanism caused it, per class, using the
extractor_diff per-branch JSON (path_a = baseline extractor, path_b = dircons):

  1. Does the retained corr shift CAUSE the naive failure?
       rho(corr_dir_retention, d(naive_gain))  -- classes the dircons shifted hardest
       (low corr_dir) are the ones whose naive update lost ground?
  2. Is the collapse LP-assignment driven (the shift broke pseudo-labels)?
       rho(d(lp_recall), d(naive_gain)) and rho(corr_dir_retention, d(lp_recall))
  3. Does the shift buy the ceiling where it costs the TTA (the inherent conflict)?
       rho(corr_dir_retention, d(oracle))  -- shift helps oracle on the same classes
       where it hurts naive?
  4. Which classes drove the collapse, and is it recoverable by a different update
       (the tta_ceiling battery naive/conf/dist/BN/kNN is the companion test).

Usage:
  uv run python robust_diagnostic/ttacollapse_diag.py
      --json robust_diagnostic/logs/extractor_diff_dircons_med.json
      --label_a robust_21ep --label_b <dircons_method> --out <out.json>
"""
import json
import argparse
import os

def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 4:
        return float('nan')
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = (sum((rx[i] - mx) ** 2 for i in range(n))) ** 0.5
    dy = (sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return (cov / (dx * dy)) if dx * dy > 0 else float('nan')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str,
                        default="robust_diagnostic/logs/extractor_diff_dircons_med.json")
    parser.add_argument("--label_a", type=str, default="robust_21ep",
                        help="baseline label in the extractor_diff JSON")
    parser.add_argument("--label_b", type=str,
                        default="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons",
                        help="dircons label in the extractor_diff JSON")
    parser.add_argument("--out", type=str,
                        default="robust_diagnostic/logs/ttacollapse_dircons_med.json")
    args = parser.parse_args()

    data = json.load(open(args.json, 'rb'))
    la, lb = args.label_a, args.label_b
    if la not in data or lb not in data:
        raise SystemExit(f"labels {la} / {lb} not both in {list(data)}")

    results = {}
    for cond in ('fog', 'crosstalk'):
        pa, pb = data[la][cond]['per_class'], data[lb][cond]['per_class']
        rows = []
        for c in sorted(pa, key=int):
            ra, rb = pa[c], pb[c]
            g = lambda k, d: d.get(k) if d.get(k) is not None else float('nan')
            if g('naive_iou', ra) != g('naive_iou', ra) or g('naive_iou', rb) != g('naive_iou', rb):
                continue
            na_a = g('naive_iou', ra) - g('zs_iou', ra)
            na_b = g('naive_iou', rb) - g('zs_iou', rb)
            rows.append({
                'cls': int(c),
                'zsA': g('zs_iou', ra), 'zsB': g('zs_iou', rb),
                'naiveA': g('naive_iou', ra), 'naiveB': g('naive_iou', rb),
                'oracleA': g('oracle_iou', ra), 'oracleB': g('oracle_iou', rb),
                'naive_gain_A': na_a, 'naive_gain_B': na_b,
                'd_naive_gain': na_b - na_a,
                'd_oracle': g('oracle_iou', rb) - g('oracle_iou', ra),
                'lpA': g('lp_recall', ra), 'lpB': g('lp_recall', rb),
                'd_lp': g('lp_recall', rb) - g('lp_recall', ra),
                'corr_dir': g('corr_dir_retention', rb),
                'inv_fc': g('inv_feat_cos', rb),
                'corr_tight': g('corr_tightness', rb),
            })
        rows.sort(key=lambda r: r['d_naive_gain'])

        def col(k):
            return [r[k] for r in rows]

        ok = lambda v: [x for x in v if x == x]
        rho = {
            'rho(corr_dir, d_naive_gain)': spearman(ok(col('corr_dir')), ok(col('d_naive_gain'))),
            'rho(corr_dir, d_oracle)':     spearman(ok(col('corr_dir')), ok(col('d_oracle'))),
            'rho(corr_dir, d_lp)':         spearman(ok(col('corr_dir')), ok(col('d_lp'))),
            'rho(d_lp, d_naive_gain)':     spearman(ok(col('d_lp')), ok(col('d_naive_gain'))),
            'rho(inv_fc, d_naive_gain)':   spearman(ok(col('inv_fc')), ok(col('d_naive_gain'))),
            'rho(d_oracle, d_naive_gain)': spearman(ok(col('d_oracle')), ok(col('d_naive_gain'))),
        }
        results[cond] = {'per_class': rows, 'correlations': rho}

        print(f"\n{'='*80}\n=== {cond}: per-class TTA collapse diagnosis ({la} -> {lb}) ===\n{'='*80}")
        print(f"{'cls':>3} {'corr_dir':>8} {'inv_fc':>7} {'naA':>6} {'naB':>6} "
              f"{'d_naive':>8} {'lpA':>6} {'lpB':>6} {'d_lp':>7} {'orA':>6} {'orB':>6} {'d_or':>7}")
        for r in rows:
            print(f"{r['cls']:>3} {r['corr_dir']:>8.2f} {r['inv_fc']:>7.2f} "
                  f"{r['naiveA']:>6.3f} {r['naiveB']:>6.3f} {r['d_naive_gain']:>+8.3f} "
                  f"{r['lpA']:>6.2f} {r['lpB']:>6.2f} {r['d_lp']:>+7.2f} "
                  f"{r['oracleA']:>6.3f} {r['oracleB']:>6.3f} {r['d_oracle']:>+7.3f}")
        print("\ncorrelations:")
        for k, v in rho.items():
            print(f"  {k:<32} {v:+.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
