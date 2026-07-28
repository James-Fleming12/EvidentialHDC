"""
analyze_ablations.py -- turn records.json into the tables that go in the paper.

Produces:
  1. Seed noise floor (so you can tell +0.01 from signal)
  2. Leave-one-out contributions, with Head/Mid/Tail
  3. Add-one-in ladder, with the interaction term vs leave-one-out
  4. Per-corruption deltas vs frozen, sorted
  5. The frozen-strength correlation (does adaptation hurt where the model is strong?)
  6. Accuracy-vs-mIoU divergence check (head/tail trade detector)

Every number is final-frozen vs initial-frozen. No cumulative-vs-frame-1 deltas.
"""

import json
import argparse
from collections import defaultdict


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def agg(records, key, field):
    """mean over corruptions, per seed -> then stats over seeds"""
    per_seed = defaultdict(list)
    for r in records:
        if r["ablation"] == key:
            per_seed[r["seed"]].append(r[field])
    seed_means = [mean(v) for v in per_seed.values()]
    return mean(seed_means), std(seed_means), len(seed_means)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="logs/ablation_v2/records.json")
    ap.add_argument("--pct", action="store_true", help="print as percentages")
    a = ap.parse_args()

    recs = json.load(open(a.records))
    s = 100.0 if a.pct else 1.0
    f = "%.2f" if a.pct else "%.4f"

    keys = []
    for r in recs:
        if r["ablation"] not in keys:
            keys.append(r["ablation"])
    names = {r["ablation"]: r["name"] for r in recs}
    fams = {r["ablation"]: r["family"] for r in recs}
    corrs = []
    for r in recs:
        if r["corruption"] not in corrs:
            corrs.append(r["corruption"])
    protocols = sorted({r["protocol"] for r in recs})
    nseeds = len(sorted({r["seed"] for r in recs}))

    print(f"\nprotocol={protocols}  seeds={nseeds}  corruptions={len(corrs)}")

    # ---------------- 1. noise floor ----------------
    print("\n" + "=" * 78)
    print("1. SEED NOISE FLOOR (std over seeds of the panel-mean final mIoU)")
    print("=" * 78)
    if nseeds < 2:
        print("  only one seed -- rerun with --seeds 42,43,44 before reading any small delta")
        floor = None
    else:
        floors = []
        for k in ("frozen", "full_method"):
            if k in keys:
                m, sd, n = agg(recs, k, "final_miou")
                print(f"  {k:<18} mean={f % (m*s)}  std={f % (sd*s)}  (n={n} seeds)")
                floors.append(sd)
        floor = max(floors) if floors else None
        if floor:
            print(f"\n  => treat |delta| < {f % (2*floor*s)} (2 sigma) as indistinguishable from noise")

    # ---------------- 2. leave-one-out ----------------
    print("\n" + "=" * 78)
    print("2. LEAVE-ONE-OUT  (component contribution = full - ablated)")
    print("=" * 78)
    have = lambda k: k in keys
    if have("full_method"):
        full_m = agg(recs, "full_method", "final_miou")[0]
        froz_m = agg(recs, "frozen", "final_miou")[0] if have("frozen") else float("nan")
        print(f"  frozen        final mIoU = {f % (froz_m*s)}")
        print(f"  full method   final mIoU = {f % (full_m*s)}   "
              f"(net adaptation {f % ((full_m-froz_m)*s)})\n")
        hdr = f"  {'ablation':<26}{'final':>9}{'contrib':>10}{'head':>9}{'mid':>9}{'tail':>9}{'acc':>9}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for k in keys:
            if fams.get(k) != "loo" or k == "full_method":
                continue
            v = agg(recs, k, "final_miou")[0]
            contrib = full_m - v
            flag = ""
            if floor and abs(contrib) < 2 * floor:
                flag = "  (< noise)"
            elif contrib < 0:
                flag = "  (HURTS)"
            print(f"  {names[k][:25]:<26}{f % (v*s):>9}{f % (contrib*s):>10}"
                  f"{f % (agg(recs,k,'final_head')[0]*s):>9}"
                  f"{f % (agg(recs,k,'final_mid')[0]*s):>9}"
                  f"{f % (agg(recs,k,'final_tail')[0]*s):>9}"
                  f"{f % (agg(recs,k,'final_acc')[0]*s):>9}{flag}")

    # ---------------- 3. add-one-in ----------------
    aoi = [k for k in keys if fams.get(k) == "aoi"]
    if aoi:
        print("\n" + "=" * 78)
        print("3. ADD-ONE-IN LADDER (marginal gain of each component in order)")
        print("=" * 78)
        order = ["frozen"] + aoi
        prev = None
        for k in order:
            if k not in keys:
                continue
            v = agg(recs, k, "final_miou")[0]
            marg = "" if prev is None else f % ((v - prev) * s)
            print(f"  {names[k][:34]:<36}{f % (v*s):>9}   marginal {marg:>9}")
            prev = v
        print("\n  Compare each marginal gain here against its leave-one-out contribution")
        print("  above. Large disagreement = the component is redundant with another one.")

    # ---------------- 4. per-corruption ----------------
    print("\n" + "=" * 78)
    print("4. PER-CORRUPTION: full method vs frozen (final frozen mIoU)")
    print("=" * 78)
    rows = []
    for c in corrs:
        fz = mean([r["final_miou"] for r in recs if r["ablation"] == "frozen" and r["corruption"] == c])
        fl = mean([r["final_miou"] for r in recs if r["ablation"] == "full_method" and r["corruption"] == c])
        if fz == fz and fl == fl:
            rows.append((c, fz, fl, fl - fz))
    rows.sort(key=lambda t: t[3])
    print(f"  {'corruption':<20}{'frozen':>10}{'full':>10}{'delta':>10}")
    for c, fz, fl, d in rows:
        print(f"  {c:<20}{f % (fz*s):>10}{f % (fl*s):>10}{f % (d*s):>10}")

    if len(rows) > 2:
        xs = [r[1] for r in rows]; ys = [r[3] for r in rows]
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        r = num / den if den else 0.0
        print(f"\n  corr(frozen mIoU, adaptation gain) = {r:+.3f}")
        if r < -0.2:
            print("  => adaptation hurts where the frozen model is STRONG.")
            print("     This is the case for domain-gap gain control: run --ablations new")
            print("     with --gap_lo from `--calibrate_gap`.")

    # ---------------- 5. head/tail trade ----------------
    print("\n" + "=" * 78)
    print("5. HEAD/TAIL TRADE CHECK (accuracy up + mIoU down = majority bias)")
    print("=" * 78)
    print(f"  {'ablation':<26}{'d mIoU':>10}{'d acc':>10}{'d head':>10}{'d mid':>10}{'d tail':>10}")
    for k in keys:
        d_miou = agg(recs, k, "final_miou")[0] - agg(recs, k, "init_miou")[0]
        d_acc = agg(recs, k, "final_acc")[0] - agg(recs, k, "init_acc")[0]
        d_h = agg(recs, k, "final_head")[0] - agg(recs, k, "init_head")[0]
        d_m = agg(recs, k, "final_mid")[0] - agg(recs, k, "init_mid")[0]
        d_t = agg(recs, k, "final_tail")[0] - agg(recs, k, "init_tail")[0]
        warn = "   <-- majority bias" if (d_acc > 0 and d_miou < 0) else ""
        print(f"  {names[k][:25]:<26}{f % (d_miou*s):>10}{f % (d_acc*s):>10}"
              f"{f % (d_h*s):>10}{f % (d_m*s):>10}{f % (d_t*s):>10}{warn}")

    # ---------------- 6. sanity ----------------
    print("\n" + "=" * 78)
    print("6. SANITY CHECKS")
    print("=" * 78)
    fr = [r for r in recs if r["ablation"] == "frozen"]
    bad = [r for r in fr if abs(r["final_miou"] - r["init_miou"]) > 1e-9]
    if bad:
        print(f"  !! FROZEN CHANGED on {len(bad)} records -- the protocol is still broken")
        for r in bad[:5]:
            print(f"     {r['corruption']}: {r['init_miou']:.6f} -> {r['final_miou']:.6f}")
    else:
        print("  OK frozen initial == frozen final (3-pass protocol is sound)")
    if len(protocols) > 1:
        print(f"  !! mixed protocols in one file: {protocols} -- do not compare across them")
    else:
        print(f"  OK single protocol: {protocols[0]}")
    print()


if __name__ == "__main__":
    main()