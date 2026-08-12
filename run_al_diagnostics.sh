#!/usr/bin/env bash
# Diagnostic batch for the Iteration-13 AL-readiness findings + the "is the combined
# objective too muddled?" check. Eval-only (fast).
#
#   [1] Medium extractors:  plain DGLSS++ / robust 21ep / blend05 / supcon_vib med
#       -> point-level AL purity (nn1/nnk), ceilings, per-class entanglement, label budget.
#   [2] Micro piecewise variants: full corsupcon vs nocons / supcon-only / cor-only
#       -> does removing a piece raise the point-level purity (i.e. is the FULL
#          combination what entangles the neighborhoods, even though each piece is
#          individually useful)?
#   [3] Python analysis: entanglement map + label-budget multipliers + muddle summary.
#
# Usage:
#   bash run_al_diagnostics.sh            # GPU 3

set -u
GPU="${1:-3}"
echo "Using GPU $GPU"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

echo "=== [1/3] AL readiness: medium extractors ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_readiness_diag.py \
  --checkpoints \
"dglsspp_med:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp,\
robust_21ep:supcon_vib_dglsspp_corsupcon:robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon,\
blend05_med:supcon_vib_dglsspp_corsupcon_blend05:robust_diagnostic/logs/med_blend05/supcon_vib_dglsspp_corsupcon_blend05,\
supcon_vib_med:supcon_vib:logs/med_pretrain_supcon_vib" \
  --out "robust_diagnostic/logs/al_readiness_med.json" \
  2>&1 | tee "logs/al_readiness_med.log" || fail "med set"

echo "=== [2/3] AL readiness: micro piecewise variants (muddle check) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_readiness_diag.py \
  --checkpoints \
"corsupcon_full:supcon_vib_dglsspp_corsupcon:robust_diagnostic/logs/micro_corsupcon/supcon_vib_dglsspp_corsupcon,\
nocons:supcon_vib_dglsspp_corsupcon_nocons:robust_diagnostic/logs/micro_abl_nocons/supcon_vib_dglsspp_corsupcon_nocons,\
supcon_only:supcon_vib_dglsspp_supcon:robust_diagnostic/logs/micro_supcon/supcon_vib_dglsspp_supcon,\
cor_only:supcon_vib_dglsspp_cor:robust_diagnostic/logs/micro_cor/supcon_vib_dglsspp_cor" \
  --out "robust_diagnostic/logs/al_readiness_micro.json" \
  2>&1 | tee "logs/al_readiness_micro.log" || fail "micro set"

echo "=== [3/3] analysis ==="
uv run python - <<'PY'
import json

def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        print(f"  (missing {p}: {e})"); return {}

med = load('robust_diagnostic/logs/al_readiness_med.json')
mic = load('robust_diagnostic/logs/al_readiness_micro.json')

print("\n== Medium extractors: nn1 purity, ceiling, label-budget multiplier ==")
print(f"{'extractor':<14} {'cond':<10} {'nn1':>5} {'nnk':>5} {'oracle':>6} {'1/nn1':>6}")
for lab in ['dglsspp_med', 'robust_21ep', 'blend05_med', 'supcon_vib_med']:
    if lab not in med:
        continue
    for cond in ['fog', 'crosstalk']:
        d = med[lab].get(cond, {})
        nn1 = d.get('nn1_mean', float('nan'))
        print(f"{lab:<14} {cond:<10} {nn1:>5.3f} {d.get('nnk_mean', float('nan')):>5.3f} "
              f"{d.get('oracle_mean', float('nan')):>6.3f} {(1/nn1 if nn1==nn1 and nn1>0 else float('nan')):>6.2f}")

print("\n== Entanglement map (per-class nn1 on fog) ==")
for lab in ['dglsspp_med', 'robust_21ep', 'blend05_med']:
    if lab not in med:
        continue
    pc = med[lab].get('fog', {}).get('per_class', {})
    row = {c: pc[c]['nn1_purity'] for c in pc if pc[c].get('nn1_purity') == pc[c].get('nn1_purity')}
    ent = sorted(row.items(), key=lambda kv: kv[1])[:5]
    print(f"  {lab:<14} entangled(5 lowest nn1): " + ", ".join(f"{c}={v:.2f}" for c, v in ent))

print("\n== Muddle check (micro): does removing a piece raise nn1? ==")
print(f"{'variant':<14} {'cond':<10} {'nn1':>5} {'nnk':>5}")
for lab in ['corsupcon_full', 'nocons', 'supcon_only', 'cor_only']:
    if lab not in mic:
        continue
    for cond in ['fog', 'crosstalk']:
        d = mic[lab].get(cond, {})
        print(f"{lab:<14} {cond:<10} {d.get('nn1_mean', float('nan')):>5.3f} "
              f"{d.get('nnk_mean', float('nan')):>5.3f}")

print("\nReadout:")
print("  - If supcon_vib_med / the untrained baseline also has nn1 ~0.4-0.5, the low")
print("    purity is intrinsic to corruption for ALL trained extractors (item 1).")
print("  - If nocons / supcon_only / cor_only have HIGHER nn1 than corsupcon_full, the")
print("    FULL combination is what entangles the neighborhoods -> the muddle hypothesis.")
PY
