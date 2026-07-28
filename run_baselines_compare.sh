#!/bin/bash
# ==============================================================================
# Section 7.2 Method Comparison -- corrected
# ==============================================================================
# Fixes vs the previous version:
#
#  1. BASELINES ACTUALLY RUN. The old dispatch was `elif` on `fired_mask.any()`,
#     which is always True for baseline methods, so D3CTTA / ConformalHDC /
#     HyperDUM were never constructed. All three executed the generic HDC path,
#     producing bit-identical numbers. Apply unsup_kitti-c_baseline_patch.md
#     before running this.
#
#  2. BASELINES RUN ONE PER PROCESS. `--method frozen,d3ctta,conformalhdc,hyperdum`
#     shared one model object; the adapter cache and BN/domain state leaked
#     across methods. Each baseline now gets its own invocation and log_dir.
#
#  3. CONTINUAL PROTOCOL FOR CONTINUAL METHODS. D3CTTA's contribution IS its
#     cross-domain memory (G_d, C_d, domains_bn_stats). --reset_per_corruption
#     wipes it every corruption, which is the setting its paper does not use.
#     Both protocols are now run and reported separately.
#
#  4. THREAD CAPS. Same oversubscription fix as the ablation suite; D3CTTA's
#     prior_filter also calls cKDTree(..., workers=-1) per frame, which grabs
#     every core on its own.
#
#  5. Detached-safe logging (per-stage `tee -a`, launch under setsid).
#
# LAUNCH:
#   mkdir -p logs/baseline_comparisons
#   setsid nohup bash run_baselines_compare.sh \
#       > logs/baseline_comparisons/console.log 2>&1 < /dev/null &
#   disown
# ==============================================================================

set -uo pipefail

export ABLATION_THREADS="${ABLATION_THREADS:-2}"
export OMP_NUM_THREADS="$ABLATION_THREADS"
export MKL_NUM_THREADS="$ABLATION_THREADS"
export OPENBLAS_NUM_THREADS="$ABLATION_THREADS"
export NUMEXPR_NUM_THREADS="$ABLATION_THREADS"
export VECLIB_MAXIMUM_THREADS="$ABLATION_THREADS"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

PRETRAINED="logs/kitti_pretrain/hdc_sub.pth"
ROOT="logs/baseline_comparisons"
PANEL="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"
SEV=3
TAU="-1.0"
IC="ic4"
METHOD="evidential_hdc_tta"

mkdir -p "$ROOT"
MAIN_LOG="$ROOT/run_baselines_compare.log"
say () { echo "$@" | tee -a "$MAIN_LOG"; }

say "=========================================================="
say "Section 7.2 Method Comparison (corrected)"
say "started      : $(date)"
say "host / cores : $(hostname) / $(nproc 2>/dev/null || echo '?')"
say "threads/proc : $ABLATION_THREADS"
say "=========================================================="

# ---------- Stage 0: verify the baselines are actually constructed ----------
say ""
say "########## STAGE 0: baseline wiring check (dry run) ##########"
for m in d3ctta conformalhdc hyperdum; do
  say ""
  say "--- dry run: $m ---"
  uv run unsup_kitti-c.py \
    --pretrained_path "$PRETRAINED" \
    --method $m --corruptions snow --severity $SEV \
    --chunked --reset_per_corruption --dry_run \
    --log_dir "$ROOT/dry_$m" 2>&1 | tee -a "$MAIN_LOG"
  if grep -q "\[baselines\] built $m" "$ROOT/dry_$m/kitti_c.log" 2>/dev/null; then
    say ">>> OK: $m adapter constructed."
  else
    say ">>> !!! $m adapter was NOT constructed. The patch is not applied."
  fi
  grep -n "FATAL" "$ROOT/dry_$m/kitti_c.log" 2>/dev/null | head -3 | tee -a "$MAIN_LOG"
done
say ""
say ">>> Do not continue past here unless all three report 'adapter constructed'"
say ">>> and no FATAL lines. Identical numbers downstream mean the patch failed."

# ---------- Stage 1: frozen reference ----------
say ""
say "########## STAGE 1: frozen reference ##########"
uv run unsup_kitti-c.py \
  --pretrained_path "$PRETRAINED" --log_dir "$ROOT/frozen" \
  --corruptions "$PANEL" --severity $SEV --chunked --reset_per_corruption \
  --method frozen 2>&1 | tee -a "$MAIN_LOG"

# ---------- Stage 2: baselines, per-corruption reset (matches your protocol) ----------
say ""
say "########## STAGE 2: baselines, reset-per-corruption ##########"
for m in d3ctta conformalhdc hyperdum; do
  say ""
  say "--- $m (reset) ---"
  uv run unsup_kitti-c.py \
    --pretrained_path "$PRETRAINED" --log_dir "$ROOT/${m}_reset" \
    --corruptions "$PANEL" --severity $SEV --chunked --reset_per_corruption \
    --method $m 2>&1 | tee -a "$MAIN_LOG"
done

# ---------- Stage 3: baselines, CONTINUAL (the setting D3CTTA is designed for) ----------
say ""
say "########## STAGE 3: baselines, continual (no reset) ##########"
say "D3CTTA's domain memory (G_d, C_d, BN stats) is its core contribution."
say "Reporting it only under reset-per-corruption understates it by construction."
for m in d3ctta conformalhdc hyperdum; do
  say ""
  say "--- $m (continual) ---"
  uv run unsup_kitti-c.py \
    --pretrained_path "$PRETRAINED" --log_dir "$ROOT/${m}_continual" \
    --corruptions "$PANEL" --severity $SEV --chunked --continual \
    --method $m 2>&1 | tee -a "$MAIN_LOG"
done

# ---------- Stage 4: our method ----------
say ""
say "########## STAGE 4: this method ##########"
uv run unsup_kitti-c.py \
  --pretrained_path "$PRETRAINED" --log_dir "$ROOT/ours_nomv" \
  --corruptions "$PANEL" --severity $SEV --chunked --reset_per_corruption \
  --method $METHOD --gate_mode soft_dual_weight --ic_method $IC --tau $TAU \
  --mv_tta none --dynamic_geom 2>&1 | tee -a "$MAIN_LOG"

uv run unsup_kitti-c.py \
  --pretrained_path "$PRETRAINED" --log_dir "$ROOT/ours_mv" \
  --corruptions "$PANEL" --severity $SEV --chunked --reset_per_corruption \
  --method $METHOD --gate_mode soft_dual_weight --ic_method $IC --tau $TAU \
  --mv_tta veto_disagree --dynamic_geom 2>&1 | tee -a "$MAIN_LOG"

# ---------- Stage 5: identical-row guard ----------
say ""
say "########## STAGE 5: identical-row guard ##########"
uv run python - << 'PYEOF' 2>&1 | tee -a "$MAIN_LOG"
import json, glob, os, itertools
rows = {}
for f in glob.glob("logs/baseline_comparisons/*/global_results.json"):
    tag = os.path.basename(os.path.dirname(f))
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for meth, per_c in d.get("mIoU", {}).items():
        vals = []
        for c in sorted(per_c):
            for s in sorted(per_c[c]):
                vals.append(round(float(per_c[c][s][1]), 6))
        if vals:
            rows[f"{tag}:{meth}"] = tuple(vals)

print(f"\ncollected {len(rows)} result vectors")
dupes = [(a, b) for a, b in itertools.combinations(sorted(rows), 2) if rows[a] == rows[b]]
if dupes:
    print("\n!!! IDENTICAL RESULT VECTORS -- these did not run as distinct methods:")
    for a, b in dupes:
        print(f"    {a}\n    {b}\n")
else:
    print("OK: every method produced a distinct result vector.")
PYEOF

say ""
say "=========================================================="
say "completed $(date)"
say "results: $ROOT/*/global_results.json"
say "=========================================================="
