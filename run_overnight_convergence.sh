#!/usr/bin/env bash
# run_overnight_convergence.sh: does the AL-relevant ceiling keep rising with more
# training? Trains the two extractors that matter for the AL/ceiling story past
# their current checkpoints (DGLSS++ at 24ep, cov-shift at 21ep) to 40ep/100%,
# then evaluates the frozen zero-shot + labeled ceiling in the R4 setup.
#
# Rationale:
#   - DGLSS++ is the AL PRIMARY target (big closeable gap: fog +12.7, crosstalk
#     +17.5 on KITTI-C 3-sev) and the "ceiling extractor", but its checkpoint is
#     a 24ep run with NO convergence evidence. If its ceiling keeps rising with
#     training, the AL story gets bigger; if it plateaus, we've confirmed 24ep is
#     converged (a useful negative).
#   - cov-shift (inputin_in_chan) is the current method at ~21ep; a longer run
#     tests whether its (already-small) gaps close further or it is at its ceiling.
#
# Usage:
#   DRY_RUN=1 bash run_overnight_convergence.sh 2
#   SMOKE=1   bash run_overnight_convergence.sh 2
#   bash run_overnight_convergence.sh 2
#   EPOCHS=40 CUTOFF=1.0 bash run_overnight_convergence.sh 2   # full (default)
#   ONLY=dglsspp  bash run_overnight_convergence.sh 2           # dglsspp only
#   ONLY=covshift bash run_overnight_convergence.sh 2           # covshift only
#
# Output:
#   robust_diagnostic/logs/conv40_supcon_vib_dglsspp/supcon_vib_dglsspp/   (ckpt)
#   robust_diagnostic/logs/conv40_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan/
#   robust_diagnostic/logs/probe_conv40_{dglsspp,covshift}.json           (R4 eval)
#   logs/overnight_convergence_{p1_train_dglsspp,...}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-40}"
CUTOFF="${CUTOFF:-1.0}"
ONLY="${ONLY:-both}"
MAX_FRAMES="${MAX_FRAMES:-200}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Overnight convergence | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (${EPOCHS}ep/${CUTOFF})"
echo "  ONLY=$ONLY"

rc=0
run_phase() {
  local name="$1"; shift
  local env_pre="$1"; shift
  echo ""
  echo "==================== $name ===================="
  echo "  CMD: $env_pre $*"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [DRY] not executed"
    return 0
  fi
  local logf="logs/overnight_convergence_${name}.log"
  if [ "$SMOKE" = "1" ]; then
    echo "  [SMOKE] executing (streaming to terminal + $logf)..."
    if eval "$env_pre $*" 2>&1 | tee "$logf"; then
      echo "  [SMOKE] $name OK"
    else
      echo "  [SMOKE] $name FAILED -- tail of $logf:"
      tail -25 "$logf"
      return 1
    fi
  else
    echo "  [RUN] full phase, streaming to terminal + $logf"
    eval "unset MAX_FRAMES CONDS; $env_pre $*" 2>&1 | tee "$logf"
    local c=${PIPESTATUS[0]}
    echo "  [$name] exit=$c"
    if [ $c -ne 0 ]; then
      echo "  WARNING: $name failed, tail of $logf:"
      tail -25 "$logf"
    fi
    return $c
  fi
}

train_and_eval() {
  local method="$1"; local logbase="$2"; local label="$3"
  local ep="$EPOCHS"; local cut="$CUTOFF"; local log="robust_diagnostic/logs/${logbase}"
  if [ "$SMOKE" = "1" ]; then ep="1"; cut="0.01"; log="${log}_smoke"; fi
  local env="CUDA_VISIBLE_DEVICES=$GPU"
  local train_cmd="uv run python robust_diagnostic/isotropy_diag.py --methods $method --epochs $ep --cutoff $cut --log_dir $log"
  run_phase "p1_train_${logbase}" "$env" "$train_cmd" || rc=1
  local ckpt="$log/$method"
  if [ "$SMOKE" = "1" ]; then ckpt="${log}/$method"; fi
  local eval_cmd="uv run python robust_diagnostic/probe_linear_prop_diag.py --max_frames $MAX_FRAMES --conds fog,crosstalk --extractors \"$label:$method:$ckpt\" --out robust_diagnostic/logs/probe_conv40_${label}.json"
  run_phase "p2_eval_${logbase}" "$env" "$eval_cmd" || rc=1
}

if [ "$ONLY" = "both" ] || [ "$ONLY" = "dglsspp" ]; then
  train_and_eval "supcon_vib_dglsspp" "conv40_supcon_vib_dglsspp" "dglsspp"
fi
if [ "$ONLY" = "both" ] || [ "$ONLY" = "covshift" ]; then
  train_and_eval "supcon_vib_dglsspp_inputin_in_chan" "conv40_supcon_vib_dglsspp_inputin_in_chan" "covshift"
fi

echo ""
if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN: commands printed, nothing executed ==="
  exit 0
fi
if [ "$SMOKE" = "1" ] && [ $rc -ne 0 ]; then
  echo "=== SMOKE RESULT: FAILURES DETECTED ==="
  exit 1
fi
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE RESULT: ALL PHASES OK ==="
  echo "Launch the full run with: bash run_overnight_convergence.sh $GPU"
  exit 0
fi
echo "=== OVERNIGHT CONVERGENCE DONE ==="
echo "Checkpoints: robust_diagnostic/logs/conv40_*/"
echo "R4 evals:    robust_diagnostic/logs/probe_conv40_{dglsspp,covshift}.json"
echo "Compare frozen zs + labeled ceiling against the current checkpoints:"
echo "  dglsspp  24ep  (fog zs 22.5 / ceil 35.2, crosstalk 11.9 / 29.4 on KITTI-C 3-sev)"
echo "  covshift 21ep  (fog zs 40.7 / ceil 43.6, crosstalk 48.3 / 49.4)"
echo "If the 40ep ceiling is higher, the AL-relevant gap grew -> bigger AL story."
echo "If it plateaus, the current checkpoints are converged (useful negative)."
exit 0
