#!/usr/bin/env bash
# cleanup_diag_outputs.sh: delete OLD diagnostic JSONs and console logs on the
# server, KEEPING weights and checkpoints.
#
# Deletes ONLY top-level diagnostic files (never anything inside checkpoint
# directories):
#   robust_diagnostic/logs/al_*.json        AL diagnostic result JSONs
#   robust_diagnostic/logs/*_results.json   other diagnostic result JSONs
#   robust_diagnostic/logs/*.log            any diagnostic logs there
#   logs/*.log                              runner console logs (al_*, geoid_*,
#                                           noiseinv_*, overnight_*)
#
# Keeps intact (untouched):
#   robust_diagnostic/logs/<ckpt-dir>/**    checkpoint/weight dirs, e.g.
#                                           supcon_vib_dglsspp/SENet* (the
#                                           model weights + any config/logs
#                                           inside those dirs)
#   logs/kitti_c_test/**                    any non-*.log content under logs/
#   every other directory
#
# Safety:
#   default is a DRY RUN (prints what would be deleted); pass --apply to
#   actually delete. Prints per-dir file counts and a grand total.
#
# Usage:
#   bash cleanup_diag_outputs.sh               # dry run
#   bash cleanup_diag_outputs.sh --apply       # delete
#   bash cleanup_diag_outputs.sh --apply --verbose

set -u
set -o pipefail

APPLY=0
VERBOSE=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --verbose) VERBOSE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# A checkpoint dir is any subdirectory of robust_diagnostic/logs (e.g.
# supcon_vib_dglsspp). Verify at least one exists so we never run in a wrong CWD.
if [ ! -d "robust_diagnostic/logs" ]; then
  echo "ERROR: robust_diagnostic/logs not found (run from the EvidentialHDC repo root)" >&2
  exit 1
fi
CKPT_DIRS=$(find robust_diagnostic/logs -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "checkpoint/weight dirs present under robust_diagnostic/logs: $CKPT_DIRS"
if [ "$CKPT_DIRS" -eq 0 ]; then
  echo "WARNING: no subdirectories under robust_diagnostic/logs -- double-check" \
       "the ckpts are elsewhere before --apply" >&2
fi

TOTAL=0
SEEN=$(mktemp)
cleanup_seen() { rm -f "$SEEN"; }
trap cleanup_seen EXIT

delete() {  # delete <glob> <description>  (dedupes across overlapping globs)
  local n=0
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    local key="$PWD/$f"
    if ! grep -qxF "$key" "$SEEN" 2>/dev/null; then
      n=$((n + 1))
      echo "$key" >> "$SEEN"
      if [ "$VERBOSE" = "1" ]; then
        echo "      ${f#./}"
      fi
      if [ "$APPLY" = "1" ]; then
        rm -f -- "$f"
      fi
    fi
  done < <(find . -maxdepth 1 -name "$1" -type f 2>/dev/null)
  if [ "$n" -gt 0 ]; then
    echo "  [$n files] $2"
    TOTAL=$((TOTAL + n))
  fi
}

echo ""
echo "== robust_diagnostic/logs (top level only; ckpt dirs untouched) =="
pushd robust_diagnostic/logs >/dev/null
delete 'al_*.json'       'AL diagnostic JSONs (al_*.json)'
delete '*_results.json'  'other diagnostic JSONs (*_results.json)'
delete '*.log'           'diagnostic logs (*.log)'
popd >/dev/null

echo ""
echo "== logs (top level only; non-*.log content like kitti_c_test kept) =="
pushd logs >/dev/null
delete '*.log'           'runner console logs (*.log)'
popd >/dev/null

echo ""
if [ "$APPLY" = "1" ]; then
  echo "DONE: deleted $TOTAL diagnostic output files. Weights/checkpoints untouched."
else
  echo "DRY RUN: would delete $TOTAL files. Re-run with --apply to actually delete."
fi
