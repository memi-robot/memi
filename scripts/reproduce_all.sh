#!/usr/bin/env bash
# Runs every rollout of one family except the principle ablations on the simulator with the LLM API, then prints
# their tables. See README.md.
#   [FAMILY=gpt|claude|act] [PROMPTS_FROM=release|outputs] bash scripts/reproduce_all.sh
set -uo pipefail
. "$(dirname "$0")/common.sh"
if [ "$PROMPTS_FROM" = outputs ]; then
  bash scripts/init.sh || exit 1
  for p in $STRONG $WEAK; do
    POLICY=$p bash scripts/optimize.sh || exit 1
  done
fi
rc=0
for p in $STRONG $WEAK; do
  for c in $($PY -m memi.conditions "$p"); do
    case "$c" in */memi_no_*) continue ;; esac
    bash scripts/rollout.sh "$c" || rc=1
  done
done
$PY -m memi.summarize "$OUT/results/$STRONG" ${WEAK:+"$OUT/results/$WEAK"}
exit $rc
