#!/usr/bin/env bash
# Runs the initializer of every task on the frontier model with the LLM API, and for Plan writes the policy prompt
# file of the Init conditions. See README.md.
#   [FAMILY=gpt|claude|act] [TASKS="BinFill PickXtimes"] bash scripts/init.sh
set -uo pipefail
. "$(dirname "$0")/common.sh"
if [ "$FAMILY" = act ]; then
  per_task "$FAMILY.init" $PY -m memi.act.init --runs-dir "$RUNS/opt_astra_act" || exit 1
  echo "[init] done. Texts under $RUNS/opt_astra_act/<Task>/texts.iteration0.json"
  exit 0
fi
per_task "$FAMILY.init" $PY -m memi.plan.init --runs-dir "$RUNS/init" --model "$($PY -m memi.conditions --model "$STRONG")" \
  || exit 1
# shellcheck disable=SC2046
$PY -m memi.plan.export --runs-dir "$RUNS/init" --run 1 --tasks $(ls "$RUNS/init") --out "$PROMPTS/initial.json" || exit 1
echo "[init] done. Policy prompts under $RUNS/init/<Task>/policy_prompt/ and in $PROMPTS/initial.json"
