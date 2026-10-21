#!/usr/bin/env bash
# Runs the optimization iterations of every task from its initial prompts with the LLM API, and for Plan writes the
# policy prompt files the rollouts read. See README.md.
#   POLICY=astra_plan|luna|fable|haiku|astra_act [ABLATE=sufficiency|predictability|necessity] [RESUME=1]
#     [ITERATIONS=5] [TASKS=StopCube] bash scripts/optimize.sh
set -uo pipefail
POLICY=${POLICY:?set POLICY=astra_plan, luna, fable, haiku or astra_act}
. "$(dirname "$0")/common.sh"
ABLATE=${ABLATE:-}
case "$ABLATE" in
  ""|sufficiency|predictability|necessity) ;;
  *) echo "ABLATE must be sufficiency, predictability or necessity"; exit 2 ;;
esac
ITERATIONS=${ITERATIONS:-5}
NAME=opt_$POLICY${ABLATE:+_no_$ABLATE}
DEST=$RUNS/$NAME
if [ "$FAMILY" = act ]; then
  [ -z "$ABLATE" ] || { echo "ABLATE is not released for Act"; exit 2; }
  # memi.act.optimize resumes on its own and initializes a task that has no initial prompts.
  per_task "$FAMILY.$NAME" $PY -m memi.act.optimize --runs-dir "$DEST" --iterations "$ITERATIONS" || exit 1
  echo "[optimize] done. Texts under $DEST/<Task>/texts.iteration<k>.json"
  exit 0
fi
INIT=$RUNS/init
ARGS=(--runs-dir "$DEST" --model "$($PY -m memi.conditions --model "$STRONG")" --max-iterations "$ITERATIONS")
# The cost-efficient policy runs only its own forward pass. Every other role stays on the frontier model.
[ "$POLICY" != "$WEAK" ] || ARGS+=(--policy-model "$($PY -m memi.conditions --model "$WEAK")" --policy-effort low)
[ -z "$ABLATE" ] || ARGS+=(--ablate "$ABLATE")
if [ "${RESUME:-0}" = 1 ]; then
  ARGS+=(--resume)
else
  # Every task is checked before any is launched, so a refusal leaves nothing running.
  for t in $TASKS; do
    [ -f "$INIT/$t/policy_prompt/policy_prompt.run1.json" ] || { echo "[optimize] no initial policy prompt for $t under $INIT. Run scripts/init.sh first."; exit 1; }
    if [ -f "$DEST/$t/policy_prompt/policy_prompt.run2.json" ]; then
      echo "[optimize] $DEST/$t already has an optimized policy prompt. Use RESUME=1 to continue a stopped optimization run, or another OUT."
      exit 1
    fi
  done
  for t in $TASKS; do
    mkdir -p "$DEST/$t"
    cp -r "$INIT/$t/policy_prompt" "$INIT/$t/annotator_prompt" "$INIT/$t/initializer" "$DEST/$t/"
  done
fi
per_task "$FAMILY.$NAME" $PY -m memi.plan.optimize "${ARGS[@]}" || exit 1
# shellcheck disable=SC2046
$PY -m memi.plan.export --runs-dir "$DEST" --stage optimized --tasks $(ls "$DEST") \
    --out "$PROMPTS/optimized_${NAME#opt_}.json" || exit 1
echo "[optimize] done. Policy prompts under $DEST/<Task>/policy_prompt/ and in $PROMPTS/optimized_${NAME#opt_}.json"
[ -z "$ABLATE" ] || exit 0
# The policy prompts after iterations 1 to 4, for the iteration conditions of Figures 1 and 5.
ONLY=$($PY -m memi.conditions --tasks "$POLICY/memi_iteration1") || exit 0
ITERATION_TASKS=$(for t in $(ls "$DEST"); do case " ${ONLY:-$t} " in *" $t "*) echo "$t" ;; esac; done)
for k in 1 2 3 4; do
  [ "$k" -le "$ITERATIONS" ] && [ -n "$ITERATION_TASKS" ] || break
  # shellcheck disable=SC2086
  $PY -m memi.plan.export --runs-dir "$DEST" --run 2 --after-iteration "$k" --tasks $ITERATION_TASKS \
      --out "$PROMPTS/${POLICY}_after_iteration$k.json" || exit 1
done
