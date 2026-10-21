# Sourced by every script. It moves to the repository root and sets the variables the scripts share.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=${PYTHON:-.venv/bin/python}
TASKS=${TASKS:-$($PY -c "from memi.tasks import TASKS; print(' '.join(TASKS))")}
# Everything a script writes goes under OUT, in the layout of the release.
OUT=${OUT:-outputs}
# FAMILY names the folder of artifacts/ a run writes in. POLICY sets it when given.
case "${POLICY:-}" in
  astra_plan|luna) POLICY_FAMILY=gpt ;;
  fable|haiku) POLICY_FAMILY=claude ;;
  astra_act) POLICY_FAMILY=act ;;
  "") POLICY_FAMILY= ;;
  *) echo "POLICY must be astra_plan, luna, fable, haiku or astra_act"; exit 2 ;;
esac
FAMILY=${FAMILY:-${POLICY_FAMILY:-gpt}}
if [ -n "$POLICY_FAMILY" ] && [ "$POLICY_FAMILY" != "$FAMILY" ]; then
  echo "POLICY=$POLICY belongs to FAMILY=$POLICY_FAMILY, not FAMILY=$FAMILY"
  exit 2
fi
# The frontier policy and the cost-efficient one.
case "$FAMILY" in
  gpt) STRONG=astra_plan; WEAK=luna ;;
  claude) STRONG=fable; WEAK=haiku ;;
  act) STRONG=astra_act; WEAK= ;;
  *) echo "FAMILY must be gpt, claude or act"; exit 2 ;;
esac
PROMPTS_FROM=${PROMPTS_FROM:-release}
case "$PROMPTS_FROM" in
  release|outputs) ;;
  *) echo "PROMPTS_FROM must be release or outputs"; exit 2 ;;
esac
RUNS=$OUT/artifacts/$FAMILY/runs
PROMPTS=$OUT/artifacts/$FAMILY/policy_prompts

# Runs `"$@" --task <Task>` for every task at once, each logged to $OUT/logs/<name>.<Task>.log. It fails when any
# task failed, which a bare `wait` never reports.
per_task() {
  local name=$1 rc=0 t p pids=()
  shift
  mkdir -p "$OUT/logs"
  for t in $TASKS; do
    "$@" --task "$t" >> "$OUT/logs/$name.$t.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc = 0 ] || echo "a task failed. See $OUT/logs/$name.<Task>.log"
  return $rc
}
