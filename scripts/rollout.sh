#!/usr/bin/env bash
# Rolls out one condition of configs/conditions.json on the simulator with the LLM API and prints its success table.
# See README.md.
#   [PROMPTS_FROM=release|outputs] [WORKERS=1] [TASKS=PickXtimes] [EPISODES="0 1 2"]
#     bash scripts/rollout.sh <policy>/<method>
set -uo pipefail
. "$(dirname "$0")/common.sh"
# shellcheck disable=SC2086
$PY -m memi.rollout "${1:?usage: scripts/rollout.sh POLICY/METHOD, e.g. luna/memi}" --tasks $TASKS \
    ${EPISODES:+--episodes $EPISODES} --workers "${WORKERS:-1}" --prompts-from "$PROMPTS_FROM" --out "$OUT"
