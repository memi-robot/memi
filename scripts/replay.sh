#!/usr/bin/env bash
# Replays every released optimization run offline, with no API key and no simulator. See README.md.
#   [FAMILY=gpt|claude|act] [TASKS=StopCube] bash scripts/replay.sh
set -uo pipefail
ONLY_FAMILY=${FAMILY:-}
unset FAMILY POLICY
. "$(dirname "$0")/common.sh"
# shellcheck disable=SC2086
$PY -m memi.replay ${ONLY_FAMILY:+--family "$ONLY_FAMILY"} --tasks $TASKS
