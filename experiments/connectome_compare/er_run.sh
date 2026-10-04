#!/usr/bin/env bash
# Random-graph reservoirs: 10 %-activity gain, condition list, shuffled-text grid. Skips finished steps.
#   er_run.sh NAME DEVICE LO HI
cd "$(dirname "$0")"
PY="${PY:-/d/flybrain_lm_cuda/.venv/Scripts/python.exe}"
NAME=$1; DEV=$2; LO=$3; HI=$4
mkdir -p target
[ -s "target/$NAME.json" ] || "$PY" target_gain.py "specs/$NAME.json" --lo "$LO" --hi "$HI" --iters 12 \
    --device "$DEV" > "target/$NAME.json" 2> "target/$NAME.log"
"$PY" er_lists.py "$NAME" > /dev/null
bash grid.sh "specs/$NAME.json" "$NAME" "$DEV" "lists/$NAME.list"
echo "$NAME ALL DONE"
