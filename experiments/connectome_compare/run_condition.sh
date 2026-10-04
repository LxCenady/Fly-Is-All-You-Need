#!/usr/bin/env bash
# One spec variant end to end: gain for ~10 % activity (target_gain.py), then the real wiring
# for input codes 3, 4, 5 (simulate.py). Skips steps whose outputs already exist.
#   run_condition.sh NAME LO HI DEVICE [ITERS]      e.g. run_condition.sh worm_tau05 0.5 6 cpu
set -u
cd "$(dirname "$0")"
PY="${PY:-/d/flybrain_lm_cuda/.venv/Scripts/python.exe}"
NAME=$1; LO=$2; HI=$3; DEV=$4; ITERS=${5:-12}
mkdir -p target features logs
[ -s "target/$NAME.json" ] || "$PY" target_gain.py "specs/$NAME.json" --lo "$LO" --hi "$HI" \
    --iters "$ITERS" --device "$DEV" > "target/$NAME.json" 2> "target/$NAME.log"
G=$("$PY" -c "import json; d=json.load(open('target/$NAME.json')); print(d['chosen_gain'] or d['closest_gain'])")
echo "$NAME: gain $G"
for C in 3 4 5; do
  F="features/${NAME}_real_c$C.npz"
  [ -f "$F" ] || "$PY" simulate.py "specs/$NAME.json" --gain "$G" --code "$C" --device "$DEV" \
      --out "$F" > "logs/${NAME}_real_c$C.log" 2>&1
done
echo "$NAME: done"
