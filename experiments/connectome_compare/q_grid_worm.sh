#!/usr/bin/env bash
# q study, worm, shuffled text (seed 0): one condition per line "NAME GAIN [--set ...]". Skips existing outputs.
set -u
cd "$(dirname "$0")"
PY="${PY:-/d/flybrain_lm_cuda/.venv/Scripts/python.exe}"
mkdir -p features_q logs_q
while read -r NAME GAIN SETS; do
  [ -z "$NAME" ] && continue
  for C in 3 4 5; do
    F="features_q/worm_${NAME}_c$C.npz"
    [ -f "$F" ] || "$PY" simulate.py specs/worm.json --gain "$GAIN" --code "$C" --shuffle 0 --device cpu $SETS \
        --out "$F" > "logs_q/worm_${NAME}_c$C.log" 2>&1
  done
  echo "$NAME done"
done <<'LIST'
base 2.5875
steps3 2.5875 --set input.steps=3
steps12 2.5875 --set input.steps=12
tau05 7.97 --set neuron.params.tau=0.05
tau20 1.1422 --set neuron.params.tau=0.2
g150 1.5
g200 2.0
g230 2.3
sus0 2.5875 --set input.params.sustain=0
sus1 2.5875 --set input.params.sustain=1
LIST
