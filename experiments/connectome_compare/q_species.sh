#!/usr/bin/env bash
# q study across species, shuffled text (seed 0), base conditions. Skips existing outputs.
set -u
cd "$(dirname "$0")"
PY="${PY:-/d/flybrain_lm_cuda/.venv/Scripts/python.exe}"
mkdir -p features_q logs_q
run() { # NAME SPEC GAIN DEVICE
  for C in 3 4 5; do
    F="features_q/$1_c$C.npz"
    [ -f "$F" ] || "$PY" simulate.py "$2" --gain "$3" --code "$C" --shuffle 0 --device "$4" \
        --out "$F" > "logs_q/$1_c$C.log" 2>&1
  done
  echo "$1 done"
}
run larvaactive_base specs/larva.json 1.4375 cpu
run fly_base specs/fly.json 4.5 cuda
