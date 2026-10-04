#!/usr/bin/env bash
# Run a list of conditions on shuffled text (seed 0). One job: list -> features_q/<prefix>_<name>_c<code>.npz.
#   grid.sh SPEC PREFIX DEVICE LIST [CODES]     LIST lines: "NAME GAIN [--set PATH=VALUE ...]"
set -u
cd "$(dirname "$0")"
PY="${PY:-/d/flybrain_lm_cuda/.venv/Scripts/python.exe}"
SPEC=$1; PREFIX=$2; DEV=$3; LIST=$4; CODES=${5:-"3 4 5"}
mkdir -p features_q logs_q
while read -r NAME GAIN SETS; do
  case "$NAME" in ""|\#*) continue;; esac
  for C in $CODES; do
    F="features_q/${PREFIX}_${NAME}_c$C.npz"
    [ -f "$F" ] || "$PY" simulate.py "$SPEC" --gain "$GAIN" --code "$C" --shuffle 0 --device "$DEV" $SETS \
        --out "$F" > "logs_q/${PREFIX}_${NAME}_c$C.log" 2>&1
  done
  echo "$NAME done"
done < "$LIST"
