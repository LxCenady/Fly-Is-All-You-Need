#!/usr/bin/env bash
# x0 part 2: input-spike grid, then activity-matched gains for three high-dose settings, then their simulations.
cd "$(dirname "$0")"
PY="${PY:-/d/flybrain_lm_cuda/.venv/Scripts/python.exe}"
bash grid.sh specs/worm_inp.json x0inp cpu x0_worm2.list
for v in act18 sus075 drv3; do
  [ -s target/worm_inp_$v.json ] || "$PY" target_gain.py specs/worm_inp_$v.json --lo 0.3 --hi 2.6 --iters 12 --device cpu \
      > target/worm_inp_$v.json 2> target/worm_inp_$v.log
  G=$("$PY" -c "import json; d=json.load(open('target/worm_inp_$v.json')); print(d['chosen_gain'] or d['closest_gain'])")
  echo "${v}_matched $G" > x0_matched_$v.list
  bash grid.sh specs/worm_inp_$v.json x0inp cpu x0_matched_$v.list
done
echo ALL DONE
