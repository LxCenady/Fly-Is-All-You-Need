"""Encoding table. One job: analyze.py json (+ the simulation logs) -> x0 / x1 per condition and M.

x = M/(1-M), M = (acc - b)/(1 - b) as in q_fit.py; x0 from k = 0, x1 from k = 1. Also the mean
fraction of readout neurons firing per character (from the simulation log) and the share of
readout neurons that never fire.

    python x0_table.py results_x0_worm.json --prefix x0worm_
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def xval(acc, b):
    m = min(max((acc - b) / (1 - b), 1e-6), 1 - 1e-6)
    return m / (1 - m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    g = defaultdict(list)
    for r in json.load(open(HERE / a.results, encoding="utf-8")):
        cond = re.sub(r"_c\d+\.npz$", "", r["file"]).removeprefix(a.prefix)
        g[(cond, r["M"])].append(r)
    act = defaultdict(list)
    for log in HERE.glob(f"logs_q/{a.prefix}*_c*.log"):
        cond = re.sub(r"_c\d+$", "", log.stem).removeprefix(a.prefix)
        last = log.read_text(encoding="utf-8").strip().splitlines()[-1]
        act[cond].append(json.loads(last)["readout_active_mean"])
    out = {}
    print(f"{'condition':10s} {'M':>4s} {'activity':>8s} {'silent':>6s} {'acc0':>6s} {'acc1':>6s} "
          f"{'x0':>6s} {'x1':>6s} {'x1/x0':>6s}")
    for (cond, m), rows in sorted(g.items()):
        b = np.mean([r["majority"] for r in rows])
        a0 = np.mean([r["span"][0] for r in rows])
        a1 = np.mean([r["span"][1] for r in rows])
        x0, x1 = xval(a0, b), xval(a1, b)
        sil = np.mean([r["silent_share"] for r in rows])
        ac = float(np.mean(act[cond])) if act[cond] else float("nan")
        out[f"{cond}|{m}"] = dict(activity=round(ac, 4), silent=round(float(sil), 3), acc0=round(a0, 4),
                                  acc1=round(a1, 4), x0=round(x0, 3), x1=round(x1, 3))
        print(f"{cond:10s} {m:4d} {ac:8.3f} {sil:6.2f} {a0:6.3f} {a1:6.3f} {x0:6.2f} {x1:6.3f} "
              f"{x1 / x0 if x0 > 1e-3 else float('nan'):6.2f}")
    (HERE / a.results.replace(".json", "_x0.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
