"""Boundary-leak test. One job: window-feature npz files -> window_test.json (+ table on stdout).

Hypothesis: memory of the previous character lives in spikes that arrive late, in the first
steps of the next character's window. For each feature block (full counts, step 0, step 1,
step 2, steps 3-5) decode the character k = 0..3 back from the same M neurons (draw seeds as
analyze.py). Shuffled text: the majority class is the baseline.

    python window_test.py features_q/worm_win_c*.npz [--m 300] [--draws 3]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "uctf"))
from uctf.plugins.probes import ridge_span  # noqa: E402
from simulate import text_ids  # noqa: E402

BLOCKS = ["spikes", "w0", "w1", "w2", "w35"]
N_TR = 20001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--draws", type=int, default=3)
    a = ap.parse_args()
    res = defaultdict(lambda: defaultdict(list))
    for f in a.files:
        z = np.load(f)
        X, info = z["X"], json.loads(str(z["info"]))
        ids, _ = text_ids(shuffle=info.get("shuffle"))
        dims = info["dims"]
        start, off = {}, 0
        for k, n in dims.items():
            start[k] = off
            off += n
        n = dims["spikes"]
        cond = re.sub(r"_c\d+$", "", Path(f).stem)
        for d in range(a.draws):
            pick = np.sort(np.random.default_rng(1000 + d).choice(n, min(a.m, n), replace=False))
            for b in BLOCKS:
                F = X[:, start[b] + pick].astype(np.float32)
                span, maj = ridge_span(F, ids, N_TR, 3)
                res[cond][b].append(span)
                res[cond]["majority"].append(maj[0])
            print(f, d, file=sys.stderr, flush=True)
    out = {}
    for cond, r in res.items():
        print(f"\n{cond} (M = {a.m}, majority {np.mean(r['majority']):.3f})")
        print(f"{'block':8s} " + " ".join(f"k={k}".rjust(6) for k in range(4)))
        out[cond] = {}
        for b in BLOCKS:
            m = np.mean(r[b], 0)
            out[cond][b] = [round(float(v), 4) for v in m]
            print(f"{b:8s} " + " ".join(f"{v:6.3f}" for v in m))
    Path(Path(__file__).parent / f"window_test_m{a.m}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
