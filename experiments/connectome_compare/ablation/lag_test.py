"""Readout delay-line test. One job: features/*.npz -> ablation/lag_test.json.

Spike counts alone, counts + traces, and counts + the counts of 1 or 2 earlier characters
(explicit delay lines). If counts + traces behaves like counts + 1 lag, the trace acts as a
readout-side delay line. M = 300, neuron draw 0 (seed 1000, as analyze.py), memory k = 0..8.

    python ablation/lag_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "uctf"))
from uctf.plugins.probes import ridge_span  # noqa: E402
from simulate import text_ids  # noqa: E402

FILES = ["worm_real_c3", "larvaactive_real_c3", "fly_real_c3", "fly_rewire_s0"]
N_TR = 20001


def lag(A, k):
    B = np.zeros_like(A)
    B[k:] = A[:-k]
    return B


def main():
    ids, _ = text_ids()
    out = {}
    for f in FILES:
        z = np.load(ROOT / "features" / f"{f}.npz")
        X, n = z["X"], json.loads(str(z["info"]))["dims"]["spikes"]
        pick = np.sort(np.random.default_rng(1000).choice(n, 300, replace=False))
        S, T = X[:, pick], X[:, n + pick]
        sets = {"counts": S, "counts+trace": np.hstack([S, T]),
                "counts+lag1": np.hstack([S, lag(S, 1)]),
                "counts+lag1+lag2": np.hstack([S, lag(S, 1), lag(S, 2)])}
        out[f] = {}
        for name, F in sets.items():
            span, _ = ridge_span(F.astype(np.float32), ids, N_TR, 8)
            out[f][name] = [round(float(v), 4) for v in span]
            print(f, name, out[f][name][:4], file=sys.stderr, flush=True)
    (HERE / "lag_test.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
