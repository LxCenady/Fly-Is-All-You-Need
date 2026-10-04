"""Readout-size-controlled metrics from cached protocol-B features.

One job: features/*.npz -> results.json. For every file and readout size M (neurons; spike
counts + traces), --draws random neuron samples are refitted (no simulation):
  memory span  ridge decoding of the character k back, k = 0..8 (uctf memory_span probe)
  val_bpc      softmax readout + hashed context table, the bench's fixed settings, clip 3
Rows: features after character i; train = first 20,000 rows, validation = the rest.

    python analyze.py features/*.npz --sizes 100,300,1000,3000 --draws 3 > results.json

--features spikes|trace keeps one feature kind (same neuron draws as "both", so paired).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "uctf"))
from uctf.plugins.probes import ridge_span  # noqa: E402
from uctf.plugins.readouts import ReadoutConfig, context_ids, fit_calibrated  # noqa: E402
from simulate import text_ids  # noqa: E402

N_TR = 20001                      # training characters; rows 0..19999 are training rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sizes", default="100,300,1000,3000")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--no-bpc", action="store_true")
    ap.add_argument("--features", choices=("both", "spikes", "trace"), default="both")
    a = ap.parse_args()
    ids, V = text_ids()
    y, ntr = ids[1:], N_TR - 1
    cfg = ReadoutConfig()
    cx = context_ids(ids, cfg.context_order, V, cfg.buckets)
    out, shuffled = [], None
    for f in a.files:
        z = np.load(f)
        X, info = z["X"], json.loads(str(z["info"]))
        if info.get("shuffle") != shuffled:               # labels follow the simulated sequence
            shuffled = info.get("shuffle")
            ids, V = text_ids(shuffle=shuffled)
            y = ids[1:]
            cx = context_ids(ids, cfg.context_order, V, cfg.buckets)
        n = info["dims"]["spikes"]
        for m in [int(s) for s in a.sizes.split(",")]:
            if m > n:
                continue
            for d in range(a.draws if m < n else 1):
                pick = np.sort(np.random.default_rng(1000 + d).choice(n, m, replace=False))
                cols = {"both": np.concatenate([pick, n + pick]), "spikes": pick,
                        "trace": n + pick}[a.features]
                Xm = X[:, cols]
                span, majority = ridge_span(Xm, ids, N_TR, 8)
                row = {"file": Path(f).name, **{k: info[k] for k in ("gain", "code", "rewire_seed")},
                       "M": m, "draw": d, "features": a.features, "span": [round(v, 4) for v in span],
                       "majority": round(majority[2], 4),
                       "readout_active": round(float((X[:, pick] > 0).mean()), 4),
                       "silent_share": round(float((X[:ntr, pick].max(0) == 0).mean()), 4)}
                if not a.no_bpc:
                    row["val_bpc"] = round(float(fit_calibrated(Xm, cx, y, ntr, V, cfg, 3.0)[3]), 4)
                out.append(row)
                print(json.dumps(row), file=sys.stderr, flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
