"""Input-side vs readout-side encoding. One job: x0inp feature files -> x0_split.json.

For each condition: x0_in from the INPUT population's own spikes (all input neurons), x0_out
from M readout neurons (3 draws), both decoding the current character (shuffled text), and
their ratio (transmission). x = M/(1-M), M = (acc - b)/(1 - b).

    python x0_split.py [--prefix x0inp_] [--m 300] [--m-in N] [--out x0_split.json]

--m-in samples N input neurons per draw (default: all recorded input neurons, one fit).
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "uctf"))
from uctf.plugins.probes import ridge_span  # noqa: E402
from simulate import text_ids  # noqa: E402


def xval(acc, b):
    m = min(max((acc - b) / (1 - b), 1e-6), 1 - 1e-6)
    return m / (1 - m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="x0inp_")
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--m-in", type=int)
    ap.add_argument("--out", default="x0_split.json")
    a = ap.parse_args()
    r = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(HERE / "features_q" / f"{a.prefix}*_c*.npz"))):
        z = np.load(f)
        X, info = z["X"], json.loads(str(z["info"]))
        d = info["dims"]
        ids, _ = text_ids(shuffle=info.get("shuffle"))
        o = d["spikes"] + d["trace"]
        cond = re.sub(r"_c\d+$", "", Path(f).stem).removeprefix(a.prefix)
        I = X[:, o:o + d["input_spk"]]
        for dr in range(3 if a.m_in else 1):
            cols = (np.sort(np.random.default_rng(3000 + dr).choice(I.shape[1], a.m_in, replace=False))
                    if a.m_in else slice(None))
            span, maj = ridge_span(I[:, cols].astype(np.float32), ids, 20001, 0)
            r[cond]["x0_in"].append(xval(span[0], maj[0]))
        for dr in range(3):
            pick = np.sort(np.random.default_rng(1000 + dr).choice(d["spikes"], a.m, replace=False))
            span, maj = ridge_span(X[:, pick].astype(np.float32), ids, 20001, 0)
            r[cond]["x0_out"].append(xval(span[0], maj[0]))
        r[cond]["gain"].append(info["gain"])
        r[cond]["set"] = info["set"]
        print(f, file=sys.stderr, flush=True)
    out = {}
    for c, v in r.items():
        xi, xo = float(np.mean(v["x0_in"])), float(np.mean(v["x0_out"]))
        out[c] = {"x0_in": round(xi, 4), "x0_out": round(xo, 4),
                  "transmission": round(xo / xi, 4) if xi > 1e-3 else None,
                  "gain": float(np.mean(v["gain"])), "set": v["set"]}
        print(f"{c:16s} x0_in {xi:6.2f}  x0_out {xo:6.2f}")
    (HERE / a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
