"""Encoding x0 and per-character decay q from memory curves. One job: analyze.py json -> table (+ q_fit.json).

Following Takasu & Aoyagi (arXiv 2504.19657), memory at delay k is M_k = x_k / (1 + x_k) with
x_k = x0 q^k. Here M_k is the decoding accuracy rescaled to the majority baseline b,
M = (acc - b) / (1 - b), so x = M / (1 - M). Per condition (mean over files and draws):
  x0      encoding term (k = 0)
  q1..q3  successive ratios x_k / x_{k-1}
  q_fit   exp of the slope of log x_k over k = 1..K where M_k > --floor (least squares)
Classification accuracy is not Takasu's R^2, so this is an approximation.

    python q_fit.py results.json [more.json] [--group REGEX] [--floor 0.02]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def curves(files, group):
    g = defaultdict(list)
    for f in files:
        for r in json.load(open(f, encoding="utf-8")):
            m = re.match(group, Path(r["file"]).stem)
            key = (m.group(1) if m else r["file"], r["M"], r.get("features", "both"))
            g[key].append((np.array(r["span"]), r["majority"]))
    return g


def fit(span, b, floor):
    M = np.clip((span - b) / (1 - b), 1e-6, 1 - 1e-6)
    x = M / (1 - M)
    ks = [k for k in range(1, len(x)) if M[k] > floor]
    qf = float(np.exp(np.polyfit(ks, np.log(x[ks]), 1)[0])) if len(ks) >= 2 else None
    return x, qf, len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--group", default=r"(.+)_c\d+$")
    ap.add_argument("--floor", type=float, default=0.02)
    a = ap.parse_args()
    out = {}
    print(f"{'condition':28s} {'M':>5s} {'feat':>6s} {'acc0':>5s} {'acc1':>5s} {'acc2':>5s} "
          f"{'x0':>6s} {'q1':>5s} {'q2':>5s} {'q3':>5s} {'q_fit':>6s} {'n_k':>4s}")
    for key in sorted(curves(a.files, a.group).items()):
        (cond, m, feat), rows = key
        span = np.mean([s for s, _ in rows], 0)
        b = float(np.mean([bb for _, bb in rows]))
        x, qf, nk = fit(span, b, a.floor)
        q = [x[k] / x[k - 1] if x[k - 1] > 1e-4 else float("nan") for k in (1, 2, 3)]
        out[f"{cond}|{m}|{feat}"] = {"acc": [round(v, 4) for v in span], "baseline": round(b, 4),
                                     "x0": round(float(x[0]), 3), "q": [round(v, 3) for v in q],
                                     "q_fit": None if qf is None else round(qf, 3), "n_k": nk}
        print(f"{cond:28s} {m:5d} {feat:>6s} {span[0]:5.2f} {span[1]:5.2f} {span[2]:5.2f} "
              f"{x[0]:6.2f} " + " ".join(f"{v:5.2f}" for v in q)
              + f" {qf if qf is not None else float('nan'):6.2f} {nk:4d}")
    Path("q_fit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
