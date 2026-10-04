"""Operating point for a protocol-B spec, by activity alone (no prediction results).

Criterion: mean 5-30 % of all neurons firing per character, never above 60 %, on the whole
training text, for every input code. Gains are scanned from --max-gain down in steps of 0.1.
A gain is out as soon as one character exceeds 60 % (runaway). The chosen gain is the first
(largest) one without runaway whose mean is at most 30 %; if its mean is below 5 % it is
still used and flagged (as for the larva), since every lower gain is quieter still.

    python select_gain.py specs/worm.json [--max-gain 4.0] [--codes 3,4,5] > gains/worm.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "uctf"))
from uctf import Network  # noqa: E402

DATA = REPO / "playground/gpf/data"
LOW, HIGH, RUNAWAY = 0.05, 0.30, 0.60


def train_ids(n=20001):
    chars = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))["chars"]
    text = (DATA / "tinyshakespeare.txt").read_text(encoding="utf-8")[:n]
    return [chars.index(c) for c in text if c in chars], len(chars)


def activity(net, ids):
    net.reset()
    net.record_activity = True
    frac = []
    for t in ids:
        net.step_token(int(t))
        frac.append(len(net.last_fired["ids"]) / net.cx.n)
        if frac[-1] > RUNAWAY:
            break
    return float(np.mean(frac)), float(np.max(frac)), len(frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--max-gain", type=float, default=4.0)
    ap.add_argument("--codes", default="3,4,5")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    ids, V = train_ids()
    net = Network(a.spec, V, device=a.device, log=lambda m: None)
    codes = [int(c) for c in a.codes.split(",")]
    rows, chosen, flag = [], None, None
    g = a.max_gain
    while g >= 0.1 - 1e-9:
        g = round(g, 1)
        net.respec("synapses.chemical.params.gain", g)
        res, runaway = [], False
        for c in codes:
            net.respec("input.params.seed", c)
            t0 = time.time()
            mean, peak, n = activity(net, ids)
            res.append({"code": c, "mean": round(mean, 4), "max": round(peak, 4), "chars": n})
            print(f"gain {g:.1f} code {c}: mean {mean:.1%} max {peak:.1%} over {n} chars "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
            if peak > RUNAWAY:
                runaway = True
                break
        rows.append({"gain": g, "codes": res, "runaway": runaway})
        if not runaway and max(r["mean"] for r in res) <= HIGH:
            chosen = g
            flag = "meets" if min(r["mean"] for r in res) >= LOW else "below 5 % mean (quiet)"
            break
        g -= 0.1
    print(json.dumps({"spec": a.spec, "criterion": {"mean": [LOW, HIGH], "max": RUNAWAY},
                      "chars": len(ids), "codes": codes, "sweep": rows,
                      "chosen_gain": chosen, "flag": flag}, indent=1))


if __name__ == "__main__":
    main()
