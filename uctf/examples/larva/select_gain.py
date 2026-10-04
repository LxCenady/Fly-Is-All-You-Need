"""Choose the larva's synaptic gain by activity alone, before any prediction result is seen.

Criterion (the worm example's): on average 5-30 % of all neurons fire per character, and never
more than 60 %, for every input code. Rule: the largest gain in 0.5, 0.6, ..., --max-gain that
meets it. Gains are tried from the top down; a gain is ruled out as soon as one character
exceeds 60 %, and the scan stops at the first gain that meets the criterion.

    python select_gain.py --max-gain 3.0 --chars 20001 --code-seeds 3,4,5 > gains.json

History, kept for honesty (see the README): a first selection used only the first 500
characters and one code and chose 1.8; with code 3 that gain ran away at about character
4,500 and stayed there. The criterion is unchanged; it is now checked on the whole
training text and on all codes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from uctf import Network  # noqa: E402

DATA = HERE.parents[2] / "playground" / "gpf" / "data"
MEAN_RANGE, MAX_LIMIT = (0.05, 0.30), 0.60


def activity(net, ids):
    """(mean, max) fraction of all neurons firing per character; stops early on runaway."""
    net.reset()
    net.record_activity = True
    frac = []
    for t in ids:
        net.step_token(int(t))
        frac.append(len(net.last_fired["ids"]) / net.cx.n)
        if frac[-1] > MAX_LIMIT:
            break
    return float(np.mean(frac)), float(np.max(frac)), len(frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=str(DATA / "tinyshakespeare.txt"))
    ap.add_argument("--vocab", default=str(DATA / "vocab.json"))
    ap.add_argument("--chars", type=int, default=500)
    ap.add_argument("--max-gain", type=float, default=1.5)
    ap.add_argument("--code-seeds", default="3")
    a = ap.parse_args()
    chars = json.loads(Path(a.vocab).read_text(encoding="utf-8"))["chars"]
    text = Path(a.text).read_text(encoding="utf-8")[:a.chars]
    ids = [chars.index(c) for c in text if c in chars]
    seeds = [int(s) for s in a.code_seeds.split(",")]
    net = Network(HERE / "larva-winding2023.json", len(chars), device="cpu", log=lambda m: None)
    steps = int(round((a.max_gain - 0.5) / 0.1))
    rows, chosen = [], None
    for g in [round(0.5 + 0.1 * i, 1) for i in range(steps, -1, -1)]:
        net.respec("synapses.chemical.params.gain", g)
        codes, ok = [], True
        for s in seeds:
            net.respec("input.params.seed", s)
            mean, peak, n = activity(net, ids)
            meets = MEAN_RANGE[0] <= mean <= MEAN_RANGE[1] and peak <= MAX_LIMIT
            codes.append({"seed": s, "mean_active": round(mean, 4), "max_active": round(peak, 4),
                          "chars_run": n, "meets": meets})
            print(f"gain {g:.1f} code {s}: mean {mean:.1%}, max {peak:.1%} over {n} chars"
                  f"{'  meets' if meets else ''}", file=sys.stderr, flush=True)
            ok = ok and meets
            if not meets:
                break
        rows.append({"gain": g, "codes": codes, "meets": ok})
        if ok:
            chosen = g
            break
    print(json.dumps({"criterion": {"mean": MEAN_RANGE, "max": MAX_LIMIT}, "chars": len(ids),
                      "code_seeds": seeds, "sweep": rows, "chosen_gain": chosen}, indent=1))


if __name__ == "__main__":
    main()
