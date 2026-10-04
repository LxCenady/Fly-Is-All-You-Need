"""Operating point by a common activity target (protocol B, revised 2026-10-03).

One job: bisect the chemical gain of a spec so that, averaged over the input codes, the mean
fraction of all neurons firing per character over the whole training text is --target
(default 10 %), within --tol. A gain at which any code runs away (a character above 60 %)
counts as too high. Activity only; no prediction results are used.

Why: "the largest gain without runaway" left the worm, larva and fly at 9-17 %, 22-24 % and
about 8 % activity, a confound for the comparison. A common target removes it; how close each
network sits to runaway is recorded separately (gains/*.json from select_gain.py).

    python target_gain.py specs/worm.json --lo 0.5 --hi 2.8 [--target 0.10] > target/worm.json
"""
import argparse
import json
import sys
import time

from select_gain import RUNAWAY, Network, activity, train_ids


def measure(net, ids, codes, g):
    net.respec("synapses.chemical.params.gain", g)
    res = []
    for c in codes:
        net.respec("input.params.seed", c)
        t0 = time.time()
        mean, peak, n = activity(net, ids)
        res.append({"code": c, "mean": round(mean, 4), "max": round(peak, 4), "chars": n})
        print(f"  gain {g:.4f} code {c}: mean {mean:.1%} max {peak:.1%} ({time.time() - t0:.0f}s)",
              file=sys.stderr, flush=True)
        if peak > RUNAWAY:
            return res, None
    return res, sum(r["mean"] for r in res) / len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--codes", default="3,4,5")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    ids, V = train_ids()
    net = Network(a.spec, V, device=a.device, log=lambda m: None)
    codes = [int(c) for c in a.codes.split(",")]
    lo, hi, steps, best = a.lo, a.hi, [], None
    for _ in range(a.iters):
        g = (lo + hi) / 2
        res, mean = measure(net, ids, codes, g)
        steps.append({"gain": round(g, 4), "codes": res, "mean": mean})
        if mean is not None and abs(mean - a.target) <= a.tol:
            best = g
            break
        if mean is None or mean > a.target:
            hi = g
        else:
            lo = g
    ok = [s for s in steps if s["mean"] is not None]
    closest = min(ok, key=lambda s: abs(s["mean"] - a.target))["gain"] if ok else None
    print(json.dumps({"spec": a.spec, "target": a.target, "tol": a.tol, "steps": steps,
                      "chosen_gain": None if best is None else round(best, 4),
                      "closest_gain": closest}, indent=1))


if __name__ == "__main__":
    main()
