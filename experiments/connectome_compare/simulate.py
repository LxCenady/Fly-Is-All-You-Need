"""Simulate one protocol-B variant over the whole text and cache its readout features.

One job: spec + gain + input code (+ optional rewiring) -> features/<name>.npz with
X (rows = characters, float32), the readout sizes and the variant's settings.
Rewired variants are matched in activity to the real wiring by bisection on the gain
(uctf.bench.match_activity, first --match-chars training characters); the activity over
the whole run is stored so a runaway can be seen afterwards.

    python simulate.py specs/worm.json --gain 2.7 --code 3 [--rewire-seed 0] --out features/x.npz

--shuffle SEED permutes the characters (same frequencies, no sequential structure), so k-back
decoding can only use the network's memory. --set PATH=VALUE overrides any spec value
(e.g. input.steps=12). Both are stored in info; analyze.py rebuilds the same sequence.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "uctf"))
from uctf import run  # noqa: E402
from uctf.bench import match_activity  # noqa: E402
from uctf.core.spec import set_path  # noqa: E402

DATA = REPO / "playground/gpf/data"


def text_ids(n=25001, shuffle=None):
    chars = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))["chars"]
    text = (DATA / "tinyshakespeare.txt").read_text(encoding="utf-8")[:n]
    ids = np.asarray([chars.index(c) for c in text if c in chars], np.int64)
    if shuffle is not None:
        ids = np.random.default_rng(shuffle).permutation(ids)
    return ids, len(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--gain", type=float, required=True)
    ap.add_argument("--code", type=int, required=True)
    ap.add_argument("--rewire-seed", type=int)
    ap.add_argument("--match-chars", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shuffle", type=int)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids, V = text_ids(shuffle=a.shuffle)
    sets = [f"synapses.chemical.params.gain={a.gain}", f"input.params.seed={a.code}", *a.set]
    real = run.brain_spec(a.spec, overrides=sets)
    info = {"spec": a.spec, "gain": a.gain, "code": a.code, "rewire_seed": a.rewire_seed,
            "shuffle": a.shuffle, "set": a.set}
    bspec = real
    if a.rewire_seed is not None:
        target = match_activity(real, ids, V, a.device, None, a.match_chars)[1]
        bspec = run.brain_spec(a.spec, overrides=sets, control="rewire-full",
                               control_seed=a.rewire_seed)
        g, got = match_activity(bspec, ids, V, a.device, target, a.match_chars)
        set_path(bspec["spec"], "synapses.chemical.params.gain", g)
        info.update(matched_gain=g, target_activity=target, matched_activity=got)
        print(f"rewired seed {a.rewire_seed}: gain {g:.4f}, activity {got:.4f} (real {target:.4f})",
              flush=True)
    X, dims = run.simulate(ids, V, bspec, a.device, with_dims=True)
    n = dims["spikes"]
    act = (X[:, :n] > 0).mean(1)
    info.update(readout_active_mean=float(act.mean()), readout_active_max=float(act.max()),
                dims=dims)
    np.savez_compressed(a.out, X=X, info=json.dumps(info))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
