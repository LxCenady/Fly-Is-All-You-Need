"""Protocol B: one annotation-free spec per connectome, the same rules for all.

One job: write specs/<name>.json from each connectome's own UCTF spec.
  - connectome source and neuron model: the base spec's; chemical gain left to select_gain.py
  - synapses onto sensory neurons (chemical layer) are cut, as in every base protocol
  - INPUT: a fixed random 10 % of the non-sensory neurons (seed 0); each character drives
    24 % of them (drive 1.5, 6 steps, sustain 0.5), so about 2.4 % of the network
  - READOUT: a fixed random sample of at most 3,000 neurons that are neither input nor sensory
    (seed 1); features: spike counts and spike traces (no voltages, see PLAN.md 3b)
  - the worm keeps its gap junctions (electrical layer, gain 0.5), part of its connectome

    python protocol.py
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

from paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "uctf"))
from uctf.core.network import apply_transforms, load_connectome  # noqa: E402
from uctf.core.select import Selector  # noqa: E402
from uctf.core.spec import load_spec  # noqa: E402

BASES = {"worm": REPO / "uctf/examples/celegans/celegans-cook2019.json",
         "larva": REPO / "uctf/examples/larva/larva-winding2023.json",
         "fly": "malecns-v1"}
INPUT_SHARE, DRIVEN_SHARE, MAX_READOUT = 0.10, 0.24, 3000
OUT = Path(__file__).parent / "specs"


def make(name, base_ref):
    base = load_spec(base_ref)
    cx = load_connectome(base)
    sensory = Selector(cx, base["populations"]).mask("SENSORY")
    pool = np.flatnonzero(~sensory)
    inp = np.sort(np.random.default_rng(0).choice(pool, round(INPUT_SHARE * len(pool)), replace=False))
    rest = np.setdiff1d(pool, inp)
    out = np.sort(np.random.default_rng(1).choice(rest, min(MAX_READOUT, len(rest)), replace=False))
    syn = [copy.deepcopy(s) for s in base["synapses"]]
    syn[0]["params"]["gain"] = 1.0                       # set by select_gain.py
    spec = {
        "uctf": 2, "name": f"{name}-protocolB",
        "description": f"Protocol B ({name}): random input and readout, see protocol.py",
        "connectome": copy.deepcopy(base["connectome"]),
        "transforms": [{"transform": "cut_inputs",
                        "params": {"population": "SENSORY", "layers": ["chemical"]}}],
        "populations": {"SENSORY": base["populations"]["SENSORY"],
                        "INPUT": {"ids": inp.tolist()}, "READOUT": {"ids": out.tolist()}},
        "neuron": copy.deepcopy(base["neuron"]),
        "synapses": syn,
        "input": {"encoder": "random_subset", "population": "INPUT", "steps": 6,
                  "params": {"active": round(DRIVEN_SHARE * len(inp)), "drive": 1.5,
                             "sustain": 0.5, "seed": 3}},
        "readout": [{"name": "spikes", "feature": "counts", "population": "READOUT", "params": {}},
                    {"name": "trace", "feature": "trace", "population": "READOUT",
                     "params": {"tau": 0.1}}],
        "controls": {"rewire-full": [{"transform": "rewire", "params": {
            "layers": {"chemical": "permute_presynaptic", "electrical": "double_edge_swap"},
            "classes": None, "skip_missing": True}}]},
        "activity_match": {"param": "synapses.chemical.params.gain", "readouts": ["spikes"]},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    print(f"{name}: {cx.n:,} neurons, {int(sensory.sum()):,} sensory, input {len(inp):,} "
          f"(driven {spec['input']['params']['active']}), readout {len(out):,}")


def main():
    for name, ref in BASES.items():
        make(name, ref)


if __name__ == "__main__":
    main()
