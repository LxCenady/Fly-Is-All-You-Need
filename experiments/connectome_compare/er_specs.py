"""Random-graph (Erdos-Renyi) reservoirs under protocol B. One job: write specs/er*.json.

The same rules as protocol.py (input: a fixed random 10 % of the neurons, 24 % of them driven per
character; readout: at most 3,000 of the rest; counts and traces), plus an "input_spk" block
(counts of all input neurons, or a fixed sample of 1,000) to split input-side encoding from
transmission. No sensory neurons. Chemical gain left to target_gain.py.

    python er_specs.py
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "specs"
NETS = {"er448": (448, 10, 0.10), "er2952": (2952, 37, 0.10), "er2952i35": (2952, 37, 0.35),
        "er20000": (20000, 50, 0.10)}
NEURON = {"model": "lif", "params": {"dt": 0.02, "tau": 0.1, "threshold": 1.0, "reset": 0.0,
                                     "tonic": 0.05, "noise_hz": 0.0, "seed": 20260920}}


def make(name, n, k, inh):
    inp = np.sort(np.random.default_rng(0).choice(n, round(0.10 * n), replace=False))
    rest = np.setdiff1d(np.arange(n), inp)
    out = np.sort(np.random.default_rng(1).choice(rest, min(3000, len(rest)), replace=False))
    rec = inp if len(inp) <= 1000 else np.sort(np.random.default_rng(2).choice(inp, 1000, replace=False))
    spec = {
        "uctf": 2, "name": f"{name}-protocolB",
        "description": f"Random graph n={n}, k={k}, inhibitory {inh}; protocol B, see er_specs.py",
        "connectome": {"source": "random", "params": {"n": n, "k": k, "inhibitory": inh, "seed": 0}},
        "populations": {"INPUT": {"ids": inp.tolist()}, "READOUT": {"ids": out.tolist()},
                        "INPUT_REC": {"ids": rec.tolist()}},
        "neuron": NEURON,
        "synapses": [{"layer": "chemical", "model": "chemical", "params": {"gain": 1.0}}],
        "input": {"encoder": "random_subset", "population": "INPUT", "steps": 6,
                  "params": {"active": round(0.24 * len(inp)), "drive": 1.5, "sustain": 0.5, "seed": 3}},
        "readout": [{"name": "spikes", "feature": "counts", "population": "READOUT", "params": {}},
                    {"name": "trace", "feature": "trace", "population": "READOUT", "params": {"tau": 0.1}},
                    {"name": "input_spk", "feature": "counts", "population": "INPUT_REC", "params": {}}],
    }
    (OUT / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    print(f"{name}: n {n}, input {len(inp)} (driven {spec['input']['params']['active']}), "
          f"readout {len(out)}, input recorded {len(rec)}")


for name, args in NETS.items():
    make(name, *args)
