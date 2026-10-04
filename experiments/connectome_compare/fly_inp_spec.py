"""Fly spec that also records input spikes. One job: specs/fly.json -> specs/fly_inp.json.

Adds an INPUT_SAMPLE population (1,000 input neurons, fixed random sample, seed 0; all
14,876 would make each feature file ~2 GB) and an "input_spk" readout block of its counts.

    python fly_inp_spec.py
"""
import json
from pathlib import Path

import numpy as np

SPECS = Path(__file__).parent / "specs"
spec = json.loads((SPECS / "fly.json").read_text(encoding="utf-8"))
inp = np.asarray(spec["populations"]["INPUT"]["ids"])
sample = np.sort(np.random.default_rng(0).choice(inp, 1000, replace=False))
spec["populations"]["INPUT_SAMPLE"] = {"ids": sample.tolist()}
spec["readout"].append({"name": "input_spk", "feature": "counts", "population": "INPUT_SAMPLE", "params": {}})
spec["name"] += "-inp"
(SPECS / "fly_inp.json").write_text(json.dumps(spec), encoding="utf-8")
print(len(inp), "input neurons; sample", len(sample))
