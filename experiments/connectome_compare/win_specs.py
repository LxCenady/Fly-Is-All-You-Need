"""Window-feature spec variants. One job: specs/<name>.json -> specs/<name>_win.json.

Readout = full counts ("spikes", kept first so simulate.py's dims work), trace, then counts in
single steps of the 6-step token: w0 (step 0), w1, w2, and w35 (steps 3-5).

    python win_specs.py
"""
import json
from pathlib import Path

SPECS = Path(__file__).parent / "specs"
WINDOWS = {"w0": [0, 1], "w1": [1, 2], "w2": [2, 3], "w35": [3, 6]}

for name in ("worm", "larva", "fly"):
    spec = json.loads((SPECS / f"{name}.json").read_text(encoding="utf-8"))
    pop = spec["readout"][0]["population"]
    spec["readout"] += [{"name": k, "feature": "counts", "population": pop, "params": {"window": w}}
                        for k, w in WINDOWS.items()]
    spec["name"] += "-win"
    (SPECS / f"{name}_win.json").write_text(json.dumps(spec), encoding="utf-8")
    print(name, [f["name"] for f in spec["readout"]])
