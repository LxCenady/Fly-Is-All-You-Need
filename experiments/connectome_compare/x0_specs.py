"""Spec variants for the x0 study. One job: specs/worm.json -> specs/worm_inp*.json.

worm_inp.json adds an "input_spk" readout block: spike counts of the INPUT population, so the
input spikes each character actually causes can be counted. worm_inp_<v>.json also bakes in an
input setting, for target_gain.py (which has no --set).

    python x0_specs.py
"""
import json
from pathlib import Path

SPECS = Path(__file__).parent / "specs"
BAKED = {"act18": ("active", 18), "sus075": ("sustain", 0.75), "drv3": ("drive", 3.0)}

base = json.loads((SPECS / "worm.json").read_text(encoding="utf-8"))
base["readout"].append({"name": "input_spk", "feature": "counts", "population": "INPUT", "params": {}})
base["name"] += "-inp"
(SPECS / "worm_inp.json").write_text(json.dumps(base), encoding="utf-8")
for v, (k, val) in BAKED.items():
    s = json.loads(json.dumps(base))
    s["input"]["params"][k] = val
    s["name"] = f"worm-inp-{v}"
    (SPECS / f"worm_inp_{v}.json").write_text(json.dumps(s), encoding="utf-8")
print(sorted(p.name for p in SPECS.glob("worm_inp*.json")))
