"""Spec variants for the two follow-up experiments. One job: specs/<name>.json -> specs/<name>_<v>.json.

  in9    every character drives 9 input neurons (absolute, the same for all connectomes)
  tau05  membrane time constant 50 ms (base 100 ms)
  tau20  membrane time constant 200 ms

    python variants.py
"""
import json
from pathlib import Path

SPECS = Path(__file__).parent / "specs"
VARIANTS = {"in9": ("input.params.active", 9),
            "tau05": ("neuron.params.tau", 0.05),
            "tau20": ("neuron.params.tau", 0.2)}


def main():
    for name in ("worm", "larva", "fly"):
        base = json.loads((SPECS / f"{name}.json").read_text(encoding="utf-8"))
        for v, (path, value) in VARIANTS.items():
            spec = json.loads(json.dumps(base))
            obj = spec
            *keys, last = path.split(".")
            for k in keys:
                obj = obj[k]
            obj[last] = value
            spec["name"] = f"{base['name']}-{v}"
            (SPECS / f"{name}_{v}.json").write_text(json.dumps(spec), encoding="utf-8")
    print(sorted(p.name for p in SPECS.glob("*_*.json")))


if __name__ == "__main__":
    main()
