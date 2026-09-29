"""A made-up 1,200-neuron "connectome" in the plain CSV format, to show how any wiring diagram
goes into UCTF (and to test the framework without a GPU).

    python make_toy.py                          writes neurons.csv and edges.csv here
    python -m uctf import --neurons neurons.csv --edges edges.csv --out toy --sign-col nt
    python -m uctf bench --substrate toy.json --data ../../../playground/gpf/data/tinyshakespeare.txt \\
        --train-chars 5000 --val-chars 1500 --device cpu

Layout: 100 input neurons -> 800 sparse "expansion" cells (a few inputs each, a loose copy of
the fly's projection-neuron -> Kenyon-cell layer) -> 100 output cells, 150 recurrent "hub" cells
and 50 inhibitory cells feeding back onto the expansion layer.  Nothing here is biological.
"""
import csv

import numpy as np

rng = np.random.default_rng(0)
layers = [("IN", 100, "acetylcholine"), ("EXP", 800, "acetylcholine"), ("OUT", 100, "acetylcholine"),
          ("HUB", 150, "acetylcholine"), ("INH", 50, "gaba")]
rows, start = [], {}
for name, n, nt in layers:
    start[name] = len(rows)
    for i in range(n):
        side = "L" if i % 2 == 0 else "R"
        x = {"IN": 0, "EXP": 1, "OUT": 2, "HUB": 1.5, "INH": 0.5}[name] + rng.normal(0, 0.12)
        rows.append({"id": f"{name}{i}", "cell_type": name, "side": side, "nt": nt,
                     "x": round((-1 if side == "L" else 1) * (0.3 + x), 3), "y": 0.0,
                     "z": round(rng.normal(0, 0.5) + {"IN": 0, "EXP": 0.8, "OUT": 1.6, "HUB": -0.8, "INH": 0.4}[name], 3)})
ids = {name: [f"{name}{i}" for i in range(n)] for name, n, _ in layers}


def connect(pre, post, k, count=(1, 6)):
    """Each post cell receives k inputs from random pre cells."""
    for q in ids[post]:
        for p in rng.choice(ids[pre], size=k, replace=False):
            yield p, q, int(rng.integers(*count))


edges = []
edges += connect("IN", "EXP", 6)
edges += connect("EXP", "OUT", 60)
edges += connect("EXP", "HUB", 20)
edges += connect("HUB", "HUB", 10)
edges += connect("HUB", "INH", 15)
edges += connect("INH", "EXP", 3)
edges += connect("OUT", "HUB", 5)

with open("neurons.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open("edges.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["pre", "post", "weight"]); w.writerows(edges)
print(f"{len(rows)} neurons, {len(edges)} edges")
