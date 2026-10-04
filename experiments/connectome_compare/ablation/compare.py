"""Feature ablation table. One job: results (both) + ablation/{spikes,trace}.json -> a markdown table.

Rows are paired by (file, M, draw): the neuron draws are the same in all three feature sets.

    python ablation/compare.py > ablation/TABLE.md
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent


def load(paths):
    rows = {}
    for p in paths:
        for r in json.load(open(p, encoding="utf-8")):
            rows[(r["file"], r["M"], r["draw"])] = r
    return rows


def group(stem):
    return re.match(r"(\w+?_(?:real|rewire))", stem.removesuffix(".npz")).group(1)


def main():
    both = load(sorted(ROOT.glob("results_*.json")) + sorted(HERE.glob("both_*.json")))
    kinds = {"spikes": load([HERE / "spikes.json"]), "trace": load([HERE / "trace.json"])}
    table = defaultdict(lambda: defaultdict(list))
    for key, r in kinds["spikes"].items():
        if key not in both or key not in kinds["trace"]:
            continue
        g = (group(key[0]), key[1])
        for name, src in (("both", both), ("spikes", kinds["spikes"]), ("trace", kinds["trace"])):
            table[g][name].append(src[key]["span"][1:5])
    print("| condition | M | features | k=1 | k=2 | k=3 | k=4 | n |")
    print("|---|---|---|---|---|---|---|---|")
    for (g, m) in sorted(table):
        for name in ("both", "spikes", "trace"):
            v = np.mean(table[(g, m)][name], 0)
            print(f"| {g} | {m} | {name} | " + " | ".join(f"{x:.2f}" for x in v)
                  + f" | {len(table[(g, m)][name])} |")


if __name__ == "__main__":
    main()
