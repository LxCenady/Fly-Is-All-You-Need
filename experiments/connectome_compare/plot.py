"""Figures for the cross-connectome comparison. One job: results_*.json -> figures/*.png.

    python plot.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
EXCLUDE = {"worm_rewire_s2"}                     # failed activity match (48 % active)
# one fixed colour per connectome (Okabe-Ito, colour-blind safe); rewired = dashed
STYLE = {
    "worm": ("C. elegans (448)", "#0072B2"),
    "larvaactive": ("Larva, active state (2,952)", "#E69F00"),
    "larvaquiet": ("Larva, quiet state (2,952)", "#CC79A7"),
    "fly": ("Adult fly (166,700)", "#009E73"),
}


def load():
    groups = defaultdict(list)
    for f in sorted(HERE.glob("results_*.json")):
        if f.stem == "results_worm_real":
            continue
        for r in json.load(open(f, encoding="utf-8")):
            stem = r["file"].removesuffix(".npz")
            if stem in EXCLUDE:
                continue
            name, kind = re.match(r"(\w+?)_(real|rewire)_", stem).groups()
            groups[(name, kind, r["M"])].append(r)
    return groups


def main():
    g = load()
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # A: memory (character 2 back) vs readout size
    for name, (label, col) in STYLE.items():
        for kind, ls in (("real", "-"), ("rewire", "--")):
            ms = sorted(m for (n, k, m) in g if n == name and k == kind)
            if not ms:
                continue
            y = [np.mean([r["span"][2] for r in g[(name, kind, m)]]) for m in ms]
            lo = [min(r["span"][2] for r in g[(name, kind, m)]) for m in ms]
            hi = [max(r["span"][2] for r in g[(name, kind, m)]) for m in ms]
            ax[0].plot(ms, y, ls, color=col, lw=2, marker="o", ms=5,
                       label=f"{label}{', rewired' if kind == 'rewire' else ''}")
            ax[0].fill_between(ms, lo, hi, color=col, alpha=0.12, lw=0)
    ax[0].set_xscale("log")
    ax[0].axhline(0.15, color="#888", lw=1, ls=":")
    ax[0].text(105, 0.16, "always guess the most frequent character", fontsize=8, color="#666")
    ax[0].set(xlabel="readout neurons M (log scale)", ylabel="decoding accuracy, 2 characters back",
              title="A. Memory grows with the number of readout neurons", ylim=(0, 1.02))
    handles, labels = ax[0].get_legend_handles_labels()

    # B: memory curve at M = 300
    ks = np.arange(1, 9)
    for name, (label, col) in STYLE.items():
        for kind, ls in (("real", "-"), ("rewire", "--")):
            rows = g.get((name, kind, 300))
            if not rows:
                continue
            y = np.mean([r["span"][1:9] for r in rows], 0)
            ax[1].plot(ks, y, ls, color=col, lw=2, marker="o", ms=4)
    ax[1].axhline(0.15, color="#888", lw=1, ls=":")
    ax[1].set(xlabel="characters back (k)", ylabel="decoding accuracy",
              title="B. Memory curves at the same readout size (M = 300)", ylim=(0, 1.02))
    ax[1].text(4.3, 0.85, "solid: real wiring\ndashed: rewired, activity matched", fontsize=8.5)

    # C: silent readout share vs memory, M = 300 and 1000
    for name, (label, col) in STYLE.items():
        for kind, mk in (("real", "o"), ("rewire", "^")):
            for m, fill in ((300, True), (1000, False)):
                rows = g.get((name, kind, m))
                if not rows:
                    continue
                x = [r["silent_share"] for r in rows]
                y = [r["span"][2] for r in rows]
                ax[2].scatter(x, y, s=28, marker=mk, color=col if fill else "white",
                              edgecolors=col, linewidths=1.3, alpha=0.9)
    ax[2].set(xlabel="share of readout neurons that never fire",
              ylabel="decoding accuracy, 2 characters back",
              title="C. Silent readout neurons vs memory", xlim=(0, 1), ylim=(0, 1.02))
    ax[2].text(0.02, 0.04, "circles: real, triangles: rewired\nfilled: M = 300, open: M = 1000",
               fontsize=8.5)

    fig.suptitle("Connectomes as text reservoirs, protocol B (random input, ~10 % activity; "
                 "larva: bistable, two states). Exploratory.", fontsize=11)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "comparison.png", dpi=170)
    print("wrote", out / "comparison.png")


if __name__ == "__main__":
    main()
