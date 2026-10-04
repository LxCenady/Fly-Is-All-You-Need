"""Follow-up figure (tau and input size). One job: results + ablation/*.json -> figures/followup.png.

Columns: worm, larva (active state), fly. Rows: spike counts + traces (the original readout),
spike counts only (the network's own memory). M = 300, mean over codes and draws.

    python plot_followup.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SOURCES = ["results_worm.json", "results_larva.json", "results_fly.json", "results_tau_worm.json",
           "results_tau_larva.json", "ablation/spikes.json", "ablation/both_fly_tau20.json",
           "ablation/both_fly_new.json", "ablation/spikes_fly_new.json"]
PANELS = [("C. elegans", [("worm", "τ 100 ms (base)"), ("worm_tau05", "τ 50 ms"),
                          ("worm_tau20", "τ 200 ms")]),
          ("Larva, active state", [("larvaactive", "τ 100 ms (base)"), ("larva_tau20", "τ 200 ms")]),
          ("Adult fly", [("fly", "τ 100 ms (base)"), ("fly_tau05", "τ 50 ms"),
                         ("fly_tau20", "τ 200 ms"), ("fly_in9", "9 driven neurons")])]
COLOURS = {"τ 100 ms (base)": "#000000", "τ 50 ms": "#56B4E9", "τ 200 ms": "#0072B2",
           "9 driven neurons": "#D55E00"}


def load():
    g = defaultdict(dict)
    for f in SOURCES:
        for r in json.load(open(HERE / f, encoding="utf-8")):
            if r["M"] != 300 or "_real_" not in r["file"]:
                continue
            cond = re.match(r"(\w+?)_real", r["file"]).group(1)
            g[(cond, r.get("features", "both"))][(r["file"], r["draw"])] = r["span"]
    return g


def main():
    g = load()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    ks = np.arange(9)
    for j, (title, conds) in enumerate(PANELS):
        for i, feats in enumerate(("both", "spikes")):
            a = ax[i, j]
            for cond, label in conds:
                spans = list(g.get((cond, feats), {}).values())
                if not spans:
                    continue
                a.plot(ks, np.mean(spans, 0), "-o", ms=3.5, lw=1.8, color=COLOURS[label], label=label)
            a.axhline(0.15, color="#999", lw=1, ls=":")
            a.set_ylim(0, 1.02)
            a.grid(color="#eee", lw=0.6)
            if i == 0:
                a.set_title(title)
                a.legend(fontsize=8, frameon=False, loc="upper right")
            if j == 0:
                a.set_ylabel(("spike counts + traces\n" if feats == "both" else
                              "spike counts only (network memory)\n") + "decoding accuracy")
            if i == 1:
                a.set_xlabel("characters back (k)")
    fig.suptitle("Follow-ups, protocol B, M = 300 readout neurons, ~10 % activity. "
                 "Dotted: always guess the most frequent character. Exploratory.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "figures" / "followup.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
