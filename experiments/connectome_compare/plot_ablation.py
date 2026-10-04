"""Readout-artifact figure. One job: ablation/*.json + results -> figures/ablation.png.

A  memory at M = 300 by feature set (counts + traces / counts / traces), three connectomes
B  delay-line test on the fly: counts + traces behaves like counts + the previous character's counts
C  the LM paper's worm numbers by feature set (three input codes)

    python plot_ablation.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FEATS = [("both", "counts + traces", "#000000"), ("spikes", "counts only", "#E69F00"),
         ("trace", "traces only", "#56B4E9")]
CONN = [("worm", "Worm"), ("larvaactive", "Larva"), ("fly", "Fly")]


def spans_m300():
    g = defaultdict(dict)
    srcs = [("both", f) for f in ("results_worm.json", "results_larva.json", "results_fly.json")]
    srcs += [("spikes", "ablation/spikes.json"), ("trace", "ablation/trace.json")]
    for feat, f in srcs:
        for r in json.load(open(HERE / f, encoding="utf-8")):
            if r["M"] == 300 and "_real_" in r["file"]:
                cond = re.match(r"(\w+?)_real", r["file"]).group(1)
                g[(cond, feat)][(r["file"], r["draw"])] = r["span"]
    return g


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    g = spans_m300()
    w = 0.13
    for i, (feat, label, col) in enumerate(FEATS):
        for j, (cond, _) in enumerate(CONN):
            v = np.mean(list(g[(cond, feat)].values()), 0)
            for kk, k in enumerate((1, 2)):
                x = j + (kk - 0.5) * 0.45 + (i - 1) * w
                ax[0].bar(x, v[k], w * 0.92, color=col, label=label if (j, kk) == (0, 0) else None)
    ax[0].set_xticks([j + d for j in range(3) for d in (-0.225, 0.225)],
                     [f"{n}\nk={k}" for _, n in CONN for k in (1, 2)], fontsize=8)
    ax[0].axhline(0.15, color="#888", lw=1, ls=":")
    ax[0].set(ylabel="decoding accuracy, k characters back", ylim=(0, 1),
              title="A. Neither feature alone carries the memory (M = 300)")
    ax[0].legend(frameon=False, fontsize=8)

    lag = json.load(open(HERE / "ablation/lag_test.json"))["fly_real_c3"]
    ks = np.arange(9)
    for name, col, ls in (("counts", "#E69F00", "-"), ("counts+trace", "#000000", "-"),
                          ("counts+lag1", "#009E73", "--"), ("counts+lag1+lag2", "#CC79A7", "--")):
        ax[1].plot(ks, lag[name], ls, marker="o", ms=3.5, lw=1.8, color=col,
                   label=name.replace("lag1", "previous char.").replace("lag2", "2 back"))
    ax[1].axhline(0.15, color="#888", lw=1, ls=":")
    ax[1].set(xlabel="characters back (k)", ylabel="decoding accuracy", ylim=(0, 1.02),
              title="B. A trace acts as a 1-character delay line (fly)")
    ax[1].legend(frameon=False, fontsize=7.5)

    pc = json.load(open(HERE / "ablation/paper_check.json"))["worm"]
    sets = [("all", "all (paper)"), ("no trace", "counts + voltage"), ("voltage", "voltage"),
            ("counts+trace", "counts + trace"), ("counts", "counts")]
    for i, (name, label) in enumerate(sets):
        for kk, k in enumerate((2, 3)):
            vals = [pc[f][name][k] for f in pc]
            x = kk * 6 + i
            ax[2].bar(x, np.mean(vals), 0.8, color="#0072B2" if name != "all" else "#D55E00")
            ax[2].plot([x] * 3, vals, "k.", ms=3)
    ax[2].set_xticks([kk * 6 + i for kk in range(2) for i in range(5)],
                     [lab for _ in range(2) for _, lab in sets], rotation=60, ha="right", fontsize=7.5)
    ax[2].text(2, 0.9, "k = 2", ha="center"); ax[2].text(8, 0.9, "k = 3", ha="center")
    ax[2].axhline(0.15, color="#888", lw=1, ls=":")
    ax[2].set(ylabel="decoding accuracy", ylim=(0, 1),
              title="C. LM paper, worm: traces inflate k = 2 by ~0.12")
    fig.suptitle("Readout-side memory: a spike trace plus the current count recovers the previous "
                 "character. Dotted: most frequent character.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "figures" / "ablation.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
