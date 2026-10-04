"""Worm q / x0 figure. One job: result json files -> figures/worm_q_x0.png.

A  memory on real vs shuffled text, with the zero-memory text oracle
B  where in the 6-step token window the information sits
C  driven inputs per character -> input-side and readout-side encoding
D  recurrent gain -> input-side and readout-side encoding
E  activity matching: x0 at the base gain vs at ~10 % activity
F  residue of the previous character (x1) vs encoding (x0)

    python plot_worm.py
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ORACLE = [0.287, 0.212, 0.178, 0.157]          # k = 1..4, perfect code of the current char, English text
BLUE, ORANGE, GREEN, PINK, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#888888", "#56B4E9"


def load(name):
    return json.load(open(HERE / name, encoding="utf-8"))


def mean_span(rows, cond, m=300):
    v = [r["span"] for r in rows if r["M"] == m and re.sub(r"_c\d+\.npz$", "", r["file"]) == cond]
    return np.mean(v, 0), np.mean([r["majority"] for r in rows if r["M"] == m
                                   and re.sub(r"_c\d+\.npz$", "", r["file"]) == cond])


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
    ax = ax.ravel()

    # A
    real, b = mean_span(load("ablation/spikes.json"), "worm_real")
    shuf, _ = mean_span(load("results_q_worm_spikes.json"), "worm_base")
    ks = np.arange(len(real))
    ax[0].plot(ks, real, "-o", color=GREY, ms=4, lw=1.8, label="English text")
    ax[0].plot(ks[:len(shuf)], shuf, "-o", color=BLUE, ms=4, lw=1.8, label="shuffled text (network only)")
    ax[0].plot([1, 2, 3, 4], ORACLE, "--", color=ORANGE, lw=1.6,
               label="zero memory, perfect current code\n(text statistics alone)")
    ax[0].axhline(b, color=GREY, lw=1, ls=":")
    ax[0].set(xlabel="characters back (k)", ylabel="decoding accuracy (M = 300, counts)",
              title="A. Much of the 'memory' on text is text statistics", ylim=(0, 0.65), xlim=(-0.2, 4.2))
    ax[0].set_xticks(range(5))
    ax[0].legend(frameon=False, fontsize=7.5)

    # B
    w = load("window_test_m300.json")["worm_win"]
    blocks = [("spikes", "whole token"), ("w0", "step 1"), ("w1", "step 2"), ("w2", "step 3"), ("w35", "steps 4-6")]
    x = np.arange(len(blocks))
    ax[1].bar(x - 0.18, [w[k][0] for k, _ in blocks], 0.34, color=BLUE, label="current character (k = 0)")
    ax[1].bar(x + 0.18, [w[k][1] for k, _ in blocks], 0.34, color=ORANGE, label="previous character (k = 1)")
    ax[1].axhline(0.145, color=GREY, lw=1, ls=":")
    ax[1].set_xticks(x, [lab for _, lab in blocks])
    ax[1].set(ylabel="decoding accuracy (shuffled text)", ylim=(0, 0.6),
              title="B. The previous character sits in the next token's first step")
    ax[1].legend(frameon=False, fontsize=7.5)

    s = load("x0_split.json")
    # C
    acts = [(2, "act2"), (4, "act4"), (9, "base"), (12, "act12"), (18, "act18"), (24, "act24"),
            (30, "act30"), (36, "act36")]
    n = [a for a, _ in acts]
    ax[2].plot(n, [s[c]["x0_in"] for _, c in acts], "-o", color=GREEN, ms=4, lw=1.8,
               label="input side (36 input neurons)")
    ax[2].plot(n, [s[c]["x0_out"] for _, c in acts], "-o", color=BLUE, ms=4, lw=1.8,
               label="readout side (300 neurons)")
    ax[2].set(xlabel="input neurons driven per character (of 36)", ylabel="encoding x0",
              title="C. Encoding peaks when half the inputs are driven")
    ax[2].legend(frameon=False, fontsize=7.5)

    # D
    gains = [(1.0, "g100"), (1.5, "g150"), (2.0, "g200"), (2.3, "g230"), (2.5875, "base")]
    g = [a for a, _ in gains]
    ax[3].plot(g, [s[c]["x0_in"] for _, c in gains], "-o", color=GREEN, ms=4, lw=1.8, label="input side")
    ax[3].plot(g, [s[c]["x0_out"] for _, c in gains], "-o", color=BLUE, ms=4, lw=1.8, label="readout side")
    for gg, c in gains:
        ax[3].annotate(f"{s[c]['transmission']:.2f}", (gg, s[c]["x0_out"]), textcoords="offset points",
                       xytext=(0, -13), ha="center", fontsize=7, color=BLUE)
    ax[3].set(xlabel="recurrent gain", ylabel="encoding x0", ylim=(0, 5),
              title="D. Gain cleans the input or spreads it, not both")
    ax[3].text(1.0, 2.0, "numbers: readout / input (transmission)", fontsize=7.5, color=GREY)
    ax[3].legend(frameon=False, fontsize=7.5, loc="upper right")

    # E
    pairs = [("base", "base", "base"), ("act18", "act18_matched", "18 driven"),
             ("sus075", "sus075_matched", "sustain 0.75"), ("drv3", "drv3_matched", "drive 3.0")]
    x = np.arange(len(pairs))
    ax[4].bar(x - 0.18, [s[a]["x0_out"] for a, _, _ in pairs], 0.34, color=SKY, label="base gain 2.59")
    ax[4].bar(x + 0.18, [s[m]["x0_out"] for _, m, _ in pairs], 0.34, color=BLUE,
              label="gain lowered to ~10 % activity")
    ax[4].set_xticks(x, [lab for _, _, lab in pairs])
    ax[4].set(ylabel="readout encoding x0 (M = 300)", title="E. More input still wins at equal activity")
    ax[4].legend(frameon=False, fontsize=7.5)

    # F
    t = load("results_x0_worm_x0.json")
    pts = [(v["x0"], v["x1"]) for k, v in t.items() if k.endswith("|300") and v["x0"] > 0.02]
    xs, ys = zip(*pts)
    ax[5].scatter(xs, ys, s=30, color=BLUE, zorder=3)
    xx = np.linspace(0.02, 4.5, 50)
    for q, ls in ((0.1, ":"), (0.2, "--"), (0.3, "-.")):
        ax[5].plot(xx, q * xx, ls, color=GREY, lw=1)
        xl = min(4.4, 0.55 / q)
        ax[5].text(xl, q * xl, f"x1 = {q} x0", fontsize=7, color=GREY, va="bottom", ha="right")
    ax[5].set(xlabel="encoding x0 (current character)", ylabel="residue x1 (previous character)",
              xlim=(0, 5.2), ylim=(0, 0.6), title="F. The residue grows less than the encoding")

    fig.suptitle("C. elegans connectome reservoir (448 neurons): where the memory comes from. "
                 "Shuffled text unless noted; spike counts; 3 input codes. Exploratory.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "figures" / "worm_q_x0.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
