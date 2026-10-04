"""Worm vs fly encoding figure. One job: results_x0inp_worm_x0.json + results_x0_fly_x0.json -> figures/x0_species.png.

A  encoding vs fraction of the input population driven per character
B  encoding vs readout activity, all conditions, marked by what was changed
C  residue of the previous character vs encoding
M = 300 readout neurons, spike counts, shuffled text, 3 input codes.

    python plot_x0_species.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SPECIES = {"worm": ("results_x0inp_worm_x0.json", 36, 9, "#0072B2", "C. elegans (36 input neurons)"),
           "fly": ("results_x0_flyinp_x0.json", 14876, 3570, "#009E73", "Adult fly (14,876 input neurons)")}
KIND = {"act": ("o", "driven inputs"), "base": ("*", "base"), "g": ("s", "recurrent gain"),
        "sus": ("^", "sustain"), "drv": ("D", "drive strength")}


def kind(cond):
    for k in ("act", "sus", "drv", "g", "base"):
        if cond.startswith(k):
            return k
    return "base"


def load(fname):
    t = json.load(open(HERE / fname, encoding="utf-8"))
    return {k.split("|")[0]: v for k, v in t.items() if k.endswith("|300")}


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for sp, (fname, n_in, base_act, col, label) in SPECIES.items():
        d = load(fname)
        acts = sorted([(int(c[3:]), v) for c, v in d.items() if c.startswith("act")] + [(base_act, d["base"])],
                      key=lambda t: t[0])
        acts = [(a, v) for a, v in acts if v["x0"] > 0.01]           # x0 = 0 cannot be drawn on a log axis
        ax[0].plot([a / n_in for a, _ in acts], [v["x0"] for _, v in acts], "-o", color=col, ms=4,
                   lw=1.8, label=label)
        for c, v in d.items():
            mk, _ = KIND[kind(c)]
            if v["x0"] <= 0.01:
                continue
            ax[1].scatter(v["activity"], v["x0"], marker=mk, s=40 if mk != "*" else 110,
                          color=col, edgecolors="white", linewidths=0.6, zorder=3)
            if v["x0"] > 0.02:
                ax[2].scatter(v["x0"], v["x1"] / v["x0"], marker=mk, s=40 if mk != "*" else 110, color=col,
                              edgecolors="white", linewidths=0.6, zorder=3)
    ax[0].set(xscale="log", yscale="log", xlabel="fraction of input neurons driven per character",
              ylabel="encoding x0 (log)", title="A. Both peak when about half the inputs are driven", ylim=(0.05, 30))
    ax[0].text(0.03, 0.06, "x0 = 0: fly driving 9 neurons (network silent);\n"
               "worm driving all 36 (no difference between characters)",
               fontsize=7.5, color="#444", transform=ax[0].transAxes)
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set(xscale="log", yscale="log", xlabel="readout neurons firing per character (fraction, log)",
              ylabel="encoding x0 (log)", ylim=(0.05, 30),
              title="B. Fly encoding follows activity; worm's does not")
    for k, (mk, lab) in KIND.items():
        ax[1].scatter([], [], marker=mk, color="#666", label=lab)
    ax[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax[2].set(xscale="log", xlabel="encoding x0 (log)", ylabel="residue / encoding (x1 / x0)",
              title="C. The previous character keeps ~5-20 % of the encoding", ylim=(0, 0.6))
    ax[2].text(0.55, 0.53, "fly, drive 0.75 (below threshold): slow, delayed response",
               fontsize=7.5, color="#444", va="top")
    fig.suptitle("Encoding strength x0 in two connectome reservoirs (M = 300, spike counts, shuffled text). "
                 "Colours: species; marker: what was changed. Exploratory.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "figures" / "x0_species.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
