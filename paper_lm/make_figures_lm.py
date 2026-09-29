"""Figures for the LM report: project timeline and 20k/5k results.

Numbers are read from the metrics / result JSON files, not typed in.
Palette: reference categorical slots (gray = earlier work, orange = n-gram,
blue = revised brain models); every bar is also labelled directly.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
LM = Path(r"E:\mechanism_20260928\lm")
OLD = Path(r"D:\flybrain_lm_cuda\models")
C1, C2, GRAY, INK, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def j(p):
    return json.load(open(p, encoding="utf-8"))


def fig_timeline():
    old = j(OLD / "flybrain_memoryrank512_skip3h32k_softmax.metrics.json")
    dual = j(OLD / "plastic_memory_lm_biological_dual_probe_20k_l2p1.metrics.json")
    exp7 = j(OLD / "exp7_metrics.json")
    stages = [
        ("Phase 1\nsynthetic tasks", "interface found:\ndense codes, K>=4\nidentity 0.95 (8 symbols)"),
        ("Phase 2\nfrozen brain LM", f"pilot {exp7.get('val_acc', 0.146):.1%}\n-> {old['val_acc']:.1%} / "
                                     f"{old['bpc']:.3f} BPC\n(+ hashed 3-char context)"),
        ("Phase 3\nMB plasticity", f"{dual['val_acc']:.1%} / {dual['bpc']:.3f} BPC\nwritten but not read"),
        ("Mechanism study", "memory on KC->MBON\nweights; sparse KC code\n= retrieval key; read out\nonly via MBON spikes"),
        ("Audit + revision", "indexing bug found;\nbenchmark ~ n-gram;\nsparse brain adds to it\nonly with little data"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.axis("off")
    for i, (t, d) in enumerate(stages):
        x = i * 1.0
        col = C1 if i >= 3 else GRAY
        ax.add_patch(plt.Rectangle((x + 0.04, 0.55), 0.92, 0.38, color=col, alpha=0.18, lw=0))
        ax.text(x + 0.5, 0.74, t, ha="center", va="center", fontsize=7.5, color=INK, weight="bold")
        ax.text(x + 0.5, 0.28, d, ha="center", va="center", fontsize=6.4, color=INK2)
        if i < len(stages) - 1:
            ax.annotate("", (x + 1.04, 0.74), (x + 0.96, 0.74),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_xlim(0, len(stages)); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_timeline.pdf"); fig.savefig(OUT / "fig1_timeline.png", dpi=200)
    plt.close(fig)


def fig_results():
    old = j(OLD / "flybrain_memoryrank512_skip3h32k_softmax.metrics.json")
    ctx = j(LM / "ctx_only_skip3_20k.json") if (LM / "ctx_only_skip3_20k.json").exists() else None
    rows = [
        ("earlier frozen reference (dense)", old["bpc"], old["val_acc"], GRAY),
        ("hashed 3-char context only", j(LM / "ctx_only_20k.json")["val_bpc"], j(LM / "ctx_only_20k.json")["val_acc"], C2),
        ("sparse 160: KC code + context", j(LM / "s160_skip3_kc_only.json")["val_bpc"], j(LM / "s160_skip3_kc_only.json")["val_acc"], C1),
        ("sparse 160: all features + context", j(LM / "s160_skip3_l25e-3.json")["val_bpc"], j(LM / "s160_skip3_l25e-3.json")["val_acc"], C1),
        ("sparse 192: all features + context", j(LM / "s192_frozen_20k_skip3_l23e-2.json")["val_bpc"], j(LM / "s192_frozen_20k_skip3_l23e-2.json")["val_acc"], C1),
        ("sparse 160: brain only", j(LM / "s160_frozen_20k_skip0_l21e-2.json")["val_bpc"], j(LM / "s160_frozen_20k_skip0_l21e-2.json")["val_acc"], C1),
        ("sparse 192 + plasticity, teacher every token", j(LM / "s192_biopulse_20k_skip3_l21e-2.json")["val_bpc"], j(LM / "s192_biopulse_20k_skip3_l21e-2.json")["val_acc"], C1),
    ]
    big = j(LM / "ctx_only_20k.json")["bigram_bpc"]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for i, (lab, b, a, col) in enumerate(rows):
        ax.barh(i, b - 3.0, left=3.0, color=col, height=0.62)
        ax.text(b + 0.01, i, f"{b:.3f}  ({a:.1%})", va="center", fontsize=6.3, color=INK2)
    ax.axvline(rows[1][1], color=C2, lw=0.8, ls="--")
    ax.axvline(big, color=MUTED, lw=0.8, ls=":")
    ax.text(big + 0.01, -0.55, "bigram", color=INK2, fontsize=6, ha="left")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(3.0, 3.85)
    ax.set_xlabel("validation BPC, 20k/5k (lower is better)\naccuracy in brackets")
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_results20k.pdf"); fig.savefig(OUT / "fig2_results20k.png", dpi=200)
    plt.close(fig)


def fig_paired():
    """Brain+context minus context-only BPC, holdout-selected, per cache
    (3 text segments x 3 PN-code seeds), from night/select_*.json."""
    night = Path(r"E:\mechanism_20260928\night")
    groups = [("sparse 160", "s160", C1), ("sparse 192", "s192", C1), ("dense 512", "d512", GRAY)]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, (lab, tag, col) in enumerate(groups):
        rs = j(night / f"select_{tag}.json")
        d = [r["brain+context"]["val_bpc"] - r["context_only"]["val_bpc"] for r in rs]
        offs = {0: -0.18, 300000: 0.0, 600000: 0.18}
        for r, v in zip(rs, d):
            ax.plot(i + offs[r["offset"]], v, "o", ms=4.5, color=col, mec="white", mew=0.8)
        m = sum(d) / len(d)
        ax.plot([i - 0.3, i + 0.3], [m, m], color=INK, lw=1.4)
        wins = sum(v < 0 for v in d)
        ax.text(i, 0.045, f"{m:+.3f}\n{wins}/9 better", ha="center", va="bottom", fontsize=6.3, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4); ax.set_ylim(-0.13, 0.09)
    ax.set_ylabel("BPC change from adding the brain\n(below 0 = brain helps)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_paired.pdf"); fig.savefig(OUT / "fig2_paired.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_timeline(); fig_results(); fig_paired(); print("ok")
