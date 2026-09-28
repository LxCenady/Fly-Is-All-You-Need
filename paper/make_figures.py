"""Figures for paper_v2 (vector PDF, one column = 3.5 in).

Palette: reference categorical slots 1-3 (blue, orange, aqua), fixed order;
nulls in muted gray.  Every series also has its own marker and a legend or
direct label, so identity never rests on colour alone.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E = Path(r"E:\mechanism_20260928")
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.5,
    "lines.markersize": 4.5, "pdf.fonttype": 42, "figure.dpi": 150})
W = 3.5


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def load(name):
    return json.load(open(E / name, encoding="utf-8"))


# ---- Fig 1: transplant decomposition -------------------------------------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(W * 2, 2.1))
    for ax, (fn, title) in zip(axes, [("m1_cross.json", "Legacy input (512 PNs)"),
                                      ("m1_cross_a192.json", "Sparse input (192 PNs)")]):
        d = load(fn)
        gaps = sorted({c["gap"] for c in d["cells"]})
        for key, col, mk, lab in (("D_m", C1, "o", "swap synaptic state $m$ only"),
                                  ("D_s", C2, "s", "swap fast state $s$ only")):
            vals = [[c["stats"]["mbon"][key] / c["stats"]["mbon"]["D_nat"]
                     for c in d["cells"] if c["gap"] == g] for g in gaps]
            med = [np.median(v) for v in vals]
            x = np.arange(len(gaps))
            for i, v in enumerate(vals):
                ax.scatter(np.full(len(v), x[i]) + (0.08 if key == "D_s" else -0.08), v,
                           s=9, color=col, alpha=0.35, linewidths=0)
            ax.plot(x, med, color=col, marker=mk, label=lab)
        ax.set_xticks(np.arange(len(gaps)), [f"{g}\n({g * 0.12:.2f} s)" for g in gaps])
        ax.set_xlabel("gap after write, tokens")
        ax.set_ylabel("fraction of content difference")
        ax.set_ylim(-0.05, 1.12)
        ax.set_title(title, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / "fig1_transplant.pdf")
    fig.savefig(OUT / "fig1_transplant.png", dpi=200)
    plt.close(fig)


# ---- Fig 2: sparse memory curve with spike gating ------------------------
def fig2():
    d = load("m4_curve_a192.json")
    rs = sorted([r for r in d["runs"] if r["tau"] == 16.0 and r["pair"] == ["a", "c"]],
                key=lambda r: r["gap_tokens"])
    t = np.array([r["t_s"] for r in rs])
    syn = np.array([r["mod_diff"] for r in rs]); syn /= syn[0]
    vm = np.array([r["mbon|a|D"] for r in rs]); vm /= vm[0]
    vc = np.array([r["central|a|D"] for r in rs]); vc /= vc[0]
    spk = [r["spk|a|X0"] for r in rs]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.plot(t, syn, color=MUTED, marker="^", label="synapse (= input current)")
    ax.plot(t, vm, color=C1, marker="o", label="MBON voltage")
    ax.plot(t, vc, color=C2, marker="s", label="downstream (central)")
    ax.set_xscale("log")
    ax.set_xlabel("time after write, s")
    ax.set_ylabel("relative to 0.96 s")
    ax.set_ylim(-0.05, 1.1)
    first_loss = next(i for i, s in enumerate(spk) if s[0] == s[1])
    ax.axvspan(t[first_loss - 1], t[first_loss], color=GRID, alpha=0.6, lw=0)
    ax.text(np.sqrt(t[first_loss - 1] * t[first_loss]), 0.55, "extra MBON\nspike lost",
            ha="center", va="center", color=INK2, fontsize=6.5)
    ax.legend(loc="lower left", fontsize=6.5, bbox_to_anchor=(0.0, 0.02))
    ax.set_title(r"Sparse input, $\tau_w=16$ s, cue = written item", color=INK)
    save(fig, "fig2_curve_gating")


# ---- Fig 3: operating point sweep ----------------------------------------
def fig3():
    rows = [r for f in glob.glob(str(E / "sweep_s1.5.json")) for r in load(Path(f).name)]
    rows.sort(key=lambda r: r["active"])
    n = [r["active"] for r in rows]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.plot(n, [r["density"]["probe"]["kc_frac"] for r in rows], color=C1, marker="o",
            label="KCs active at probe")
    ax.plot(n, [r["density"]["probe"]["kc_jaccard"] for r in rows], color=C2, marker="s",
            label="KC overlap between symbols (Jaccard)")
    ax.axvspan(150, 200, color=GRID, alpha=0.6, lw=0)
    ax.text(175, 0.97, "sparse", ha="center", va="top", color=INK2, fontsize=6.5)
    ax.scatter([512], [0.87], color=C1, marker="o", zorder=3)
    ax.scatter([512], [0.93], color=C2, marker="s", zorder=3)
    ax.annotate("legacy\n(scale 1.0)", (512, 0.87), (470, 0.55), fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("PNs per symbol (of 675)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=6.5, bbox_to_anchor=(1.0, 0.02))
    save(fig, "fig3_operating_point")


# ---- Fig 4: addressing (6 pairs) -----------------------------------------
def fig4():
    d = load("m2_current_6pairs.json")
    fig, axes = plt.subplots(1, 2, figsize=(W * 2, 2.1))
    labels = []
    for i, pt in enumerate(d):
        own = [c for c in pt["cells"] if c["probe"] == c["pair"][0]]
        spec = []
        for c in own:
            oth = [cc for cc in pt["cells"] if cc["pair"] == c["pair"] and cc["probe"] != c["probe"]]
            spec.append(c["I_x_norm"] / np.mean([cc["I_x_norm"] for cc in oth]))
        kc = [c["scramble"]["within_mbon"]["median"][0] for c in own]
        rt = [c["scramble"]["within_kc"]["median"][0] for c in own]
        labels.append(f"{pt['active']} PNs")
        jit = np.linspace(-0.12, 0.12, len(spec))
        axes[0].scatter(i + jit, spec, color=C1, s=14, zorder=3)
        axes[0].plot([i - 0.2, i + 0.2], [np.median(spec)] * 2, color=INK, lw=1)
        axes[1].scatter(i - 0.15 + jit / 2, kc, color=C1, marker="o", s=14, zorder=3,
                        label="shuffle KC identity (keep MBON totals)" if i == 0 else None)
        axes[1].scatter(i + 0.15 + jit / 2, rt, color=C2, marker="s", s=14, zorder=3,
                        label="shuffle MBON routing (keep KC totals)" if i == 0 else None)
    axes[0].set_yscale("log")
    axes[0].axhline(1, color=MUTED, lw=0.8, ls="--")
    axes[0].set_ylabel("cue specificity\n(written item / other probes)")
    axes[1].set_ylabel("cued readout kept after shuffle")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower right", fontsize=6.5)
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels)
    axes[0].set_title("Six write pairs", color=INK)
    axes[1].set_title("Which wiring carries the cued readout", color=INK)
    save(fig, "fig4_addressing")


# ---- Fig 5: output distinctness under rewiring ---------------------------
def fig5():
    rand, glom = [], []
    for fn in ("m3_kcmbon_levels.json", "m3_kcmbon_profile.json", "m3_kcmbon_pninput.json",
               "m3_kcmbon.json", "m6_side_a160.json", "m6_side_a192.json", "m6_glom_route.json"):
        if (E / fn).exists():
            for r in load(fn):
                (glom if r.get("input_code") == "glom" else rand).append(r)
    order = [("real", "connectome"),
             ("profile_type_side", "swap KC output profiles: same subtype, same side"),
             ("profile_type", "swap KC output profiles: same subtype, sides mixed"),
             ("profile_side", "swap KC output profiles: same side only"),
             ("kc_perm_side", "shuffle KC→MBON within side"),
             ("kc_perm", "shuffle KC→MBON, all"),
             ("pn_perm_type_side", "shuffle PN→KC: same PN type, same side"),
             ("pn_perm_side", "shuffle PN→KC within side"),
             ("pn_perm_type", "shuffle PN→KC: same PN type, sides mixed"),
             ("pn_perm", "shuffle PN→KC, all")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, rows, title in ((axes[0], rand, "Side-blind random symbol codes"),
                            (axes[1], glom, "Bilateral glomerular symbol codes")):
        pts = ((160, C1, "o"), (192, C2, "s"), (224, C3, "D"))
        for j, (active, col, mk) in enumerate(pts):
            for i, (mode, _) in enumerate(order):
                v = sorted({(r["seed"], r["item_cos_mean"]) for r in rows
                            if r["active"] == active and r["mode"] == mode})
                if v:
                    ax.scatter([x[1] for x in v], np.full(len(v), i) + (j - 1) * 0.22, color=col,
                               marker=mk, s=14, zorder=3, label=f"{active} PNs")
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), loc="lower left", fontsize=6.5)
        ax.set_xlim(0.7, 1.0)
        ax.set_xlabel("item-to-item output cosine\n(lower = more distinct)")
        ax.set_title(title, color=INK)
    axes[0].set_yticks(range(len(order)), [o[1] for o in order], fontsize=6.3)
    axes[0].invert_yaxis()
    save(fig, "fig5_routing")


# ---- Fig 6: spike gating --------------------------------------------------
def fig6():
    d = load("m5b_timing.json")
    cls = [("identical", "MBON rasters identical"), ("timing_only", "spike moved,\nsame count"),
           ("count_diff", "spike count changed")]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for i, (k, lab) in enumerate(cls):
        w = np.array([max(r["W_central"], 1e-9) for r in d if r["class"] == k])
        x = i + np.random.default_rng(0).uniform(-0.18, 0.18, len(w))
        ax.scatter(x, w, s=10, color=[MUTED, C2, C1][i], marker=["o", "s", "D"][i],
                   alpha=0.8, linewidths=0)
        ax.text(i, 3, f"n={len(w)}", ha="center", color=INK2, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(5e-10, 8)
    ax.axhline(1e-4, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(range(3), [c[1] for c in cls], fontsize=6.5)
    ax.set_ylabel("downstream change\n(central, write vs no write)")
    ax.set_title("Downstream change vs MBON raster", color=INK)
    save(fig, "fig6_spike_gating")


# ---- Fig 7: output map (subtype -> MBON) and item placement -------------
def fig7():
    d = load("m7_map.json")
    subs = d["subtypes"]
    M = np.asarray(d["subtype_to_mbon_share"])
    main = d["mbon_main_subtype"]
    names = d["mbon_names"]
    # order MBONs by their main subtype, then by weight from it
    order = sorted([m for m in range(len(names)) if main[m] is not None],
                   key=lambda m: (subs.index(main[m]), -M[subs.index(main[m]), m]))
    S = np.asarray(d["item_output_share"])[:, order]
    items = [c["item"] for c in d["item_top_mbons"]]
    lab = [s.replace("KCapbp", "KCa'b'") for s in subs]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), height_ratios=[len(subs), 13],
                             sharex=True)
    im0 = axes[0].imshow(M[:, order], aspect="auto", cmap="Blues", interpolation="nearest")
    axes[0].set_yticks(range(len(subs)), lab, fontsize=6.5)
    axes[0].set_title("Share of each KC subtype's output weight per MBON", color=INK)
    im1 = axes[1].imshow(S[::2], aspect="auto", cmap="Blues", interpolation="nearest")
    axes[1].set_yticks(range(len(items[::2])), items[::2], fontsize=6.5)
    axes[1].set_title("Share of each item's MBON output (every second item shown)", color=INK)
    bounds = [i for i in range(1, len(order)) if main[order[i]] != main[order[i - 1]]]
    for ax in axes:
        ax.grid(False)
        for x in bounds:
            ax.axvline(x - 0.5, color=MUTED, lw=0.6)
    centers = []
    start = 0
    for x in bounds + [len(order)]:
        centers.append((start + x - 1) / 2)
        start = x
    groups = [main[order[0]]] + [main[order[b]] for b in bounds]
    axes[1].set_xticks(centers, [g.replace("KCapbp", "a'b'").replace("KC", "") for g in groups],
                       fontsize=6.5)
    axes[1].set_xlabel("MBONs grouped by the KC subtype that supplies most of their input")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)
    save(fig, "fig7_output_map")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print("ok", f.__name__)
