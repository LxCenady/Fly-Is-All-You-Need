"""Memory curve: readout retention vs preset synaptic decay.

For each tau/pair/probe: values normalised to the first checkpoint (8 tokens),
next to exp(-(t - t0)/tau).  Effective tau = -1/slope of log(D) vs t over the
checkpoints where D is above 10x the per-element replay floor scale.
"""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1], encoding="utf-8"))
lv = sys.argv[2] if len(sys.argv) > 2 else "mbon"
runs = d["runs"]
keys = sorted({(r["tau"], tuple(r["pair"])) for r in runs})
for tau, pair in keys:
    rs = sorted([r for r in runs if r["tau"] == tau and tuple(r["pair"]) == pair],
                key=lambda r: r["gap_tokens"])
    t = np.array([r["t_s"] for r in rs])
    md = np.array([r["mod_diff"] for r in rs])
    pred = np.exp(-(t - t[0]) / tau)
    print(f"\ntau={tau:g}s pair={''.join(pair)}  t(s): " + " ".join(f"{x:7.2f}" for x in t))
    print(f"  {'preset exp':>16}: " + " ".join(f"{x:7.3f}" for x in pred))
    print(f"  {'|mX-mY| rel':>16}: " + " ".join(f"{x:7.3f}" for x in md / md[0]))
    for pch in (pair[0], pair[1], "z"):
        for q in ("D", "W"):
            v = np.array([r[f"{lv}|{pch}|{q}"] for r in rs])
            ok = v > 1e-5
            tau_eff = float("nan")
            if ok.sum() >= 3:
                sl = np.polyfit(t[ok], np.log(v[ok]), 1)[0]
                tau_eff = -1 / sl if sl < 0 else float("inf")
            print(f"  {pch}:{q} rel (abs0 {v[0]:.2e}): " + " ".join(f"{x:7.3f}" for x in v / v[0])
                  + f"   tau_eff={tau_eff:.1f}s")
