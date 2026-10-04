"""Which variable predicts memory best? One job: results + ablation json -> predictors.json (+ a table on stdout).

Every row is one measurement (connectome, condition, code or rewiring seed, readout draw, M).
Target: decoding accuracy of the character k = 1 back, for counts + traces ("both") and counts
only ("spikes", the network's own memory). For each candidate variable:
  rho     Spearman correlation over all rows
  loco    leave-one-connectome-out R^2 of a 1-variable linear fit (fit on two species, predict
          the third; mean over the three held-out species). Negative = worse than the mean.
Rows are not independent (draws share simulations), so treat rho as descriptive.

    python predictors.py
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
BOTH = ["results_worm.json", "results_larva.json", "results_fly.json", "results_tau_worm.json",
        "results_tau_larva.json", "ablation/both_fly_tau20.json", "ablation/both_fly_new.json"]
SPIKES = ["ablation/spikes.json", "ablation/spikes_fly_new.json"]
EXCLUDE = re.compile(r"worm_rewire_s2|larvaactive_rewire|larva_in9|larva_tau05")
SPECIES = {"worm": dict(n=448, inh=0.079), "larva": dict(n=2952, inh=0.064),
           "fly": dict(n=166700, inh=0.448)}
DRIVEN = {"worm": 9, "larva": 60, "fly": 3570}


def meta(stem):
    sp = "larva" if stem.startswith("larva") else stem.split("_")[0]
    tau = 0.05 if "tau05" in stem else 0.2 if "tau20" in stem else 0.1
    driven = 9 if "in9" in stem else DRIVEN[sp]
    return sp, dict(tau=tau, driven=driven, rewired=float("rewire" in stem), **SPECIES[sp])


def rows(files):
    out, seen = [], set()
    for f in files:
        for r in json.load(open(HERE / f, encoding="utf-8")):
            stem = r["file"].removesuffix(".npz")
            key = (stem, r["M"], r["draw"])
            if EXCLUDE.search(stem) or key in seen:
                continue
            seen.add(key)
            sp, m = meta(stem)
            active_n = r["M"] * (1 - r["silent_share"])
            out.append(dict(species=sp, y=r["span"][1], k0=r["span"][0], logM=np.log10(r["M"]),
                            log_active_readout=np.log10(max(active_n, 1)),
                            readout_activity=r["readout_active"], silent_share=r["silent_share"],
                            log_driven=np.log10(m["driven"]), tau=m["tau"], log_N=np.log10(m["n"]),
                            inhibitory_share=m["inh"], rewired=m["rewired"]))
    return out


VARS = ["log_active_readout", "logM", "readout_activity", "silent_share", "log_driven", "tau",
        "log_N", "inhibitory_share", "rewired", "k0"]


def loco(R, v):
    scores = []
    for held in SPECIES:
        tr = [r for r in R if r["species"] != held]
        te = [r for r in R if r["species"] == held]
        x, y = np.array([r[v] for r in tr]), np.array([r["y"] for r in tr])
        if np.ptp(x) == 0 or not te:
            continue
        a, b = np.polyfit(x, y, 1)
        yt = np.array([r["y"] for r in te])
        pred = a * np.array([r[v] for r in te]) + b
        scores.append(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    return float(np.mean(scores)) if scores else None


def main():
    res = {}
    for name, files in (("both", BOTH), ("spikes", SPIKES)):
        R = rows(files)
        res[name] = {"rows": len(R)}
        print(f"\n== target: k=1 accuracy, features {name} ({len(R)} rows)")
        print(f"{'variable':22s} {'rho':>6s} {'LOCO R2':>8s}")
        for v in VARS:
            x = [r[v] for r in R]
            rho = spearmanr(x, [r["y"] for r in R]).statistic if np.ptp(x) else float("nan")
            lc = loco(R, v)
            res[name][v] = {"rho": round(float(rho), 3), "loco_r2": None if lc is None else round(lc, 3)}
            print(f"{v:22s} {rho:6.2f} {lc if lc is not None else float('nan'):8.2f}")
    (HERE / "predictors.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
