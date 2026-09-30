"""Summary statistics with 95% confidence intervals for the key mechanism results, over
independent input-code seeds (repro/replicate.json; confirmatory: repro/confirm.json).

    python repro/replication_stats.py RUNDIR [--ref results] [--out repro/REPLICATION.md] [--json F]

Unit of replication = one input-code seed (a fresh set of random PN codes, or glomerular codes
for the teacher experiment): write pairs within a seed are not independent, so intervals come
from a cluster bootstrap: resample seeds with replacement, then pairs within each resampled
seed (10,000 resamples, percentile interval, seed 0).  Metrics are computed exactly as in the
paper figures (paper/make_figures.py).  The reference run (code seed 3 / glom seed 11, in
results/) is reported next to the replications but not pooled with them.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = 10_000
FLOOR_CENTRAL = 1e-4       # "downstream changed": above the dashed line of Fig. 7 (float floor ~2e-7)


def boot(groups, stat=np.median, rng=None):
    """Cluster bootstrap of stat(pooled values); groups = list of 1-D arrays (one per seed)."""
    rng = rng or np.random.default_rng(0)
    groups = [np.asarray(g, float) for g in groups if len(g)]
    est = float(stat(np.concatenate(groups)))
    if len(groups) < 2:
        return est, float("nan"), float("nan")
    vals = np.empty(B)
    for i in range(B):
        gs = [groups[j] for j in rng.integers(0, len(groups), len(groups))]
        vals[i] = stat(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in gs]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def seed_of(name):
    m = re.search(r"_[cg](\d+)\.json$", name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- per-claim metrics
def m8(d):
    """Per operating point: cue specificity and KC-identity-scramble kept fraction, one per pair."""
    out = {}
    for pt in d:
        own = [c for c in pt["cells"] if c["probe"] == c["pair"][0]]
        spec, kept = [], []
        for c in own:
            oth = [cc for cc in pt["cells"] if cc["pair"] == c["pair"] and cc["probe"] != c["probe"]]
            spec.append(c["I_x_norm"] / np.mean([cc["I_x_norm"] for cc in oth]))
            kept.append(c["scramble"]["within_mbon"]["median"][0])
        out[pt["active"]] = {"spec": spec, "kept": kept}
    return out


def m10(d):
    """Per operating point: item cosine (real), change after side+subtype swap, after KC->MBON shuffle."""
    by = {}
    for r in d:
        by.setdefault(r["active"], {})[r["mode"]] = r["item_cos_mean"]
    return {a: {"real": g["real"], "d_swap": g["profile_type_side"] - g["real"], "d_perm": g["kc_perm"] - g["real"]}
            for a, g in by.items() if {"real", "profile_type_side", "kc_perm"} <= set(g)}


def m16(d):
    return {"teacher": d["variance_share"]["teacher"], "item": d["variance_share"]["item"]}


def m19(d):
    """Cells (pair x probe x bias) by whether the write changed the MBON spike count, and whether
    the central (downstream) response changed."""
    rows = [r for pt in d for r in pt["rows"]]
    ch = [r for r in rows if r["mbon_spikes"]["X"] != r["mbon_spikes"]["0"]]
    un = [r for r in rows if r["mbon_spikes"]["X"] == r["mbon_spikes"]["0"]]
    return {"changed": [float(r["W_central"] > FLOOR_CENTRAL) for r in ch],
            "unchanged": [float(r["W_central"] > FLOOR_CENTRAL) for r in un]}


def m2(d):
    """D_m / D_nat at the MBON level for gaps >= 8 tokens (one value per pair x probe x gap)."""
    return [c["stats"]["mbon"]["D_m"] / c["stats"]["mbon"]["D_nat"] for c in d["cells"]
            if c["gap"] >= 8 and c["stats"]["mbon"]["D_nat"] > 0]


# ---------------------------------------------------------------- report
def summarise(rundir: Path, ref: Path):
    files = {p.name: p for p in Path(rundir).glob("*.json") if not p.name.endswith("manifest.json")}
    res, lines = {}, []

    def group(prefix):
        return sorted(((seed_of(n), load(p)) for n, p in files.items() if n.startswith(prefix) and seed_of(n)),
                      key=lambda t: t[0])

    # M8
    g = group("m2_")
    if g:
        refm = m8(load(ref / "m2_current_6pairs.json"))
        lines += ["## M8: KC identity is the retrieval key (sparse input)", "",
                  f"{len(g)} input-code seeds ({', '.join(str(s) for s, _ in g)}) x 6 write pairs. "
                  "Cue specificity = current for the written item / mean of the other probes; "
                  "kept = fraction of the cued readout left after shuffling KC identity within each MBON.", "",
                  "| PNs | cue specificity, median [95% CI] | reference (seed 3) | KC-identity shuffle keeps, median [95% CI] | reference |",
                  "|---|---|---|---|---|"]
        for a in (160, 192, 512):
            spec = [m8(d)[a]["spec"] for _, d in g]; kept = [m8(d)[a]["kept"] for _, d in g]
            e1, l1, h1 = boot(spec); e2, l2, h2 = boot(kept)
            res[f"M8_{a}"] = {"spec": [e1, l1, h1], "kept": [e2, l2, h2],
                              "per_seed_spec_median": [float(np.median(s)) for s in spec]}
            lines.append(f"| {a} | {e1:.2f} [{l1:.2f}, {h1:.2f}] | {np.median(refm[a]['spec']):.2f} | "
                         f"{e2:.2f} [{l2:.2f}, {h2:.2f}] | {np.median(refm[a]['kept']):.2f} |")
        lines.append("")
    # M10
    g = group("route_")
    if g:
        refm = {r_a: v for r_a, v in m10(load(ref / "m3_kcmbon.json")).items()}
        refs = {}
        for f in ("m6_side_a160.json", "m6_side_a192.json"):
            refs.update(m10(load(ref / f)) if (ref / f).exists() else {})
        lines += ["## M10: output identity needs body side and KC subtype, not individual wiring", "",
                  f"{len(g)} input-code seeds. Item cosine = mean pairwise cosine of the 8 items' MBON currents "
                  "(lower = more distinct). Changes relative to the real wiring.", "",
                  "| PNs | real | side+subtype swap: change [95% CI] | KC->MBON shuffle: change [95% CI] | reference changes |",
                  "|---|---|---|---|---|"]
        for a in (160, 192):
            vals = [m10(d)[a] for _, d in g if a in m10(d)]
            real = boot([[v["real"] for v in vals]], np.mean)
            ds = boot([[v["d_swap"] for v in vals]], np.mean); dp = boot([[v["d_perm"] for v in vals]], np.mean)
            res[f"M10_{a}"] = {"real": real, "d_swap": ds, "d_perm": dp}
            rr = refs.get(a, {})
            lines.append(f"| {a} | {real[0]:.3f} | {ds[0]:+.3f} [{ds[1]:+.3f}, {ds[2]:+.3f}] | "
                         f"{dp[0]:+.3f} [{dp[1]:+.3f}, {dp[2]:+.3f}] | swap {rr.get('d_swap', float('nan')):+.3f}, "
                         f"shuffle {refm.get(a, {}).get('d_perm', float('nan')):+.3f} |")
        lines.append("")
    # M16
    g = group("teacher_")
    if g:
        t = [[m16(d)["teacher"]] for _, d in g]
        e, lo, hi = boot(t, np.mean)
        refv = m16(load(ref / "m7_teacher.json"))
        res["M16"] = {"teacher_share": [e, lo, hi], "per_seed": [x[0] for x in t]}
        lines += ["## M16: when teachers differ, the teacher decides the compartment", "",
                  f"{len(g)} glomerular-code seeds. Share of the variance of the unit output vectors explained by the teacher.", "",
                  f"Teacher share: mean {e:.2f} [95% CI {lo:.2f}, {hi:.2f}] (per seed: "
                  f"{', '.join(f'{x[0]:.2f}' for x in t)}); reference (seed 11) {refv['teacher']:.2f}.", ""]
    # M19
    g = group("down_")
    if g:
        ch = [m19(d)["changed"] for _, d in g]; un = [m19(d)["unchanged"] for _, d in g]
        e1, l1, h1 = boot(ch, np.mean); e2, l2, h2 = boot(un, np.mean)
        nch, nun = sum(len(x) for x in ch), sum(len(x) for x in un)
        res["M19"] = {"p_central_if_mbon_changed": [e1, l1, h1], "p_central_if_unchanged": [e2, l2, h2],
                      "n_changed": nch, "n_unchanged": nun}
        lines += ["## M19: downstream neurons see the memory only through MBON spikes", "",
                  f"{len(g)} input-code seeds, 4 operating points x 5 bias levels x 2 pairs x 2 probes each. "
                  "Spike-count changes only (the reference also counts moved spikes, m5b_timing).", "",
                  f"Central response changed when the MBON spike count changed: {e1:.2f} [95% CI {l1:.2f}, {h1:.2f}] "
                  f"({int(round(e1 * nch))}/{nch} cells); when it did not: {e2:.3f} [{l2:.3f}, {h2:.3f}] "
                  f"({int(round(e2 * nun))}/{nun}).", ""]
    # M2
    g = group("m1x_")
    if g:
        v = [m2(d) for _, d in g]
        e, lo, hi = boot(v, np.median)
        res["M2"] = {"Dm_over_Dnat_gap8plus": [e, lo, hi], "min": float(min(min(x) for x in v))}
        lines += ["## M2: from ~1 s after writing, the memory is carried by the KC->MBON weights", "",
                  f"{len(g)} input-code seeds at 192 PNs. D_m / D_nat at the MBON level (swap only the synaptic state), "
                  "gaps of 8 and 32 tokens.", "",
                  f"Median {e:.4f} [95% CI {lo:.4f}, {hi:.4f}], minimum {res['M2']['min']:.4f}.", ""]
    return res, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--ref", default=str(ROOT / "results"))
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--title", default="Replications with fresh input codes (exploratory)")
    a = ap.parse_args()
    res, lines = summarise(Path(a.rundir), Path(a.ref))
    text = "\n".join([f"# {a.title}", "", "Generated by repro/replication_stats.py from " + Path(a.rundir).name + ".", ""] + lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
