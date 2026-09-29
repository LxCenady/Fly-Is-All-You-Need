"""Summarise the sensitivity queue (E:\\mechanism_20260928\\sens) -> SENS_SUMMARY.md."""
import json
from pathlib import Path

import numpy as np

S = Path(r"E:\mechanism_20260928\sens")
VARS = ["base", "gain1.2", "gain1.8", "tonic0.03", "tonic0.08", "noise2", "noise5"]


def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


L = ["# Sensitivity summary", "",
     "## Output identity (item_cos_mean; lower = items more distinct)", "",
     "| variant | code | real | side+subtype swap | subtype swap (sides mixed) | kc_perm | top5 real |",
     "|---|---|---|---|---|---|---|"]
for v in VARS:
    for tag, lab in (("route160r", "160 random"), ("route192g", "192 glom")):
        d = load(S / f"{tag}_{v}.json")
        if not d:
            L.append(f"| {v} | {lab} | n/a | | | | |"); continue
        g = {r["mode"]: r for r in d}
        f = lambda k, key="item_cos_mean": f"{g[k][key]:.3f}" if k in g and g[k].get(key) is not None else "-"
        L.append(f"| {v} | {lab} | {f('real')} | {f('profile_type_side')} | {f('profile_type')} | "
                 f"{f('kc_perm')} | {f('real', 'mbon_share_top5')} |")

L += ["", "## Retrieval key (M2, 6 pairs; cue probe = written item x)", "",
      "| variant | point | cue specificity (median) | KC-identity scramble: magnitude kept | direction cos | content cos |",
      "|---|---|---|---|---|---|"]
for v in VARS:
    d = load(S / f"m2_{v}.json")
    if not d:
        L.append(f"| {v} | n/a | | | | |"); continue
    for pt in d:
        cells = pt["cells"]
        spec = []
        for pair in {tuple(c["pair"]) for c in cells}:
            cs = {c["probe"]: c["I_x_norm"] for c in cells if tuple(c["pair"]) == pair}
            others = [x for p, x in cs.items() if p != pair[0]]
            if others and np.mean(others) > 0:
                spec.append(cs[pair[0]] / np.mean(others))
        xs = [c for c in cells if c["probe"] == c["pair"][0]]
        med = np.array([c["scramble"]["within_mbon"]["median"] for c in xs], float)
        mm = np.nanmedian(med, 0) if np.isfinite(med).any() else [np.nan] * 4
        L.append(f"| {v} | {pt['active']}x{pt['scale']} | {np.median(spec) if spec else float('nan'):.2f} | "
                 f"{mm[0]:.2f} | {mm[1]:.2f} | {mm[3]:.2f} |")
(S / "SENS_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
