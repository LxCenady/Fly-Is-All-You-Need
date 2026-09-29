"""Summarise the overnight queue (FLY_OUT/night) into SUMMARY.md.

Safe to run on partial results: missing files are listed, not guessed.
Paired statistic: delta = BPC(brain+context) - BPC(context only) on the same
segment and code seed (negative = the brain helps).
"""
import json
import statistics as st
from pathlib import Path

import paths  # noqa: E402
N = paths.OUT / "night"
OUT = N / "SUMMARY.md"
CODES = ["s160", "s192", "d512"]
OFFS = [0, 300000, 600000]
SEEDS = [-1, 11, 23]


def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ms(xs):
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}"


lines = ["# Overnight summary", ""]
status = load(N / "logs" / "queue_status.json") or {}
lines += [f"queue status at {status.get('time')}: running {status.get('running')}, "
          f"{len(status.get('remaining', []))} remaining", ""]

# 1. fixed-config runs (all features + context, L2 1e-2; settings not selected)
lines += ["## 1. Fixed config (all features + context, L2 1e-2), 20k/5k", "",
          "| code | runs | val BPC | val acc | train acc |", "|---|---|---|---|---|"]
missing = []
for c in CODES:
    rs = []
    for o in OFFS:
        for s in SEEDS:
            r = load(N / f"{c}_o{o}_c{s}.json")
            (rs.append(r) if r else missing.append(f"{c}_o{o}_c{s}"))
    lines.append(f"| {c} | {len(rs)}/9 | {ms([r['val_bpc'] for r in rs])} | "
                 f"{ms([r['val_acc'] for r in rs])} | {ms([r['train_acc'] for r in rs])} |")
lines += ["", f"missing: {', '.join(missing) or 'none'}", ""]

# 2. holdout-selected runs (selection never sees validation)
lines += ["## 2. Holdout-selected (lm_select), paired against context only", ""]
for tag in [f"select_{c}" for c in CODES] + ["select_s160_100k", "select_d512_100k"]:
    rs = load(N / f"{tag}.json")
    if not rs:
        lines += [f"- {tag}: not available", ""]
        continue
    lines += [f"### {tag} ({len(rs)} caches)", "",
              "| segment | seed | selected | brain+ctx BPC | ctx-only BPC | brain-only BPC | delta |",
              "|---|---|---|---|---|---|---|"]
    d = []
    for r in rs:
        name = Path(r["cache"]).stem
        bc, co, bo = (r[k]["val_bpc"] for k in ("brain+context", "context_only", "brain_only"))
        d.append(bc - co)
        lines.append(f"| {r['offset']} | {name.rsplit('_c', 1)[-1]} | {r['selected']['features']}, "
                     f"{r['selected']['l2']:g} | {bc:.3f} | {co:.3f} | {bo:.3f} | {bc - co:+.3f} |")
    wins = sum(x < 0 for x in d)
    lines += ["", f"delta mean {ms(d)}; brain helps in {wins}/{len(d)} caches", ""]

# 3. mechanism jobs: record presence and top-level keys only (interpret by hand)
lines += ["## 3. Mechanism jobs", ""]
for tag in ["mech_route_160_random", "mech_route_160_glom", "mech_route_192_random",
            "mech_route_192_glom", "mech_m2_12pairs"]:
    r = load(N / f"{tag}.json")
    if r is None:
        lines.append(f"- {tag}: not available")
    else:
        keys = list(r.keys())[:12] if isinstance(r, dict) else f"list[{len(r)}]"
        lines.append(f"- {tag}: present; keys {keys}")
lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT)
