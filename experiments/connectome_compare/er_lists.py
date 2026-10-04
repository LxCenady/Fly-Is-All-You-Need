"""Condition list for a random-graph reservoir. One job: spec + target gain -> lists/<name>.list.

Driven share of the input population: 2, 5, 10, 24 (base), 50, 75, 95 %; gain x0.5, x0.75,
x1.25 of the 10 %-activity gain (target/<name>.json from target_gain.py).

    python er_lists.py er448
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
name = sys.argv[1]
spec = json.loads((HERE / "specs" / f"{name}.json").read_text(encoding="utf-8"))
t = json.loads((HERE / "target" / f"{name}.json").read_text(encoding="utf-8"))
g = t["chosen_gain"] or t["closest_gain"]
n_in = len(spec["populations"]["INPUT"]["ids"])
lines = [f"base {g}"]
for pct in (2, 5, 10, 50, 75, 95):
    lines.append(f"act{pct:02d} {g} --set input.params.active={max(1, round(pct / 100 * n_in))}")
for f in (0.5, 0.75, 1.25):
    lines.append(f"g{int(f * 100):03d} {round(g * f, 4)}")
(HERE / "lists").mkdir(exist_ok=True)
(HERE / "lists" / f"{name}.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
