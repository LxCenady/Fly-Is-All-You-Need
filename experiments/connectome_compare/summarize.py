"""Group analyze.py results: mean and range per (file group, readout size M).

One job: results*.json -> a markdown table on stdout. A file's group is its name without the
trailing code / seed (worm_real_c3 -> worm_real). --exclude drops files (e.g. failed matches).

    python summarize.py results_worm.json [--exclude worm_rewire_s2]
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--exclude", default="")
    a = ap.parse_args()
    skip = {s for s in a.exclude.split(",") if s}
    groups = defaultdict(list)
    for path in a.results:
        for r in json.load(open(path, encoding="utf-8")):
            stem = r["file"].removesuffix(".npz")
            if stem in skip:
                continue
            groups[(re.sub(r"_(c|s)\d+$", "", stem), r["M"])].append(r)
    print("| group | M | n | val BPC | k=1 | k=2 | k=3 | k=4 | readout active | silent share |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (g, m), rows in sorted(groups.items()):
        def rng(vals):
            v = np.asarray(vals, float)
            return f"{v.mean():.3f} ({v.min():.3f}–{v.max():.3f})"
        bpc = rng([r["val_bpc"] for r in rows]) if "val_bpc" in rows[0] else "—"
        spans = [rng([r["span"][k] for r in rows]) for k in (1, 2, 3, 4)]
        print(f"| {g} | {m} | {len(rows)} | {bpc} | " + " | ".join(spans)
              + f" | {np.mean([r['readout_active'] for r in rows]):.3f}"
              + f" | {np.mean([r['silent_share'] for r in rows]):.2f} |")


if __name__ == "__main__":
    main()
