"""95% intervals for M10 (output identity), which repro/replication_stats.py leaves as nan.

replication_stats.py passes the per-seed values of M10 to its cluster bootstrap as a single
cluster, so no interval is produced; its point estimates (used by the pre-registered H2) are
correct. The file is frozen by the pre-registration, so the intervals are computed here: one
value per input-code seed, bootstrap over seeds (10,000 resamples, seed 0).  The reference
changes come from two committed files: the swap from m6_side_a*.json, the shuffle from
m3_kcmbon.json.

    python repro/replication_m10_ci.py RUNDIR
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replication_stats as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(d):
    runs = sorted(p for p in Path(d).glob("route_c*.json") if not p.name.endswith(".manifest.json"))
    rng = np.random.default_rng(0)
    ref_perm = {r["active"]: r for r in json.loads((ROOT / "results/m3_kcmbon.json").read_text()) if r["mode"] == "kc_perm" and r["seed"] == 1}
    ref_real = {r["active"]: r for r in json.loads((ROOT / "results/m3_kcmbon.json").read_text()) if r["mode"] == "real"}
    lines = [f"M10 over {len(runs)} input-code seeds (mean change of item cosine vs the real wiring, 95% bootstrap over seeds):", ""]
    for a in (160, 192):
        vals = [R.m10(json.loads(p.read_text()))[a] for p in runs]
        out = []
        for k in ("d_swap", "d_perm"):
            x = np.array([v[k] for v in vals])
            b = x[rng.integers(0, len(x), (10_000, len(x)))].mean(1)
            out.append(f"{k} {x.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
        side = json.loads((ROOT / f"results/m6_side_a{a}.json").read_text())
        g = {(r["mode"], r["seed"]): r["item_cos_mean"] for r in side}
        ref_swap = g[("profile_type_side", 1)] - g[("real", 0)]
        ref_shuf = ref_perm[a]["item_cos_mean"] - ref_real[a]["item_cos_mean"]
        lines.append(f"- {a} PNs: " + "; ".join(out) + f" (reference: swap {ref_swap:+.3f}, shuffle {ref_shuf:+.3f})")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
