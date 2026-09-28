"""Is real PN->KC wiring more cue-specific than degree-matched / uniform nulls?

Per (code seed, point, pair): percentile of the real cue specificity within the
8 null values (5 pn_perm + 3 pn_uniform).  Pooled over pairs: fraction of
pairs where real > null median, and where real > all nulls; exact one-sided
sign test on real > null median (pairs are not independent draws of anything
larger than this connectome; the test is descriptive).
Pre-stated rule: call the hint 'supported' only if real > null median in
>= 75 % of pairs in every (seed, point) setting.
"""
import glob
import json
from math import comb

import numpy as np

for f in sorted(glob.glob(r"E:\mechanism_20260928\m3_pnkc_glom_s*.json")):
    rows = json.load(open(f, encoding="utf-8"))
    for active in sorted({r["active"] for r in rows}):
        rs = [r for r in rows if r["active"] == active]
        real = [r for r in rs if r["mode"] == "real"]
        nulls = [r for r in rs if r["mode"] != "real"]
        if not real or len(nulls) < 8:
            print(f"{f[-8:-5]} {active}: incomplete ({len(nulls)} nulls)")
            continue
        R = np.log(np.asarray(real[0]["cue_spec_per_pair"]))
        N = np.log(np.asarray([n["cue_spec_per_pair"] for n in nulls]))     # nulls x pairs
        above_med = R > np.median(N, 0)
        above_all = R > N.max(0)
        k, n = int(above_med.sum()), len(R)
        p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
        pct = [(N[:, j] < R[j]).mean() for j in range(n)]
        print(f"seed {real[0].get('glom_seed')} PN {active}: real>null-median {k}/{n}, real>all nulls "
              f"{int(above_all.sum())}/{n}, sign-test p={p:.3f}; per-pair percentile {np.round(pct, 2).tolist()}; "
              f"median log-ratio real/null-median {np.median(R - np.median(N, 0)):+.2f}")
        kcf = [n_["kc_probe_frac"] for n_ in nulls]
        print(f"      KC frac real {real[0]['kc_probe_frac']:.3f} vs nulls {min(kcf):.3f}-{max(kcf):.3f}; "
              f"KC-id scramble real {real[0]['own_kcid_scramble_eratio_median']:.2f} vs nulls "
              f"{min(n_['own_kcid_scramble_eratio_median'] for n_ in nulls):.2f}-"
              f"{max(n_['own_kcid_scramble_eratio_median'] for n_ in nulls):.2f}")
