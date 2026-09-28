"""How dense are PN input codes and KC responses along the actual protocol?

For each character: reset -> 32 write tokens (drive 1.0, DAN pulse, plasticity
OFF so all characters see the same weights) -> 32 silent tokens -> one probe
(drive 2.5, settle 6).  Record the KC set that spikes at write tokens
1, 2, 4, 8, 16, 32 and at the probe; MBON active counts; pairwise Jaccard of
KC sets across characters at the same protocol point.
Optional PN code size override (a mechanism manipulation, reported separately).
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
import mb_persistent_memory_probe as pmp  # noqa: E402

CH = list("abcdefgh") + ["z"]
MARK = [1, 2, 4, 8, 16, 32]


def jacc(a, b):
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else float("nan")


def run(active: int):
    args, chars = C.protocol_args()
    args.active = int(active)
    st = C.build(args)
    rec = {}
    for ch in CH:
        C.reset(st, args)
        tid = chars.index(ch)
        sets = {}
        for t in range(1, 33):
            pmp._advance_token(st, args, tid, allow_plastic=False, pulse_dan=True)
            if t in MARK:
                sets[f"w{t}"] = (st["_last_kc_counts"] > 0, int((st["_last_mbon_counts"] > 0).sum()))
        C.gap(st, args, 32)
        C.probe(st, args, tid)
        sets["probe"] = (st["_last_kc_counts"] > 0, int((st["_last_mbon_counts"] > 0).sum()))
        rec[ch] = sets
    out = {"active": int(active), "points": {}}
    for key in [f"w{t}" for t in MARK] + ["probe"]:
        fr = [float(rec[c][key][0].mean()) for c in CH]
        js = [jacc(rec[a][key][0], rec[b][key][0]) for a, b in combinations(CH, 2)]
        out["points"][key] = {"kc_frac_mean": float(np.mean(fr)), "kc_frac_min": float(np.min(fr)),
                              "kc_frac_max": float(np.max(fr)),
                              "kc_jaccard_mean": float(np.nanmean(js)) if not all(np.isnan(js)) else None,
                              "mbon_active_mean": float(np.mean([rec[c][key][1] for c in CH]))}
    return out


if __name__ == "__main__":
    res = []
    for act in [int(x) for x in sys.argv[2].split(",")]:
        t0 = time.time()
        res.append(run(act))
        print(f"active={act} {time.time() - t0:.0f}s", flush=True)
        Path(sys.argv[1]).write_text(json.dumps(res, indent=2), encoding="utf-8")
    for r in res:
        print(f"\nPN code size {r['active']}/675")
        for k, v in r["points"].items():
            j = v["kc_jaccard_mean"]
            print(f"  {k:>6}: KC active {v['kc_frac_mean']:.3f} [{v['kc_frac_min']:.3f},{v['kc_frac_max']:.3f}]  "
                  f"KC Jaccard {('%.3f' % j) if j is not None else '  -  '}  MBON active {v['mbon_active_mean']:.1f}/97")
