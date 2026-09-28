"""Operating-point sweep: is there a regime with sparse KC codes AND a readable memory?

For each (PN code size, drive scale):
  density  9 characters, plasticity off: KC active fraction at write tokens
           2/8/32 and at the probe, pairwise KC Jaccard, MBON active counts
  memory   pair (a, c): 32-token plastic writes, 32-token gap, probes a, c, z on
           the common no-write fast state; W = |r(m_a) - r(m_0)|, D = |r(m_a) - r(m_c)|
           (MBON and central v_pre, settle window), plus |modulation| and the
           number of KCs written
Drive scale multiplies both the write drive (1.0) and the probe drive (2.5).
Target regime (stated before data): probe KC active 0.03-0.15, KC Jaccard
across characters < 0.3, and D above 1e3 x the replay floor (~2e-7).
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


def jacc(a, b):
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else float("nan")


def one(active: int, scale: float) -> dict:
    args, chars = C.protocol_args()
    args.active = int(active)
    args.drive = 1.0 * scale
    args.probe_drive = 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    tid = {c: chars.index(c) for c in CH}
    sets = {}
    for ch in CH:
        C.reset(st, args)
        s = {}
        for t in range(1, 33):
            pmp._advance_token(st, args, tid[ch], allow_plastic=False, pulse_dan=True)
            if t in (2, 8, 32):
                s[f"w{t}"] = (st["_last_kc_counts"] > 0, int((st["_last_mbon_counts"] > 0).sum()))
        C.gap(st, args, 32)
        C.probe(st, args, tid[ch])
        s["probe"] = (st["_last_kc_counts"] > 0, int((st["_last_mbon_counts"] > 0).sum()))
        sets[ch] = s
    dens = {}
    for key in ("w2", "w8", "w32", "probe"):
        fr = [float(sets[c][key][0].mean()) for c in CH]
        js = [jacc(sets[a][key][0], sets[b][key][0]) for a, b in combinations(CH, 2)]
        js = [j for j in js if not np.isnan(j)]
        dens[key] = {"kc_frac": float(np.mean(fr)), "kc_frac_min": float(np.min(fr)),
                     "kc_frac_max": float(np.max(fr)),
                     "kc_jaccard": float(np.mean(js)) if js else None,
                     "mbon_active": float(np.mean([sets[c][key][1] for c in CH]))}

    def episode(item, plastic):
        C.reset(st, args)
        C.write_item(st, args, tid[item], plastic=plastic)
        C.gap(st, args, 32)
        return C.snapshot(st)

    SA, SC, S0 = episode("a", True), episode("c", True), episode("a", False)
    p = st["plastic"]
    ma = SA["modulation"].get()
    mem = {"mod_norm_a": float(np.linalg.norm(ma)),
           "mod_diff": float(np.linalg.norm(ma - SC["modulation"].get())),
           "kc_written_a": int(len(np.unique(p.edge_kc_slot[np.abs(ma) > 0])))}
    for pch in ("a", "c", "z"):
        R = {}
        for k, slow in (("A", SA), ("C", SC), ("0", "zero")):
            C.restore(st, S0, slow)
            R[k] = C.probe(st, args, tid[pch])
        for lv in ("mbon", "central"):
            mem[f"{lv}|{pch}|W"] = float(np.linalg.norm(R["A"][lv] - R["0"][lv]))
            mem[f"{lv}|{pch}|D"] = float(np.linalg.norm(R["A"][lv] - R["C"][lv]))
    del st
    return {"active": int(active), "scale": float(scale), "density": dens, "memory": mem}


if __name__ == "__main__":
    out_path = sys.argv[1]
    actives = [int(x) for x in sys.argv[2].split(",")]
    scales = [float(x) for x in sys.argv[3].split(",")]
    res = []
    for sc in scales:
        for a in actives:
            t0 = time.time()
            res.append(one(a, sc))
            Path(out_path).write_text(json.dumps(res, indent=2), encoding="utf-8")
            r = res[-1]
            print(f"active={a} scale={sc} probeKC={r['density']['probe']['kc_frac']:.3f} "
                  f"J={r['density']['probe']['kc_jaccard']} D_mbon_z={r['memory']['mbon|z|D']:.2e} "
                  f"{time.time() - t0:.0f}s", flush=True)
