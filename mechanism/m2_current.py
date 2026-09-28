"""M2 on the MBON input current (before the spiking nonlinearity).

The voltage-based M2 metric is dominated by 1-3 MBONs whose write-induced
threshold crossing shifts by one step under any perturbation (m2_diag.json:
per-MBON energy cos 0.997 but time-course cos 0.75).  The address question is
cleaner on the write-attributable input current each MBON receives from the
probe's KC activity:

    I_m(dW) = sum_k dW[k, m] * kc_probe[k] / k_steps        (97-vector)

kc_probe = KC spike counts of the probe token on the common no-write state.
Scrambles as in m2_address (ΔW multiset exactly preserved, same sigma for X
and Y, 20 draws).  Reported: effect ratio / cos for I(sigma dW_X) vs I(dW_X),
content ratio / cos for I(sigma dW_X) - I(sigma dW_Y) vs I(dW_X) - I(dW_Y).
Operating points: PN code 160 / 192 / 224 at x1.5 drive, and 512 (standard).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
from m2_address import group_perm  # noqa: E402

PAIRS = [("a", "c"), ("b", "d"), ("e", "g")]
POINTS = [(160, 1.5), (192, 1.5), (224, 1.5), (512, 1.0)]
TYPES = ["all", "within_mbon", "within_kc"]
DRAWS = 20


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def run_point(active, scale):
    args, chars = C.protocol_args()
    args.active, args.drive, args.probe_drive = int(active), 1.0 * scale, 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    p = st["plastic"]
    w0 = p.w0_gpu.get().astype(np.float64)
    kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
    groups = {"all": np.zeros(p.n_edges, np.int64), "within_mbon": mslot, "within_kc": kslot}
    tid = {c: chars.index(c) for c in set("".join(x + y for x, y in PAIRS)) | {"z"}}

    def episode(item, plastic):
        C.reset(st, args)
        C.write_item(st, args, tid[item], plastic=plastic)
        C.gap(st, args, 32)
        return C.snapshot(st)

    def current(dw, kc):
        return np.bincount(mslot, weights=dw * kc[kslot] / args.k, minlength=len(p.mbon_ids))

    cells = []
    for x, y in PAIRS:
        SX, SY, S0 = episode(x, True), episode(y, True), episode(x, False)
        dwx = SX["w_slots"].get().astype(np.float64) - w0
        dwy = SY["w_slots"].get().astype(np.float64) - w0
        for pch in (x, y, "z"):
            C.restore(st, S0, "zero")
            C.probe(st, args, tid[pch])
            kc = st["_last_kc_counts"].astype(np.float64)
            ix, iy = current(dwx, kc), current(dwy, kc)
            cell = {"pair": [x, y], "probe": pch, "kc_probe_active": int((kc > 0).sum()),
                    "kc_written_x": int(len(np.unique(kslot[dwx != 0]))),
                    "overlap": int(len(np.intersect1d(np.flatnonzero(kc > 0), kslot[dwx != 0]))),
                    "I_x_norm": float(np.linalg.norm(ix)), "c_norm": float(np.linalg.norm(ix - iy)),
                    "scramble": {}}
            for typ in TYPES:
                rs = []
                for d in range(DRAWS):
                    sg = group_perm(groups[typ], np.random.default_rng(7919 * d + TYPES.index(typ)))
                    jx, jy = current(dwx[sg], kc), current(dwy[sg], kc)
                    rs.append([np.linalg.norm(jx) / max(np.linalg.norm(ix), 1e-30), cos(jx, ix),
                               np.linalg.norm(jx - jy) / max(np.linalg.norm(ix - iy), 1e-30),
                               cos(jx - jy, ix - iy)])
                rs = np.asarray(rs)
                cell["scramble"][typ] = {"median": np.nanmedian(rs, 0).round(4).tolist(),
                                         "p10": np.nanpercentile(rs, 10, 0).round(4).tolist(),
                                         "p90": np.nanpercentile(rs, 90, 0).round(4).tolist()}
            cells.append(cell)
    del st
    return {"active": active, "scale": scale, "cells": cells}


if __name__ == "__main__":
    # optional: argv[2] = pairs like "ac,bd,eg"; argv[3] = points like "160:1.5,192:1.5"
    if len(sys.argv) > 2:
        PAIRS = [(s[0], s[1]) for s in sys.argv[2].split(",")]
    if len(sys.argv) > 3:
        POINTS = [(int(s.split(":")[0]), float(s.split(":")[1])) for s in sys.argv[3].split(",")]
    out, t0 = [], time.time()
    for a, s in POINTS:
        out.append(run_point(a, s))
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"done {a} x{s} {time.time() - t0:.0f}s", flush=True)
    for pt in out:
        print(f"\n### PN {pt['active']} x{pt['scale']}   cols: [e_ratio e_cos c_ratio c_cos] medians")
        for c in pt["cells"]:
            s = c["scramble"]
            print(f"{''.join(c['pair'])} p={c['probe']} kc act={c['kc_probe_active']:>4} wr={c['kc_written_x']:>4} "
                  f"ov={c['overlap']:>4} |I|={c['I_x_norm']:.2e} |c|={c['c_norm']:.2e} | "
                  + " | ".join(f"{t}: " + " ".join(f"{v:+.2f}" for v in s[t]['median']) for t in TYPES))
