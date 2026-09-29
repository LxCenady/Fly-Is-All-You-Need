"""Second implementation of the retrieval-key metric, independent of the plasticity bookkeeping.

m2_current reads dW from the plasticity object's slot arrays (edge_kc_slot / edge_mbon_slot /
w_slots), the bookkeeping in which the CSR canonicalisation bug once lived.  Here dW is read
directly from the live CSR entries whose row is an MBON and whose column is a KC (cell types
from the connectome), and the MBON input current I_m = sum_k dW_mk * c_k / k is computed from
those entries.  The probe KC counts come from an episode written without plasticity.
Output: per pair and probe, |I| here vs I_x_norm in m2 (same pairs, same operating points).
Usage: m9_independent.py OUT.json M2_REFERENCE.json
"""
import json
import sys

import numpy as np

from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_core as C  # noqa: E402

PAIRS = [("a", "c"), ("b", "d"), ("e", "g"), ("f", "h"), ("i", "k"), ("j", "l")]


def run_point(active, scale):
    args, chars = C.protocol_args()
    args.active, args.drive, args.probe_drive = int(active), 1.0 * scale, 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    br = st["brain"]; xp = br.xp; W = br._W
    ct = np.asarray(br.cell_type).astype(str)
    is_kc = np.char.find(ct, "KC") >= 0
    is_mbon = np.char.find(ct, "MBON") >= 0
    indptr = xp.asnumpy(W.indptr); indices = xp.asnumpy(W.indices)
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    sel = np.flatnonzero(is_mbon[rows] & is_kc[indices])
    sel_g = xp.asarray(sel)
    post, pre = rows[sel], indices[sel]
    mb_ids = np.unique(post)
    mrow = np.searchsorted(mb_ids, post)
    kc_list = np.asarray(st["kc"]).astype(np.int64)
    tid = {c: chars.index(c) for c in "abcdefghijklz"}

    def live():
        return xp.asnumpy(W.data[sel_g]).astype(np.float64)

    C.reset(st, args)
    base = live()

    def written(item):
        C.reset(st, args)
        assert np.allclose(live(), base), "reset did not restore KC->MBON weights"
        C.write_item(st, args, tid[item], plastic=True)
        C.gap(st, args, 32)
        return live() - base

    def probe_counts(item, probe):
        C.reset(st, args)
        C.write_item(st, args, tid[item], plastic=False)
        C.gap(st, args, 32)
        C.probe(st, args, tid[probe])
        c = np.zeros(len(ct)); c[kc_list] = st["_last_kc_counts"]
        return c

    out = []
    for x, y in PAIRS:
        dwx = written(x)
        for p in (x, y, "z"):
            c = probe_counts(x, p)
            I = np.bincount(mrow, weights=dwx * c[pre] / args.k, minlength=len(mb_ids))
            out.append({"pair": [x, y], "probe": p, "I_norm": float(np.linalg.norm(I)),
                        "n_entries": int(len(sel)), "dw_nonzero": int((dwx != 0).sum())})
    del st
    return {"active": active, "scale": scale, "cells": out}


if __name__ == "__main__":
    res = [run_point(160, 1.5), run_point(512, 1.0)]
    ref = json.load(open(sys.argv[2], encoding="utf-8"))
    for pt, rp in zip(res, ref):
        refmap = {(tuple(c["pair"]), c["probe"]): c["I_x_norm"] for c in rp["cells"]}
        for c in pt["cells"]:
            r = refmap.get((tuple(c["pair"]), c["probe"]))
            c["m2_I_x_norm"] = r
            c["rel_diff"] = None if r is None else abs(c["I_norm"] - r) / max(r, 1e-30)
        d = [c["rel_diff"] for c in pt["cells"] if c["rel_diff"] is not None]
        print(pt["active"], "cells", len(d), "max rel diff", max(d), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
