"""Is output identity carried by KC routing, or by item-specific dopamine gates?

Pass 1 (natural): write each of 8 items; record the per-MBON gate gamma_m of
every write token; item gate = mean over tokens.  Report pairwise cosine of
item gates, and item distinctness of the MBON readout current (as in m3_kcmbon).
Pass 2 (gate clamped): the gate is replaced by the across-item mean gate for
every token of every item (a mechanism intervention), so items differ only in
which KCs they activate.  Re-measure item distinctness.
Pass 3 (gate uniform): gate = +1 on every MBON with a nonzero natural gate
(sign and compartment pattern removed).
Stated reading: if clamping leaves distinctness near the natural value, the
dopamine gate does not carry item identity; KC routing does.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402

ITEMS = list("abcdefgh")


def run(active, scale):
    args, chars = C.protocol_args()
    args.active, args.drive, args.probe_drive = active, 1.0 * scale, 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    p, xp = st["plastic"], st["brain"].xp
    kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
    w0 = p.w0_gpu.get().astype(np.float64)
    tid = {c: chars.index(c) for c in ITEMS + ["z"]}
    orig_gate = p._gate
    rec = {"gates": [], "fixed": None}

    def gate_hook(dan_counts, teaching_signal):
        g = orig_gate(dan_counts, teaching_signal)
        rec["gates"].append(g.get().copy())
        if rec["fixed"] is not None:
            return xp.asarray(rec["fixed"], dtype=xp.float32)
        return g

    p._gate = gate_hook

    def cur(dw, kc):
        return np.bincount(mslot, weights=dw * kc[kslot] / args.k, minlength=len(p.mbon_ids))

    def distinctness():
        I, G = [], []
        for x in ITEMS:
            rec["gates"] = []
            C.reset(st, args); C.write_item(st, args, tid[x], plastic=True); C.gap(st, args, 32)
            G.append(np.mean(rec["gates"], 0))
            SX = C.snapshot(st)
            C.reset(st, args); C.write_item(st, args, tid[x], plastic=False); C.gap(st, args, 32)
            S0 = C.snapshot(st)
            C.restore(st, S0, "zero")
            C.probe(st, args, tid[x])
            I.append(cur(SX["w_slots"].get().astype(np.float64) - w0,
                         st["_last_kc_counts"].astype(np.float64)))
        I, G = np.asarray(I), np.asarray(G)
        U = I / np.maximum(np.linalg.norm(I, axis=1, keepdims=True), 1e-30)
        Ug = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-30)
        iu = np.triu_indices(len(ITEMS), 1)
        return {"item_cos_mean": float((U @ U.T)[iu].mean()),
                "item_cos_min": float((U @ U.T)[iu].min()),
                "gate_cos_mean": float((Ug @ Ug.T)[iu].mean()),
                "gate_cos_min": float((Ug @ Ug.T)[iu].min()),
                "gate_nonzero_mbons": int((np.abs(G).mean(0) > 1e-6).sum()),
                "I_norm_median": float(np.median(np.linalg.norm(I, axis=1)))}, G

    natural, G = distinctness()
    rec["fixed"] = G.mean(0)
    clamped, _ = distinctness()
    rec["fixed"] = (np.abs(G.mean(0)) > 1e-6).astype(np.float32)
    uniform, _ = distinctness()
    return {"active": active, "scale": scale, "natural": natural,
            "gate_clamped_to_item_mean": clamped, "gate_uniform_plus1": uniform}


if __name__ == "__main__":
    out, t0 = [], time.time()
    for s in sys.argv[2].split(","):
        a, sc = s.split(":")
        out.append(run(int(a), float(sc)))
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out[-1]), f"{time.time() - t0:.0f}s", flush=True)
