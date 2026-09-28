"""Item-specific teaching: does the dopaminergic teacher take part in what is stored?

So far every write used one teacher (a pulse on all PAM neurons), and the gate
was identical for every item.  Here the same items are written with different
teachers, i.e. pulses on different DAN groups:
  PAM      all PAM (the default so far)
  PPL      all PPL (opposite valence sign in the gate)
  PAM_A    PAM cell types in the first half of the sorted type list
  PAM_B    PAM cell types in the second half
Input: bilateral glomerular symbol codes (seed 11) at 192 PNs x1.5.
For every (item, teacher): MBON readout current I (97) of the cued probe on the
common no-write fast state, the mean gate vector during writing, and the sign
balance of the stored change.
Reported:
  gate cosine between teachers
  cos(I) between teachers for the same item   (teacher moves the output?)
  cos(I) between items for the same teacher   (item distinctness per teacher)
  two-way decomposition of the unit output vectors: share of variance
  explained by item, by teacher, and residual (interaction)
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
from m3_pnkc import set_glom_codes  # noqa: E402

ITEMS = list("abcdef")


def main(out_path, active=192, scale=1.5, code="glom"):
    t0 = time.time()
    args, chars = C.protocol_args()
    args.active, args.drive, args.probe_drive = active, 1.0 * scale, 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    if code == "glom":
        set_glom_codes(st, active, 11)
    p, b = st["plastic"], st["brain"]
    ct = np.asarray(b.cell_type).astype(str)
    pam = np.flatnonzero(np.char.find(ct, "PAM") >= 0)
    ppl = np.flatnonzero(np.char.find(ct, "PPL") >= 0)
    pam_types = sorted(set(ct[pam]))
    half = set(pam_types[: len(pam_types) // 2])
    teachers = {"PAM": pam, "PPL": ppl,
                "PAM_A": np.asarray([i for i in pam if ct[i] in half], np.int64),
                "PAM_B": np.asarray([i for i in pam if ct[i] not in half], np.int64)}
    kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
    w0 = p.w0_gpu.get().astype(np.float64)
    tid = {c: chars.index(c) for c in ITEMS}
    orig_gate, rec = p._gate, {"g": []}

    def hook(dan_counts, teaching_signal):
        g = orig_gate(dan_counts, teaching_signal)
        rec["g"].append(g.get().copy())
        return g

    p._gate = hook

    def cur(dw, kc):
        return np.bincount(mslot, weights=dw * kc[kslot] / args.k, minlength=len(p.mbon_ids))

    I, G, SIGN = {}, {}, {}
    for tname, ids in teachers.items():
        st["pam_ids"] = ids                              # the pulse target for writes
        gs = []
        for x in ITEMS:
            rec["g"] = []
            C.reset(st, args); C.write_item(st, args, tid[x], plastic=True); C.gap(st, args, 32)
            gs.append(np.mean(rec["g"], 0))
            SX = C.snapshot(st)
            C.reset(st, args); C.write_item(st, args, tid[x], plastic=False); C.gap(st, args, 32)
            S0 = C.snapshot(st)
            C.restore(st, S0, "zero")
            C.probe(st, args, tid[x])
            dw = SX["w_slots"].get().astype(np.float64) - w0
            I[(x, tname)] = cur(dw, st["_last_kc_counts"].astype(np.float64))
            SIGN[(x, tname)] = float((dw < 0).sum() / max((dw != 0).sum(), 1))
        G[tname] = np.mean(gs, 0)
        print(f"teacher {tname}: {len(ids)} DANs, {time.time() - t0:.0f}s", flush=True)

    def ucos(a, c):
        return float(a @ c / max(np.linalg.norm(a) * np.linalg.norm(c), 1e-30))

    T = list(teachers)
    out = {"code": code, "active": active, "teachers": {t: int(len(v)) for t, v in teachers.items()},
           "pam_types_A": sorted(half), "pam_types_B": sorted(set(pam_types) - half)}
    out["gate_cos_between_teachers"] = {f"{a}|{c}": ucos(G[a], G[c]) for a, c in combinations(T, 2)}
    out["gate_active_mbons"] = {t: int((np.abs(G[t]) > 1e-3).sum()) for t in T}
    out["same_item_cos_between_teachers"] = {
        f"{a}|{c}": float(np.mean([ucos(I[(x, a)], I[(x, c)]) for x in ITEMS])) for a, c in combinations(T, 2)}
    out["item_distinctness_per_teacher"] = {
        t: float(np.mean([ucos(I[(x, t)], I[(y, t)]) for x, y in combinations(ITEMS, 2)])) for t in T}
    out["depression_fraction"] = {t: float(np.mean([SIGN[(x, t)] for x in ITEMS])) for t in T}
    U = np.asarray([[I[(x, t)] / max(np.linalg.norm(I[(x, t)]), 1e-30) for t in T] for x in ITEMS])
    grand = U.mean((0, 1))
    item_m, teach_m = U.mean(1), U.mean(0)
    ss_tot = ((U - grand) ** 2).sum()
    ss_item = len(T) * ((item_m - grand) ** 2).sum()
    ss_teach = len(ITEMS) * ((teach_m - grand) ** 2).sum()
    out["variance_share"] = {"item": float(ss_item / ss_tot), "teacher": float(ss_teach / ss_tot),
                             "interaction_residual": float(1 - (ss_item + ss_teach) / ss_tot)}
    Path(out_path).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 192,
         float(sys.argv[3]) if len(sys.argv) > 3 else 1.5, sys.argv[4] if len(sys.argv) > 4 else "glom")
