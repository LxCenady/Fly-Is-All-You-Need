"""Interference between two memories written one after the other.

Sequence:  write X (32 tokens) -> gap 8 -> write Y (32 tokens) -> gap 32 -> probe X
Reference: write X (32 tokens) -> gap 8 -> 32 silent tokens -> gap 32 -> probe X
(same age and decay for X; only Y's write differs).  Readout = MBON input
current of cue X, I = sum_k dW_k,m * c_k(probe X) / 6, on the common no-write
fast state of the reference episode.
  interference  |I_seq - I_ref| / |I_ref|
  kept          cos(I_seq, I_ref)
  prediction    |I_Y->X| / |I_ref|, with I_Y->X = Y's write alone read by cue X
                (the linear, additive expectation)
  overlap       Jaccard between the KCs written by Y and those active at probe X
Conditions:
  teacher  same   : X and Y both taught by all PAM
           diff   : X taught by PAM01-07, Y by PAM08-15 (different compartments)
  point    sparse : 192 PNs x1.5, bilateral glomerular codes
           dense  : 512 PNs x1.0, legacy random codes
Pairs: 6.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
from m3_pnkc import set_glom_codes  # noqa: E402

PAIRS = [("a", "c"), ("b", "d"), ("e", "g"), ("f", "h"), ("i", "k"), ("j", "l")]


def run(active, scale, code):
    args, chars = C.protocol_args()
    args.active, args.drive, args.probe_drive = active, 1.0 * scale, 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    if code == "glom":
        set_glom_codes(st, active, 11)
    p, b = st["plastic"], st["brain"]
    ct = np.asarray(b.cell_type).astype(str)
    pam = np.flatnonzero(np.char.find(ct, "PAM") >= 0)
    types = sorted(set(ct[pam])); half = set(types[: len(types) // 2])
    T = {"all": pam, "A": np.asarray([i for i in pam if ct[i] in half]),
         "B": np.asarray([i for i in pam if ct[i] not in half])}
    kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
    w0 = p.w0_gpu.get().astype(np.float64)

    def cur(dw, kc):
        return np.bincount(mslot, weights=dw * kc[kslot] / args.k, minlength=len(p.mbon_ids))

    def write(item, teacher, plastic=True):
        st["pam_ids"] = T[teacher]
        C.write_item(st, args, chars.index(item), plastic=plastic)

    rows = []
    for x, y in PAIRS:
        for cond, (tx, ty) in (("same", ("all", "all")), ("diff", ("A", "B"))):
            # reference: X, gap 8, 32 silent, gap 32
            C.reset(st, args); write(x, tx); C.gap(st, args, 8 + 32 + 32)
            S_ref = C.snapshot(st)
            # sequence: X, gap 8, Y, gap 32
            C.reset(st, args); write(x, tx); C.gap(st, args, 8); write(y, ty); C.gap(st, args, 32)
            S_seq = C.snapshot(st)
            # Y alone (same age as in the sequence)
            C.reset(st, args); write(y, ty); C.gap(st, args, 32)
            S_y = C.snapshot(st)
            # common no-write fast state and probe-X KC activity
            C.reset(st, args); write(x, tx, plastic=False); C.gap(st, args, 8)
            write(y, ty, plastic=False); C.gap(st, args, 32)
            S0 = C.snapshot(st)
            C.restore(st, S0, "zero")
            C.probe(st, args, chars.index(x))
            kc = st["_last_kc_counts"].astype(np.float64)
            dref = S_ref["w_slots"].get().astype(np.float64) - w0
            dseq = S_seq["w_slots"].get().astype(np.float64) - w0
            dy = S_y["w_slots"].get().astype(np.float64) - w0
            i_ref, i_seq, i_y = cur(dref, kc), cur(dseq, kc), cur(dy, kc)
            wy = np.unique(kslot[dy != 0]); px = np.flatnonzero(kc > 0)
            n = lambda v: float(np.linalg.norm(v))
            rows.append({"pair": x + y, "teacher": cond,
                         "interference": n(i_seq - i_ref) / max(n(i_ref), 1e-30),
                         "kept_cos": float(i_seq @ i_ref / max(n(i_seq) * n(i_ref), 1e-30)),
                         "linear_prediction": n(i_y) / max(n(i_ref), 1e-30),
                         "overlap_Ywritten_Xprobe": float(len(np.intersect1d(wy, px)) /
                                                          max(len(np.union1d(wy, px)), 1)),
                         "clip_fraction_seq": float((np.abs(S_seq["modulation"].get()) >= 0.899).mean())})
    del st
    return rows


if __name__ == "__main__":
    out, t0 = [], time.time()
    for spec in sys.argv[2].split(","):
        a, sc, code = spec.split(":")
        rows = run(int(a), float(sc), code)
        out.append({"active": int(a), "scale": float(sc), "code": code, "rows": rows})
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
        for cond in ("same", "diff"):
            r = [x for x in rows if x["teacher"] == cond]
            print(f"PN {a} {code} teacher={cond}: interference "
                  f"{np.median([x['interference'] for x in r]):.3f} "
                  f"(range {min(x['interference'] for x in r):.3f}-{max(x['interference'] for x in r):.3f}), "
                  f"kept cos {np.median([x['kept_cos'] for x in r]):.3f}, linear pred "
                  f"{np.median([x['linear_prediction'] for x in r]):.3f}, overlap "
                  f"{np.median([x['overlap_Ywritten_Xprobe'] for x in r]):.3f}  {time.time() - t0:.0f}s",
                  flush=True)
