"""Diagnose the suspicious ~0.75 floor in M2 scrambles.

At active=192 x1.5, pair (a,c), probe a:
  identity   sigma = arange -> must reproduce r(m_a) within replay floor
  roundtrip  slow_from_dw(dw) without permutation (same as identity)
  within_kc / within_mbon draws: fraction of dW entries actually changed
  decomposition of the response change by MBON (which MBONs carry e)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
from m2_address import group_perm  # noqa: E402

args, chars = C.protocol_args()
args.active, args.drive, args.probe_drive = 192, 1.5, 3.75
st = C.build(args)
st["enc"].drive = args.drive
p, xp = st["plastic"], st["brain"].xp
w0 = p.w0_gpu.get().astype(np.float64)
tid = {c: chars.index(c) for c in "acz"}


def episode(item, plastic):
    C.reset(st, args)
    C.write_item(st, args, tid[item], plastic=plastic)
    C.gap(st, args, 32)
    return C.snapshot(st)


SA, S0 = episode("a", True), episode("a", False)
dwa = SA["w_slots"].get().astype(np.float64) - w0


def slow_from_dw(dw):
    w = (w0 + dw).astype(np.float32)
    mod = np.where(w0 != 0, dw / np.where(w0 != 0, w0, 1), 0).astype(np.float32)
    return {"modulation": xp.asarray(mod), "w_slots": xp.asarray(w)}


def resp(slow):
    C.restore(st, S0, slow)
    return C.probe(st, args, tid["a"])["mbon"]


r0, rA = resp("zero"), resp(SA)
e = rA - r0
out = {"e_norm": float(np.linalg.norm(e)),
       "identity_roundtrip_maxabs": float(np.max(np.abs(resp(slow_from_dw(dwa)) - rA))),
       "mod_roundtrip_maxabs": float(np.max(np.abs(
           slow_from_dw(dwa)["modulation"].get() - SA["modulation"].get()))),
       "n_dw_nonzero": int((dwa != 0).sum())}
ev = e.reshape(6, -1)                               # settle steps x 97 MBON
per_mbon = np.linalg.norm(ev, axis=0)
out["e_top_mbons_share"] = float(np.sort(per_mbon ** 2)[::-1][:3].sum() / (per_mbon ** 2).sum())
out["e_n_mbons_nonzero"] = int((per_mbon > 1e-6).sum())
for typ, g in (("within_kc", p.edge_kc_slot.astype(np.int64)),
               ("within_mbon", p.edge_mbon_slot.astype(np.int64))):
    rows = []
    for d in range(3):
        sg = group_perm(g, np.random.default_rng(d))
        es = resp(slow_from_dw(dwa[sg])) - r0
        pm = np.linalg.norm(es.reshape(6, -1), axis=0)
        rows.append({"frac_dw_changed": float(np.mean(dwa[sg] != dwa)),
                     "e_ratio": float(np.linalg.norm(es) / np.linalg.norm(e)),
                     "e_cos": float(es @ e / np.linalg.norm(es) / np.linalg.norm(e)),
                     "per_mbon_cos": float(pm @ per_mbon / np.linalg.norm(pm) / np.linalg.norm(per_mbon)),
                     "maxabs_vs_orig": float(np.max(np.abs(es - e)))})
    out[typ] = rows
Path(sys.argv[1]).write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=1))
