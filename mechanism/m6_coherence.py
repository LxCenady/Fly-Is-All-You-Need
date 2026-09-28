"""Population summation amplifies weak input-output matching (test).

For 8 items at a sparse operating point, take the set A_x of KCs written by
item x (nonzero weight change) and their activity a_k.  Using the static
connectome output profiles o_k (KC -> 97 MBON weights):
  rho_x      mean pairwise cosine of o_k among KCs in A_x
  rho_null   same for random KC sets with the same subtype composition
  N_x        |A_x|
  static-sum model:  S_x = sum_{k in A_x} a_k o_k ; item distinctness =
             mean pairwise cos(S_x, S_y), for real profiles and for profiles
             swapped within subtype (5 draws) -- compare to the simulated
             0.76 (real) vs 0.90-0.95 (swapped) readout.
  amplification index N_x * (rho_x - rho_null)
Also the number of MBONs addressed by each item's KC set and the names of the
top MBONs per item.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402

ITEMS = list("abcdefgh")


def main(out_path, points):
    res = []
    for active, scale in points:
        args, chars = C.protocol_args()
        args.active, args.drive, args.probe_drive = active, 1.0 * scale, 2.5 * scale
        st = C.build(args)
        st["enc"].drive = args.drive
        p, b = st["plastic"], st["brain"]
        ct = np.asarray(b.cell_type).astype(str)
        kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
        nk, nm = len(p.kc_ids), len(p.mbon_ids)
        w0 = p.w0_gpu.get().astype(np.float64)
        O = np.zeros((nk, nm))
        np.add.at(O, (kslot, mslot), w0)                   # static output profiles
        On = O / np.maximum(np.linalg.norm(O, axis=1, keepdims=True), 1e-30)
        sub = ct[p.kc_ids]
        rng = np.random.default_rng(0)
        tid = {c: chars.index(c) for c in ITEMS}
        sets, acts = [], []
        for x in ITEMS:
            C.reset(st, args)
            C.write_item(st, args, tid[x], plastic=True)
            dw = st["brain"]._W.data[p.edge_pos_gpu].get().astype(np.float64) - w0
            a = np.zeros(nk)
            np.add.at(a, kslot, np.abs(dw))
            A = np.flatnonzero(a > 0)
            sets.append(A)
            acts.append(a)

        def rho(idx):
            if len(idx) < 2:
                return float("nan")
            G = On[idx] @ On[idx].T
            return float(G[np.triu_indices(len(idx), 1)].mean())

        def matched_random(A):
            out = []
            for t in np.unique(sub[A]):
                pool = np.flatnonzero(sub == t)
                out.extend(rng.choice(pool, size=int((sub[A] == t).sum()), replace=False))
            return np.asarray(out)

        per_item = []
        for x, A, a in zip(ITEMS, sets, acts):
            r = rho(A)
            rn = [rho(matched_random(A)) for _ in range(20)]
            S = (a[A, None] * O[A]).sum(0)
            top = np.argsort(-S)[:3]
            per_item.append({"item": x, "N": int(len(A)), "rho": r, "rho_null_mean": float(np.mean(rn)),
                             "rho_null_sd": float(np.std(rn)),
                             "amplification_N_x_excess_rho": float(len(A) * (r - np.mean(rn))),
                             "top_mbons": [str(ct[p.mbon_ids[j]]) for j in top],
                             "top3_share": float(np.sort(S ** 2)[::-1][:3].sum() / (S ** 2).sum())})

        def distinct(Oprof):
            Ss = np.asarray([(a[A, None] * Oprof[A]).sum(0) for A, a in zip(sets, acts)])
            U = Ss / np.linalg.norm(Ss, axis=1, keepdims=True)
            return float((U @ U.T)[np.triu_indices(len(ITEMS), 1)].mean())

        swapped = []
        for d in range(5):
            perm = np.arange(nk)
            r2 = np.random.default_rng(100 + d)
            for t in np.unique(sub):
                idx = np.flatnonzero(sub == t)
                perm[idx] = idx[r2.permutation(len(idx))]
            swapped.append(distinct(O[perm]))
        overlap = [len(np.intersect1d(sets[i], sets[j])) / len(np.union1d(sets[i], sets[j]))
                   for i in range(len(ITEMS)) for j in range(i + 1, len(ITEMS))]
        res.append({"active": active, "scale": scale, "items": per_item,
                    "static_sum_distinct_real": distinct(O),
                    "static_sum_distinct_swapped_within_subtype": swapped,
                    "written_set_jaccard_mean": float(np.mean(overlap))})
        Path(out_path).write_text(json.dumps(res, indent=1), encoding="utf-8")
        del st
    return res


if __name__ == "__main__":
    pts = [(int(s.split(":")[0]), float(s.split(":")[1])) for s in sys.argv[2].split(",")]
    t0 = time.time()
    for r in main(sys.argv[1], pts):
        print(f"\nPN {r['active']}: static-sum distinctness real {r['static_sum_distinct_real']:.3f}, "
              f"swapped {np.round(r['static_sum_distinct_swapped_within_subtype'], 3).tolist()}, "
              f"written-set Jaccard {r['written_set_jaccard_mean']:.3f}")
        for it in r["items"]:
            print(f"  {it['item']}: N={it['N']:4d} rho={it['rho']:.4f} null={it['rho_null_mean']:.4f}"
                  f"+-{it['rho_null_sd']:.4f} N*excess={it['amplification_N_x_excess_rho']:.2f} "
                  f"top3 {it['top_mbons']} share {it['top3_share']:.2f}")
    print(f"{time.time() - t0:.0f}s")
