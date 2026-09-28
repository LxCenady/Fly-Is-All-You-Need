"""How do items become distinct through the two hemispheres?

For 26 items (a-z) at a given operating point: fraction of the item's PN code
on the left side; fraction of written KCs (nonzero weight change) on the left;
fraction of the item's MBON readout current energy on left MBONs (static sum of
written KCs' output profiles, weighted by activity, as validated in
m6_coherence).  Report correlations and the spread across items, i.e. how a
small left/right imbalance of the input is amplified by the KC threshold.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402

ITEMS = list("abcdefghijklmnopqrstuvwxyz")


def main(out_path, points, input_code="random"):
    res = []
    for active, scale in points:
        args, chars = C.protocol_args()
        args.active, args.drive, args.probe_drive = active, 1.0 * scale, 2.5 * scale
        st = C.build(args)
        st["enc"].drive = args.drive
        if input_code == "glom":
            from m3_pnkc import set_glom_codes
            set_glom_codes(st, int(active), 11)
        p, b = st["plastic"], st["brain"]
        kc_sub = np.asarray(b.cell_type).astype(str)[p.kc_ids]
        subs = sorted(set(kc_sub))
        side = np.asarray(b.side).astype(str)
        kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
        nk, nm = len(p.kc_ids), len(p.mbon_ids)
        w0 = p.w0_gpu.get().astype(np.float64)
        O = np.zeros((nk, nm)); np.add.at(O, (kslot, mslot), w0)
        kc_left = side[p.kc_ids] == "L"
        mb_left = side[p.mbon_ids] == "L"
        rows = []
        for x in ITEMS:
            if x not in chars:
                continue
            tid = chars.index(x)
            code = np.asarray(st["enc"].codes[tid])
            C.reset(st, args)
            C.write_item(st, args, tid, plastic=True)
            dw = b._W.data[p.edge_pos_gpu].get().astype(np.float64) - w0
            a = np.zeros(nk); np.add.at(a, kslot, np.abs(dw))
            A = a > 0
            S = (a[:, None] * O).sum(0)
            comp = {s: float(a[kc_sub == s].sum() / max(a.sum(), 1e-30)) for s in subs}
            rows.append({"item": x, "pn_left_frac": float((side[code] == "L").mean()),
                         "kc_subtype_activity_share": comp,
                         "n_kc_written": int(A.sum()),
                         "kc_left_frac": float(kc_left[A].mean()) if A.any() else None,
                         "mbon_left_energy_frac": float((S[mb_left] ** 2).sum() / max((S ** 2).sum(), 1e-30))})
        pl = np.array([r["pn_left_frac"] for r in rows])
        kl = np.array([r["kc_left_frac"] for r in rows], float)
        ml = np.array([r["mbon_left_energy_frac"] for r in rows])
        ok = ~np.isnan(kl)
        share = np.array([[r["kc_subtype_activity_share"][s] for s in subs] for r in rows])
        res.append({"active": active, "scale": scale, "input_code": input_code, "items": rows,
                    "subtype_share_mean": dict(zip(subs, share.mean(0).round(4).tolist())),
                    "subtype_share_sd_across_items": dict(zip(subs, share.std(0).round(4).tolist())),
                    "corr_pn_kc_left": float(np.corrcoef(pl[ok], kl[ok])[0, 1]),
                    "corr_kc_mbon_left": float(np.corrcoef(kl[ok], ml[ok])[0, 1]),
                    "sd_pn_left": float(pl.std()), "sd_kc_left": float(np.nanstd(kl)),
                    "sd_mbon_left": float(ml.std()),
                    "gain_kc_over_pn": float(np.polyfit(pl[ok], kl[ok], 1)[0])})
        Path(out_path).write_text(json.dumps(res, indent=1), encoding="utf-8")
        del st
    return res


if __name__ == "__main__":
    pts = [(int(s.split(":")[0]), float(s.split(":")[1])) for s in sys.argv[2].split(",")]
    for r in main(sys.argv[1], pts, sys.argv[3] if len(sys.argv) > 3 else "random"):
        print(f"PN {r['active']}: sd(PN left)={r['sd_pn_left']:.3f} sd(KC left)={r['sd_kc_left']:.3f} "
              f"sd(MBON left energy)={r['sd_mbon_left']:.3f} | corr PN->KC {r['corr_pn_kc_left']:.2f} "
              f"slope {r['gain_kc_over_pn']:.1f} | corr KC->MBON {r['corr_kc_mbon_left']:.2f}")
        print("   subtype share mean:", r["subtype_share_mean"])
        print("   subtype share sd across items:", r["subtype_share_sd_across_items"])
