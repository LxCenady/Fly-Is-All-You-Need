"""Output map: which KC subtypes feed which MBONs, and where each item lands.

Bilateral glomerular codes at 192 PNs x1.5, 26 items.  Per item: activity a_k
of the KCs it writes (sum |dw| per KC).  Static output profiles O[k, m] = w0.
  M[s, m]    share of subtype s's total KC->MBON weight going to MBON m
  S_x        = sum_k a_k O[k, :]   (validated proxy of the item's output)
  contrib    S_x split by KC subtype: which subtype supplies each item's top MBONs
  per MBON   which subtype supplies most of its KC input (its 'lobe')
Saved for plotting: M, S (items x MBON), subtype shares per item, MBON names
and sides.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
from m3_pnkc import set_glom_codes  # noqa: E402

ITEMS = list("abcdefghijklmnopqrstuvwxyz")


def main(out_path, active=192, scale=1.5):
    args, chars = C.protocol_args()
    args.active, args.drive, args.probe_drive = active, 1.0 * scale, 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    set_glom_codes(st, active, 11)
    p, b = st["plastic"], st["brain"]
    ct = np.asarray(b.cell_type).astype(str)
    side = np.asarray(b.side).astype(str)
    kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
    nk, nm = len(p.kc_ids), len(p.mbon_ids)
    w0 = p.w0_gpu.get().astype(np.float64)
    O = np.zeros((nk, nm)); np.add.at(O, (kslot, mslot), w0)
    sub = ct[p.kc_ids]
    subs = [s for s in sorted(set(sub)) if (sub == s).sum() >= 50]      # drop tiny groups
    M = np.asarray([O[sub == s].sum(0) for s in subs])
    Mshare = M / M.sum(1, keepdims=True)
    mb_names = [f"{ct[j]}_{side[j]}" for j in p.mbon_ids]
    mb_main_subtype = [subs[int(np.argmax(M[:, m]))] if M[:, m].sum() > 0 else None for m in range(nm)]
    mb_main_share = [float(M[:, m].max() / M[:, m].sum()) if M[:, m].sum() > 0 else None for m in range(nm)]

    S, shares, contrib_top = [], [], []
    for x in ITEMS:
        tid = chars.index(x)
        C.reset(st, args)
        C.write_item(st, args, tid, plastic=True)
        dw = b._W.data[p.edge_pos_gpu].get().astype(np.float64) - w0
        a = np.zeros(nk); np.add.at(a, kslot, np.abs(dw))
        s_x = (a[:, None] * O).sum(0)
        S.append(s_x)
        shares.append({s: float(a[sub == s].sum() / max(a.sum(), 1e-30)) for s in subs})
        top = np.argsort(-s_x)[:5]
        contrib_top.append({"item": x, "top_mbons": [
            {"mbon": mb_names[m], "share_of_item_output": float(s_x[m] / s_x.sum()),
             "supplied_by": {s: float((a[sub == s][:, None] * O[sub == s][:, [m]]).sum() / max(s_x[m], 1e-30))
                             for s in subs}} for m in top]})
    S = np.asarray(S)
    Sn = S / S.sum(1, keepdims=True)
    out = {"active": active, "subtypes": subs, "mbon_names": mb_names,
           "mbon_main_subtype": mb_main_subtype, "mbon_main_subtype_share": mb_main_share,
           "subtype_to_mbon_share": Mshare.round(5).tolist(),
           "item_output_share": Sn.round(5).tolist(), "item_subtype_share": shares,
           "item_top_mbons": contrib_top}
    # how much of the item-to-item variation of the output is explained by the
    # subtype mixture alone: predict S_x from shares @ (mean profile per subtype)
    prof = np.asarray([O[sub == s].mean(0) for s in subs])
    A = np.asarray([[sh[s] for s in subs] for sh in shares])
    Pred = A @ prof
    Pn = Pred / Pred.sum(1, keepdims=True)
    Sc, Pc = Sn - Sn.mean(0), Pn - Pn.mean(0)
    out["subtype_mixture_R2_of_item_variation"] = float(1 - ((Sc - Pc) ** 2).sum() / (Sc ** 2).sum())
    Path(out_path).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("subtypes:", subs)
    print("R2 of item-to-item output variation explained by subtype mixture alone:",
          round(out["subtype_mixture_R2_of_item_variation"], 3))
    for s_i, s in enumerate(subs):
        top = np.argsort(-Mshare[s_i])[:4]
        print(f"  {s:>12} -> " + ", ".join(f"{mb_names[m]} {Mshare[s_i, m]:.2f}" for m in top))
    for c in contrib_top[:8]:
        t = c["top_mbons"][0]
        dom = max(t["supplied_by"], key=t["supplied_by"].get)
        print(f"  item {c['item']}: top {t['mbon']} ({t['share_of_item_output']:.2f} of output), "
              f"supplied mainly by {dom} ({t['supplied_by'][dom]:.2f})")


if __name__ == "__main__":
    main(sys.argv[1])
