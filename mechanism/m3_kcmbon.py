"""M3b: does the real KC->MBON (compartment) routing shape the output direction?

Wiring conditions (only KC->MBON entries in MBON rows are touched):
  real       connectome
  kc_perm    KC presynaptic labels permuted across all 61,210 KC->MBON edges:
             every MBON keeps its KC in-degree and weight multiset, every KC keeps
             its MBON out-degree; which KC feeds which MBON (lobe/compartment
             structure) is randomised.  No duplicate KC within an MBON row.
The plasticity object is rebuilt on the rewired CSR (DAN->MBON gate axis is
unchanged, since DAN->MBON edges are untouched).  2 seeds.

Measurements (operating points PN 160 / 192 x1.5 and 512 x1.0; items a..h):
  I_x = write-attributable MBON input current for write x read by probe x
        (common no-write fast state, gap 32)
  item_cos_mean   mean pairwise cos(I_x, I_y) over 8 items  (low = distinct outputs)
  pr_items        participation ratio of the 8 x 97 matrix of I_x (output dimensionality)
  cue_spec        |I_x(probe x)| / |I_x(probe z)| median
  mbon_share_top5 share of |I|^2 carried by the top 5 MBONs (mean over items)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
from mb_plasticity import KCMBONPlasticity  # noqa: E402

ITEMS = list("abcdefgh")


def kc_class(label: str) -> str:
    if label.startswith("KCg"):
        return "gamma"
    if label.startswith("KCapbp") or label.startswith("KCa'b'"):
        return "apbp"
    if label.startswith("KCab"):
        return "ab"
    return "other"


def rewire_kcmbon(st, args, seed, level="all"):
    """Permute KC presynaptic labels over KC->MBON edges.

    level = all    : across all edges
            class  : only among KCs of the same class (gamma / ab / a'b')
            type   : only among KCs of the same annotated subtype
    """
    import cupy as cp
    from cupyx.scipy import sparse as cusparse
    b, p = st["brain"], st["plastic"]
    ct = np.asarray(b.cell_type).astype(str)
    ip, ix, dat = b._W.indptr.get(), b._W.indices.get().copy(), b._W.data.get().copy()
    pos = np.asarray(p.edge_pos, np.int64)
    rows = np.searchsorted(ip, pos, side="right") - 1
    rng = np.random.default_rng(seed)
    pre = ix[pos]
    if level.startswith("profile"):
        # swap whole KC output profiles: bijection pi on KC ids within groups;
        # edge k->m becomes pi(k)->m.  Every profile survives intact; only which
        # KC (hence which PN inputs) owns it changes.
        #   profile_type      groups = annotated subtype
        #   profile_all       one group
        #   profile_clusterK  groups = k-means cluster (K) of the KC's PN-type input
        #   profile_strength  groups = subtype x decile of total PN input weight
        kcs = np.unique(pre)
        side = np.asarray(b.side).astype(str)
        if level == "profile_type":
            lab = ct[kcs]
        elif level == "profile_all":
            lab = np.zeros(len(kcs), "<U1")
        elif level == "profile_side":
            lab = side[kcs]
        elif level == "profile_type_side":
            lab = np.char.add(np.char.add(ct[kcs], "|"), side[kcs])
        elif level.startswith("profile_cluster"):
            z = np.load(C.paths.OUT / "kc_input_clusters.npz")
            cmap = dict(zip(z["kc"].tolist(), z["k" + level[len("profile_cluster"):]].tolist()))
            lab = np.asarray([f"c{cmap[int(k)]}" if int(k) in cmap else f"solo{int(k)}" for k in kcs])
        elif level == "profile_strength":
            is_pn = np.char.find(ct, "PN") >= 0
            Wh = b._W.get()
            strength = np.asarray([Wh.data[Wh.indptr[k]:Wh.indptr[k + 1]][
                is_pn[Wh.indices[Wh.indptr[k]:Wh.indptr[k + 1]]]].sum() for k in kcs])
            lab = np.empty(len(kcs), object)
            for t in np.unique(ct[kcs]):
                idx = np.flatnonzero(ct[kcs] == t)
                q = np.quantile(strength[idx], np.linspace(0, 1, 11)[1:-1])
                lab[idx] = [f"{t}|{int(np.searchsorted(q, s))}" for s in strength[idx]]
            lab = lab.astype(str)
        else:
            raise ValueError(level)
        _, gk = np.unique(lab, return_inverse=True)
        from m2_address import group_perm
        pi = dict(zip(kcs.tolist(), kcs[group_perm(gk, rng)].tolist()))
        new = np.asarray([pi[int(k)] for k in pre], np.int64)
        changed = float(np.mean(pre != new))
        ix[pos] = new
        for r in np.unique(rows):
            lo, hi = ip[r], ip[r + 1]
            o = np.argsort(ix[lo:hi], kind="stable")
            ix[lo:hi], dat[lo:hi] = ix[lo:hi][o], dat[lo:hi][o]
        return _finish(st, args, b, ix, dat, ip, pos, changed)
    if level == "all":
        g = np.zeros(len(pos), np.int64)
    elif level == "side":
        _, g = np.unique(np.asarray(b.side).astype(str)[pre], return_inverse=True)
    else:
        lab = ct[pre] if level == "type" else np.asarray([kc_class(s) for s in ct[pre]])
        _, g = np.unique(lab, return_inverse=True)
    from m2_address import group_perm
    new = pre[group_perm(g, rng)]
    members = {int(k): np.flatnonzero(g == k) for k in np.unique(g)}
    # row -> multiset of labels; resolve duplicates only with swaps that keep
    # both rows duplicate-free (same group, so group-level structure is kept)
    from collections import Counter, defaultdict
    row_labels = defaultdict(Counter)
    for r, lab in zip(rows, new):
        row_labels[int(r)][int(lab)] += 1
    for _ in range(20):
        dup = [i for i in range(len(new)) if row_labels[int(rows[i])][int(new[i])] > 1]
        if not dup:
            break
        for d in dup:
            rd, ld = int(rows[d]), int(new[d])
            if row_labels[rd][ld] <= 1:
                continue
            cand = members[int(g[d])]
            for o in rng.choice(cand, size=min(len(cand), 400), replace=False):
                ro, lo = int(rows[o]), int(new[o])
                if ro == rd or row_labels[rd][lo] > 0 or row_labels[ro][ld] > 0:
                    continue
                for (r_, l_, s_) in ((rd, ld, -1), (ro, lo, -1), (rd, lo, 1), (ro, ld, 1)):
                    row_labels[r_][l_] += s_
                new[d], new[o] = lo, ld
                break
    if any(row_labels[int(r)][int(l)] > 1 for r, l in zip(rows, new)):
        raise RuntimeError("duplicates remain")
    assert np.array_equal(np.sort(new), np.sort(pre))
    changed = float(np.mean(ix[pos] != new))
    ix[pos] = new
    for r in np.unique(rows):
        lo, hi = ip[r], ip[r + 1]
        o = np.argsort(ix[lo:hi], kind="stable")
        ix[lo:hi], dat[lo:hi] = ix[lo:hi][o], dat[lo:hi][o]
    return _finish(st, args, b, ix, dat, ip, pos, changed)


def _finish(st, args, b, ix, dat, ip, pos, changed):
    import cupy as cp
    from cupyx.scipy import sparse as cusparse
    b._W = cusparse.csr_matrix((cp.asarray(dat), cp.asarray(ix), cp.asarray(ip)), shape=b._W.shape)
    st["plastic"] = KCMBONPlasticity(b, mode="biological", eta=args.eta,
                                     eligibility_tau=args.eligibility_tau,
                                     weight_tau=args.weight_tau,
                                     max_modulation=args.max_modulation,
                                     eligibility_mix=args.eligibility_mix)
    st["plastic"].reset()
    st["_other_mask"] = C._other_positions(st)
    assert st["plastic"].n_edges == len(pos)
    return {"frac_label_changed": changed}


def measure(st, args, chars):
    p = st["plastic"]
    kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
    w0 = p.w0_gpu.get().astype(np.float64)
    tid = {c: chars.index(c) for c in ITEMS + ["z"]}

    def cur(dw, kc):
        return np.bincount(mslot, weights=dw * kc[kslot] / args.k, minlength=len(p.mbon_ids))

    I, spec = [], []
    for x in ITEMS:
        C.reset(st, args); C.write_item(st, args, tid[x], plastic=True); C.gap(st, args, 32)
        SX = C.snapshot(st)
        C.reset(st, args); C.write_item(st, args, tid[x], plastic=False); C.gap(st, args, 32)
        S0 = C.snapshot(st)
        dw = SX["w_slots"].get().astype(np.float64) - w0
        k = {}
        for pch in (x, "z"):
            C.restore(st, S0, "zero")
            C.probe(st, args, tid[pch])
            k[pch] = st["_last_kc_counts"].astype(np.float64)
        ix_ = cur(dw, k[x])
        I.append(ix_)
        spec.append(np.linalg.norm(ix_) / max(np.linalg.norm(cur(dw, k["z"])), 1e-30))
    I = np.asarray(I)
    U = I / np.maximum(np.linalg.norm(I, axis=1, keepdims=True), 1e-30)
    G = U @ U.T
    iu = np.triu_indices(len(ITEMS), 1)
    ev = np.linalg.svd(I - I.mean(0), compute_uv=False) ** 2
    top5 = np.mean([np.sort(r ** 2)[::-1][:5].sum() / max((r ** 2).sum(), 1e-30) for r in I])
    return {"item_cos_mean": float(G[iu].mean()), "item_cos_min": float(G[iu].min()),
            "pr_items_centred": float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30)),
            "cue_spec_median": float(np.median(spec)),
            "mbon_share_top5": float(top5),
            "I_norm_median": float(np.median(np.linalg.norm(I, axis=1)))}


def main(out_path, points, conds, code="random"):
    out, t0 = [], time.time()
    for active, scale in points:
        for mode, seed in conds:
            args, chars = C.protocol_args()
            args.active, args.drive, args.probe_drive = int(active), 1.0 * scale, 2.5 * scale
            st = C.build(args)
            st["enc"].drive = args.drive
            level = ({"kc_perm": "all", "kc_perm_class": "class", "kc_perm_type": "type",
                      "kc_perm_side": "side"}.get(mode)
                     or (mode if mode.startswith("profile") else None))
            if mode.startswith("pn_"):                      # rewire the INPUT side instead
                from m3_pnkc import rewire as rewire_pnkc
                info = rewire_pnkc(st, mode, seed)
            else:
                info = rewire_kcmbon(st, args, seed, level) if mode != "real" else {}
            if code == "glom":                     # bilateral glomerular symbol codes
                from m3_pnkc import set_glom_codes
                info["glom_code_mean_size"] = set_glom_codes(st, int(active), 11)
            info["input_code"] = code
            m = measure(st, args, chars)
            out.append({"active": active, "scale": scale, "mode": mode, "seed": seed, **info, **m})
            Path(out_path).write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"{active} x{scale} {mode}#{seed}: {json.dumps(m)} {time.time() - t0:.0f}s", flush=True)
            del st


if __name__ == "__main__":
    pts = [(int(s.split(":")[0]), float(s.split(":")[1])) for s in sys.argv[2].split(",")]
    # argv[3]: conditions like "real:0,kc_perm_class:1,kc_perm_type:1"
    conds = ([(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[3].split(",")]
             if len(sys.argv) > 3 else [("real", 0), ("kc_perm", 1), ("kc_perm", 2)])
    main(sys.argv[1], pts, conds, sys.argv[4] if len(sys.argv) > 4 else "random")
