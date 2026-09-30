"""M3: does the real PN->KC wiring matter for KC sparseness and cue-specific addressing?

Wiring conditions (only PN->KC entries in KC rows are touched; KC->MBON rows,
APL, DAN, everything else unchanged; nnz and indptr unchanged):
  real        the connectome
  pn_perm     presynaptic PN labels permuted across all PN->KC edges: every KC
              keeps its PN in-degree and its weight multiset, every PN keeps its
              KC out-degree; which PN feeds which KC is randomised
  pn_uniform  each PN->KC edge gets a uniformly random PN (KC in-degree and
              weights kept; PN out-degrees no longer preserved)
  (no duplicate PN within a KC row in either null; 2 seeds each)
Measurements at PN code 160 / 192 (x1.5) and 512 (x1.0):
  KC probe active fraction and cross-character Jaccard (9 characters)
  cue specificity on the MBON input current for pairs (a,c), (b,d), (e,g):
     spec = |I_own| / mean(|I_other|, |I_z|)
  KC-identity scramble on own probe (within_mbon, 10 draws): e_ratio, e_cos
Stated reading: if nulls reproduce real within spread, the specific PN->KC
wiring is not needed for these properties (only degrees / weights are).
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
import mb_persistent_memory_probe as pmp  # noqa: E402
from m2_address import group_perm  # noqa: E402

CH = list("abcdefgh") + ["z"]
PAIRS = [("a", "c"), ("b", "d"), ("e", "g")]


def rewire(st, mode, seed):
    """Rewire PN->KC entries of the live CSR in place (host round trip)."""
    import cupy as cp
    from cupyx.scipy import sparse as cusparse
    b = st["brain"]
    ct = np.asarray(b.cell_type).astype(str)
    pn = np.flatnonzero(np.char.find(ct, "PN") >= 0)
    is_pn = np.zeros(b.n, bool); is_pn[pn] = True
    kc_rows = np.asarray(st["kc"], np.int64)
    ip, ix, dat = b._W.indptr.get(), b._W.indices.get().copy(), b._W.data.get().copy()
    pos = np.concatenate([np.arange(ip[r], ip[r + 1])[is_pn[ix[ip[r]:ip[r + 1]]]] for r in kc_rows])
    row_of = np.repeat(kc_rows, [int(is_pn[ix[ip[r]:ip[r + 1]]].sum()) for r in kc_rows])
    rng = np.random.default_rng(seed)
    side = np.asarray(b.side).astype(str)
    old = ix[pos]
    if mode == "pn_uniform":
        new = rng.choice(pn, size=len(pos))
        g = None
    else:
        # label permutations among PN->KC edges, optionally constrained:
        #   pn_perm            all edges
        #   pn_perm_side       only among PNs of the same hemisphere
        #   pn_perm_type       only among PNs of the same type (sides mixed)
        #   pn_perm_type_side  same type and same hemisphere
        lab = {"pn_perm": np.zeros(len(old), "<U1"), "pn_perm_side": side[old],
               "pn_perm_type": ct[old],
               "pn_perm_type_side": np.char.add(np.char.add(ct[old], "|"), side[old])}[mode]
        _, g = np.unique(lab, return_inverse=True)
        from m2_address import group_perm
        new = old[group_perm(g, rng)]
    members = None if g is None else {int(k): np.flatnonzero(g == k) for k in np.unique(g)}
    # resolve duplicate (row, pre) pairs by swapping within the same group
    for _ in range(500):
        key = row_of * b.n + new
        _, first = np.unique(key, return_index=True)
        dup = np.setdiff1d(np.arange(len(key)), first)
        if not len(dup):
            break
        if g is None:
            new[dup] = rng.choice(pn, size=len(dup))
        else:
            for d in dup:
                o = rng.choice(members[int(g[d])])
                new[d], new[o] = new[o], new[d]
    else:
        raise RuntimeError("could not remove duplicates")
    changed = float(np.mean(ix[pos] != new))
    ix[pos] = new
    for r in kc_rows:                              # re-sort each KC row, data follows
        lo, hi = ip[r], ip[r + 1]
        o = np.argsort(ix[lo:hi], kind="stable")
        ix[lo:hi], dat[lo:hi] = ix[lo:hi][o], dat[lo:hi][o]
    b._W = cusparse.csr_matrix((cp.asarray(dat), cp.asarray(ix), cp.asarray(ip)),
                               shape=b._W.shape)
    p = st["plastic"]
    live = p.xp.asnumpy(b._W.indices[p.edge_pos_gpu]).astype(np.int64)
    assert np.array_equal(live, p.edge_pre_ids), "KC->MBON identity moved"
    assert np.allclose(b._W.data[p.edge_pos_gpu].get(), p.w0_gpu.get())
    return {"n_pn_kc_edges": int(len(pos)), "frac_label_changed": changed}


def set_glom_codes(st, target: int, seed: int = 11):
    """Glomerular input code: each character drives ALL PNs of a random set of
    PN cell types (glomeruli), accumulated until >= target PNs.  Fixed seed, so
    every wiring condition sees the same codes.  MB_GLOM_SEED overrides the seed
    (replications with new glomerular codes); unset = the reference codes."""
    import os
    if os.environ.get("MB_GLOM_SEED"):
        seed = int(os.environ["MB_GLOM_SEED"])
    b, enc = st["brain"], st["enc"]
    ct = np.asarray(b.cell_type).astype(str)
    chans = np.asarray(enc.channels)
    types = np.unique(ct[chans])
    rng = np.random.default_rng(seed)
    codes = []
    for _ in range(len(enc.codes)):
        ids = []
        for t in rng.permutation(types):
            ids.extend(chans[ct[chans] == t].tolist())
            if len(ids) >= target:
                break
        codes.append(np.sort(np.asarray(ids, np.int64)))
    enc.codes = codes
    enc.position_codes = [codes] * enc.window
    b._lm_position_codes_gpu = None                 # drop the injector's cache
    return float(np.mean([len(c) for c in codes]))


def jacc(a, b):
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else float("nan")


def measure(st, args, chars):
    tid = {c: chars.index(c) for c in set(CH) | set("".join(x + y for x, y in PAIRS))}
    p = st["plastic"]
    kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
    w0 = p.w0_gpu.get().astype(np.float64)
    sets = {}
    for ch in CH:
        C.reset(st, args)
        C.probe(st, args, tid[ch])
        sets[ch] = st["_last_kc_counts"] > 0
    fr = [float(sets[c].mean()) for c in CH]
    js = [j for j in (jacc(sets[a], sets[b]) for a, b in combinations(CH, 2)) if not np.isnan(j)]

    def episode(item, plastic):
        C.reset(st, args)
        C.write_item(st, args, tid[item], plastic=plastic)
        C.gap(st, args, 32)
        return C.snapshot(st)

    def cur(dw, kc):
        return np.bincount(mslot, weights=dw * kc[kslot] / args.k, minlength=len(p.mbon_ids))

    spec, eratio, ecos = [], [], []
    for x, y in PAIRS:
        SX, S0 = episode(x, True), episode(x, False)
        dwx = SX["w_slots"].get().astype(np.float64) - w0
        I = {}
        kcs = {}
        for pch in (x, y, "z"):
            C.restore(st, S0, "zero")
            C.probe(st, args, tid[pch])
            kcs[pch] = st["_last_kc_counts"].astype(np.float64)
            I[pch] = np.linalg.norm(cur(dwx, kcs[pch]))
        spec.append(I[x] / max(np.mean([I[y], I["z"]]), 1e-30))
        ix_ = cur(dwx, kcs[x])
        for d in range(10):
            sg = group_perm(mslot, np.random.default_rng(d))
            j = cur(dwx[sg], kcs[x])
            eratio.append(np.linalg.norm(j) / max(np.linalg.norm(ix_), 1e-30))
            ecos.append(float(j @ ix_ / max(np.linalg.norm(j) * np.linalg.norm(ix_), 1e-30)))
    return {"kc_probe_frac": float(np.mean(fr)), "kc_jaccard": float(np.mean(js)) if js else None,
            "cue_spec_per_pair": [round(float(s), 3) for s in spec],
            "own_kcid_scramble_eratio_median": float(np.median(eratio)),
            "own_kcid_scramble_ecos_median": float(np.median(ecos))}


def main(out_path, points, code="random", glom_seed=11, conds=None):
    conds = conds or [("real", 0), ("pn_perm", 1), ("pn_perm", 2), ("pn_uniform", 1), ("pn_uniform", 2)]
    out, t0 = [], time.time()
    for active, scale in points:
        for mode, seed in conds:
            args, chars = C.protocol_args()
            args.active, args.drive, args.probe_drive = int(active), 1.0 * scale, 2.5 * scale
            st = C.build(args)
            st["enc"].drive = args.drive
            info = rewire(st, mode, seed) if mode != "real" else {}
            if code == "glom":
                info["glom_code_mean_size"] = set_glom_codes(st, int(active), glom_seed)
                info["glom_seed"] = glom_seed
            info["input_code"] = code
            m = measure(st, args, chars)
            out.append({"active": active, "scale": scale, "mode": mode, "seed": seed, **info, **m})
            Path(out_path).write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"{active} x{scale} {mode}#{seed}: KC {m['kc_probe_frac']:.3f} J {m['kc_jaccard']} "
                  f"spec {m['cue_spec_per_pair']} kcid-scr {m['own_kcid_scramble_eratio_median']:.2f}/"
                  f"{m['own_kcid_scramble_ecos_median']:.2f}  {time.time() - t0:.0f}s", flush=True)
            del st


if __name__ == "__main__":
    # argv: out points [code] [glom_seed] [pairs "ac,bd"] [conds "real:0,pn_perm:1"]
    pts = [(int(s.split(":")[0]), float(s.split(":")[1])) for s in sys.argv[2].split(",")]
    code = sys.argv[3] if len(sys.argv) > 3 else "random"
    gseed = int(sys.argv[4]) if len(sys.argv) > 4 else 11
    if len(sys.argv) > 5:
        PAIRS = [(s[0], s[1]) for s in sys.argv[5].split(",")]
    conds = ([(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[6].split(",")]
             if len(sys.argv) > 6 else None)
    main(sys.argv[1], pts, code, gseed, conds)
