"""Quantify the pre-fix CSR canonicalization bug for asset A (sensory_input=False)."""
import json, sys
import numpy as np
from scipy import sparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mechanism"))
import paths  # noqa: E402
DATA = str(paths.data_dir())
meta = np.load(DATA + r"\brain.npz")
W = sparse.load_npz(DATA + r"\weights.npz")
sup = meta["superclass"].astype(str)
ct = np.asarray(meta["cell_type"]).astype(str)
sensory = np.char.find(sup, "sensory") >= 0
Wm = (sparse.diags((~sensory).astype(np.float32)) @ W.tocsr()).tocsr()  # pre-fix path
out = {"raw_sorted": bool(W.tocsr().has_sorted_indices), "masked_nnz": int(Wm.nnz),
       "masked_has_sorted_indices_flag": bool(Wm.has_sorted_indices)}
ip, ix, dat = Wm.indptr.copy(), Wm.indices.copy(), Wm.data.copy()
# check actual sortedness per row
unsorted_rows = 0
for r in range(Wm.shape[0]):
    seg = ix[ip[r]:ip[r+1]]
    if seg.size > 1 and np.any(np.diff(seg) < 0):
        unsorted_rows += 1
out["rows_actually_unsorted"] = unsorted_rows

kc = np.flatnonzero(np.char.find(ct, "KC") >= 0)
mbon = np.flatnonzero(np.char.find(ct, "MBON") >= 0)
kcset = np.zeros(Wm.shape[0], bool); kcset[kc] = True
# emulate KCMBONPlasticity edge_pos caching on the pre-fix (unsorted) matrix
pos = []
for post in mbon:
    lo, hi = ip[post], ip[post+1]
    sel = np.flatnonzero(kcset[ix[lo:hi]]) + lo
    pos.extend(sel.tolist())
pos = np.asarray(pos, np.int64)
w0 = dat[pos].copy(); pre0 = ix[pos].copy()
# canonicalize (what cuSPARSE did in place on first SpMV per the 09-17 smoke review)
S = Wm.copy(); S.sum_duplicates(); S.sort_indices()
same_structure = np.array_equal(S.indptr, ip)
pre1 = S.indices[pos]; dat1 = S.data[pos]
moved = pre1 != pre0
out.update({
    "n_kc_mbon_edges": int(pos.size),
    "indptr_unchanged_by_sort": bool(same_structure),
    "edges_whose_slot_now_holds_a_different_presynaptic_cell": int(moved.sum()),
    "of_those_slot_now_holds_non_KC_presyn": int((~kcset[pre1][moved]).sum()),
    "mbon_rows_with_any_moved_slot": int(len(np.unique(np.searchsorted(ip, pos[moved], side='right') - 1))),
    "sum_abs_w_true_in_slots_after_sort": float(np.abs(dat1).sum()),
    "sum_abs_w0_cached": float(np.abs(w0).sum()),
    "sign_mismatch_slots(w0 vs true occupant)": int((np.sign(w0) != np.sign(dat1)).sum()),
    "rel_L2_change_of_MBON_input_rows_if_w0_written_back": None,
})
# effect of write_back with modulation=0: slots get w0 (stale) instead of true occupant
rows = np.unique(np.searchsorted(ip, pos, side='right') - 1)
row_mask = np.zeros(S.shape[0], bool); row_mask[rows] = True
seg_idx = np.concatenate([np.arange(S.indptr[r], S.indptr[r+1]) for r in rows])
before = S.data[seg_idx].copy()
S2 = S.data.copy(); S2[pos] = w0
after = S2[seg_idx]
out["rel_L2_change_of_MBON_input_rows_if_w0_written_back"] = float(np.linalg.norm(after-before)/np.linalg.norm(before))
# total MBON input from KCs (true) vs corrupted
out["n_mbon"] = int(mbon.size); out["n_kc"] = int(kc.size)
json.dump(out, open(sys.argv[1], "w"), indent=2)
