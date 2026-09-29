"""Build degree-preserving shuffled connectomes as control reservoirs (CPU).

Each edge keeps its postsynaptic neuron (row); the (presynaptic neuron, weight) pairs are
permuted among edges.  Every neuron keeps its in-degree, its out-degree and the weights it
sends (so a neuron's sign, i.e. Dale's law, is kept).  Duplicate pre->post pairs created by
the permutation are summed.
  full   one global permutation
  class  permutation only within blocks (class of pre, class of post), where class is
         KC / PN / MBON / DAN (PAM, PPL, PPM) / APL, else the superclass.  PN->KC stays
         PN->KC and KC->MBON stays KC->MBON, but the partners are random.
Usage: shuffle_connectome.py {full|class} OUTDIR [seed]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

import paths  # noqa: E402
SRC = paths.data_dir()


def classes():
    m = np.load(SRC / "brain.npz", allow_pickle=True)
    ct = m["cell_type"].astype(str); sc = m["superclass"].astype(str)
    lab = sc.copy().astype(object)
    for name, keys in (("KC", ("KC",)), ("PN", ("PN",)), ("MBON", ("MBON",)),
                       ("DAN", ("PAM", "PPL", "PPM")), ("APL", ("APL",))):
        mask = np.zeros(len(ct), bool)
        for k in keys:
            mask |= np.char.find(ct, k) >= 0
        lab[mask] = name
    _, idx = np.unique(lab.astype(str), return_inverse=True)
    return idx


def main(mode, out, seed=0):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    W = sparse.load_npz(SRC / "weights.npz").tocoo()
    row, col, dat = W.row.astype(np.int64), W.col.astype(np.int64), W.data
    rng = np.random.default_rng(seed)
    if mode == "full":
        p = rng.permutation(len(col))
        col2, dat2 = col[p], dat[p]
    else:
        cls = classes()
        key = cls[col] * (cls.max() + 1) + cls[row]
        order = np.argsort(key, kind="stable")
        bounds = np.flatnonzero(np.diff(key[order])) + 1
        col2, dat2 = col.copy(), dat.copy()
        for g in np.split(order, bounds):
            q = g[rng.permutation(len(g))]
            col2[g], dat2[g] = col[q], dat[q]
    W2 = sparse.csr_matrix((dat2, (row, col2)), shape=W.shape)
    W2.sum_duplicates()
    sparse.save_npz(out / "weights.npz", W2)
    shutil.copy(SRC / "brain.npz", out / "brain.npz")
    same = np.mean(col2 == col)
    print(f"{mode}: edges {W.nnz} -> {W2.nnz} after merging duplicates; "
          f"fraction of edges keeping their presynaptic partner {same:.4f}")
    # sanity: in-degree per neuron (distinct partners may shrink by merging) and total input weight
    print("in-weight preserved:", np.allclose(np.asarray(W.tocsr().sum(1)).ravel(),
                                              np.asarray(W2.sum(1)).ravel(), atol=1e-3))
    print("out-weight preserved:", np.allclose(np.asarray(W.tocsr().sum(0)).ravel(),
                                               np.asarray(W2.sum(0)).ravel(), atol=1e-3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
