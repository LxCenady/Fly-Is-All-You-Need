"""Null connectomes: is the real wiring needed, or would any network with the same degrees do?

rewire(cx, seed)                     permute (presynaptic neuron, weight) pairs across all edges
rewire(cx, seed, classes=labels)     ... only among edges whose (pre class, post class) match
Every neuron keeps its number of inputs, its number of outputs and the sign and weights of its
outputs; which neurons it connects to is randomised.  Duplicate edges created by the permutation
are merged (weights summed).  This is the control of the connectome-reservoir report
(paper_lm/, Section VII): there, rewired connectomes predicted text as well as the real one.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .data import Connectome, canonical


def rewire(cx: Connectome, seed: int = 0, classes: np.ndarray | None = None) -> tuple[Connectome, dict]:
    W = cx.W.tocsr()
    post = np.repeat(np.arange(cx.n), np.diff(W.indptr)).astype(np.int64)
    pre = W.indices.astype(np.int64)
    w = W.data.copy()
    rng = np.random.default_rng(seed)
    if classes is None:
        src = rng.permutation(len(w))
        new_pre, new_w = pre[src], w[src]
    else:
        _, cls = np.unique(np.asarray(classes).astype(str), return_inverse=True)
        key = cls[pre] * (cls.max() + 1) + cls[post]
        order = np.argsort(key, kind="stable")                  # edges grouped by block
        bounds = np.flatnonzero(np.diff(key[order])) + 1
        new_pre, new_w = pre.copy(), w.copy()
        for g in np.split(order, bounds):                       # one permutation per block
            q = g[rng.permutation(len(g))]
            new_pre[g], new_w[g] = pre[q], w[q]
    R = canonical(sparse.coo_matrix((new_w, (post, new_pre)), shape=W.shape))
    info = {"edges": int(len(w)), "edges_after_merge": int(R.nnz),
            "merged_fraction": float(1 - R.nnz / max(len(w), 1)),
            "mode": "within classes" if classes is not None else "whole network", "seed": int(seed)}
    G = None
    if cx.G is not None:
        G, info["gap_swaps"] = swap_symmetric(cx.G, rng, None if classes is None else cls)
    return Connectome(R, cx.annotations, cx.positions, cx.name + " (rewired)", G), info


def swap_symmetric(G, rng, cls=None, per_edge: int = 10):
    """Degree-preserving rewiring of a symmetric (electrical) network by double-edge swaps:
    a-b, c-d -> a-d, c-b, rejected when it would create a self-loop or an existing edge.  Each edge
    keeps its weight.  With classes, a swap is allowed only when b and d share a class, so every
    edge keeps the classes of its two ends."""
    U = sparse.triu(G, k=1).tocoo()
    a, b, w = U.row.astype(np.int64), U.col.astype(np.int64), U.data.copy()
    m = len(w)
    if m < 2:
        return G, 0
    edges = set(zip(a.tolist(), b.tolist())) | set(zip(b.tolist(), a.tolist()))
    done = 0
    for _ in range(per_edge * m):
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        p, q = (a[i], b[i]) if rng.random() < 0.5 else (b[i], a[i])
        r, s = (a[j], b[j]) if rng.random() < 0.5 else (b[j], a[j])
        if p == s or r == q or (p, s) in edges or (r, q) in edges:
            continue
        if cls is not None and cls[q] != cls[s]:
            continue
        for x, y in ((p, q), (r, s)):
            edges.discard((x, y)); edges.discard((y, x))
        edges |= {(p, s), (s, p), (r, q), (q, r)}
        a[i], b[i], a[j], b[j] = p, s, r, q
        done += 1
    S = sparse.coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=G.shape)
    return canonical(S), done


def class_labels(sel, spec: dict) -> np.ndarray:
    """Class of every neuron: the value of an annotation column, overridden by membership of the
    listed populations (a later population wins, as in paper_lm's shuffle_connectome.py).
    spec: {"populations": ["KC", "PN", ...], "else_col": "superclass"}."""
    col = spec.get("else_col")
    labels = (sel.cx.column(col).astype(object) if col else np.full(sel.cx.n, "other", dtype=object))
    for name in spec.get("populations", []):
        labels[sel.mask(name)] = name
    return labels.astype(str)
