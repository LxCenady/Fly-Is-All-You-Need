"""Connectome transforms: each returns a new connectome and an info dict.

Used by specs ("transforms"), by controls (rewired copies) and by the
importer. Every transform names the layers it touches; none assumes which
layers exist.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..core.connectome import Layer, canonical
from ..core.registry import register

INHIBITORY_FLY = {"gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}


def _rows(W):
    return np.repeat(np.arange(W.shape[0]), np.diff(W.indptr)).astype(np.int64)


# ---------------------------------------------------------------- structure
@register("transform", "cut_inputs")
def cut_inputs(cx, sel, population, layers):
    """Remove every synapse onto `population` in the named layers."""
    mask = sel.mask(population)
    for name in layers:
        lay = cx.layer(name)
        if lay.edges:                        # keep the edge order: zero values
            W = lay.matrix.copy()
            W.data[mask[_rows(W)]] = 0.0
            cx = cx.with_layer(name, Layer(lay.kind, W, lay.edges))
        else:
            W = sparse.diags((~mask).astype(np.float32)) @ lay.matrix
            cx = cx.with_layer(name, lay.with_matrix(W))
    return cx, {"cut": int(mask.sum()), "layers": list(layers)}


@register("transform", "lesion")
def lesion(cx, sel, population, layers, direction="both"):
    """Silence `population`: zero its outputs, inputs or both (the edges
    stay stored, so per-edge arrays keep their order)."""
    if direction not in ("both", "outputs", "inputs"):
        raise ValueError(f"lesion: direction {direction!r}")
    mask = sel.mask(population)
    for name in layers:
        lay = cx.layer(name)
        W = lay.matrix.copy()
        hit = np.zeros(W.nnz, bool)
        if direction in ("both", "outputs"):
            hit |= mask[W.indices]
        if direction in ("both", "inputs"):
            hit |= mask[_rows(W)]
        W.data[hit] = 0.0
        cx = cx.with_layer(name, Layer(lay.kind, W, lay.edges))
    return cx, {"lesioned": int(mask.sum()), "direction": direction}


@register("transform", "scale")
def scale(cx, sel, layer, factor, pre=None, post=None):
    """Multiply the weights of a layer (optionally only from `pre` / onto
    `post` populations) by `factor`."""
    lay = cx.layer(layer)
    W = lay.matrix.copy()
    hit = np.ones(W.nnz, bool)
    if pre is not None:
        hit &= sel.mask(pre)[W.indices]
    if post is not None:
        hit &= sel.mask(post)[_rows(W)]
    W.data[hit] *= np.float32(factor)
    cx = cx.with_layer(layer, Layer(lay.kind, W, lay.edges))
    return cx, {"edges": int(hit.sum())}


# ---------------------------------------------------------------- weights
@register("transform", "transmitter_signs")
def transmitter_signs(cx, sel, layer, column, table=None, default=1.0,
                      match="contains"):
    """Sign every output of a neuron from its transmitter annotation.

    `table` maps a transmitter (lower case) to a factor, e.g. {"gaba": -1};
    `match` "contains" (the annotation contains the key) or "equals". Unlisted
    transmitters get `default`. The default table is flybrain's MaleCNS rule
    (GABA, glutamate and histamine inhibitory); for another animal pass its
    own table.
    """
    table = INHIBITORY_FLY if table is None else table
    labels = np.char.lower(cx.column(column).astype(str))
    sign = np.full(cx.n, float(default), np.float32)
    for key, value in table.items():
        hit = (np.char.find(labels, key.lower()) >= 0 if match == "contains"
               else labels == key.lower())
        sign[hit] = value
    lay = cx.layer(layer)
    W = lay.matrix.copy()
    W.data *= sign[W.indices]
    info = {"negative_neurons": int((sign < 0).sum())}
    return cx.with_layer(layer, Layer(lay.kind, W, lay.edges)), info


@register("transform", "edge_labels")
def edge_labels(cx, sel, layer, column, name=None, side="pre"):
    """Give every edge of `layer` the value of a neuron column of its
    presynaptic (side "pre") or postsynaptic ("post") neuron, as the edge
    array `name` (default: the column name). E.g. a per-neuron transmitter
    annotation becomes a per-edge label for the by_transmitter synapse."""
    lay = cx.layer(layer)
    W = lay.matrix
    if side not in ("pre", "post"):
        raise ValueError(f"edge_labels: side {side!r}")
    idx = W.indices if side == "pre" else _rows(W)
    edges = {**lay.edges, name or column: cx.column(column)[idx]}
    return cx.with_layer(layer, Layer(lay.kind, W, edges)), {}


@register("transform", "normalise_inputs")
def normalise_inputs(cx, sel, layer, minimum=1.0):
    """Divide each neuron's inputs by its total absolute input (at least
    `minimum`): flybrain's recipe for MaleCNS."""
    lay = cx.layer(layer)
    W = lay.matrix.copy()
    total = np.asarray(abs(W).sum(1, dtype=np.float64)).ravel()
    total = total.astype(np.float32)
    W.data /= np.maximum(total, minimum)[_rows(W)]
    return cx.with_layer(layer, Layer(lay.kind, W, lay.edges)), {}


# ---------------------------------------------------------------- nulls
def class_labels(sel, spec: dict) -> np.ndarray:
    """Class of each neuron: the value of `else_col`, overridden by membership
    of the listed populations (a later population wins)."""
    col = spec.get("else_col")
    n = sel.cx.n
    labels = (sel.cx.column(col).astype(str).astype(object) if col
              else np.full(n, "other", dtype=object))
    for name in spec.get("populations", []):
        labels[sel.mask(name)] = name
    return labels.astype(str)


@register("transform", "rewire")
def rewire(cx, sel, layers, seed=0, classes=None, skip_missing=False):
    """Degree-preserving null model of one or more layers.

    `layers` maps layer name -> method, applied in order with one random
    stream seeded by `seed`:
      permute_presynaptic  (presynaptic neuron, weight) pairs permuted across
                           edges, within (pre class, post class) blocks if
                           `classes` is given; duplicates merged
      double_edge_swap     a-b, c-d -> a-d, c-b on a symmetric layer; with
                           classes, only when b and d share a class
    `classes`: {"populations": [...], "else_col": col} or None (whole network).
    """
    rng = np.random.default_rng(seed)
    labels = class_labels(sel, classes) if classes else None
    cls = np.unique(labels, return_inverse=True)[1] if labels is not None else None
    info = {"mode": "within classes" if classes else "whole network",
            "seed": int(seed)}
    for name, method in layers.items():
        if name not in cx.layers:
            if skip_missing:
                continue
            raise KeyError(f"rewire: no layer {name!r}")
        lay = cx.layer(name)
        if method == "permute_presynaptic":
            W, i = _permute_presynaptic(lay.matrix, rng, cls)
        elif method == "double_edge_swap":
            W, i = _double_edge_swap(lay.matrix, rng, cls)
        else:
            raise ValueError(f"rewire: unknown method {method!r}")
        cx = cx.with_layer(name, Layer(lay.kind, W))
        info[name] = i
    chem = info.get(next(iter(layers)), {})
    if "merged_fraction" in chem:
        info["message"] = (f"control: rewired ({info['mode']}, seed {seed}, "
                           f"{chem['merged_fraction']:.1%} of edges merged)")
    return cx, info


def _permute_presynaptic(W, rng, cls):
    post, pre, w = _rows(W), W.indices.astype(np.int64), W.data.copy()
    if cls is None:
        src = rng.permutation(len(w))
        new_pre, new_w = pre[src], w[src]
    else:
        key = cls[pre] * (cls.max() + 1) + cls[post]
        order = np.argsort(key, kind="stable")
        bounds = np.flatnonzero(np.diff(key[order])) + 1
        new_pre, new_w = pre.copy(), w.copy()
        for g in np.split(order, bounds):
            q = g[rng.permutation(len(g))]
            new_pre[g], new_w[g] = pre[q], w[q]
    R = canonical(sparse.coo_matrix((new_w, (post, new_pre)), shape=W.shape))
    merged = float(1 - R.nnz / max(len(w), 1))
    return R, {"edges": int(len(w)), "edges_after_merge": int(R.nnz),
               "merged_fraction": merged}


def _double_edge_swap(G, rng, cls, per_edge=10):
    U = sparse.triu(G, k=1).tocoo()
    a, b, w = U.row.astype(np.int64), U.col.astype(np.int64), U.data.copy()
    m = len(w)
    if m < 2:
        return G, {"swaps": 0}
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
            edges.discard((x, y))
            edges.discard((y, x))
        edges |= {(p, s), (s, p), (r, q), (q, r)}
        a[i], b[i], a[j], b[j] = p, s, r, q
        done += 1
    S = sparse.coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=G.shape)
    return canonical(S), {"swaps": done}
