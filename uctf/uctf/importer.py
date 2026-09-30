"""Build a connectome folder (format 2) from a neuron table and an edge list.

    python -m uctf import --neurons neurons.csv --edges edges.csv --out DIR
        [--id-col id --pre-col pre --post-col post --weight-col weight]
        [--type-col type --electrical electrical]
        [--edge-cols nt,confidence]
        [--sign-col nt --inhibitory gaba,glutamate,histamine]
        [--raw]

neurons.csv: one row per neuron; an id column, any annotation columns
(cell type, class, transmitter, ...), optional x, y, z positions.
edges.csv: one row per connection: pre id, post id, synapse count.

Layers: rows whose --type-col equals --electrical go to the symmetric
"electrical" layer (raw counts, the larger of both listings); the rest go
to the "chemical" layer. --edge-cols keeps per-edge columns (e.g. a
per-synapse transmitter or confidence) as edge arrays of the chemical
layer; this needs each (pre, post) pair listed once.

Unless --raw, the chemical layer is then passed through two transforms
(plugins/transforms.py): transmitter_signs (outputs of neurons whose
--sign-col contains an --inhibitory name become negative) and
normalise_inputs (each neuron's inputs divided by its total absolute
input, at least 1): flybrain's recipe for MaleCNS. With --raw the folder
keeps raw counts and a spec can apply its own transforms.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import sparse

from .core.connectome import Connectome, Layer, canonical, read_neuron_csv
from .core.select import Selector
from .plugins import transforms


def read_edges(path, order, a):
    """Edge rows -> (chemical rows, electrical rows, extra columns)."""
    chem, elec = [], []
    extra = {c: [] for c in a.edge_cols}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = order.get(r[a.pre_col].strip())
            j = order.get(r[a.post_col].strip())
            if i is None or j is None:
                continue
            w = float(r[a.weight_col])
            kind = r[a.type_col].strip().lower() if a.type_col else ""
            if kind == a.electrical:
                elec.append((i, j, w))
                continue
            chem.append((i, j, w))
            for c in extra:
                extra[c].append(r[c])
    return chem, elec, extra


def chemical_layer(rows, n, extra) -> Layer:
    pre, post, w = (np.asarray(c) for c in zip(*rows))
    coo = (w.astype(np.float32), (post.astype(np.int64), pre.astype(np.int64)))
    W = canonical(sparse.coo_matrix(coo, shape=(n, n)))
    if not extra:
        return Layer("chemical", W)
    if W.nnz != len(w):
        raise ValueError("--edge-cols needs each (pre, post) pair once")
    order = np.lexsort((pre, post))             # CSR order: row=post, col=pre
    edges = {c: np.asarray(v)[order] for c, v in extra.items()}
    return Layer("chemical", W, edges)


def electrical_layer(rows, n) -> Layer:
    i, j, w = (np.asarray(c) for c in zip(*rows))
    E = sparse.coo_matrix((w.astype(np.float32), (i, j)), shape=(n, n))
    E = E.tocsr()
    E.sum_duplicates()
    G = canonical(E.maximum(E.T))
    G.setdiag(0)
    G.eliminate_zeros()
    return Layer("electrical", G)


def build(a) -> Connectome:
    neurons, pos = read_neuron_csv(Path(a.neurons))
    ids = neurons[a.id_col]
    order = {v: i for i, v in enumerate(ids)}
    n = len(ids)
    chem, elec, extra = read_edges(a.edges, order, a)
    layers = {"chemical": chemical_layer(chem, n, extra)}
    if elec:
        layers["electrical"] = electrical_layer(elec, n)
    neurons["id"] = np.asarray(ids).astype(str)
    if a.id_col != "id":
        neurons.pop(a.id_col)
    meta = {"name": a.name or Path(a.out).name,
            "source": f"{Path(a.neurons).name}, {Path(a.edges).name}",
            "units": {"chemical": "synapse count"}, "applied": []}
    cx = Connectome(neurons, layers, pos, meta).validate()
    return cx if a.raw else _recipe(cx, a)


def _recipe(cx, a) -> Connectome:
    sel = Selector(cx)
    if a.sign_col:
        table = {s: -1.0 for s in a.inhibitory}
        cx, _ = transforms.transmitter_signs(
            cx, sel, "chemical", a.sign_col, table=table)
        cx.meta["applied"].append(
            {"transform": "transmitter_signs",
             "params": {"column": a.sign_col, "table": table}})
    cx, _ = transforms.normalise_inputs(cx, sel, "chemical")
    cx.meta["applied"].append({"transform": "normalise_inputs"})
    cx.meta["units"]["chemical"] = "signed synapse count / total input"
    return cx


def parser():
    ap = argparse.ArgumentParser(
        prog="uctf import", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    add = ap.add_argument
    add("--neurons", required=True)
    add("--edges", required=True)
    add("--out", required=True)
    add("--name", help="connectome name (default: the folder name)")
    add("--id-col", default="id")
    add("--pre-col", default="pre")
    add("--post-col", default="post")
    add("--weight-col", default="weight")
    add("--type-col", help="edge column marking electrical synapses")
    add("--electrical", default="electrical")
    add("--edge-cols", default="", help="per-edge columns to keep")
    add("--sign-col", help="neuron column naming the transmitter")
    add("--inhibitory", default="gaba,glutamate,histamine")
    add("--raw", action="store_true", help="keep raw synapse counts")
    add("--no-normalise", dest="raw", action="store_true",
        help=argparse.SUPPRESS)                 # the name before 0.2
    return ap


def main(argv):
    a = parser().parse_args(argv)
    a.electrical = a.electrical.lower()
    a.edge_cols = [c.strip() for c in a.edge_cols.split(",") if c.strip()]
    a.inhibitory = [s.strip().lower() for s in a.inhibitory.split(",")
                    if s.strip()]
    cx = build(a)
    cx.save(a.out)
    W = cx.layer("chemical").matrix
    gap = ""
    if "electrical" in cx.layers:
        gap = f", {cx.layer('electrical').matrix.nnz // 2:,} gap junctions"
    print(f"{cx.n:,} neurons, {W.nnz:,} connections "
          f"({int((W.data < 0).sum()):,} inhibitory){gap}")
    print(f"columns: {', '.join(cx.neurons)}; "
          f"positions: {'yes' if cx.positions is not None else 'no'}")
    print(f"written to {a.out}")
