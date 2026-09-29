"""Connectomes as data: a signed, normalised weight matrix plus a table of neuron annotations.

On disk a connectome is a folder with
    weights.npz   scipy.sparse matrix (n x n), rows = postsynaptic, columns = presynaptic neuron,
                  float32 signed weights (the same format as flybrain's MaleCNS files)
    neurons.npz   one array of length n per annotation column (e.g. cell_type, superclass,
                  side), plus optionally positions (n x 3) for the brain view
                  (or neurons.csv with a header; x, y, z columns become positions)

Loaders:
    load_folder(path)              a folder in the format above
    load_flybrain(data=None)       MaleCNS v1.0 as distributed by flybrain (brain.npz + weights.npz;
                                   downloaded on first use)
    import_edges(...)              build a folder from a neuron table and an edge list
                                   (synapse counts, sign from neurotransmitter, per-neuron input
                                   normalisation: flybrain's recipe for MaleCNS)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class Connectome:
    W: sparse.csr_matrix                        # rows = post, columns = pre, float32
    annotations: dict = field(default_factory=dict)   # column name -> array of length n (str)
    positions: np.ndarray | None = None         # (n, 3) float32, NaN where unknown
    name: str = "connectome"

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def column(self, name: str) -> np.ndarray:
        if name not in self.annotations:
            raise KeyError(f"{self.name} has no annotation column {name!r}; "
                           f"columns: {', '.join(self.annotations) or 'none'}")
        return self.annotations[name]

    def cut_inputs(self, mask: np.ndarray) -> "Connectome":
        """Remove every synapse onto the neurons in mask (e.g. sensory neurons)."""
        W = sparse.diags((~mask).astype(np.float32)) @ self.W
        return Connectome(canonical(W), self.annotations, self.positions, self.name)

    def save(self, folder: Path | str) -> Path:
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(folder / "weights.npz", self.W.tocsr())
        cols = {k: np.asarray(v).astype(str) for k, v in self.annotations.items()}
        if self.positions is not None:
            cols["positions"] = self.positions.astype(np.float32)
        np.savez_compressed(folder / "neurons.npz", **cols)
        return folder


def canonical(W) -> sparse.csr_matrix:
    """CSR with summed duplicates and sorted indices (CuPy may otherwise re-sort in place)."""
    W = sparse.csr_matrix(W, dtype=np.float32)
    W.sum_duplicates(); W.sort_indices()
    return W


def load_folder(folder: Path | str, name: str | None = None) -> Connectome:
    folder = Path(folder)
    W = canonical(sparse.load_npz(folder / "weights.npz"))
    ann, pos = {}, None
    if (folder / "neurons.npz").exists():
        z = np.load(folder / "neurons.npz", allow_pickle=False)
        for k in z.files:
            if k == "positions":
                pos = z[k].astype(np.float32)
            else:
                ann[k] = z[k].astype(str)
    elif (folder / "neurons.csv").exists():
        ann, pos = read_neuron_csv(folder / "neurons.csv")
    for k, v in ann.items():
        if len(v) != W.shape[0]:
            raise ValueError(f"annotation {k!r} has {len(v)} rows, the matrix has {W.shape[0]} neurons")
    return Connectome(W, ann, pos, name or folder.name)


def read_neuron_csv(path: Path) -> tuple[dict, np.ndarray | None]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.asarray([r[k] for r in rows]).astype(str) for k in (rows[0].keys() if rows else [])}
    pos = None
    if all(c in cols for c in ("x", "y", "z")):
        pos = np.stack([np.asarray([float(v) if v not in ("", "nan") else np.nan for v in cols.pop(c)],
                                   np.float32) for c in ("x", "y", "z")], 1)
    return cols, pos


def fly_data_dir() -> str | None:
    return os.environ.get("GPF_FLY_DATA") or os.environ.get("FLY_DATA") or None


def load_flybrain(data: str | None = None) -> Connectome:
    """MaleCNS v1.0 as prebuilt by flybrain (166,700 neurons); downloads ~260 MB on first use."""
    from flybrain.data import ensure_data
    d = ensure_data(data or fly_data_dir())
    meta = np.load(Path(d) / "brain.npz")
    W = canonical(sparse.load_npz(Path(d) / "weights.npz"))
    ann = {k: meta[k].astype(str) for k in ("cell_type", "side", "superclass") if k in meta.files}
    pos = meta["positions"].astype(np.float32) if "positions" in meta.files else None
    return Connectome(W, ann, pos, "MaleCNS v1.0 (flybrain)")


def import_edges(neurons: Path | str, edges: Path | str, out: Path | str, id_col: str = "id",
                 pre_col: str = "pre", post_col: str = "post", weight_col: str = "weight",
                 sign_col: str | None = None, inhibitory: tuple[str, ...] = ("gaba", "glutamate", "histamine"),
                 normalise: bool = True, name: str | None = None) -> Connectome:
    """Build a connectome folder from two CSV files.

    neurons: one row per neuron; column `id_col` holds its identifier, every other column is kept
             as an annotation (x, y, z become positions).  An optional `sign_col` (e.g. the predicted
             neurotransmitter) makes a neuron's outputs inhibitory when its value contains any of
             `inhibitory` (case-insensitive).
    edges:   one row per connection: pre id, post id, weight (synapse count).  Edges between neurons
             missing from the neuron table are dropped; repeated pairs are summed.
    normalise: divide each neuron's inputs by its total absolute input (at least 1), as flybrain
             does for MaleCNS, so that neurons with many synapses are not driven harder."""
    ann, pos = read_neuron_csv(Path(neurons))
    ids = ann.pop(id_col)
    order = {v: i for i, v in enumerate(ids)}
    n = len(ids)
    sign = np.ones(n, np.float32)
    if sign_col:
        labels = np.char.lower(ann[sign_col].astype(str))
        sign[np.array([any(s in lab for s in inhibitory) for lab in labels])] = -1.0
    pre, post, w = [], [], []
    with open(edges, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, b = order.get(r[pre_col]), order.get(r[post_col])
            if a is None or b is None:
                continue
            pre.append(a); post.append(b); w.append(float(r[weight_col]))
    pre, post, w = np.asarray(pre, np.int64), np.asarray(post, np.int64), np.asarray(w, np.float32)
    w = w * sign[pre]
    if normalise:
        incoming = np.bincount(post, weights=np.abs(w), minlength=n).astype(np.float32)
        w = w / np.maximum(incoming[post], 1.0)
    W = canonical(sparse.coo_matrix((w, (post, pre)), shape=(n, n)))
    ann["id"] = np.asarray(ids).astype(str)
    cx = Connectome(W, ann, pos, name or Path(out).name)
    cx.save(out)
    return cx


def load(source: dict) -> Connectome:
    """source: {"loader": "flybrain"} or {"loader": "folder", "path": ...}."""
    kind = source.get("loader", "folder")
    if kind == "flybrain":
        return load_flybrain(source.get("data"))
    if kind == "folder":
        path = Path(os.path.expandvars(os.path.expanduser(source["path"])))
        return load_folder(path, source.get("name"))
    raise ValueError(f"unknown connectome loader {kind!r} (flybrain, folder)")
