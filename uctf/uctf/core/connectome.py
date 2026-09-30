"""A connectome as data: neurons with annotations, and named synapse layers.

Nothing here knows what a neuron type or a synapse type means. A connectome is

    neurons    columns of per-neuron data, each an array of length n: annotations
               (cell type, class, side, transmitter, ...) or numbers (soma size,
               measured time constants, ...)
    positions  optional (n, 3) coordinates, NaN where unknown
    layers     named synapse layers. Each has a `kind` (free text, e.g.
               "chemical", "electrical", "neuromodulatory"), a sparse matrix
               (rows = postsynaptic, columns = presynaptic neuron) and optional
               per-edge arrays (synapse count, transmitter, receptor, delay, ...)
               in the order of the matrix's stored entries
    meta       name, species, source, version, license, citation, units, ...

On disk (format version 2) a connectome is a folder:

    connectome.json        {"format": "uctf-connectome", "version": 2,
                            "meta": {...}, "layers": {name: {"kind": ...}}}
    neurons.npz            one array per column; "positions" if known
    <layer>.npz            the layer's matrix (scipy.sparse.save_npz)
    <layer>.edges.npz      optional per-edge arrays

Folders in the earlier format (weights.npz [+ gap.npz] + neurons.npz or
neurons.csv) are read as a "chemical" layer [and an "electrical" layer].
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import sparse

FORMAT, VERSION = "uctf-connectome", 2


def canonical(matrix) -> sparse.csr_matrix:
    """float32 CSR with summed duplicates and sorted indices, so that its layout
    never changes later (a GPU library may otherwise re-sort it in place)."""
    m = sparse.csr_matrix(matrix, dtype=np.float32)
    m.sum_duplicates()
    m.sort_indices()
    return m


@dataclass
class Layer:
    """One kind of synapse: matrix rows are postsynaptic neurons."""

    kind: str
    matrix: sparse.csr_matrix
    edges: dict = field(default_factory=dict)

    def with_matrix(self, matrix, edges=None) -> "Layer":
        return Layer(self.kind, canonical(matrix), {} if edges is None else edges)


@dataclass
class Connectome:
    neurons: dict
    layers: dict
    positions: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(next(iter(self.neurons.values())))

    @property
    def name(self) -> str:
        return self.meta.get("name", "connectome")

    def column(self, name: str) -> np.ndarray:
        if name not in self.neurons:
            cols = ", ".join(self.neurons) or "none"
            raise KeyError(f"{self.name} has no neuron column {name!r} ({cols})")
        return self.neurons[name]

    def layer(self, name: str) -> Layer:
        if name not in self.layers:
            names = ", ".join(self.layers) or "none"
            raise KeyError(f"{self.name} has no synapse layer {name!r} ({names})")
        return self.layers[name]

    def with_layer(self, name: str, layer: Layer) -> "Connectome":
        return replace(self, layers={**self.layers, name: layer})

    def layers_of_kind(self, kind: str) -> list:
        return [n for n, lay in self.layers.items() if lay.kind == kind]

    def validate(self) -> "Connectome":
        n = self.n
        for col, arr in self.neurons.items():
            if len(arr) != n:
                raise ValueError(f"neuron column {col!r} has {len(arr)} rows, not {n}")
        for name, lay in self.layers.items():
            if lay.matrix.shape != (n, n):
                raise ValueError(f"layer {name!r} is {lay.matrix.shape}, not {n}x{n}")
            for key, arr in lay.edges.items():
                if len(arr) != lay.matrix.nnz:
                    raise ValueError(f"edge array {name}.{key} does not match")
        if self.positions is not None and self.positions.shape != (n, 3):
            raise ValueError("positions must be (n, 3)")
        return self

    # ------------------------------------------------------------ disk
    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        cols = dict(self.neurons)
        if self.positions is not None:
            cols["positions"] = self.positions.astype(np.float32)
        np.savez_compressed(folder / "neurons.npz", **cols)
        layers = {}
        for name, lay in self.layers.items():
            sparse.save_npz(folder / f"{name}.npz", lay.matrix)
            if lay.edges:
                np.savez_compressed(folder / f"{name}.edges.npz", **lay.edges)
            layers[name] = {"kind": lay.kind, "edges": sorted(lay.edges)}
        head = {"format": FORMAT, "version": VERSION,
                "meta": self.meta, "layers": layers}
        text = json.dumps(head, indent=1, ensure_ascii=False)
        (folder / "connectome.json").write_text(text, encoding="utf-8")
        return folder


def load_folder(folder, name: str | None = None) -> Connectome:
    """A connectome folder in format 2 or in the earlier format."""
    folder = Path(folder)
    neurons, positions = _read_neurons(folder)
    head_file = folder / "connectome.json"
    if head_file.exists():
        head = json.loads(head_file.read_text(encoding="utf-8"))
        if head.get("format") != FORMAT:
            raise ValueError(f"{head_file} is not a {FORMAT} file")
        layers = {}
        for lname, info in head["layers"].items():
            matrix = canonical(sparse.load_npz(folder / f"{lname}.npz"))
            edges = {}
            if info.get("edges"):
                z = np.load(folder / f"{lname}.edges.npz", allow_pickle=False)
                edges = {k: z[k] for k in z.files}
            layers[lname] = Layer(info["kind"], matrix, edges)
        meta = head.get("meta", {})
    else:                                           # the earlier format
        layers = {"chemical": Layer(
            "chemical", canonical(sparse.load_npz(folder / "weights.npz")))}
        if (folder / "gap.npz").exists():
            gap = canonical(sparse.load_npz(folder / "gap.npz"))
            layers["electrical"] = Layer("electrical", gap)
        meta = {}
    meta = {**meta, "name": name or meta.get("name") or folder.name}
    return Connectome(neurons, layers, positions, meta).validate()


def _read_neurons(folder: Path):
    if (folder / "neurons.npz").exists():
        z = np.load(folder / "neurons.npz", allow_pickle=False)
        cols = {k: z[k] for k in z.files if k != "positions"}
        cols = {k: (v.astype(str) if v.dtype.kind in "US" else v)
                for k, v in cols.items()}
        pos = z["positions"].astype(np.float32) if "positions" in z.files else None
        return cols, pos
    if (folder / "neurons.csv").exists():
        return read_neuron_csv(folder / "neurons.csv")
    raise FileNotFoundError(f"{folder} has no neurons.npz or neurons.csv")


def read_neuron_csv(path) -> tuple:
    """Columns of a neuron table as string arrays; x, y, z become positions."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    cols = {k: np.asarray([r[k] for r in rows]).astype(str) for k in rows[0]}
    pos = None
    if all(c in cols for c in ("x", "y", "z")):
        def num(v):
            return float(v) if v not in ("", "nan", "NaN") else np.nan
        xyz = [[num(v) for v in cols.pop(c)] for c in ("x", "y", "z")]
        pos = np.asarray(xyz, np.float32).T
    return cols, pos
