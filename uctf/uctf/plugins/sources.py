"""Connectome sources."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import sparse

from ..core.connectome import Connectome, Layer, canonical, load_folder
from ..core.registry import register


@register("source", "folder")
def folder(path, name=None):
    """A connectome folder (see core/connectome.py)."""
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not path.exists():
        raise FileNotFoundError(
            f"connectome folder {path} not found; build it with the example's "
            "prepare/import steps or fix the spec's path")
    return load_folder(path, name)


@register("source", "flybrain")
def flybrain(data=None):
    """MaleCNS v1.0 as prebuilt by flybrain (166,700 neurons): one chemical
    layer of signed, input-normalised synapse counts. Downloads ~260 MB on
    first use. Needs `flybrain` (use the patched 0.1.0.post1)."""
    from flybrain.data import ensure_data
    d = Path(ensure_data(data or os.environ.get("GPF_FLY_DATA")
                         or os.environ.get("FLY_DATA") or None))
    meta = np.load(d / "brain.npz")
    W = canonical(sparse.load_npz(d / "weights.npz"))
    cols = ("cell_type", "side", "superclass")
    neurons = {k: meta[k].astype(str) for k in cols if k in meta.files}
    pos = meta["positions"].astype(np.float32) if "positions" in meta.files else None
    info = {"name": "MaleCNS v1.0 (flybrain)", "species": "Drosophila melanogaster",
            "source": "flybrain prebuilt data", "license": "CC BY 4.0",
            "citation": "Berg et al. 2026, Cell 189(18)",
            "units": {"chemical": "signed synapse count / total input"}}
    return Connectome(neurons, {"chemical": Layer("chemical", W)}, pos, info)
