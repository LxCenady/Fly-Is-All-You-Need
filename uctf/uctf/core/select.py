"""Choosing neurons: by annotation, or by wiring in a named synapse layer.

A selector is JSON:

    "KC"                                     a population named in the spec
    {"col": "cell_type", "contains": "KC"}   also "equals", "in" (list),
                                             "startswith", "regex",
                                             "range": [lo, hi] for numbers
    {"and": [s1, s2]}  {"or": [s1, s2]}  {"not": s}
    {"all": true}
    {"ids": [0, 5, 17]}                      explicit neuron indices
    {"top_targets_of": s1, "among": s2, "n": 256, "layer": "chemical"}
        the n neurons of s2 receiving the most summed |weight| from s1 in that
        layer, strongest first. "layer" may be left out only when the
        connectome has exactly one layer of kind "chemical".

Selecting returns neuron indices: ascending, except top_targets_of.
"""
from __future__ import annotations

import re

import numpy as np

from .connectome import Connectome


class Selector:
    def __init__(self, cx: Connectome, named: dict | None = None):
        self.cx = cx
        self.named = dict(named or {})
        self._cache: dict = {}

    def mask(self, sel) -> np.ndarray:
        m = np.zeros(self.cx.n, bool)
        m[self.ids(sel)] = True
        return m

    def ids(self, sel) -> np.ndarray:
        if isinstance(sel, str):
            return self._named(sel)
        if not isinstance(sel, dict):
            raise TypeError(f"a selector is a name or a JSON object: {sel!r}")
        if "col" in sel:
            return np.flatnonzero(self._column(sel))
        if "and" in sel:
            m = np.ones(self.cx.n, bool)
            for s in sel["and"]:
                m &= self.mask(s)
            return np.flatnonzero(m)
        if "or" in sel:
            m = np.zeros(self.cx.n, bool)
            for s in sel["or"]:
                m |= self.mask(s)
            return np.flatnonzero(m)
        if "not" in sel:
            return np.flatnonzero(~self.mask(sel["not"]))
        if sel.get("all"):
            return np.arange(self.cx.n)
        if "ids" in sel:
            return np.unique(np.asarray(sel["ids"], np.int64))
        if "top_targets_of" in sel:
            return self._top_targets(sel)
        raise ValueError(f"cannot read selector {sel}")

    def _named(self, name: str) -> np.ndarray:
        if name not in self.named:
            known = ", ".join(self.named) or "none"
            raise KeyError(f"unknown population {name!r} (defined: {known})")
        if name not in self._cache:
            self._cache[name] = self.ids(self.named[name])
        return self._cache[name]

    def _column(self, sel: dict) -> np.ndarray:
        col = self.cx.column(sel["col"])
        if "contains" in sel:
            return np.char.find(col.astype(str), sel["contains"]) >= 0
        if "equals" in sel:
            return col.astype(str) == str(sel["equals"])
        if "in" in sel:
            return np.isin(col.astype(str), [str(v) for v in sel["in"]])
        if "startswith" in sel:
            return np.char.startswith(col.astype(str), sel["startswith"])
        if "regex" in sel:
            rx = re.compile(sel["regex"])
            return np.array([bool(rx.search(v)) for v in col.astype(str)])
        if "range" in sel:
            lo, hi = sel["range"]
            x = col.astype(float)
            return (x >= lo) & (x <= hi)
        raise ValueError(f"selector {sel} needs a comparison")

    def _top_targets(self, sel: dict) -> np.ndarray:
        layer = sel.get("layer") or self._only_chemical_layer()
        W = self.cx.layer(layer).matrix
        source = self.mask(sel["top_targets_of"])
        among = self.mask(sel.get("among", {"all": True}))
        rows = np.repeat(np.arange(self.cx.n), np.diff(W.indptr))
        keep = source[W.indices] & among[rows]
        w = np.abs(W.data[keep])
        score = np.bincount(rows[keep], weights=w, minlength=self.cx.n)
        degree = np.bincount(rows[keep], minlength=self.cx.n)
        ids = np.flatnonzero(degree > 0)
        ids = ids[np.argsort(-score[ids].astype(np.float32), kind="stable")]
        return ids[:min(int(sel["n"]), len(ids))].astype(np.int64)

    def _only_chemical_layer(self) -> str:
        names = self.cx.layers_of_kind("chemical")
        if len(names) != 1:
            raise ValueError("top_targets_of needs 'layer': the connectome has "
                             f"{len(names)} chemical layers")
        return names[0]
