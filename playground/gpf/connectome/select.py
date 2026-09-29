"""Choosing neurons by their annotations (and, for readouts, by their wiring).

A selector is JSON:
    "KC"                                    a named population defined elsewhere in the spec
    {"col": "cell_type", "contains": "KC"}  also "equals", "in" (list), "startswith", "regex"
    {"and": [s1, s2]}  {"or": [s1, s2]}  {"not": s}
    {"all": true}
    {"ids": [0, 5, 17]}                     explicit neuron indices
    {"top_targets_of": s1, "among": s2, "n": 256}
        the n neurons of s2 that receive the most summed |weight| from s1, strongest first
Selecting returns neuron indices: ascending, except top_targets_of (strongest first).
"""
from __future__ import annotations

import re

import numpy as np

from .data import Connectome


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
            if sel not in self.named:
                raise KeyError(f"unknown population {sel!r}; defined: {', '.join(self.named) or 'none'}")
            if sel not in self._cache:
                self._cache[sel] = self.ids(self.named[sel])
            return self._cache[sel]
        if not isinstance(sel, dict):
            raise TypeError(f"a selector is a name or a JSON object, not {sel!r}")
        if "col" in sel:
            col = self.cx.column(sel["col"])
            if "contains" in sel:
                m = np.char.find(col, sel["contains"]) >= 0
            elif "equals" in sel:
                m = col == str(sel["equals"])
            elif "in" in sel:
                m = np.isin(col, [str(v) for v in sel["in"]])
            elif "startswith" in sel:
                m = np.char.startswith(col, sel["startswith"])
            elif "regex" in sel:
                rx = re.compile(sel["regex"])
                m = np.array([bool(rx.search(v)) for v in col])
            else:
                raise ValueError(f"selector {sel} needs contains, equals, in, startswith or regex")
            return np.flatnonzero(m)
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
            return top_targets(self.cx, self.mask(sel["top_targets_of"]),
                               self.mask(sel.get("among", {"all": True})), int(sel["n"]))
        raise ValueError(f"cannot read selector {sel}")


def top_targets(cx: Connectome, source: np.ndarray, among: np.ndarray, n: int) -> np.ndarray:
    """Neurons of `among` with any input from `source`, ranked by summed |weight| from it."""
    W = cx.W
    rows = np.repeat(np.arange(cx.n), np.diff(W.indptr))
    keep = source[W.indices] & among[rows]
    score = np.bincount(rows[keep], weights=np.abs(W.data[keep]), minlength=cx.n).astype(np.float32)
    degree = np.bincount(rows[keep], minlength=cx.n)
    ids = np.flatnonzero(degree > 0)
    ids = ids[np.argsort(-score[ids], kind="stable")]
    return ids[:min(n, len(ids))].astype(np.int64)
