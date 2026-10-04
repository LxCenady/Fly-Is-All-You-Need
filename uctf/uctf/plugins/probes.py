"""Probes: analyses of recorded features."""
from __future__ import annotations

import numpy as np

from ..core.registry import register


def ridge_span(X, ids, n_tr, span, lam=10.0, clip=3.0):
    """Accuracy of decoding the token k back (k = 0..span) on validation rows,
    from standardised, clipped features (training statistics); ridge
    regression with an intercept, one factorisation for all k. Also returns
    the rate of always guessing the most frequent token."""
    from scipy.linalg import cho_factor, cho_solve
    x_ids, ntr = ids[:-1], n_tr - 1
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Z = np.clip((X - mu) / sd, -clip, clip).astype(np.float32)
    Ztr = np.c_[Z[:ntr], np.ones(ntr, np.float32)]
    Zva = np.c_[Z[ntr:], np.ones(len(Z) - ntr, np.float32)]
    eye = np.eye(Ztr.shape[1], dtype=np.float32)
    try:
        A = cho_factor(Ztr.T @ Ztr + lam * eye)
    except np.linalg.LinAlgError:
        # Near-collinear columns (counts and traces of one quiet neuron)
        # can break float32 positive-definiteness: retry in float64.
        # Results that factor in float32 are unchanged.
        Z64 = Ztr.astype(np.float64)
        A = cho_factor(Z64.T @ Z64 + lam * np.eye(Z64.shape[1]))
    V = int(ids.max()) + 1
    acc, majority = [], []
    for k in range(span + 1):
        tgt = (np.concatenate([np.zeros(k, np.int64), x_ids[:len(x_ids) - k]])
               if k else x_ids.copy())
        Y = np.eye(V, dtype=np.float32)[tgt]
        Wt = cho_solve(A, Ztr.T @ Y[:ntr])
        acc.append(float(((Zva @ Wt).argmax(1) == tgt[ntr:]).mean()))
        counts = np.bincount(tgt[ntr:], minlength=V)
        majority.append(float(counts.max() / len(tgt[ntr:])))
    return acc, majority


@register("probe", "memory_span")
class MemorySpan:
    """How well the token k steps back can be decoded (k = 0..span), from each
    readout group and from all features. Params: span, lam, clip."""

    def __init__(self, span=8, lam=10.0, clip=3.0):
        self.span, self.lam, self.clip = span, lam, clip

    def run(self, X, split, groups: dict) -> dict:
        out, lo = {}, 0
        for name, size in groups.items():
            if size:
                part = X[:, lo:lo + size]
                out[name] = ridge_span(part, split.ids, split.n_tr, self.span,
                                       self.lam, self.clip)[0]
            lo += size
        acc, majority = ridge_span(X, split.ids, split.n_tr, self.span,
                                   self.lam, self.clip)
        out["all"] = acc
        return {"memory_span": out, "majority": majority}
