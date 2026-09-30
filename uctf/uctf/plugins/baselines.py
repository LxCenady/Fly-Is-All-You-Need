"""Baselines without a connectome: unigram and Kneser-Ney n-grams."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..core.registry import register

LOG2 = math.log(2)


class KneserNey:
    """Interpolated modified Kneser-Ney n-gram over ids 0..V-1
    (Chen and Goodman 1999)."""

    def __init__(self, order: int, V: int):
        self.n, self.V = order, V

    def fit(self, seq):
        n = self.n
        c = [defaultdict(lambda: defaultdict(int)) for _ in range(n + 1)]
        for i in range(len(seq)):
            for k in range(1, n + 1):
                if i - k + 1 < 0:
                    break
                c[k][tuple(seq[i - k + 1:i])][seq[i]] += 1
        cc = [defaultdict(lambda: defaultdict(int)) for _ in range(n + 1)]
        for k in range(2, n + 1):
            for ctx, d in c[k].items():
                for w in d:
                    cc[k - 1][ctx[1:]][w] += 1
        self.c, self.cc, self.D = c, cc, []
        for k in range(n + 1):
            nr = np.zeros(5)
            for d in (c[k] if k == n else cc[k]).values():
                for v in d.values():
                    if v <= 4:
                        nr[v] += 1
            Y = nr[1] / max(nr[1] + 2 * nr[2], 1)
            disc = [max(min(r - (r + 1) * Y * nr[r + 1] / max(nr[r], 1), r), 0.0)
                    for r in (1, 2, 3)]
            self.D.append([0.0] + disc)
        return self

    def _p(self, k, ctx, top):
        if k == 0:
            return np.full(self.V, 1.0 / self.V)
        key = ctx[len(ctx) - (k - 1):] if k > 1 else ()
        d = (self.c[k] if top else self.cc[k]).get(key)
        lower = self._p(k - 1, ctx, False)
        if not d:
            return lower
        tot, D = sum(d.values()), self.D[k]
        p, nr = np.zeros(self.V), [0, 0, 0, 0]
        for w, v in d.items():
            p[w] = max(v - D[min(v, 3)], 0) / tot
            nr[min(v, 3)] += 1
        back = (D[1] * nr[1] + D[2] * nr[2] + D[3] * nr[3]) / tot
        return p + back * lower

    def logprobs(self, hist) -> np.ndarray:
        """Log probabilities of the next token after the token ids in hist."""
        ctx = tuple(hist[-(self.n - 1):]) if self.n > 1 else ()
        p = self._p(min(self.n, len(ctx) + 1), ctx, True)
        return np.log(np.maximum(p, 1e-12))


def kn_bpc(order, ids, n_tr, V):
    """(validation BPC, accuracy) of a KN model fitted on ids[:n_tr]."""
    m = KneserNey(order, V).fit([int(t) for t in ids[:n_tr]])
    hist = [int(t) for t in ids[max(0, n_tr - order):n_tr]]   # end of training
    lp, hit = 0.0, 0
    for t in ids[n_tr:]:
        lg = m.logprobs(hist)
        lp += lg[t]
        hit += int(np.argmax(lg) == t)
        hist.append(int(t))
    n = len(ids) - n_tr
    return -lp / n / LOG2, hit / n


def unigram_bpc(ids, n_tr, V):
    p = (np.bincount(ids[:n_tr], minlength=V) + 1.0) / (n_tr + V)
    return float(-np.log2(p[ids[n_tr:]]).mean())


@register("baseline", "kneser_ney")
class KneserNeyBaseline:
    def __init__(self, order=5):
        self.order = order

    def evaluate(self, split) -> dict:
        bpc, acc = kn_bpc(self.order, split.ids, split.n_tr, len(split.chars))
        return {"val_bpc": bpc, "val_acc": acc}


@register("baseline", "unigram")
class UnigramBaseline:
    def evaluate(self, split) -> dict:
        return {"val_bpc": unigram_bpc(split.ids, split.n_tr, len(split.chars))}
