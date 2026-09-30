"""Encoders: how a token becomes input to the network."""
from __future__ import annotations

import numpy as np

from ..core.registry import register


class _Pulses:
    """Shared part: token t drives the neurons codes[t] with `drive` on the
    first step and `sustain * drive` on the later steps of the token."""

    def __init__(self, codes, xp, drive, sustain):
        self.codes = [xp.asarray(c) for c in codes]
        self.drive, self.sustain = float(drive), float(sustain)

    def inject(self, token, step):
        scale = 1.0 if step == 0 else self.sustain
        return self.codes[token], np.float32(self.drive * scale)


@register("encoder", "random_subset")
class RandomSubset(_Pulses):
    """Each token drives its own fixed random set of `active` input neurons.
    Params: active, drive, sustain, seed."""

    def __init__(self, ids, n_tokens, xp, params, cx=None):
        active = int(params["active"])
        if active > len(ids):
            raise ValueError(f"random_subset: active={active} but the input "
                             f"population has {len(ids)} neurons")
        rng = np.random.default_rng(int(params.get("seed", 0)))
        codes = [np.sort(ids[rng.choice(len(ids), size=active, replace=False)])
                 for _ in range(n_tokens)]
        super().__init__(codes, xp, params.get("drive", 1.0),
                         params.get("sustain", 0.5))


@register("encoder", "by_group")
class ByGroup(_Pulses):
    """Each token drives all input neurons of a random set of groups (e.g. all
    projection neurons of a few glomeruli), added until at least `active`
    neurons are driven. Params: column (the grouping annotation), active,
    drive, sustain, seed."""

    def __init__(self, ids, n_tokens, xp, params, cx):
        labels = cx.column(params["column"]).astype(str)[ids]
        groups = np.unique(labels)
        rng = np.random.default_rng(int(params.get("seed", 0)))
        target = int(params["active"])
        codes = []
        for _ in range(n_tokens):
            chosen = []
            for g in rng.permutation(groups):
                chosen.extend(ids[labels == g].tolist())
                if len(chosen) >= target:
                    break
            codes.append(np.sort(np.asarray(chosen, np.int64)))
        super().__init__(codes, xp, params.get("drive", 1.0),
                         params.get("sustain", 0.5))
