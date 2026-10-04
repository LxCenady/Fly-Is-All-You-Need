"""Readout features: what is recorded from a population for each token."""
from __future__ import annotations

import numpy as np

from ..core.backend import to_host
from ..core.registry import register


class _Feature:
    def __init__(self, ids, xp, dt, params):
        self.ids, self.xp, self.dt = xp.asarray(ids), xp, float(dt)
        self.size = len(ids)

    def reset(self):
        pass

    def begin(self):
        pass

    def on_voltage(self, v):
        pass

    def on_spikes(self, spikes):
        pass


@register("feature", "counts")
class Counts(_Feature):
    """Spikes per neuron during the token. Optional "window": [a, b] counts
    only the token's steps a..b-1 (e.g. [0, 1] = the first step)."""

    def __init__(self, ids, xp, dt, params):
        super().__init__(ids, xp, dt, params)
        w = params.get("window")
        self.window = None if w is None else (int(w[0]), int(w[1]))

    def begin(self):
        self._n = self.xp.zeros(self.size, self.xp.float32)
        self._step = 0

    def on_spikes(self, spikes):
        if self.window is None or self.window[0] <= self._step < self.window[1]:
            self._n += spikes[self.ids]
        self._step += 1

    def value(self):
        return to_host(self._n)


@register("feature", "voltage")
class Voltage(_Feature):
    """Mean membrane voltage before firing, over the token's steps."""

    def begin(self):
        self._v = []

    def on_voltage(self, v):
        self._v.append(to_host(v[self.ids]).astype(np.float32))

    def value(self):
        return np.asarray(self._v, np.float32).mean(0)


@register("feature", "trace")
class Trace(_Feature):
    """A spike trace decaying with time constant `tau` (s), kept across
    tokens: its value at the end of the token."""

    def __init__(self, ids, xp, dt, params):
        super().__init__(ids, xp, dt, params)
        tau = float(params.get("tau", 0.1))
        self.decay = np.float32(np.exp(-self.dt / tau))

    def reset(self):
        self._t = self.xp.zeros(self.size, self.xp.float32)

    def on_spikes(self, spikes):
        self._t *= self.decay
        self._t += spikes[self.ids]

    def value(self):
        return to_host(self._t).copy()
