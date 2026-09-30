"""Neuron models."""
from __future__ import annotations

import numpy as np

from ..core.registry import register

LIF_DEFAULTS = {"dt": 0.02, "tau": 0.1, "threshold": 1.0, "reset": 0.0,
                "tonic": 0.05, "noise_hz": 0.0, "noise_amp": 0.22, "seed": 0}


@register("neuron", "lif")
class LIF:
    """Leaky integrate-and-fire point neurons (as in flybrain):

        v <- exp(-dt/tau) v + I_1 + tonic + I_2 + ... (+ noise)
        v >= threshold -> spike, v = reset

    Params: dt (s), tau (s), threshold, reset, tonic, noise_hz, noise_amp,
    seed. `overrides` [(ids, params)] give chosen neurons their own tau,
    threshold, reset or tonic (arrays are then used instead of scalars).
    """

    PER_NEURON = ("tau", "threshold", "reset", "tonic")

    def __init__(self, n, xp, params, overrides=()):
        p = {**LIF_DEFAULTS, **params}
        unknown = set(params) - set(LIF_DEFAULTS)
        if unknown:
            raise ValueError(f"lif: unknown params {sorted(unknown)}")
        self.n, self.xp = n, xp
        self.dt = float(p["dt"])
        self.noise_hz = float(p["noise_hz"])
        self.noise_amp = np.float32(p["noise_amp"])
        self.seed = int(p["seed"])
        vals = {k: float(p[k]) for k in self.PER_NEURON}
        if overrides:
            vals = {k: np.full(n, v, np.float32) for k, v in vals.items()}
            for ids, op in overrides:
                for k, v in op.items():
                    if k not in self.PER_NEURON:
                        raise ValueError(f"lif: cannot override {k!r}")
                    vals[k][ids] = v
            vals = {k: xp.asarray(v) for k, v in vals.items()}
            self.decay = xp.exp(-self.dt / vals["tau"]).astype(xp.float32)
        else:
            self.decay = np.float32(np.exp(-self.dt / vals["tau"]))
            vals = {k: np.float32(v) for k, v in vals.items()}
        self.threshold, self.v_reset = vals["threshold"], vals["reset"]
        self.tonic = vals["tonic"]

    def reset(self):
        xp = self.xp
        self.rng = xp.random.default_rng(self.seed)
        self.v = xp.zeros(self.n, xp.float32)

    def inject(self, ids, amount):
        self.v[ids] += amount

    def integrate(self, currents):
        self.v *= self.decay
        first = currents[0] if currents else 0
        self.v += first + self.tonic
        for c in currents[1:]:
            self.v += c
        if self.noise_hz:
            p = self.noise_hz * self.dt
            self.v += (self.rng.random(self.n) < p) * self.noise_amp

    def fire(self):
        fired = self.xp.flatnonzero(self.v >= self.threshold)
        reset = self.v_reset
        self.v[fired] = reset[fired] if getattr(reset, "ndim", 0) else reset
        return fired
