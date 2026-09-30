"""Synapse models: how one layer of the connectome turns activity into current."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..core.backend import sparse_on
from ..core.connectome import Layer
from ..core.registry import register


def _check(name, params, allowed):
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(f"{name}: unknown params {sorted(unknown)}")


@register("synapse", "chemical")
class Chemical:
    """Spike-triggered current through the layer's signed weights:

        I = gain * W @ s            (s = spikes of the previous step)

    Optional: "delay" (steps, uniform; default 0 = the next step, as above)
    and "tau_syn" (s; default 0 = instantaneous; else an exponentially
    decaying synaptic current). Per-edge delays, receptors and transmitter-
    specific kinetics are left to other plugins (see core/interfaces.py).
    """

    PARAMS = ("gain", "delay", "tau_syn", "dt")

    def __init__(self, layer, xp, params):
        _check("chemical", params, self.PARAMS)
        self.xp = xp
        self.W = sparse_on(xp, layer.matrix)
        self.gain = np.float32(params.get("gain", 1.0))
        self.delay = int(params.get("delay", 0))
        tau, dt = float(params.get("tau_syn", 0.0)), float(params.get("dt", 0.02))
        self.syn_decay = np.float32(np.exp(-dt / tau)) if tau > 0 else None

    def reset(self):
        self._queue = []
        self._i = None

    def current(self, spikes, v):
        if self.delay:
            self._queue.append(spikes)
            if len(self._queue) <= self.delay:
                return self.xp.zeros_like(spikes)
            spikes = self._queue.pop(0)
        i = self.W @ spikes * self.gain
        if self.syn_decay is None:
            return i
        self._i = i if self._i is None else self._i * self.syn_decay + i
        return self._i


@register("synapse", "by_transmitter")
class ByTransmitter:
    """A chemical layer split by a per-edge label (e.g. the transmitter or
    receptor of each synapse), each part a `chemical` synapse with its own
    params:

        "params": {"edge": "transmitter",
                   "table": {"gaba": {"gain": -1.5, "tau_syn": 0.01},
                             "acetylcholine": {"gain": 1.5}},
                   "default": {"gain": 1.5}}

    Labels are matched in lower case. A label missing from `table` uses
    `default`; with no default it is an error. The per-edge array comes
    from the connectome folder or from the `edge_labels` transform.
    """

    PARAMS = ("edge", "table", "default")

    def __init__(self, layer, xp, params):
        _check("by_transmitter", params, self.PARAMS)
        edge = params.get("edge", "transmitter")
        if edge not in layer.edges:
            raise KeyError(f"by_transmitter: layer has no edge array "
                           f"{edge!r} ({', '.join(layer.edges) or 'none'})")
        labels = np.char.lower(layer.edges[edge].astype(str))
        table = {k.lower(): v for k, v in params.get("table", {}).items()}
        default = params.get("default")
        self.parts, self.labels = [], []
        for label in np.unique(labels):
            cfg = table.get(label, default)
            if cfg is None:
                raise KeyError(f"by_transmitter: no params for {label!r}")
            W = layer.matrix.copy()
            W.data[labels != label] = 0
            W.eliminate_zeros()
            self.parts.append(Chemical(Layer(layer.kind, W), xp, cfg))
            self.labels.append(str(label))

    def reset(self):
        for part in self.parts:
            part.reset()

    def current(self, spikes, v):
        out = self.parts[0].current(spikes, v)
        for part in self.parts[1:]:
            out = out + part.current(spikes, v)
        return out


@register("synapse", "electrical")
class Electrical:
    """Gap junctions: I_i = gain * sum_j G_ij (v_j - v_i).

    "normalise": "row_max1" (default) divides each row of G by max(row sum, 1)
    so that no neuron is coupled with total weight above 1; "none" keeps G.
    """

    PARAMS = ("gain", "normalise")

    def __init__(self, layer, xp, params):
        _check("electrical", params, self.PARAMS)
        G = layer.matrix
        mode = params.get("normalise", "row_max1")
        if mode == "row_max1":
            deg = np.asarray(G.sum(1)).ravel()
            scale = (1.0 / np.maximum(deg, 1.0)).astype(np.float32)
            G = (sparse.diags(scale) @ G).tocsr().astype(np.float32)
        elif mode != "none":
            raise ValueError(f"electrical: unknown normalise {mode!r}")
        self.G = sparse_on(xp, G)
        rowsum = np.asarray(G.sum(1)).ravel().astype(np.float32)
        self.deg = xp.asarray(rowsum)
        self.gain = np.float32(params.get("gain", 1.0))

    def reset(self):
        pass

    def current(self, spikes, v):
        return (self.G @ v - self.deg * v) * self.gain
