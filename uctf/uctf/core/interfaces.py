"""What each plugin kind must provide.

These are structural types (typing.Protocol): a plugin needs only the
methods, not a base class. A plugin is registered as a factory (usually its
class), called as shown in the comment of each protocol. uctf/PLUGINS.md is
the full protocol, with the step loop and the spec.

Arrays live on the network's array module `xp` (numpy, or cupy on a GPU).
All per-neuron vectors have length n (every neuron of the connectome).
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .connectome import Connectome
from .select import Selector


class Source(Protocol):
    """source(**params) -> Connectome"""

    def __call__(self, **params) -> Connectome: ...


class Transform(Protocol):
    """transform(cx, sel, **params) -> (Connectome, info dict)

    Returns a new connectome; `sel` resolves the spec's populations on the
    connectome as it was before this transform."""

    def __call__(self, cx: Connectome, sel: Selector, **params): ...


class Neuron(Protocol):
    """neuron(n, xp, params, overrides) where overrides = [(ids, params)].

    One step of the network is: inject -> integrate -> read v -> fire."""

    v: Any                                  # membrane state, length n
    dt: float                               # step length (s)

    def reset(self) -> None: ...

    def inject(self, ids, amount) -> None: ...

    def integrate(self, currents: list) -> None: ...

    def fire(self):                         # -> indices of neurons that fired
        ...


class Synapse(Protocol):
    """synapse(layer, xp, params) for one named layer of the connectome.

    current() is called before the neurons integrate, with the spikes of the
    previous step (a 0/1 vector) and the membrane state; it returns a current
    vector. The synapse sees only its own layer: matrix, kind and per-edge
    arrays (transmitter, receptor, delay, ...).

    Reserved: learn(spikes, v, **signals), called after each step, for
    plasticity. The step loop does not call it yet."""

    def reset(self) -> None: ...

    def current(self, spikes, v): ...


class Encoder(Protocol):
    """encoder(ids, n_tokens, xp, params, cx): ids = the input population;
    cx is the connectome, read only (e.g. to group inputs by annotation).

    inject(token, step) -> (neuron indices, amount) or None, for each of the
    `steps` steps a token is presented."""

    def inject(self, token: int, step: int): ...


class Feature(Protocol):
    """feature(ids, xp, dt, params): ids = the readout population.

    Per token: begin(); for every step on_voltage(v) before firing and
    on_spikes(spikes) after; value() -> numpy vector. reset() clears state
    kept across tokens."""

    size: int

    def reset(self) -> None: ...

    def begin(self) -> None: ...

    def on_voltage(self, v) -> None: ...

    def on_spikes(self, spikes) -> None: ...

    def value(self) -> np.ndarray: ...


class Task(Protocol):
    """task(**params): split() -> an object with ids (int array), chars
    (the vocabulary) and n_tr (the number of training tokens)."""

    def split(self) -> Any: ...


class Readout(Protocol):
    """readout(**params): evaluate(X, split, features=True) -> dict with
    val_bpc and val_acc. Row i of X is the features after token i; with
    features=False the readout is fitted without them."""

    def evaluate(self, X, split, features: bool = True) -> dict: ...


class Probe(Protocol):
    """probe(**params): run(X, split, groups) -> dict; groups = {readout
    name: number of feature columns}, in column order."""

    def run(self, X, split, groups: dict) -> dict: ...


class Baseline(Protocol):
    """baseline(**params): evaluate(split) -> dict with at least val_bpc."""

    def evaluate(self, split) -> dict: ...
