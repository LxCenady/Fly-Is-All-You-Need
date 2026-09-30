"""Running a connectome over a token stream.

A brain spec says what is simulated:

    {"spec": {...}, "controls": [...], "substrate": name}
        a network from a UCTF spec (any version; see core/spec.py) and
        the control transforms applied to its wiring
    {"substrate": "<registered name>", ...}
        a runtime registered by another package with `register`
        (GPF registers its original GPF-1 runtime this way)

Brain specs written by UCTF 0.1 ({"spec": v1, "control": {...}}) still
load: the spec and control are migrated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np

from .core.spec import load_spec, migrate_control_v1, override

# Part of every feature cache key: bump when a simulation result changes.
SIM_VERSION = 3
_RUNTIMES: dict = {}


def register(name: str, factory) -> None:
    """factory(n_tokens, brain_spec, device) -> an object with reset()
    and step_token(token)."""
    _RUNTIMES[name] = factory


def brain_spec(ref, input_active=None, input_drive=None, overrides=(),
               control="none", control_seed=0) -> dict:
    """A spec (name, path or dict) with overrides ("path=json value", e.g.
    "synapses.chemical.params.gain=1.8"; version-1 paths such as
    "input.code_seed" are translated) and an optional named control from
    the spec's "controls" (e.g. "rewire-full")."""
    spec = load_spec(ref)
    if input_active:
        override(spec, "input.params.active", int(input_active))
    if input_drive:
        override(spec, "input.params.drive", float(input_drive))
    for item in overrides or []:
        key, _, text = item.partition("=")
        try:
            value = json.loads(text)
        except ValueError:
            value = text
        override(spec, key.strip(), value)
    return {"spec": spec, "substrate": spec["name"],
            "controls": control_steps(spec, control, control_seed),
            "control_name": None if control in (None, "none") else control}


def control_steps(spec: dict, name, seed=0) -> list:
    """The transform list of a named control, with its seed set."""
    if not name or name == "none":
        return []
    if name not in spec["controls"]:
        known = ", ".join(spec["controls"]) or "none"
        raise KeyError(f"spec has no control {name!r} (known: {known})")
    steps = copy.deepcopy(spec["controls"][name])
    for step in steps:
        step.setdefault("params", {})["seed"] = int(seed)
    return steps


class Substrate:
    """A Network with the brain view (view.py) that GPF draws."""

    def __init__(self, net):
        from .view import View
        self.net, self._view = net, View(net)
        self.readout_sizes = net.readout_sizes

    def reset(self):
        self.net.reset()

    def step_token(self, token: int) -> np.ndarray:
        return self.net.step_token(token)

    @property
    def record_activity(self) -> bool:
        return self.net.record_activity

    @record_activity.setter
    def record_activity(self, on: bool):
        self.net.record_activity = bool(on)

    @property
    def last_activity(self):
        return self._view.activity(self.net.last_fired)

    def map_payload(self, max_px: int = 4095) -> dict:
        return self._view.payload(max_px)


def make_substrate(n_tokens: int, bspec: dict, device="auto", log=print):
    if "spec" in bspec:
        from .core.network import Network
        spec = load_spec(bspec["spec"])
        controls = bspec.get("controls")
        if controls is None:                    # a UCTF 0.1 brain spec
            controls = migrate_control_v1(bspec.get("control"), spec)
        net = Network(spec, n_tokens, controls, device=device, log=log)
        return Substrate(net)
    name = bspec.get("substrate")
    if name not in _RUNTIMES:
        known = ", ".join(_RUNTIMES) or "none"
        raise ValueError(f"unknown substrate {name!r} (registered: {known})")
    return _RUNTIMES[name](n_tokens, bspec, device)


def feature_key(ids, n_tokens, bspec) -> str:
    """Identifies simulated features: the token ids, the vocabulary size,
    everything in the brain spec that changes the simulation (not its
    description or view) and SIM_VERSION."""
    b = copy.deepcopy(bspec)
    for k in ("view", "description"):
        b.get("spec", {}).pop(k, None)
    blob = json.dumps([n_tokens, b, SIM_VERSION], sort_keys=True).encode()
    ids = np.asarray(ids, np.int64).tobytes()
    return hashlib.sha256(ids + blob).hexdigest()


def log_flush(msg):
    print(msg, flush=True)


def simulate(ids, n_tokens, bspec, device="auto", log=log_flush,
             with_dims=False):
    """Run the substrate over the ids as one stream from a reset. Row i is
    the features after token i (used to predict token i+1). with_dims:
    also return the size of each readout group, in feature order."""
    what = bspec.get("control_name") or bspec.get("control")
    log(f"building {bspec.get('substrate')}"
        + (f" (control: {what})" if what else "") + "...")
    rt = make_substrate(n_tokens, bspec, device=device, log=log)
    rt.reset()
    n = len(ids) - 1
    X, t0 = None, time.time()
    for i in range(n):
        x = rt.step_token(int(ids[i]))
        if X is None:
            X = np.zeros((n, len(x)), np.float32)
        X[i] = x
        if (i + 1) % 1000 == 0 or i + 1 == n:
            rate = (i + 1) / (time.time() - t0)
            left = (n - i - 1) / rate / 60
            log(f"  simulated {i + 1:,}/{n:,} tokens "
                f"({rate:.0f}/s, {left:.1f} min left)")
    if not with_dims:
        return X
    dims = getattr(rt, "readout_sizes", None) or {"features": X.shape[1]}
    return X, dict(dims)
