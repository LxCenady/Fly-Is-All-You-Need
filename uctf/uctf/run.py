"""Running a substrate over text: what is simulated (a "brain spec"), the registry of substrates,
feature simulation, and the key under which simulated features may be cached.

A brain spec is a dict:
    {"spec": {...}, "control": None | {"rewire": "full"|"class", "seed": s}, "substrate": name}
        any connectome described by a spec (substrate.py)
    {"substrate": "<registered name>", ...}
        a substrate registered by another package with register() (GPF registers its original
        GPF-1 runtime this way)
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

_REGISTRY: dict = {}


def register(name: str, factory) -> None:
    """factory(vocab_size, brain_spec, device) -> an object with reset() and step_token(t)."""
    _REGISTRY[name] = factory


def make_substrate(vocab_size: int, bspec: dict, device: str = "auto", log=print):
    if "spec" in bspec:
        from .substrate import LIFSubstrate
        return LIFSubstrate(vocab_size, bspec["spec"], bspec.get("control"), device=device, log=log)
    name = bspec.get("substrate")
    if name not in _REGISTRY:
        raise ValueError(f"unknown substrate {name!r}; registered: {', '.join(_REGISTRY) or 'none'}")
    return _REGISTRY[name](vocab_size, bspec, device)


def set_path(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def brain_spec(substrate, input_active=None, input_drive=None, overrides=(), control="none",
               control_seed=0) -> dict:
    """A spec (built-in name, JSON path or dict) with overrides ("neuron.gain=1.8", JSON values)
    and an optional control ("rewire-full" or "rewire-class")."""
    from .substrate import load_spec
    spec = json.loads(json.dumps(load_spec(substrate)))               # a private copy
    if input_active:
        spec["input"]["active"] = input_active
    if input_drive:
        spec["input"]["drive"] = input_drive
    for item in overrides or []:
        key, _, val = item.partition("=")
        try:
            val = json.loads(val)
        except ValueError:
            pass
        set_path(spec, key.strip(), val)
    ctl = None
    if control and control != "none":
        ctl = {"rewire": {"rewire-full": "full", "rewire-class": "class"}[control], "seed": int(control_seed)}
    return {"spec": spec, "control": ctl, "substrate": spec.get("name", "custom")}


def feature_key(ids, V, bspec) -> str:
    """Identifies simulated features: the text, the character set, everything in the brain spec
    that changes the simulation (not its description or brain-view settings) and the simulator
    version (substrate.SIM_VERSION), so a cache is never reused across simulator changes."""
    from .substrate import SIM_VERSION
    b = json.loads(json.dumps(bspec))
    for k in ("view", "description"):
        b.get("spec", {}).pop(k, None)
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()
                          + json.dumps([V, b, SIM_VERSION], sort_keys=True).encode()).hexdigest()


def log_flush(msg):
    print(msg, flush=True)


def simulate(ids, V, bspec: dict, device="auto", log=log_flush, with_dims=False):
    """Run the substrate over the text as one continuous stream from a reset; row i = features
    after character i (used to predict character i+1).  with_dims: also return the size of each
    readout group, in feature order."""
    ctl = bspec.get("control")
    log(f"building the substrate ({bspec.get('substrate')}" + (f", {ctl['rewire']} rewiring, seed {ctl['seed']}"
                                                              if ctl else "") + ")...")
    rt = make_substrate(V, bspec, device=device, log=log)
    rt.reset()
    n = len(ids) - 1
    X = None; t0 = time.time()
    for i in range(n):
        x = rt.step_token(int(ids[i]))
        if X is None:
            X = np.zeros((n, len(x)), np.float32)
        X[i] = x
        if (i + 1) % 1000 == 0 or i + 1 == n:
            r = (i + 1) / (time.time() - t0)
            log(f"  simulated {i + 1:,}/{n:,} characters  ({r:.0f}/s, {(n - i - 1) / r / 60:.1f} min left)")
    if with_dims:
        dims = getattr(rt, "readout_sizes", None) or {"features": X.shape[1]}
        return X, dict(dims)
    return X
