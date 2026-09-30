"""Assemble a network from a spec and run it one token at a time.

The network only wires plugins together; it knows no neuron types, synapse
types or tasks. One token is presented for `input.steps` steps. Each step:

    1. the encoder injects input into the neurons
    2. every synapse layer computes its current from the previous step
    3. the neuron model integrates the currents
    4. features see the membrane state before firing
    5. the neurons fire; features see the spikes

Controls (e.g. a rewired copy) are transforms applied to the raw connectome
before the spec's own transforms. Populations are always resolved on the
connectome without controls, so a control reads out the same neurons.
"""
from __future__ import annotations

import copy

import numpy as np

from . import registry
from .backend import array_module, to_host
from .connectome import Connectome
from .select import Selector
from .spec import load_spec


def load_connectome(spec: dict) -> Connectome:
    src = spec["connectome"]
    return registry.get("source", src["source"])(**src.get("params", {}))


def apply_transforms(cx: Connectome, steps, populations, log=print):
    """Apply a list of {"transform": name, "params": {...}} in order."""
    infos = []
    for step in steps:
        fn = registry.get("transform", step["transform"])
        cx, info = fn(cx, Selector(cx, populations), **step.get("params", {}))
        infos.append({"transform": step["transform"], **info})
        if info.get("message"):
            log(info["message"])
    return cx, infos


class Network:
    def __init__(self, spec, n_tokens: int, controls=(), device="auto",
                 connectome: Connectome | None = None, log=print):
        self.spec = spec = load_spec(spec)
        pops = spec["populations"]
        raw = connectome if connectome is not None else load_connectome(spec)
        real, _ = apply_transforms(raw, spec["transforms"], pops, log)
        self.sel = Selector(real, pops)             # populations: real wiring
        self.info = []
        cx = real
        if controls:
            cx, self.info = apply_transforms(raw, controls, pops, log)
            cx, _ = apply_transforms(cx, spec["transforms"], pops, log)
        self.cx = cx.validate()
        self.xp = xp = array_module(device)
        self.n_tokens = n_tokens
        self._build(spec, n_tokens, xp)
        self.record_activity = False
        self.last_fired = None
        self.reset()

    def _build(self, spec, n_tokens, xp):
        sel, cx = self.sel, self.cx
        nr = spec["neuron"]
        overrides = [(sel.ids(o["population"]), o["params"])
                     for o in nr.get("overrides", [])]
        make = registry.get("neuron", nr.get("model", "lif"))
        self.neuron = make(cx.n, xp, nr.get("params", {}), overrides)
        self.synapses = []
        for s in spec["synapses"]:
            params = dict(s.get("params", {}))
            optional = params.pop("optional", False)
            if s["layer"] not in cx.layers and optional:
                continue
            make = registry.get("synapse", s["model"])
            self.synapses.append(make(cx.layer(s["layer"]), xp, params))
        inp = spec["input"]
        self.steps = int(inp["steps"])
        make = registry.get("encoder", inp["encoder"])
        in_ids = sel.ids(inp["population"])
        self.encoder = make(in_ids, n_tokens, xp, inp.get("params", {}), cx)
        dt = self.neuron.dt
        self.features = []
        for r in spec["readout"]:
            make = registry.get("feature", r["feature"])
            ids = sel.ids(r["population"])
            self.features.append(make(ids, xp, dt, r.get("params", {})))
        self.readout_sizes = {r["name"]: f.size
                              for r, f in zip(spec["readout"], self.features)}
        self.n_features = sum(self.readout_sizes.values())

    # ------------------------------------------------------------ dynamics
    def reset(self):
        self.neuron.reset()
        for s in self.synapses:
            s.reset()
        for f in self.features:
            f.reset()
        self.spikes = self.xp.zeros(self.cx.n, self.xp.float32)

    def step_token(self, token: int) -> np.ndarray:
        xp, neuron = self.xp, self.neuron
        for f in self.features:
            f.begin()
        fired_steps = []
        for step in range(self.steps):
            inj = self.encoder.inject(int(token), step)
            if inj is not None:
                neuron.inject(*inj)
            currents = [s.current(self.spikes, neuron.v) for s in self.synapses]
            neuron.integrate(currents)
            for f in self.features:
                f.on_voltage(neuron.v)
            fired = neuron.fire()
            self.spikes = xp.zeros(self.cx.n, xp.float32)
            self.spikes[fired] = 1.0
            for f in self.features:
                f.on_spikes(self.spikes)
            if self.record_activity:
                fired_steps.append(fired)
        if self.record_activity:
            self._remember(fired_steps)
        out = [f.value() for f in self.features]
        return np.concatenate(out).astype(np.float32)

    def _remember(self, fired_steps):
        xp = self.xp
        total = int(sum(int(f.size) for f in fired_steps))
        ids = xp.unique(xp.concatenate(fired_steps)) if total else xp.zeros(0)
        self.last_fired = {"ids": to_host(ids).astype(np.uint32), "spikes": total}

    # ------------------------------------------------------------ changes
    FIXED = ("connectome", "transforms", "populations")

    def respec(self, dotted: str, value) -> None:
        """Change one dynamics setting (e.g. "synapses.chemical.params.gain")
        and rebuild the parts that use it; the wiring is kept, so the path
        may not lie under connectome, transforms or populations."""
        from .spec import override, v2_path
        if v2_path(dotted).split(".")[0] in self.FIXED:
            raise ValueError(f"{dotted!r} changes the wiring; build anew")
        spec = copy.deepcopy(self.spec)
        override(spec, dotted, value)
        self.spec = spec
        self._build(spec, self.n_tokens, self.xp)
        self.reset()
