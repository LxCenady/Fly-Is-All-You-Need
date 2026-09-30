"""Plugin registry.

Every replaceable part of UCTF is a named plugin of one kind:

    source     where a connectome comes from (a folder, flybrain, ...)
    transform  a change to a connectome (cut inputs, rewire, apply signs, ...)
    neuron     membrane dynamics (leaky integrate-and-fire, ...)
    synapse    how a synapse layer turns activity into current
    encoder    how a token becomes input to the network
    feature    what is read out of the network per token
    task       what the network is trained on (next character, ...)
    readout    the trained decoder on top of the features
    probe      an analysis of recorded features (memory span, ...)
    baseline   a model without a connectome, for comparison

Register with a decorator:

    @register("feature", "counts")
    class Counts: ...

Third-party packages add plugins through the entry-point group "uctf.plugins":
each entry point is a module, imported once, whose decorators register it.
"""
from __future__ import annotations

from importlib import import_module, metadata

KINDS = (
    "source", "transform", "neuron", "synapse", "encoder",
    "feature", "task", "readout", "probe", "baseline",
)
BUILTIN = (
    "sources", "transforms", "neurons", "synapses", "encoders",
    "features", "tasks", "readouts", "probes", "baselines",
)
_plugins: dict = {kind: {} for kind in KINDS}
_missing: dict = {}                 # plugin module -> the import error
_loaded = False


def register(kind: str, name: str):
    """Decorator: make `obj` available as plugin `name` of `kind`."""
    _check_kind(kind)

    def add(obj):
        _plugins[kind][name] = obj
        return obj

    return add


def get(kind: str, name: str):
    """The plugin `name` of `kind`; a KeyError lists what exists."""
    _check_kind(kind)
    _load_all()
    try:
        return _plugins[kind][name]
    except KeyError:
        known = ", ".join(sorted(_plugins[kind])) or "none"
        msg = f"no {kind} plugin {name!r} (known: {known})"
        if _missing:
            skipped = "; ".join(f"{m}: {e}" for m, e in _missing.items())
            msg += f"; not loaded: {skipped}"
        raise KeyError(msg) from None


def available(kind: str | None = None) -> dict:
    """Names of the registered plugins, for one kind or for all."""
    _load_all()
    if kind is not None:
        _check_kind(kind)
        return {kind: sorted(_plugins[kind])}
    return {k: sorted(v) for k, v in _plugins.items()}


def skipped() -> dict:
    """Plugin modules that could not be imported, with the reason."""
    _load_all()
    return {m: str(e) for m, e in _missing.items()}


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise KeyError(f"unknown plugin kind {kind!r} (kinds: {', '.join(KINDS)})")


def _load_all() -> None:
    """Import the built-in plugins, then third-party entry points, once."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in BUILTIN:
        _try(import_module, f"uctf.plugins.{module}")
    for ep in metadata.entry_points(group="uctf.plugins"):
        _try(lambda _: ep.load(), ep.value)


def _try(load, module: str) -> None:
    """Import a plugin module; one whose dependencies are missing (e.g.
    scipy in a numpy-only install) is skipped and named in errors."""
    try:
        load(module)
    except ImportError as e:
        _missing[module] = e
