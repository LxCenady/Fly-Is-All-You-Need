"""UCTF, the Universal Connectome Training Framework.

A small kernel that turns any connectome into a model that reads tokens, and
plugins for everything that is a modelling choice:

    uctf/core      connectome data, spec, selectors, plugin registry, step loop
    uctf/plugins   sources, transforms, neurons, synapses, encoders, features,
                   tasks, readouts, probes, baselines
    uctf/specs     built-in specs (malecns-v1)

Public API (imported on first use, so `import uctf` stays light):

    from uctf import Network, load_spec, register, available
    net = Network("my-spec.json", n_tokens=65)
    features = net.step_token(token)

Command line: python -m uctf import | bench | plugins
"""
__version__ = "0.2.0"

_API = {
    "Connectome": "core.connectome", "Layer": "core.connectome",
    "load_folder": "core.connectome", "Network": "core.network",
    "available": "core.registry", "get": "core.registry",
    "register": "core.registry", "load_spec": "core.spec",
}


def __getattr__(name):
    if name in _API:
        from importlib import import_module
        return getattr(import_module(f".{_API[name]}", __name__), name)
    raise AttributeError(name)
