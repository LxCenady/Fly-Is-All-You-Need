"""Connectome language models: any wiring diagram as a character-level reservoir.

    data.py       connectomes as data (weights.npz + neurons table), loaders and an edge-list importer
    select.py     choosing neurons by annotation or wiring
    controls.py   degree-preserving rewiring (the null that asks whether the real wiring matters)
    substrate.py  LIFSubstrate: a spec (specs/*.json) turned into characters -> neuron features
    bench.py      the standard comparison: real vs rewired connectome vs context-only vs n-gram

Train on a spec with  python -m gpf train brain --substrate <spec> --data text.txt --name my-model
"""
from .data import Connectome, import_edges, load, load_flybrain, load_folder  # noqa: F401
from .substrate import LIFSubstrate, builtin_specs, load_spec  # noqa: F401
