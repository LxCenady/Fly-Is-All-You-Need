"""UCTF, the Universal Connectome Training Framework: any wiring diagram as a character-level
language model, with the controls that ask whether the wiring matters.

    data.py       connectomes as data (weights.npz + neuron table [+ gap.npz]), loaders, CSV importer
    select.py     choosing neurons by annotation or by wiring
    controls.py   degree-preserving rewiring (chemical and electrical synapses)
    substrate.py  LIFSubstrate: a spec (specs/*.json) turned into characters -> neuron features
    run.py        substrate registry, feature simulation and caching keys
    readout.py    the linear readout (softmax over neuron features + hashed context table)
    text.py       train/validation split and character set
    baselines.py  unigram and Kneser-Ney n-gram baselines
    bench.py      the standard comparison (n-grams, context table, connectome, rewired controls,
                  memory span, activity matching)

    python -m uctf import --neurons n.csv --edges e.csv --out my-connectome
    python -m uctf bench --substrate my-spec.json --data text.txt

Modules are imported on demand; the package itself imports nothing heavy.
"""
__version__ = "0.1.0"
