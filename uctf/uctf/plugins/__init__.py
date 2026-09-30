"""Built-in plugins. The registry imports these modules on first use
(core/registry.py, BUILTIN), so importing one plugin module stays light.

    sources      folder, flybrain
    transforms   cut_inputs, lesion, scale, transmitter_signs,
                 normalise_inputs, rewire
    neurons      lif
    synapses     chemical, electrical
    encoders     random_subset, by_group
    features     counts, voltage, trace
    tasks        next_char
    readouts     softmax_context
    probes       memory_span
    baselines    kneser_ney, unigram
"""
