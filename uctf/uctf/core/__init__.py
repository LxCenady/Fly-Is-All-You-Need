"""The UCTF kernel: data model, spec, selectors, registry and the step loop.

The kernel knows no neuron types, synapse types, input codes or tasks; those
are plugins (uctf/plugins, or third-party packages via the "uctf.plugins"
entry-point group).
"""
