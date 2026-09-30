# UCTF plugin protocol

UCTF 0.2 is a small kernel plus plugins. The kernel is in `uctf/core`. It
holds a connectome, resolves neuron selections, reads the spec, keeps the
plugin registry and runs the step loop.

The kernel does not know any of these things:
- what a neuron type or synapse type is;
- how input is encoded;
- what is read out;
- what the task is.

Each of those is a named plugin. A spec composes them, and each plugin sees
only the arguments listed here, never another plugin's internals.

This file is the contract for plugins. It covers the data model, every
plugin kind with its call signature, the spec that composes plugins, how to
add plugins from your own package, and which extension points are
reserved.

- [The data model](#the-data-model)
- [The step loop](#the-step-loop)
- [Plugin kinds](#plugin-kinds)
- [The spec (version 2)](#the-spec-version-2)
- [Writing a plugin](#writing-a-plugin)
- [Reserved extension points](#reserved-extension-points)
- [Rules](#rules)

## The data model

A `Connectome` (`uctf/core/connectome.py`) is plain data:

| Field | Contents |
|---|---|
| `neurons` | columns of per-neuron data, each an array of length n: annotations (cell type, class, side, transmitter, ...) or numbers (size, measured time constants, ...) |
| `positions` | optional (n, 3) coordinates, NaN where unknown |
| `layers` | named synapse layers, `{name: Layer}` |
| `meta` | name, species, source, version, license, citation, units, transforms already applied, ... |

A `Layer` has three parts:
- `kind`: free text, such as `"chemical"`, `"electrical"` or `"neuromodulatory"`.
- `matrix`: an n × n scipy CSR matrix with rows = postsynaptic neuron and
  columns = presynaptic neuron. It is kept canonical (sorted indices, no
  duplicates) by `canonical()`.
- `edges`: per-edge arrays in the order of the matrix's stored entries.
  Examples are synapse count, transmitter, receptor, delay and confidence.

A layer name is a handle. Nothing in the kernel reads meaning into it.

On disk, a connectome is a folder (format 2):

```
connectome.json      {"format": "uctf-connectome", "version": 2,
                      "meta": {...}, "layers": {"chemical": {"kind": "chemical",
                                                             "edges": ["transmitter"]}}}
neurons.npz          one array per column; "positions" if known
<layer>.npz          the layer's matrix (scipy.sparse.save_npz)
<layer>.edges.npz    the layer's per-edge arrays, if any
```

- `Connectome.save(folder)` writes this format and `load_folder(folder)`
  reads it back.
- `load_folder` also reads the earlier format: `weights.npz` [+ `gap.npz`]
  plus `neurons.npz` or `neurons.csv`.
- `python -m uctf import` builds a folder from CSV files (see the
  [README](README.md)).

**Selectors** (`uctf/core/select.py`) choose neurons in JSON. A selector is
one of:
- a population name;
- a column test: `{"col": ..., "contains" | "equals" | "in" | "startswith" | "regex" | "range": ...}`;
- `{"and" | "or": [...]}` or `{"not": ...}`;
- `{"all": true}` or `{"ids": [...]}`;
- a test on the wiring: `{"top_targets_of": s1, "among": s2, "n": 256, "layer": "chemical"}`.

A plugin that takes a population receives the resolved neuron indices, not
the selector.

## The step loop

`Network(spec, n_tokens, controls=(), device="auto")` (`uctf/core/network.py`)
builds a network in five steps:

1. `source(**params)` returns the raw connectome.
2. Resolve the populations on the connectome after the spec's
   `transforms`, which is the real wiring. A control therefore always reads
   out the same neurons.
3. Build the wiring that is simulated: `controls` first (for example a
   rewiring), then the spec's `transforms`, each as
   `transform(cx, selector, **params)`.
4. Build the plugins: `neuron`, one `synapse` per entry of `synapses`,
   `encoder`, and one `feature` per `readout` entry.
5. `reset()`.

`step_token(token)` presents one token for `input.steps` steps. Each step:

```
for f in features: f.begin()                      # once per token
for step in range(steps):
    amount = encoder.inject(token, step)          # -> (ids, amount) or None
    neuron.inject(ids, amount)
    currents = [s.current(spikes, neuron.v) for s in synapses]   # spikes of the previous step
    neuron.integrate(currents)
    for f in features: f.on_voltage(neuron.v)     # membrane state before firing
    fired = neuron.fire()                         # indices; spikes = 0/1 vector
    for f in features: f.on_spikes(spikes)
return concat(f.value() for f in features)        # one numpy vector per token
```

`Network.respec("synapses.chemical.params.gain", 1.8)` changes one dynamics
setting and rebuilds the plugins on the same wiring. Activity matching uses
it. A change under `connectome`, `transforms` or `populations` is refused,
because it would change the wiring.

## Plugin kinds

All arrays inside the step loop live on the array module `xp`: numpy, or
CuPy on an NVIDIA GPU. Per-neuron vectors have length n. The protocols are
also written as `typing.Protocol` classes in `uctf/core/interfaces.py`.

| Kind | Factory call | Must provide | Built-in |
|---|---|---|---|
| `source` | `source(**params)` | returns a `Connectome` | `folder` (path), `flybrain` (MaleCNS v1.0) |
| `transform` | `transform(cx, sel, **params)` | returns `(Connectome, info dict)`; `info["message"]` is logged | `cut_inputs`, `lesion`, `scale`, `transmitter_signs`, `edge_labels`, `normalise_inputs`, `rewire` |
| `neuron` | `neuron(n, xp, params, overrides)` | `v`, `dt`, `reset()`, `inject(ids, amount)`, `integrate(currents)`, `fire()` → indices | `lif` |
| `synapse` | `synapse(layer, xp, params)` | `reset()`, `current(spikes, v)` → current vector | `chemical`, `by_transmitter`, `electrical` |
| `encoder` | `encoder(ids, n_tokens, xp, params, cx)` | `inject(token, step)` → `(ids, amount)` or `None` | `random_subset`, `by_group` |
| `feature` | `feature(ids, xp, dt, params)` | `size`, `reset()`, `begin()`, `on_voltage(v)`, `on_spikes(spikes)`, `value()` → numpy vector | `counts`, `voltage`, `trace` |
| `task` | `task(**params)` | `split()` → object with `ids`, `chars`, `n_tr` | `next_char` |
| `readout` | `readout(**params)` | `evaluate(X, split, features=True)` → dict with `val_bpc`, `val_acc` | `softmax_context` |
| `probe` | `probe(**params)` | `run(X, split, groups)` → dict | `memory_span` |
| `baseline` | `baseline(**params)` | `evaluate(split)` → dict with `val_bpc` | `unigram`, `kneser_ney` |

Details that matter when writing one:

- **transform**:
  - It returns a new connectome and must not modify the one it was given.
    Use `cx.with_layer(name, Layer(...))`.
  - `sel` resolves the spec's populations on the connectome as passed in.
  - Keep per-edge arrays aligned. If a transform changes which entries are
    stored, it must reorder the arrays or drop them. `rewire` drops them;
    `lesion` zeroes weights but keeps the entries.
- **neuron**:
  - `overrides` is a list of `(ids, params)` from the spec's
    `neuron.overrides`. Use it for per-population constants.
  - `integrate(currents)` receives one current per synapse plugin, in spec
    order. `lif` adds tonic drive after the first one, as flybrain does.
  - Unknown params should raise an error, not be ignored.
- **synapse**:
  - It receives only its own `Layer`: matrix, kind and edge arrays. It
    never sees other layers or the neuron model.
  - It gets the previous step's spikes and the current membrane state.
  - It keeps its own state (delay queues, synaptic currents) and clears it
    in `reset()`.
- **encoder**:
  - `ids` is the resolved input population.
  - `cx` is read-only and lets an encoder group inputs by an annotation
    (see `by_group`).
- **feature**:
  - `ids` is the resolved readout population and `dt` is the neuron
    model's step.
  - State kept across tokens (for example a trace) is cleared in
    `reset()`; state per token is cleared in `begin()`.

Built-in parameters:

| Plugin | Params |
|---|---|
| `lif` | `dt` (s), `tau` (s), `threshold`, `reset`, `tonic`, `noise_hz`, `noise_amp`, `seed`; overrides of `tau`, `threshold`, `reset`, `tonic` |
| `chemical` | `gain`, `delay` (steps), `tau_syn` (s, 0 = instantaneous), `dt` |
| `by_transmitter` | `edge` (edge array, default `transmitter`), `table` {label: chemical params}, `default` |
| `electrical` | `gain`, `normalise` (`row_max1` or `none`) |
| `random_subset` | `active`, `drive`, `sustain`, `seed` |
| `by_group` | `column`, `active`, `drive`, `sustain`, `seed` |
| `trace` | `tau` (s) |
| `cut_inputs` | `population`, `layers` |
| `lesion` | `population`, `layers`, `direction` (`both`, `inputs`, `outputs`) |
| `scale` | `layer`, `factor`, `pre`, `post` |
| `transmitter_signs` | `layer`, `column`, `table` {transmitter: factor}, `default`, `match` (`contains` or `equals`) |
| `edge_labels` | `layer`, `column`, `name`, `side` (`pre` or `post`) |
| `normalise_inputs` | `layer`, `minimum` |
| `rewire` | `layers` {layer: `permute_presynaptic` or `double_edge_swap`}, `seed`, `classes`, `skip_missing` |

Every synapse entry in a spec also accepts `"optional": true`. The kernel
then skips the entry when the connectome has no such layer, so one spec can
serve connectomes with and without gap junctions.

## The spec (version 2)

A spec is JSON. Each section names a plugin and passes it parameters, and
no section reads another section's settings.

```json
{
 "uctf": 2,
 "name": "my-connectome",
 "connectome": {"source": "folder", "params": {"path": "data"}},
 "transforms": [{"transform": "cut_inputs",
                 "params": {"population": "SENSORY", "layers": ["chemical"]}}],
 "populations": {"SENSORY": {"col": "class", "equals": "sensory"},
                 "READOUT": {"not": "SENSORY"}},
 "neuron": {"model": "lif", "params": {"dt": 0.02, "tau": 0.1},
            "overrides": [{"population": "SENSORY", "params": {"threshold": 1.2}}]},
 "synapses": [{"layer": "chemical", "model": "chemical", "params": {"gain": 1.5}},
              {"layer": "electrical", "model": "electrical",
               "params": {"gain": 0.5, "optional": true}}],
 "input": {"encoder": "random_subset", "population": "SENSORY", "steps": 6,
           "params": {"active": 20, "drive": 1.5, "sustain": 0.5, "seed": 3}},
 "readout": [{"name": "spikes", "feature": "counts", "population": "READOUT"},
             {"name": "trace", "feature": "trace", "population": "READOUT",
              "params": {"tau": 0.1}}],
 "controls": {"rewire-full": [{"transform": "rewire", "params": {
                 "layers": {"chemical": "permute_presynaptic",
                            "electrical": "double_edge_swap"},
                 "skip_missing": true}}]},
 "activity_match": {"param": "synapses.chemical.params.gain",
                    "readouts": ["spikes"]},
 "view": {"groups": [{"name": "sensory", "population": "SENSORY"}]}
}
```

- `connectome.params.path` is relative to the spec file.
- `controls` maps a name to a list of transforms. The controls are applied
  to the raw connectome before the spec's own `transforms`. `bench
  --controls NAME` and `gpf train brain --control NAME` pick them by name,
  and the seed is set per run.
- `activity_match` names the parameter that `bench --match-activity` tunes
  and the readouts whose activity it matches. Paths address list items by
  their `layer` or `name`.
- `--spec-set PATH=VALUE` (bench and `gpf train brain`) overrides a value
  the same way. An example is `synapses.chemical.params.gain=1.8`.
  - An override may change an existing entry or add a key directly under a
    `params` dict. Anything else is an error, so a typo cannot be silently
    ignored.
  - Version-1 paths are translated: `input.code_seed` becomes
    `input.params.seed`, and `neuron.gain` becomes
    `synapses.chemical.params.gain`.
- **Unknown keys are errors.** A spec is checked when loaded. An unknown
  key in any section is an error, and so is an `activity_match.param` that
  is not set explicitly.
- `view` is used only for drawing (`uctf/view.py`) and never changes the
  simulation or the feature-cache key.
- Version-1 specs (UCTF 0.1) are migrated when loaded (`migrate_v1`), and
  models trained with them still load.

Built-in specs: [`uctf/specs/malecns-v1.json`](uctf/specs/malecns-v1.json).
Examples: [`examples/`](examples/).

## Writing a plugin

A plugin is a class or function registered under a kind and a name:

```python
# my_package/uctf_plugins.py
import numpy as np
from uctf import register


@register("feature", "first_spike")
class FirstSpike:
    """Step of each neuron's first spike in the token (steps + 1 if none)."""

    def __init__(self, ids, xp, dt, params):
        self.ids, self.xp, self.size = xp.asarray(ids), xp, len(ids)

    def reset(self):
        pass

    def begin(self):
        self.step, self.first = 0, None

    def on_voltage(self, v):
        self.step += 1

    def on_spikes(self, spikes):
        s = spikes[self.ids] > 0
        if self.first is None:
            self.first = self.xp.full(self.size, np.inf, self.xp.float32)
        new = s & ~self.xp.isfinite(self.first)
        self.first[new] = self.step

    def value(self):
        f = self.first.get() if self.xp is not np else self.first
        return np.where(np.isfinite(f), f, self.step + 1).astype(np.float32)
```

To make UCTF find the plugin, do one of two things:
- Import the module yourself before building a network.
- Declare it as an entry point, and UCTF imports it on first use:

```toml
# my_package/pyproject.toml
[project.entry-points."uctf.plugins"]
my_package = "my_package.uctf_plugins"
```

Then use it by name in a spec (`{"name": "latency", "feature": "first_spike",
"population": "READOUT"}`). `python -m uctf plugins` lists every registered
plugin and every module that could not be imported. `python -m uctf check
SPEC` builds a network and runs one token, which is the quickest test of a
new connectome, spec or plugin.

A package can also add a whole runtime that is not built from a spec, as
GPF does for its original GPF-1 simulator. It calls
`uctf.run.register(name, factory)`, where
`factory(n_tokens, brain_spec, device)` returns an object with `reset()` and
`step_token(token)`.

## Reserved extension points

The data model and protocols already have room for connectome detail that
no built-in plugin uses yet. The table says where each kind of detail goes.
**Built in** means a plugin exists and is tested. **Reserved** means the
interface carries the data, but the model that uses it is still to be
written as a plugin.

| Detail | Where it goes | Status |
|---|---|---|
| Neurotransmitter per neuron | a neuron column; `transmitter_signs` (sign table), `edge_labels` (copy to edges) | built in |
| Neurotransmitter or receptor per synapse | a per-edge array; `by_transmitter` synapse (gain, delay, `tau_syn` per label) | built in |
| Synapse counts, confidence, NT prediction scores | per-edge arrays or extra layers | carried; used by transforms you write |
| Gap junctions | a layer of kind `electrical`; `electrical` synapse | built in |
| Uniform synaptic delay and kinetics | `chemical` params `delay`, `tau_syn` | built in |
| Per-edge delays | a per-edge array plus a synapse plugin with a delay buffer | reserved |
| Neuromodulation (dopamine, serotonin, peptides) | a layer of kind `neuromodulatory` or a neuron column; a synapse or neuron plugin that scales gains from it | reserved |
| Per-neuron biophysics (τ, threshold, size, measured constants) | neuron columns; `neuron.overrides` per population; a neuron plugin may read columns passed as params | overrides built in; per-neuron columns reserved |
| Multi-compartment or conductance-based neurons | a `neuron` plugin (it owns its state; the kernel only needs `v` and `fire()`) | reserved |
| Plasticity while reading | a synapse plugin that updates its own weights in `current()`, or a `learn(spikes, v, **signals)` hook | reserved (hook named in `core/interfaces.py`, not yet called by the loop) |
| Positions, regions, hemispheres | `positions` and neuron columns; selectors; the view | built in |
| Several connectomes of one species (individuals, sexes, stages) | one folder each; the same spec with another `connectome.params.path` | built in |
| Other tasks or token types | a `task` plugin (the loop only sees integer tokens) | reserved |

## Rules

These keep the parts independent:

1. **Explicit only.** A plugin gets everything it uses as arguments or
   params. It does not read another plugin's attributes, global settings or
   another section of the spec.
2. **Reject unknown params.** A typo must fail loudly (see
   `neurons.LIF` and `synapses._check`).
3. **Pure transforms.** Return a new connectome, keep per-edge arrays
   aligned, and report what changed in `info`.
4. **Populations are names.** Plugins receive neuron indices. Only the
   spec says which neurons they are.
5. **Deterministic.** Randomness comes from a `seed` param. The same spec
   and seed give the same features on the CPU.
6. **Cache honesty.** When a change alters simulation results for the same
   spec, bump `uctf.run.SIM_VERSION` so cached features are not reused.
7. **Short lines.** At most 88 characters; `tests/test_uctf.py` checks
   this.
