# UCTF: Universal Connectome Training Framework

UCTF turns a connectome, any wiring diagram of a nervous system, into a character-level
language model. A simulated network reads text one character at a time, and a linear readout
learns to predict the next character from its neurons. UCTF also runs the controls that ask
whether the wiring matters.

The idea of UCTF is Quanxi Li's. The implementation reorganises and debugs tools that already
existed in this project: the GPF-1 runtime, the training and evaluation scripts of the
language-model report (`../paper_lm`), and its rewiring controls. UCTF is tested on two
connectomes from two species, so it is general in design but not yet shown to be general in
practice (see [Limits](#limits)).

[GPF](../playground) is the chat-style app built on it.

## Design

UCTF is a minimal kernel that operates on connectomes. Everything that is a
modelling choice is a named plugin:
- where the connectome comes from;
- how it is changed;
- the neuron and synapse models;
- how tokens enter;
- what is read out;
- the task, readout, probes and baselines.

A JSON spec composes the plugins, and no part knows another part's
internals.

```
uctf/core      data model (Connectome, Layer), selectors, spec, registry, step loop
uctf/plugins   source · transform · neuron · synapse · encoder · feature ·
               task · readout · probe · baseline
uctf/specs     built-in specs (malecns-v1)
```

**[PLUGINS.md](PLUGINS.md) is the protocol.** It covers:
- the data model and the step loop;
- the call signature of every plugin kind;
- every field of the spec;
- how to ship plugins from your own package (entry-point group
  `uctf.plugins`);
- which extension points are reserved for detail no built-in model uses yet
  (per-edge delays, neuromodulation, plasticity, multi-compartment neurons).

Neurotransmitters are handled at two levels:
- a neuron column, used by the `transmitter_signs` and `edge_labels`
  transforms;
- a per-synapse edge array, used by the `by_transmitter` synapse, which
  sets gain, delay and kinetics per label.

## Install

```
pip install -e uctf                  # numpy + scipy; add [gpu] for CuPy, [malecns] for the fly
python -m uctf                       # lists the commands
python -m uctf plugins               # every registered plugin
python uctf/tests/test_uctf.py       # invariants, CPU, seconds
```

## 1. Import a connectome

A connectome is a folder (format 2; details in [PLUGINS.md](PLUGINS.md#the-data-model)):

| File | Contents |
|---|---|
| `connectome.json` | format, metadata (name, source, units, transforms applied) and the synapse layers with their kind |
| `neurons.npz` | one array per neuron column (any annotations), `positions` if known |
| `<layer>.npz` | the layer's sparse matrix, n × n, rows = postsynaptic, columns = presynaptic |
| `<layer>.edges.npz` (optional) | per-edge arrays (transmitter, receptor, confidence, ...) |

Folders in the earlier format (`weights.npz`, `gap.npz`, `neurons.npz` or
`neurons.csv`) still load.

From a neuron table and an edge list:

```
python -m uctf import --neurons neurons.csv --edges edges.csv --out my-connectome \
    --sign-col nt [--type-col type] [--edge-cols nt,confidence] [--raw]
```

The layers:
- **chemical**: synapse counts, from every edge row not marked electrical.
- **electrical**: rows whose `--type-col` is `electrical` become this
  symmetric layer of gap junctions.

Per-edge columns named by `--edge-cols` are kept as edge arrays.

Unless `--raw` is given, the chemical layer then goes through two
transforms:
1. `transmitter_signs`: outputs are negative when the presynaptic neuron's
   `--sign-col` names an inhibitory transmitter (default: GABA, glutamate,
   histamine).
2. `normalise_inputs`: each neuron's inputs are divided by its total
   absolute input.

This is flybrain's recipe for MaleCNS, so different connectomes start on
the same scale. With `--raw` the folder keeps raw counts, and the spec's
`transforms` can apply its own recipe, such as a species-specific
transmitter table.

## 2. Describe it in a spec

A JSON spec names a plugin and its params for every part. The built-in
spec is [`uctf/specs/malecns-v1.json`](uctf/specs/malecns-v1.json), the
adult fly CNS with GPF-1's protocol. The sections:

- `connectome`: a source plugin, for example `{"source": "folder",
  "params": {"path": ...}}` (relative to the spec) or `{"source":
  "flybrain"}`.
- `transforms`: changes applied to it in order, for example `cut_inputs`,
  `lesion`, `scale`, `transmitter_signs`, `edge_labels` or
  `normalise_inputs`.
- `populations`: named neuron sets.
  - By annotation: `contains`, `equals`, `in`, `startswith`, `regex`,
    `range`, combined with `and`, `or`, `not`.
  - By wiring: `top_targets_of`, the cells that receive the most input from
    another set in a named layer.
- `neuron`: the neuron model and its params. The built-in `lif` is leaky
  integrate-and-fire, as in flybrain, with per-population `overrides`.
- `synapses`: one entry per layer, each with a synapse model:
  - `chemical`: `gain · W · spikes`, with optional delay and synaptic time
    constant;
  - `by_transmitter`: the same, per transmitter label;
  - `electrical`: `gain · Σ_j Ĝ_ij (v_j − v_i)`, where Ĝ has rows scaled to
    a total of at most 1.

  `"optional": true` skips a layer that the connectome lacks.
- `input`: the encoder, its input population and the steps per token.
  `random_subset` drives a fixed random set of `active` neurons per
  character.
- `readout`: named features per token: `counts` (spikes), `voltage` (mean
  membrane voltage) or `trace` (decaying spike trace) of a population.
  Read out neurons downstream of the input, not the driven input neurons
  themselves: their own membrane leak would otherwise count as memory.
- `controls`: named transform lists for null models, such as `rewire-full`
  and `rewire-class`.
- `activity_match`: the param `--match-activity` tunes, and the readouts
  whose activity it matches.
- `view`: groups, colours, labels and captions for GPF's brain view.
  Without positions, the view is a schematic with one column per group.

Version-1 specs (UCTF 0.1) are migrated when loaded.
`python -m uctf check my-spec.json` builds the network and runs one token.

## 3. Benchmark it

```
python -m uctf bench --substrate my-spec.json --data text.txt [--match-activity]
```

On one train/validation split this reports validation bits per character (lower is better)
for:

- unigram, and Kneser-Ney 3- and 5-gram models;
- the readout's hashed 3-character context table alone;
- the context table plus the connectome;
- the same for degree-preserving **rewired copies** of the connectome.

There are two kinds of rewiring:

- `rewire-full`: (presynaptic neuron, weight) pairs are permuted across the whole network.
- `rewire-class`: the same, but only within blocks of (presynaptic class, postsynaptic class).

Every neuron keeps its number of inputs and its outputs' signs and weights. Gap junctions
are rewired by degree-preserving double-edge swaps.

Rewiring changes how active the network is: weights move to neurons they were not normalised
for. With `--match-activity`, each control is also run at the gain that makes its readout
neurons as active as the real network's. The gain is set on the first 500 training
characters, with no prediction results involved. The report lists the activity of every
variant.

It also reports a **memory span**: how well the character k steps back can be decoded from
each readout group (ridge regression, validation rows).

The questions, from the connectome-reservoir report ([`../paper_lm`](../paper_lm)):

- Does the brain add anything over a simple context table?
- Does it come near a smoothed n-gram?
- How far back does it remember?
- Does any of it depend on the real wiring?

## Then train and play

```
python -m gpf train brain --substrate my-spec.json --data text.txt --name my-model
python -m gpf --web
```

In the web UI, pick my-model under "Your models"; the brain view follows your spec. Models
store their full spec. The connectome folder is referenced by path, so keep it where it was.

## Examples and results

| Example | |
|---|---|
| [`uctf/specs/malecns-v1.json`](uctf/specs/malecns-v1.json) | the adult fly CNS, 166,700 neurons (MaleCNS v1.0 via flybrain) |
| [`examples/celegans/`](examples/celegans/) | the whole C. elegans worm (Cook et al. 2019): 300 neurons, gap junctions, a hand-chosen operating point |
| [`examples/toy_connectome/`](examples/toy_connectome/) | a made-up network, showing the CSV format end to end |

Setup for both species: TinyShakespeare, 20k training / 5k validation characters, validation
BPC. Ranges are over 3 rewiring seeds; for the worm's real wiring, over 3 input codes.

| | Fly (MaleCNS v1.0) | Worm (C. elegans) |
|---|---|---|
| Kneser-Ney 5-gram | 2.680 | 2.680 |
| context table only | 3.205 | 3.205 |
| + real connectome | 3.122 | 3.135–3.160 |
| + rewired, whole network, same gain | 3.139 (1 seed) | 3.121–3.126 |
| + rewired, whole network, activity matched | 3.126 (1 seed)* | 3.151–3.171 |
| + rewired, within classes, activity matched | 3.120 (1 seed) | 3.169–3.186 |
| character 2 back decoded / 3 back | 84% / 50% | 66–67% / 33% |

In both species the connectome improves a little on the context table, stays far behind a
5-gram, and remembers only a few characters. Rewired copies at matched activity do about as
well as the real wiring. In the worm they are on average 0.014 (whole network) and 0.033
(within classes) BPC worse; in the fly they are within 0.01. Without activity matching, the
fully rewired worm looked better than the real one. That came only from it being much
quieter (2–4% of readout neurons active instead of 6%).

\* The fully rewired fly's Kenyon cells are silent at the original gain. The gain that
matched activity over the first 500 characters (3.7 instead of 1.5) left its readout neurons
more active over the whole run (1.4% against 0.2%), so this match is imperfect.

## Checks

`tests/test_uctf.py` checks these properties:
- rewiring keeps in-degrees, output weights and class blocks, and gap
  junctions stay symmetric with unchanged degrees;
- CPU simulation is deterministic, and a control reads out the same
  neurons;
- `respec` changes dynamics only and refuses wiring changes;
- the feature-cache key follows the simulation settings (including
  `SIM_VERSION`) but not the display settings;
- brain specs saved by UCTF 0.1 load and give the same features;
- connectome folders round-trip, including per-edge arrays;
- `by_transmitter` with one gain for all labels equals the plain chemical
  synapse;
- readout rows are aligned: features that leak the target reach near-zero
  BPC, and features holding only the current character add nothing;
- every built-in plugin registers;
- no line is longer than 88 characters.

Other checks:
- **0.2 against 0.1.** The plugin kernel reproduces UCTF 0.1's features
  bit for bit on the toy network and the worm (real wiring and both
  rewirings). With `malecns-v1` on a GPU they agree to 9e-8, which is GPU
  float32 nondeterminism. The worm benchmark reproduces 0.1's report.
- **Against GPF-1.** With `malecns-v1`, the simulator gives the same spikes
  as GPF-1's original runtime, and voltages equal to float32 rounding.
- **Against the report's controls.** Both rewirings with seed 0 reproduce
  the language-model report's rewired connectomes exactly (identical
  matrices).
- **Against the report's memory span.** The fly benchmark reproduces the
  report's Kenyon-cell memory span: 100, 90, 59, 36, 25% for k = 0–4.

## Limits

- **Only two connectomes** (fly and worm), one task (TinyShakespeare), one input protocol and
  a linear readout.
- **Hand-chosen operating points.** The worm's was chosen by a criterion fixed before looking
  at results, but it was still chosen by hand.
- **Simplified worm signs.** Only the 26 GABAergic neurons are inhibitory.
- **Model gaps.** No built-in neuromodulation, plasticity during reading or multi-compartment
  neurons. Their interfaces are reserved ([PLUGINS.md](PLUGINS.md#reserved-extension-points)); the
  models are not written.
- **Speed.** CuPy is used when an NVIDIA GPU is present (`--device cpu` to force the CPU).
  MaleCNS runs at about 70–85 characters per second on a laptop GPU; the worm runs on a CPU in
  seconds.
