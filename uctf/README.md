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

## Install

```
pip install -e uctf                  # numpy + scipy; add [gpu] for CuPy, [malecns] for the fly
python -m uctf                       # lists the commands
python uctf/tests/test_uctf.py       # invariants, CPU, seconds
```

## 1. Import a connectome

A connectome is a folder:

| File | Contents |
|---|---|
| `weights.npz` | scipy sparse matrix, n × n, rows = postsynaptic neuron, columns = presynaptic neuron, signed float32 weights (chemical synapses) |
| `gap.npz` (optional) | symmetric matrix of electrical synapses (gap junctions), raw weights |
| `neurons.npz` or `neurons.csv` | one entry per neuron: any annotation columns and optionally `x`, `y`, `z` |

From a neuron table and an edge list:

```
python -m uctf import --neurons neurons.csv --edges edges.csv --out my-connectome \
    --sign-col nt [--type-col type]
```

Chemical weights are synapse counts. They are negative when the presynaptic neuron's
`--sign-col` names an inhibitory transmitter (default: GABA, glutamate, histamine), and each
neuron's inputs are divided by its total absolute input. This is flybrain's recipe for
MaleCNS, so different connectomes start on the same scale. Edges whose `--type-col` says
`electrical` become gap junctions.

## 2. Describe it in a spec

A JSON spec says which neurons receive the text, what is read out and how neurons behave.
Built-in: [`uctf/specs/malecns-v1.json`](uctf/specs/malecns-v1.json), the adult fly CNS with
GPF-1's protocol. The fields:

- `connectome`: `{"loader": "folder", "path": ...}` (relative to the spec) or `{"loader": "flybrain"}`.
  `cut_inputs_to` silences the inputs onto some neurons.
- `populations`: named neuron sets.
  - By annotation: `contains`, `equals`, `in`, `startswith`, `regex`, combined with `and`,
    `or`, `not`.
  - By wiring: `top_targets_of`, the cells that receive the most input from another set.
- `neuron`: leaky integrate-and-fire, as in flybrain,
  `v ← e^(−dt/τ)·v + gain·W·spikes + tonic + gap_gain·Σ_j Ĝ_ij (v_j − v_i)`, where Ĝ is the
  gap matrix with rows scaled to total at most 1. A neuron spikes and resets at threshold.
- `input`: each character drives a fixed random set of `active` neurons of one population for
  `steps` steps.
- `readout`: per character, `counts` (spikes), `voltage` (mean membrane voltage) or `trace`
  (decaying spike trace) of a population. Read out neurons downstream of the input, not the
  driven input neurons themselves: their own membrane leak would otherwise count as memory.
- `classes`: the cell classes for class-preserving rewiring.
- `view`: groups, colours, labels and captions for GPF's brain view. Without positions, the
  view is a schematic with one column per group.

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

- `tests/test_uctf.py` checks four properties:
  - rewiring keeps in-degrees, output weights and class blocks, and gap junctions stay
    symmetric with unchanged degrees;
  - CPU simulation is deterministic;
  - the feature-cache key follows the simulation settings (including `SIM_VERSION`) but not
    the display settings;
  - readout rows are aligned: features that leak the target reach near-zero BPC, and features
    holding only the current character add nothing.
- With `malecns-v1`, the simulator gives the same spikes as GPF-1's original runtime, and
  voltages equal to float32 rounding.
- Both rewirings with seed 0 reproduce the language-model report's rewired connectomes
  exactly (identical matrices).
- The fly benchmark reproduces the report's Kenyon-cell memory span: 100, 90, 59, 36, 25%
  for k = 0–4.

## Limits

- **Only two connectomes** (fly and worm), one task (TinyShakespeare), one input protocol and
  a linear readout.
- **Hand-chosen operating points.** The worm's was chosen by a criterion fixed before looking
  at results, but it was still chosen by hand.
- **Simplified worm signs.** Only the 26 GABAergic neurons are inhibitory.
- **Model gaps.** No neuromodulation, plasticity during reading or multi-compartment neurons.
- **Speed.** CuPy is used when an NVIDIA GPU is present (`--device cpu` to force the CPU).
  MaleCNS runs at about 70–85 characters per second on a laptop GPU; the worm runs on a CPU in
  seconds.
