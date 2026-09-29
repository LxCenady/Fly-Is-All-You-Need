# Bring your own connectome

GPF can turn any wiring diagram into a character-level language model: a simulated network
reads text one character at a time, and a linear readout learns to predict the next character
from its neurons. The fly (MaleCNS v1.0) is one example. This page shows how to plug in
another connectome and how to test, honestly, what its wiring contributes.

Three steps: **import** the connectome, **describe** it in a spec, **benchmark** it.

## 1. Import

A connectome is a folder with two files:

| File | Contents |
|---|---|
| `weights.npz` | scipy sparse matrix, n × n, rows = postsynaptic neuron, columns = presynaptic neuron, signed float32 weights |
| `neurons.npz` or `neurons.csv` | one entry per neuron: any annotation columns (cell type, class, side, …) and optionally `x`, `y`, `z` for the brain view |

From two CSV files (a neuron table and an edge list):

```
python -m gpf import --neurons neurons.csv --edges edges.csv --out my-connectome --sign-col nt
```

Weights are synapse counts, negative when the presynaptic neuron's `--sign-col` names an
inhibitory transmitter (default: GABA, glutamate, histamine), and each neuron's inputs are
divided by its total absolute input. This is flybrain's recipe for MaleCNS, so different
connectomes start on the same scale. `--no-normalise` keeps raw counts.

## 2. Describe it in a spec

A spec is a JSON file. It says which neurons receive the text, what is read out and how the
neurons behave. Built-in: [`gpf/connectome/specs/malecns-v1.json`](gpf/connectome/specs/malecns-v1.json)
(the fly, GPF-1's protocol). A minimal one is in
[`examples/toy_connectome/toy.json`](examples/toy_connectome/toy.json):

```json
{
 "name": "toy",
 "connectome": {"loader": "folder", "path": "toy"},
 "populations": {"IN": {"col": "cell_type", "equals": "IN"},
                 "EXP": {"col": "cell_type", "equals": "EXP"}},
 "neuron": {"dt": 0.02, "tau": 0.1, "threshold": 1.0, "gain": 1.5, "tonic": 0.05},
 "input": {"population": "IN", "active": 20, "drive": 1.5, "steps": 6, "sustain": 0.5},
 "readout": [{"name": "exp", "population": "EXP", "feature": "counts"}]
}
```

- **Neurons** are leaky integrate-and-fire, as in flybrain:
  `v ← e^(−dt/τ)·v + gain·W·spikes + tonic`, spike and reset at threshold.
- **Input**: each character gets a fixed random set of `active` neurons from the input
  population, driven for `steps` steps.
- **Readout features** per character: `counts` (spikes), `voltage` (mean membrane voltage)
  or `trace` (a decaying spike trace).
- **Populations** are chosen by annotation (`contains`, `equals`, `in`, `startswith`,
  `regex`, combined with `and`, `or`, `not`) or by wiring (`top_targets_of`: the neurons that
  receive the most input from another population).
- **Other fields:**
  - `connectome.cut_inputs_to`: silence the inputs onto some neurons.
  - `classes`: the cell classes used by class-preserving rewiring.
  - `view`: groups, colours and labels for the web UI's brain view.

## 3. Benchmark

```
python -m gpf bench --substrate my-spec.json --data text.txt
```

On one train/validation split this reports validation bits per character for a unigram, 3-
and 5-gram Kneser-Ney models, the readout's context table alone, and the context table plus
the connectome. It does the same for **degree-preserving rewired copies** of the connectome:

- `rewire-full`: all (presynaptic neuron, weight) pairs permuted across the whole network.
- `rewire-class`: the same, but only within blocks of (presynaptic class, postsynaptic class).

In both, every neuron keeps its number of inputs and outputs and its output signs and weights;
only who connects to whom is randomised. The report also gives a **memory span**: how well the
character k steps back can be decoded from each readout group.

These are the questions of the connectome-reservoir report ([`paper_lm/`](../paper_lm)):

- Does the brain add anything over a simple context table?
- Does it come anywhere near a smoothed n-gram?
- How far back does it remember?
- Does any of it depend on the real wiring?

On MaleCNS the answers were: a little, no, about four characters, and no. A new connectome
deserves the same questions.

## Then train and play

```
python -m gpf train brain --substrate my-spec.json --data text.txt --name my-model
python -m gpf --web          # pick my-model under "Your models"; the brain view uses your spec
```

`--control rewire-full` trains on a rewired copy instead. `--spec-set neuron.gain=1.8` changes
one spec entry for one run. Models store their full spec, so they can be reloaded later exactly.

## Notes

- **Speed:** CuPy on an NVIDIA GPU is used when available (`--device cpu` to force the CPU).
  MaleCNS (166,700 neurons, 25 M synapses) runs at about 85 characters per second on a laptop
  GPU. Networks of a few thousand neurons run fine on a CPU.
- **Checks:** the generic simulator with `malecns-v1` gives the same spikes as GPF-1's runtime
  and voltages equal to float32 rounding. Whole-network rewiring with seed 0 reproduces the
  paper's rewired connectome exactly.
- **Gap junctions** (electrical synapses) are supported. Import them with `--type-col` (edges
  marked `electrical`); they are stored as `gap.npz` and switched on by `neuron.gap_gain` in
  the spec. The rewiring controls rewire them too, by degree-preserving double-edge swaps.
- **Not modelled yet:** neuromodulation, plasticity during reading, and multi-compartment
  neurons.

## Examples

| Example | What it shows |
|---|---|
| [`gpf/connectome/specs/malecns-v1.json`](gpf/connectome/specs/malecns-v1.json) | the adult fly CNS (166,700 neurons), GPF-1's protocol |
| [`examples/celegans/`](examples/celegans/) | a different species: the whole C. elegans worm (Cook et al. 2019), with gap junctions and a hand-chosen operating point. Its result has the same pattern as the fly's. |
| [`examples/toy_connectome/`](examples/toy_connectome/) | a made-up network, showing the CSV format end to end |
