# Fly Is All You Need

Independent research (high-school project) on a leaky integrate-and-fire model
of the complete adult male *Drosophila* CNS connectome (MaleCNS v1.0,
166,700 neurons), in two parts:

1. **Language model** (`paper_lm/`) — the project started as an attempt to use
   the frozen connectome as a reservoir for character-level language modelling
   on TinyShakespeare, later with dopamine-gated plasticity on the 61,210
   Kenyon-cell (KC) -> mushroom-body output neuron (MBON) synapses. Plasticity
   wrote decodable information into the synapses but did not improve
   prediction.
2. **Mechanism study** (`paper/`) — grew out of that failure: where is a
   mushroom-body memory stored, what retrieves it, what sets its lifetime, and
   how does it reach downstream neurons? Answered with causal interventions
   (state transplants, weight scrambles, wiring nulls, teacher swaps).

The mechanism results were then used to revise and test the language model.

**Try it:** [GPF, the Generative Pretrained Fly](playground/) continues a prompt with the
fly connectome or with n-gram/GRU comparison models (downloads on the
[Releases](https://github.com/LxCenady/Fly-Is-All-You-Need/releases) page).

## Main findings (model-internal; not claims about the fly)

**Mechanism** — a mushroom-body memory has three addresses:

| Question | Answer in this model |
|---|---|
| Carrier after ~1 s | KC->MBON weights only; fast activity decays with the membrane time constant |
| Lifetime | exactly the prescribed synaptic decay; the network adds no forgetting or consolidation |
| Retrieval key | identity of the written KCs, only when the KC code is sparse (160-192 of 675 PNs) |
| Output identity | body side and KC subtype on both sides of the KCs; within those, KCs are interchangeable |
| Write location | which dopaminergic neurons teach (87 % of output variance when teachers differ) |
| Interference | linear superposition in the weights, set by KC overlap |
| Downstream readout | only when the memory flips an MBON spike (35/35 vs 0/93 cells) |

**Language model** — a clear negative result. The earlier benchmark (42.55 % /
3.298 bits per character) was matched by a hashed three-character context
alone. With sparse input the brain beats that weak context head (18/18
held-out cases), but a Kneser-Ney 5-gram (2.47-2.68 BPC) beats every brain
model, and adding the brain to it makes prediction worse. The frozen
connectome's KC code remembers about four characters (the character two back
is decoded in 57-60 % of cases, four back in 24-26 %), and degree-preserving
rewired connectomes do as well as the real one. A content-dependent
dopaminergic teacher (a GRU-like write gate) lets the mushroom body learn
associations online within two or three exposures, but on real text it adds
nothing beyond the 5-gram.

Every number is mapped to its script and output file in
[`docs/CLAIMS_EVIDENCE.md`](docs/CLAIMS_EVIDENCE.md).

## Known problems and corrections

- Before a simulator fix (2026-09-17) the masked weight matrix was re-sorted
  on the GPU after the plasticity module had cached synapse positions, so
  plastic writes landed on the wrong synapses (`audit_20260928/`). All
  plasticity results produced before the fix, including every plasticity
  language-model run, are superseded; frozen results are unaffected.
- The earlier language-model benchmark was essentially an n-gram score, and
  its summary document quotes 42.71 % / 3.227 while the metrics file records
  42.55 % / 3.298.
- An intermediate conclusion of the mechanism study (output identity needs
  individual KC wiring) was withdrawn after side-preserving controls.
- Claim-by-claim corrections of the first paper draft:
  `mechanism/PAPER_OVERTURN_20260928.md`.

## Layout

```
paper/            mechanism paper (IEEEtran main.tex, make_figures.py, figures/)
paper_lm/         language-model report (main.tex, make_figures_lm.py, figures/)
mechanism/        experiment scripts; paths.py = where inputs and outputs live
                  m1_core.py = validated snapshot/restore interface
                  lm_mech.py = revised language model; lm_select.py = held-out selection
                  queue_runner.py = job queue; RESULTS_20260928.md = result register
harness/          experiment harness from the earlier project stage (15 modules used by mechanism/)
playground/       GPF, the Generative Pretrained Fly (web UI, terminal UI, packages)
results/          JSON outputs (results/lm/, results/lm_limits/, results/sens/, results/apl/)
audit_20260928/   evidence for the simulator bug
docs/             CLAIMS_EVIDENCE.md, LITERATURE_INDEX.md
```

## Reproducing

```
pip install "flybrain[gpu]==0.1.0" scipy
python mechanism/lm_mech.py --out outputs/test.json --train 900 --val 100 --active 160 --scale 1.5
```

- **Simulator:** [flybrain](https://github.com/alextitonis/fly.ai) (MIT) with GPU support
  and an NVIDIA GPU. On first use it downloads the MaleCNS v1.0 connectome (~260 MB) into
  `~/fly-data`; its checksums match the data used here.
- **Paths:** nothing is hard-coded. `mechanism/paths.py` reads `FLY_DATA`, `FLY_OUT`
  (default `outputs/`), `FLY_HARNESS`, `FLY_CORPUS`, `FLY_LEGACY` and `FLY_PYTHON` from the
  environment or from a git-ignored `mechanism/paths_local.json`.
- **Speed:** RTX 3070 Ti Laptop, about 40 characters/s for one process, about 70/s in
  total with 3-4 processes. CPU-only: `m6_structure.py`, `subtype_lobe.py`, the language-model
  analyses on cached features, and the audit CSR check.
- **Figures:** `paper/make_figures.py` and `paper_lm/make_figures_lm.py` read the
  experiment outputs from `FLY_OUT`; the JSON results they use are in `results/`.

## Data, citation and license

Connectome: MaleCNS v1.0 by FlyEM (HHMI Janelia), University of Cambridge, MRC
LMB and Google Research, CC BY 4.0; cite Berg, S. et al. (2026), *Sexual
dimorphism in the complete Drosophila male central nervous system
connectome*, Cell 189(18), 5504-5526.e15. Simulator: flybrain by Alex Titonis (MIT).
Code in this repository: MIT (see `LICENSE`).

## Use of AI tools

This work used AI agents extensively (Anthropic Claude, OpenAI Codex,
DeepSeek via OpenCode) to write and run code, run experiments, audit results
and draft the papers; the author directed the research and is responsible for
the content. See the statements in both papers.
