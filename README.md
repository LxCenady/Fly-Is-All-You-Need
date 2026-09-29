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

The mechanism results were then used to revise the language model.

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

**Language model** — the earlier benchmark (42.55 % / 3.298 bits per
character on 20k/5k) is matched by a hashed three-character context alone
(42.9 % / 3.205). With sparse input, the brain adds to that context: over
3 text segments x 3 PN-code draws, with readouts chosen on held-out training
data, BPC drops by 0.078 +- 0.020 (160 PNs) and 0.076 +- 0.014 (192) in 9/9
cases each, but only by 0.007 +- 0.019 with the old dense input (5/9). The
gain is in calibration (BPC); top-1 accuracy is unchanged. With 100k
training characters (one run) the gain falls to 0.01 BPC, within fit noise:
the brain helps the n-gram only while data are scarce.

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
paper_lm/         language-model report (main.tex)
mechanism/        experiment scripts; m1_core.py = validated snapshot/restore interface
                  lm_mech.py = revised language model; lm_select.py = held-out selection
                  queue_runner.py = overnight job queue
                  RESULTS_20260928.md = result register
results/          JSON outputs (results/lm/: language model)
audit_20260928/   evidence for the simulator bug
docs/             CLAIMS_EVIDENCE.md, LITERATURE_INDEX.md
```

## Reproducing

The scripts run on top of the `flybrain_lm` code base and the `flybrain`
simulator package with the 2026-09-17 CSR canonicalisation fix; neither is
included here, and paths point to the original machine
(`D:\flybrain_lm_cuda`, `E:\mechanism_20260928`). A CUDA GPU (CuPy) is
required for simulations (RTX 3070 Ti Laptop: ~40 tokens/s for one process,
~70 tokens/s total with 3-4 processes). CPU-only: `m6_structure.py` and the
audit CSR check.

## Data, citation and license

Connectome: MaleCNS v1.0 by FlyEM (HHMI Janelia), University of Cambridge, MRC
LMB and Google Research, CC BY 4.0; cite Berg, S. et al. (2026), *Sexual
dimorphism in the complete Drosophila male central nervous system
connectome*, Cell 189(18), 5504-5526.e15. Code in this repository: MIT
(see `LICENSE`).

## Use of AI tools

This work used AI agents extensively (Anthropic Claude, OpenAI Codex,
DeepSeek via OpenCode) to write and run code, run experiments, audit results
and draft the papers; the author directed the research and is responsible for
the content. See the statements in both papers.
