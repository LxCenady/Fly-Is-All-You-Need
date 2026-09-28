# Fly Is All You Need

Mechanism analysis of a dopamine-gated memory in a connectome-constrained
leaky integrate-and-fire model of the adult male *Drosophila* CNS
(166,700 neurons; plasticity on the 61,210 Kenyon-cell -> MBON synapses).

The question is not how much the model can remember but **where a memory is
stored, what retrieves it, what sets its lifetime and how it leaves the
mushroom body**, answered with causal interventions (state transplants,
weight scrambles that preserve the multiset of weight changes, wiring nulls,
teacher swaps) and matched controls.

## Main results (model-internal; not claims about the fly)

| Question | Answer in this model | Evidence |
|---|---|---|
| What carries the memory after ~1 s? | The KC->MBON weights only; the fast state relaxes with the membrane time constant, no reverberation | transplant of synaptic vs fast state, `m1_cross.py` |
| What sets its lifetime? | Exactly the prescribed synaptic decay; with decay off the readout is constant for 123 s | `m4_curve.py` |
| When is the KC code sparse? | Only at small PN codes (160-192 of 675 PNs: 9-16 % of KCs, Jaccard 0.08-0.14); the legacy setting (512 PNs) activates 87 % of KCs | `sweep_opoint.py`, `kc_density.py` |
| What retrieves it? | The identity of the written KCs (cue specificity 3.6-4.7 at sparse input, 1.06 at dense); random PN->KC wiring suffices | `m2_current.py`, `m3_pnkc.py` |
| What makes different items' outputs different? | Coarse wiring only: body side and KC subtype on both sides of the KCs; within a subtype and side, KCs are interchangeable | `m3_kcmbon.py`, `m6_*.py` |
| Role of the dopaminergic teacher | Chooses the compartment: with different teachers it explains 87 % of output variance, the item 5 % | `m7_teacher.py`, `m7_map.py` |
| Interference between memories | Linear superposition in the weights, set by KC overlap (0.10-0.13 sparse vs 0.86-0.90 dense) | `m8_interference.py` |
| How does it reach downstream? | Only by flipping MBON spikes (35/35 changed-raster cells vs 0/93 identical) | `m5_downstream.py`, `m5b_timing.py` |

Corrections to an earlier analysis of the same model (simulator bug that wrote
to the wrong synapses, predictions true by construction, dense operating point)
are listed in `mechanism/PAPER_OVERTURN_20260928.md` and in the paper.

## Layout

```
mechanism/     experiment scripts (m1_core.py = validated snapshot/restore interface)
               RESULTS_20260928.md      result register (every number -> JSON file)
               PAPER_OVERTURN_20260928.md  claim-by-claim correction of the earlier draft
results/       JSON outputs of every experiment (+ KC input clusters .npz)
paper/         main.tex (IEEEtran), make_figures.py, figures/
audit_20260928/  CSR canonicalisation bug: evidence scripts and outputs
```

## Reproducing

The scripts run on top of the `flybrain_lm` code base and the `flybrain`
simulator package (with the CSR canonicalisation fix of 2026-09-17), which are
**not included here**; paths in the scripts point to the original machine
(`D:\flybrain_lm_cuda`, `E:\mechanism_20260928`). A CUDA GPU (CuPy) is
required for the simulations; `m6_structure.py` and the audit CSR check run on
CPU. Figures are regenerated from `results/` with `paper/make_figures.py`
(adjust the input path at the top).

## Status

Draft. The paper compiles on Overleaf with no errors or warnings. The
connectome is MaleCNS v1.0 (Berg et al., Cell 2026). Related work and the
novelty boundary are in `docs/LITERATURE_INDEX.md`.

## License

MIT (see `LICENSE`).
