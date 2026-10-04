# Drosophila larva (Winding et al. 2023) with UCTF

The complete brain of a first-instar *Drosophila* larva: 2,952 neurons and 352,611 chemical
synapses, reconstructed from electron microscopy (Winding et al. 2023, *Science*). This is the
third connectome run through UCTF, after the adult fly (MaleCNS) and the worm (*C. elegans*).
It is the same species as the adult fly at a different stage of life, about 50 times smaller.

```
cd examples/larva
python prepare_winding2023.py        # downloads ~1.1 MB (the paper's Supplementary Data S1), writes neurons.csv, edges.csv
python -m uctf import --neurons neurons.csv --edges edges.csv --out winding2023 \
    --sign-col nt_assumed --inhibitory inhibitory
python select_gain.py --max-gain 3.0 > gains.json      # operating point, by activity only
python -m uctf bench --substrate larva-winding2023.json --data <text> --vocab <vocab.json> \
    --train-chars 20001 --val-chars 5000 --device cpu --control-seeds 0,1,2 --match-activity
```

Everything runs on a CPU.

## Data

- **Source.** Supplementary Data S1 of Winding et al. (2023): `all-all_connectivity_matrix.csv`
  and `annotations.csv`. The prepare script downloads it from the Betzel lab's mirror at
  [brain-networks/larval-drosophila-connectome](https://github.com/brain-networks/larval-drosophila-connectome),
  because Science's own site blocks scripted downloads. The script checks the SHA-256 of the zip.
- **Orientation.** In the matrix, rows are presynaptic and columns postsynaptic. Known
  connections confirm this:
  - KC→MBON has 17,752 synapses, MBON→KC 195;
  - PN→KC has 4,710 synapses, KC→PN 207.
- **Weights.** Synapse counts summed over all four compartment pairings: axon→axon,
  axon→dendrite, dendrite→axon and dendrite→dendrite. Each neuron's inputs are then divided by
  its total absolute input, the same recipe as for the fly and the worm.
- **Cell types.** 2,606 neurons carry a cell type in `annotations.csv`. The other 346 are
  "unannotated".

## Modelling choices

- **Signs.** The supplementary data has no neurotransmitters.
  - Neurons of cell type `LN` (local neurons, 110 cells, including the GABAergic broad LNs and
    the APL) are taken as inhibitory. Every other neuron is excitatory.
  - This is a strong simplification: other larval neurons are inhibitory too, and not every LN
    may be.
- **Input.** Each character drives its own random 50 of the 206 second-order projection
  neurons (`PN`). That is about the same share (24%) as in the fly (160 of 675) and the worm
  (20 of 83).
- **Sensory neurons.** Synapses onto sensory neurons are removed, as in the fly protocol.
- **Readout.** Spike counts, mean voltage and a spike trace of every neuron that is neither a
  PN nor sensory: 2,316 neurons, 6,948 features per character. As with the fly and the worm,
  the driven input neurons are left out.
- **Operating point.** The criterion is the worm's, fixed before any prediction result: on
  average 5–30% of all neurons active per character, never above 60%. The rule is the largest
  gain that meets it (`select_gain.py`, first 500 training characters).
  - No gain from 0.5 to the pre-stated limit of 1.5 was active enough: at 1.5 only 4.0% of
    neurons fire. The larva is much quieter than the worm, which runs away at the fly's gain
    of 1.5.
  - The grid was therefore extended to 3.0, with the same criterion and rule, still before any
    prediction result was seen.
  - On the first 500 characters gains 1.7 and 1.8 met it; over the whole training text 1.8 runs away (see Result), so 1.7 is used.
  - Activity jumps abruptly from 1.8 to 1.9 (from 6% to 50–80%), so the operating point lies
    just below runaway. The rewired controls are compared at matched activity.
- **No anatomy.** The data has no positions; the brain view shows a schematic, one column per
  cell class.

## Result

TinyShakespeare, 20k training / 5k validation characters, validation BPC, gain 1.7.

| Model | Validation BPC | Readout neurons active |
|---|---|---|
| Kneser-Ney 5-gram | 2.680 | |
| context table only (no brain) | 3.205 | |
| + larva connectome (3 input codes) | 3.415–3.432 | 3.5–4.4% |
| + fully rewired, same gain / activity matched (3 seeds) | 3.434–3.491 | 3.9–4.6% |
| + rewired within cell classes, same gain / activity matched (3 seeds) | 3.373–3.501 | 4.5–6.0% |

- **With all three feature types the readout does worse than the context table alone, and the
  voltage features cause it.** Refitting the same readout on the cached features of each code
  (`overfit_check.py`, exploratory diagnosis, no retuning):

  | Readout features | Validation BPC (codes 3 / 4 / 5) | Training BPC |
  |---|---|---|
  | context table only | 3.205 | 1.84 |
  | spike counts only (2,316) | 3.125 / 3.102 / 3.094 | 1.76–1.77 |
  | spike traces only | 3.126 / 3.117 / 3.100 | 1.75–1.76 |
  | mean voltages only | 3.348 / 3.356 / 3.317 | 2.06–2.12 |
  | all three (6,948) | 3.432 / 3.415 / 3.427 | 2.08–2.09 |
  | all three, rows shuffled (control) | 3.827 / 3.782 / 3.744 | 2.11–2.15 |

  - **This is not overfitting.** With the voltages the readout fits the training set *worse*
    than the context table alone. A stronger L2 penalty did not fix it.
  - **The likely cause is the voltage features.** There are 2,316 continuous, strongly
    correlated voltage features, and the fixed readout schedule (15 Adam epochs) cannot fit
    them. The voltages are not near-constant: every column has SD above 1e-3. This
    explanation is not yet tested directly.
  - **With spike features alone the larva improves on the context table by 0.08–0.11 BPC.**
    That is like the fly (about 0.08) and the worm (0.05–0.07).
- **Its state holds more history than the fly's or the worm's.** From all features, the
  character 2 back is decoded in 98% of cases, 3 back in 71% and 4 back in 42% (fly: 84% and
  50% at 2 and 3 back).
- **Most readout neurons are silent.** 1,568 of the 2,316 readout neurons never fire.
- **The real wiring is again not special.** Rewired copies at matched activity fall on both
  sides of the real connectome.

**History of the operating point, kept for honesty.**

1. A first selection checked only the first 500 characters and one input code, and chose gain
   1.8.
2. With code 3, gain 1.8 ran away at about character 4,500 and stayed there: 95% of the readout
   neurons active, 5.89 BPC. Codes 4 and 5 gave 3.46–3.48 BPC.
3. The check was then extended to the whole training text and all three codes:
   - no gain met the criterion's 5% lower bound without running away;
   - 1.7 is the largest gain that never runs away (mean 4.6–5.3%, at most 8.5%);
   - it is used although two codes sit just below 5%.

   The prediction results at 1.8 had been seen by then, but the choice of 1.7 is set by runaway
   alone.

The narrow window between a quiet network and runaway is probably due to the sign
assumption: only 110 LNs (under 4% of neurons) inhibit, far fewer than in the real larva.

## Sources

- Winding, M. et al. (2023). The connectome of an insect brain. *Science* 379, eadd9330.
- Data mirror: Betzel, R., Puxeddu, M. G. & Seguin, C. (2023), bioRxiv;
  [brain-networks/larval-drosophila-connectome](https://github.com/brain-networks/larval-drosophila-connectome).
- Related: a frozen-operator analysis of the same connectome (arXiv:2606.17745).
