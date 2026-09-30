# C. elegans (Cook et al. 2019) in UCTF

The whole hermaphrodite worm: 300 neurons, 133 muscles and a few other cells, with 4,681
chemical connections and 1,345 gap junctions (electrical synapses). This is a different species
with a different kind of nervous system, used to test whether UCTF generalises beyond the fly.

Run from `uctf/` (or `pip install -e uctf`). Everything runs on a CPU in minutes.

```
cd examples/celegans
python prepare_cook2019.py        # downloads ~320 KB from OpenWorm, writes neurons.csv, edges.csv
python -m uctf import --neurons neurons.csv --edges edges.csv --out cook2019 \
    --sign-col nt --inhibitory gaba --type-col type
python -m uctf bench --substrate celegans-cook2019.json --data <text> --vocab <vocab.json> \
    --train-chars 20001 --val-chars 5000 --device cpu --control-seeds 0,1,2 --match-activity
python -m gpf train brain --substrate celegans-cook2019.json --data <text> --vocab <vocab.json> \
    --name worm-cook2019 --device cpu
```

## Modelling choices

- **Signs.** The 26 GABAergic neurons of McIntire et al. (1993) are inhibitory; every other
  cell is excitatory. This is a simplification: the worm also has inhibitory glutamate and
  acetylcholine receptors, and a full neurotransmitter atlas exists (Wang et al. 2024, eLife).
- **Gap junctions** couple membrane voltages: `I_i = 0.5 · Σ_j Ĝ_ij (v_j − v_i)`, where each row
  of Ĝ is scaled to total at most 1.
- **Input and readout.** Each character drives its own random 20 of the 83 sensory neurons.
  The readout takes spike counts, mean voltage and a spike trace of all non-sensory neurons
  (interneurons, motor and pharyngeal neurons). The driven sensory neurons are left out, as the
  fly's readout leaves out its input neurons: their own membrane leak would otherwise be
  counted as memory.
- **Operating point.** This was chosen by hand, with the criterion fixed before looking at any
  prediction results: 5–30% of all neurons active per character on average, and never above
  60%.
  - With the fly's gain of 1.5, the worm network runs away: about 90% of neurons fire on
    every character.
  - Between gains 1.0 and 1.1 it switches abruptly from moderate to runaway activity.
  - Gains 0.8 and 0.9 meet the criterion; 0.9 is used (16% of all neurons active on average,
    at most 25%; 6% of the readout neurons).
- **No anatomy.** The brain view shows a schematic, one column per cell class.

## Result

TinyShakespeare, 20k training / 5k validation characters, validation BPC.

| Model | Validation BPC | Readout neurons active |
|---|---|---|
| Kneser-Ney 5-gram | 2.680 | |
| context table only (no brain) | 3.205 | |
| + worm connectome (3 input codes) | 3.135–3.160 | 6% |
| + fully rewired, same gain (3 seeds) | 3.121–3.126 | 2–4% |
| + fully rewired, activity matched (3 seeds) | 3.151–3.171 | 6% |
| + rewired within cell classes, activity matched (3 seeds) | 3.169–3.186 | 6–7% |

The pattern is the same as in the fly:

- The connectome improves a little on the readout's own context table (0.05–0.07 BPC).
- It stays far behind a smoothed 5-gram.
- It remembers about two characters: the character two back is decoded in 66–67% of cases and
  three back in 33%, against 15% for always guessing the most frequent character.
- Rewired copies at matched activity do about as well as the real wiring. They are 0.01–0.04
  BPC worse, which is small but consistent in direction.

**A correction from the review.** At the same gain, the fully rewired networks look *better*
than the real one. That came only from their being much quieter: 2–4% of readout neurons
active, against 6%. Matching activity removes it. An earlier version of this example also read
out the driven sensory neurons; that version is superseded.

## Sources

- Cook, S. J. et al. (2019). Whole-animal connectomes of both *Caenorhabditis elegans* sexes.
  *Nature* 571, 63–71.
- Data files from OpenWorm's [ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox)
  (MIT): `herm_full_edgelist.csv`, `all_cell_info.csv`.
- McIntire, S. L. et al. (1993). The GABAergic nervous system of *Caenorhabditis elegans*.
  *Nature* 364, 337–341.
