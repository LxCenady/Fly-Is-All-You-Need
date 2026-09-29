# C. elegans (Cook et al. 2019) in GPF

The whole hermaphrodite worm: 300 neurons, 133 muscles and a few other cells, with 4,681
chemical connections and 1,345 gap junctions (electrical synapses). This is a different
species with a different kind of nervous system, used here to test that the connectome
framework is general.

```
python prepare_cook2019.py              # downloads ~320 KB from OpenWorm, writes neurons.csv, edges.csv
python -m gpf import --neurons neurons.csv --edges edges.csv --out cook2019 \
    --sign-col nt --inhibitory gaba --type-col type
python -m gpf bench --substrate celegans-cook2019.json --data ../../gpf/data/tinyshakespeare.txt \
    --vocab ../../gpf/data/vocab.json --train-chars 20001 --val-chars 5000 --device cpu
python -m gpf train brain --substrate celegans-cook2019.json --data ../../gpf/data/tinyshakespeare.txt \
    --vocab ../../gpf/data/vocab.json --name worm-cook2019 --device cpu
```

Everything runs on a CPU in about two minutes.

## Modelling choices

- **Signs.** The 26 GABAergic neurons of McIntire et al. (1993) are inhibitory; every other
  cell is excitatory. This is a simplification: the worm also has inhibitory glutamate and
  acetylcholine receptors, and a full neurotransmitter atlas exists (Wang et al. 2024, eLife).
- **Gap junctions** couple membrane voltages: `I_i = 0.5 · Σ_j G_ij (v_j − v_i)`. Each row of
  G is divided by its total, so no neuron is coupled with total weight above 1.
- **Input and readout.** Each character drives its own random 20 of the 83 sensory neurons.
  The readout takes spike counts, mean voltage and a spike trace of all 300 neurons.
- **Operating point.** This was chosen by hand, with the criterion fixed before looking at
  any text results: 5–30% of neurons active per character on average, and never above 60%.
  With the fly's gain of 1.5 the worm network runs away: about 90% of neurons fire on every
  character. Between gains 1.0 and 1.1 it switches abruptly from moderate to runaway activity.
  At 0.8 and 0.9 it meets the criterion; 0.9 is used (16% active on average, at most 25%).
- **No anatomy.** The brain view shows a schematic, one column per cell class.

## Result (TinyShakespeare, 20k training / 5k validation, one seed)

| Model | Validation BPC |
|---|---|
| Kneser-Ney 5-gram | 2.680 |
| context table only (no brain) | 3.205 |
| context table + worm connectome | 3.147 |
| context table + worm, fully rewired | 3.106 |
| context table + worm, rewired within cell classes | 3.180 |

The pattern is the same as in the fly. The connectome improves a little on the readout's own
context table, stays far behind a smoothed 5-gram, and the real wiring is not special:
degree-preserving rewired copies do as well. The worm's activity retains about two to three
characters (the character two back is decoded in 80% of cases, three back in 37%); the fly's
retains about four.

These are single runs. The readout fit alone varies by about 0.03 BPC, so the differences
between the three worm variants are within noise.

## Sources

- Cook, S. J. et al. (2019). Whole-animal connectomes of both *Caenorhabditis elegans* sexes.
  *Nature* 571, 63–71.
- Data files from OpenWorm's [ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox)
  (MIT): `herm_full_edgelist.csv`, `all_cell_info.csv`.
- McIntire, S. L. et al. (1993). The GABAergic nervous system of *Caenorhabditis elegans*.
  *Nature* 364, 337–341.
