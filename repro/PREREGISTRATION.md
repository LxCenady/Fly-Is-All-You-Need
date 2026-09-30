# Pre-registration: confirmatory run with fresh seeds and unseen text

Registered on 2026-09-30, **before** any of the runs below and before the exploratory
replications (seeds 101-105) were run. The only data seen when setting the thresholds were the
results already committed in `results/` (reference code seed 3, glomerular seed 11, text
segments at 0 / 300,000 / 600,000).

## Frozen evaluation

The evaluation is `repro/confirm_eval.py`, which uses `repro/replication_stats.py`, as of the
commit that adds this file. After the runs, `git diff <that commit> -- repro/confirm_eval.py
repro/replication_stats.py` must be empty. Every hypothesis is reported as CONFIRMED or NOT
CONFIRMED with its numbers. Nothing is re-tested with other seeds or thresholds after the
results are seen.

## Fresh data

- **Mechanism:** random PN codes with seeds 201, 202, 203 (`MB_CODE_SEED`) and glomerular codes
  with seeds 21, 22, 23 (`MB_GLOM_SEED`). Neither was used before.
- **Language model:** the text segment starting at character 900,000 of TinyShakespeare (20,001
  training / 5,000 validation characters), never used before. PN-code seeds 201, 202, 203.

## Hypotheses (units: input-code seeds; statistics as in the paper figures)

| | Claim | Test (all conditions must hold) |
|---|---|---|
| H1 | M8: with sparse input, KC identity is the retrieval key; with dense input it is not | pooled median cue specificity > 2 at 160 and at 192 PNs, and < 1.5 at 512 PNs; pooled median fraction of the cued readout kept after shuffling KC identity within each MBON < 0.6 at 160 and at 192 PNs |
| H2 | M10: output identity needs body side and KC subtype, not individual wiring | mean change of item cosine after swapping output profiles within subtype and side: \|change\| < 0.03 at 160 and at 192 PNs; after shuffling KC->MBON wiring: > +0.10 at both |
| H3 | M16: when teachers differ, the teacher decides the compartment | mean share of output variance explained by the teacher > 0.60 |
| H4 | M19: downstream neurons see the memory only through MBON spikes | central response changed (> 1e-4) in >= 90 % of cells where the write changed the MBON spike count, and in <= 10 % of cells where it did not |
| H5 | M2: from about 1 s after writing, the memory is carried by the weights | median D_m / D_nat at the MBON level for gaps of 8 and 32 tokens >= 0.99, minimum >= 0.95 |
| H6 | LM: the fly connectome helps a context table a little, is far behind a 5-gram and remembers a few characters | in each of the 3 codes: connectome + context below context alone; KN-5 more than 0.3 BPC below the connectome; memory span (all features) k = 2 in [0.60, 0.95], k = 4 < 0.45 |
| H7 | LM: the specific fly wiring does not matter | in each code, the connectome rewired within cell classes (activity matched) within 0.05 BPC of the real one |
| H8 | UCTF: the worm connectome shows the same pattern | in each of the 3 codes: connectome below context; KN-5 more than 0.3 below the connectome; class-rewired (activity matched) within 0.06; k = 2 in [0.50, 0.90] |

## Commands

```
python repro/run.py repro/confirm.json CONFIRM                                   # H1-H5
for c in 201 202 203:                                                          # H6-H7
  python -m uctf bench --substrate malecns-v1 --data playground/gpf/data/tinyshakespeare.txt \
      --vocab playground/gpf/data/vocab.json --offset 900000 --train-chars 20001 --val-chars 5000 \
      --controls rewire-class --control-seeds 0 --match-activity --spec-set input.code_seed=$c \
      --out CONFIRM/lm_fly_c$c
for c in 201 202 203:                                                          # H8
  python -m uctf bench --substrate uctf/examples/celegans/celegans-cook2019.json ... (same text and
      vocabulary) --controls rewire-class,rewire-full --control-seeds 0 --match-activity \
      --spec-set input.code_seed=$c --device cpu --out CONFIRM/lm_worm_c$c
python repro/confirm_eval.py CONFIRM > repro/CONFIRMATORY.md
```

## Known limits of this design

- **Same connectome and simulator.** Everything runs on one connectome with the same simulator
  and the same authors' code. "Independent" means fresh inputs and unseen text, not an
  independent lab or implementation. A second implementation exists only for the MBON
  current (M9 in the mechanism paper).
- **Threshold choice.** The thresholds were chosen from the reference results with margins, so a
  pass means "the effect holds on new inputs at roughly the reported size", not a precise
  replication of each number.
