# Claims and evidence

Every claim of the two papers, with the script that produced it, the output
file (in `results/`, or `results/lm/` for the language model), and the numbers.
"Mech" = *Three Addresses of a Connectome-Constrained Mushroom-Body Memory*;
"LM" = *Teaching a Fly Connectome to Predict Text*.
Status: **solid** = direct measurement with controls; **limited** = holds with
the stated caveat; **withdrawn** = earlier claim no longer supported.

## Mechanism paper

| # | Claim | Section | Script | Output | Key numbers | Status |
|---|---|---|---|---|---|---|
| M1 | Snapshot/restore reproduces natural continuation; injected errors are detected | Methods | `m1_validate.py` | `m1_validate_g0.json`, `m1_validate_g32.json` | restore max 2.4e-7 = replay floor; negatives ~1000x, ~3e6x, 4-5x floor | solid |
| M2 | From ~1 s after writing, content is carried only by the KC->MBON weights | Results A | `m1_cross.py` | `m1_cross.json` (512 PN), `m1_cross_a192.json` (192 PN) | D_m/D_nat = 1.000, D_s <= 0.007 from 8 tokens | solid |
| M3 | Fast state relaxes with the membrane time constant; no reverberation | Results A | `m1_cross.py` | same | v difference 2.6 -> 2e-3 -> 7e-6 (float32 floor) | solid |
| M4 | Retention equals the prescribed decay; with decay off the readout is constant for 123 s | Results B | `m4_curve.py` | `m4_curve.json`, `m4_curve_a192.json` | synaptic difference = exp(-t/tau) exactly; tau=1e12 -> 1.000 at all checkpoints | solid |
| M5 | Readout lifetime = lifetime of an extra MBON spike | Results B | `m4_curve.py` | `m4_curve_a192.json` | 6 vs 5 MBON spikes until 15.4 s; downstream 0.31 -> 0 at 30.7 s with 16 % trace left | solid (one pair shown) |
| M6 | Legacy input (512 PN) gives dense KC codes | Results C | `kc_density.py`, `sweep_opoint.py` | `kc_density.json`, `sweep_s*.json` | probe KC 87 %, Jaccard 0.93 | solid |
| M7 | Sparse regime exists at 160-192 PN | Results C | `sweep_opoint.py` | `sweep_s1.5.json` | probe KC 9-16 %, Jaccard 0.08-0.14 | solid |
| M8 | In the sparse regime KC identity is the retrieval key | Results D | `m2_current.py` | `m2_current_6pairs.json` | cue specificity 4.7 / 3.6 vs 1.06; KC-identity scramble keeps 11-66 % vs 82-88 % | solid (6 pairs; 12-pair rerun overnight) |
| M9 | Random PN->KC wiring reproduces sparseness and cue specificity | Results E | `m3_pnkc.py` | `m3_pnkc.json`, `m3_pnkc_glom*.json` | nulls match real; pre-stated test for real>null failed (4/8, 6/8, 6/8, 7/8) | solid |
| M10 | Output identity needs coarse wiring only: body side and KC subtype | Results E | `m3_kcmbon.py` | `m6_side_a*.json`, `m3_kcmbon*.json` | real 0.756/0.764; same subtype+side swap 0.745-0.756; mixing sides 0.90-0.95; full shuffle 0.96-0.98 | solid (2 draws; more overnight) |
| M11 | Side-blind codes -> lateralised storage by threshold amplification | Results E | `m6_hemi.py` | `m6_hemi.json` | PN left-share SD 0.03 -> KC left-share SD 0.23-0.26; r(KC, MBON) = 0.96 | solid |
| M12 | Bilateral codes -> items differ by KC subtype mix | Results E | `m6_hemi.py`, `m3_kcmbon.py` | `m6_hemi_glom.json`, `m6_glom_route.json` | lateralisation SD 0.06-0.08; 0.88 vs 0.96-0.99; ab share 0.42 +- 0.15 | solid |
| M13 | Static sum of written KCs' output profiles reproduces distinctness | Results E | `m6_coherence.py` | `m6_coherence.json` | 0.76 / 0.72 vs 0.91-0.93 / 0.86-0.87 | solid |
| M14 | Subtype mixture explains 66 % of item output variation | Results E | `m7_map.py` | `m7_map.json` | R^2 = 0.66 (26 items) | solid |
| M15 | With one teacher the DAN gate carries no item identity | Results E | `m6_gate.py` | `m6_gate.json` | gate cos 1.000; clamped 0.756/0.764 unchanged | solid |
| M16 | With different teachers the teacher decides the compartment | Results F | `m7_teacher.py` | `m7_teacher.json` | teacher 87 %, item 5 %, interaction 7 % of output variance | solid (6 items, 4 teachers) |
| M17 | Location results hold under a depression-only rule | Results F | `MB_SIGN=depress` reruns | `sign_*.json` | same values; largely by symmetry | limited |
| M18 | Interference = linear superposition set by KC overlap | Results G | `m8_interference.py` | `m8_interference.json` | predicted = measured (0.29/0.29, 0.55/0.55, 1.57/1.57); overlap 0.10-0.13 vs 0.86-0.90 | solid; known in principle (Shen 2023) |
| M19 | Downstream sees the memory only via MBON spike changes | Results H | `m5_downstream.py`, `m5b_timing.py` | `m5_downstream.json`, `m5b_timing.json` | 31/31 count, 4/4 timing, 0/93 identical | solid (point neurons) |
| M20 | Pre-fix simulator wrote plastic changes to wrong synapses | Corrections | `audit_20260928/csr_check.py`, `gpu_sort_check.py` | same folder | 61,164/61,210 slots misplaced; 63 % change of MBON rows | solid |
| M21 | Post-fix, plasticity lowers activity-based recall; weight state holds lags 4-12 | Corrections | `rerun_asset_a.py` | `asset_a_rerun.json` | 1.126-1.131 vs 1.162-1.186; lags 4-12: 0.11-0.16 vs 0.004-0.028 | limited (3 seeds, one stream) |
| W1 | "Output identity needs individual KC input-output matching" | (withdrawn) | - | - | all earlier nulls mixed hemispheres | withdrawn |
| W2 | "Neutral probe readout depends on KC identity" (voltage metric) | (withdrawn) | `m2_diag.py` | `m2_diag.json` | spike-timing artefact | withdrawn |

## Language-model paper

| # | Claim | Section | Script / source | Output | Key numbers | Status |
|---|---|---|---|---|---|---|
| L1 | Earlier frozen reference | History | `flybrain_lm/lm/train_lm.py` | `flybrain_memoryrank512_skip3h32k_softmax.metrics.json` | 42.55 % / 3.298 (summary doc says 42.71 / 3.227) | solid (metrics file) |
| L2 | Earlier plasticity runs did not improve prediction | History | `train_plastic_memory_lm.py` | `plastic_memory_lm_biological_dual_probe_20k_l2p1.metrics.json` | 42.88 % / 3.346 | limited (pre-fix) |
| L3 | All plasticity LM results predate the simulator fix | Audit | file timestamps | - | `brain.py` newer than every `plastic_*`, `pilot*`, `overnight_*` | solid |
| L4 | A hashed trigram alone matches the earlier benchmark | Audit | `lm_mech.py --features none` | `lm/ctx_only_20k.json` | 42.9 % / 3.205 | limited (split differs) |
| L5 | Sparse input makes the brain informative | Revision | `lm_mech.py` | `lm/s192_6k_*`, `lm/s160_frozen_20k_*` | 6k: brain only 43.0/3.178 vs ctx 42.8/3.378; 20k: KC+ctx 43.2/3.146, all+ctx 42.4/3.110 | limited (one seed; settings chosen on validation) -> overnight multi-seed rerun with holdout selection |
| L6 | Uniform teaching pulse hurts prediction | Revision | `lm_mech.py --pulse` | `lm/s192_biopulse_20k_*` | 40.2 % / 3.450 | limited (one run) |
