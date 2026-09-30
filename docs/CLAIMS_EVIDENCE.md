# Claims and evidence

Every claim of the two papers, with the script that produced it, the output file (in `results/`),
the numbers, what kind of evidence it is, and how often it has been replicated.
"Mech" = *Three Addresses of a Connectome-Constrained Mushroom-Body Memory*;
"LM" = *Teaching a Fly Connectome to Predict Text*.

## Classes

Nothing here is called "solid". Each claim is one of:

- **E, in-model empirical.** Measured in the simulation. It could have come out otherwise:
  neither the model's rules nor the metric force it. It is still a statement about the model,
  not about the fly.
- **C, by construction.** Follows from the model's rules or from how the metric is defined. The
  simulation only confirms or quantifies it; its contribution is a number, not a mechanism.
- **X, exploratory.** Hypothesis-generating: a single run or very few units, settings chosen
  after looking at the data, or an in-sample fit. Not to be cited as established.

A claim can have parts in different classes (for example "C; the rarity of flips: E"). Earlier
claims no longer supported are listed as **withdrawn**.

## Replication and checks

- The **Replication** column gives the independent units behind a number. Write pairs within
  one input code are not independent of each other. Input-code seeds, text segments and rewiring
  seeds are the units that are.
- Intervals are 95% intervals:
  - Mech: a cluster bootstrap over input-code seeds, in `repro/REPLICATION.md`;
  - LM: t-intervals, cluster bootstraps or block bootstraps, in `repro/LM_STATS.md`.
- **Rerun** reports whether rerunning the committed command, in the pinned environment with the
  released flybrain 0.1.0.post1, reproduces the committed output (`repro/RERUN_REPORT.md`):
  - "match" means every number agrees within 1e-5 + 1e-3 relative; voltages differ at the last
    float32 bit because GPU sparse products are not bit-deterministic.
- **Confirmatory** results come from a pre-registered run with fresh seeds
  (`repro/PREREGISTRATION.md`, `repro/CONFIRMATORY.md`).

## Mechanism paper

| # | Claim | Section | Script | Output | Key numbers | Class | Replication | Rerun |
|---|---|---|---|---|---|---|---|---|
| M1 | Snapshot/restore reproduces natural continuation; injected errors are detected | Methods | `m1_validate.py` | `m1_validate_g0.json`, `m1_validate_g32.json` | restore max 2.4e-7 = replay floor; negatives ~1000x, ~3e6x, 4-5x floor | E (method check) | 2 gaps | RR:m1_validate_g0 RR:m1_validate_g32 |
| M2 | From ~1 s after writing, content is carried only by the KC->MBON weights | Results A | `m1_cross.py` | `m1_cross.json` (512 PN), `m1_cross_a192.json` (192 PN) | D_m/D_nat = 1.000, D_s <= 0.007 from 8 tokens | E; the time scale is the 100 ms membrane constant (C) | 2 pairs x 2 operating points; + 3 fresh code seeds (REPLICATION) | RR:m1_cross RR:m1_cross_a192 |
| M3 | Fast state relaxes with the membrane time constant; no reverberation | Results A | `m1_cross.py` | same | v difference 2.6 -> 2e-3 -> 7e-6 (float32 floor) | E | as M2 | as M2 |
| M4 | Retention equals the prescribed decay; with decay off the readout is constant for 123 s | Results B | `m4_curve.py` | `m4_curve.json`, `m4_curve_a192.json` | synaptic difference = exp(-t/tau) exactly; tau=1e12 -> 1.000 at all checkpoints | C (shows only that no other forgetting emerges) | 2 pairs | RR:m4_curve_a192 |
| M5 | Readout lifetime = lifetime of an extra MBON spike | Results B | `m4_curve.py` | `m4_curve_a192.json` | 6 vs 5 MBON spikes until 15.4 s; downstream 0.31 -> 0 at 30.7 s with 16 % trace left | X (one pair shown); the mechanism is C (spike-based transmission, M19) | 1 pair | RR:m4_curve_a192 |
| M6 | Legacy input (512 PN) gives dense KC codes | Results C | `kc_density.py`, `sweep_opoint.py` | `kc_density.json`, `sweep_s*.json` | probe KC 87 %, Jaccard 0.93 | E | 1 code | RR:sweep_s1.5 |
| M7 | Sparse regime exists at 160-192 PN | Results C | `sweep_opoint.py` | `sweep_s1.5.json` | probe KC 9-16 %, Jaccard 0.08-0.14 | E; the operating points were then chosen by us | 1 code; + sensitivity (7 settings) | RR:sweep_s1.5 |
| M8 | In the sparse regime KC identity is the retrieval key | Results D | `m2_current.py` | `m2_current_6pairs.json` | cue specificity 4.7 / 3.6 vs 1.06; KC-identity scramble keeps 11-66 % vs 82-88 % | E | 6 pairs (12-pair repeat); 5 fresh code seeds (REPLICATION); confirmatory (CONFIRMATORY) | RR:m2_current_6pairs |
| M9 | Random PN->KC wiring reproduces sparseness and cue specificity | Results E | `m3_pnkc.py` | `m3_pnkc.json`, `m3_pnkc_glom*.json` | nulls match real; pre-stated test for real>null failed (4/8, 6/8, 6/8, 7/8) | E (a negative result) | 8 pairs x 4 settings | - |
| M10 | Output identity needs coarse wiring only: body side and KC subtype | Results E | `m3_kcmbon.py` | `m6_side_a*.json`, `m3_kcmbon*.json` | real 0.756/0.764; same subtype+side swap 0.745-0.756; mixing sides 0.90-0.95; full shuffle 0.96-0.98 | E | 2-3 shuffle seeds, 2 codes types; + 5 fresh code seeds; confirmatory | RR:m3_kcmbon RR:m6_side_a160 RR:m6_side_a192 |
| M11 | Side-blind codes -> lateralised storage by threshold amplification | Results E | `m6_hemi.py` | `m6_hemi.json` | PN left-share SD 0.03 -> KC left-share SD 0.23-0.26; r(KC, MBON) = 0.96 | E | 26 items, 1 code set | RR:m6_hemi |
| M12 | Bilateral codes -> items differ by KC subtype mix | Results E | `m6_hemi.py`, `m3_kcmbon.py` | `m6_hemi_glom.json`, `m6_glom_route.json` | lateralisation SD 0.06-0.08; 0.88 vs 0.96-0.99; ab share 0.42 +- 0.15 | E | 1 glomerular code set, 1-3 shuffle seeds | RR:m6_glom_route |
| M13 | Static sum of written KCs' output profiles reproduces distinctness | Results E | `m6_coherence.py` | `m6_coherence.json` | 0.76 / 0.72 vs 0.91-0.93 / 0.86-0.87 | C, largely: the readout current is linear in the written KCs' output weights | 1 code set | - |
| M14 | Subtype mixture explains 66 % of item output variation | Results E | `m7_map.py` | `m7_map.json` | R^2 = 0.66 (26 items) | X (in-sample regression) | 26 items, 1 code set | RR:m7_map |
| M15 | With one teacher the DAN gate carries no item identity | Results E | `m6_gate.py` | `m6_gate.json` | gate cos 1.000; clamped 0.756/0.764 unchanged | C (one teacher = the same pulse for every item) | - | - |
| M16 | With different teachers the teacher decides the compartment | Results F | `m7_teacher.py` | `m7_teacher.json` | teacher 87 %, item 5 %, interaction 7 % of output variance | C, largely (the rule applies the DAN signal through DAN->MBON edges); how little teachers overlap is set by the connectome (E) | 6 items, 4 teachers; + 5 fresh glomerular seeds; confirmatory | RR:m7_teacher |
| M17 | Location results hold under a depression-only rule | Results F | `MB_SIGN=depress` reruns | `sign_*.json` | same values | C (largely by symmetry) | - | - |
| M18 | Interference = linear superposition set by KC overlap | Results G | `m8_interference.py` | `m8_interference.json` | predicted = measured (0.29/0.29, 0.55/0.55, 1.57/1.57); overlap 0.10-0.13 vs 0.86-0.90 | additivity C (the current metric is linear); the dependence on overlap E, expected from theory (Shen 2023) | 12 rows x 2 settings | RR:m8_interference |
| M19 | Downstream sees the memory only via MBON spike changes | Results H | `m5_downstream.py`, `m5b_timing.py` | `m5_downstream.json`, `m5b_timing.json` | 31/31 count, 4/4 timing, 0/93 identical | C (spike-based synapses); the rarity of flips E | 128 cells, 1 code; + 3 fresh code seeds; confirmatory | RR:m5_downstream RR:m5b_timing |
| M20 | Pre-fix simulator wrote plastic changes to wrong synapses | Corrections | `audit_20260928/csr_check.py`, `gpu_sort_check.py`; regression test `third_party/flybrain/tests/test_csr_identity.py` | same folders | 61,164/61,210 slots misplaced; 63 % change of MBON rows; test fails on 0.1.0, passes on 0.1.0.post1 | E (audit) | - | test re-run 2026-09-30 |
| M21 | Post-fix, plasticity lowers activity-based recall; weight state holds lags 4-12 | Corrections | `rerun_asset_a.py` | `asset_a_rerun.json` | 1.126-1.131 vs 1.162-1.186; lags 4-12: 0.11-0.16 vs 0.004-0.028 | X (3 seeds, one stream) | 3 seeds | - |
| W1 | "Output identity needs individual KC input-output matching" | (withdrawn) | - | - | all earlier nulls mixed hemispheres | withdrawn | | |
| W2 | "Neutral probe readout depends on KC identity" (voltage metric) | (withdrawn) | `m2_diag.py` | `m2_diag.json` | spike-timing artefact | withdrawn | | |

## Language-model paper

| # | Claim | Section | Script / source | Output | Key numbers | Class | Replication |
|---|---|---|---|---|---|---|---|
| L1 | Earlier frozen reference | History | `harness/lm/train_lm.py` | `lm/legacy/flybrain_memoryrank512_skip3h32k_softmax.metrics.json` | 42.55 % / 3.298 (summary doc said 42.71 / 3.227) | X (one historical run; a record, superseded as a benchmark by L4) | 1 run |
| L2 | Earlier plasticity runs did not improve prediction | History | `harness/lm/train_plastic_lm.py` | `lm/legacy/plastic_memory_lm_biological_dual_probe_20k_l2p1.metrics.json` | 42.88 % / 3.346 | superseded (pre-fix: plastic writes hit the wrong synapses) | 1 run |
| L3 | All plasticity LM results predate the simulator fix | Audit | file timestamps | - | the fixed `brain.py` is newer than every `plastic_*`, `pilot*`, `overnight_*` | E (audit) | - |
| L4 | A hashed trigram alone matches the earlier benchmark | Audit | `lm_mech.py --features none` | `lm/ctx_only_20k.json` | 42.9 % / 3.205 | E (one split; the split differs from the earlier one) | 1 split |
| L5 | With sparse input the brain improves the context table | Revision | `lm_select.py` | `lm/night/select_{s160,s192,d512}.json` | dBPC -0.078 [95% CI -0.093, -0.063] (s160, 9/9, sign test p = 0.004), -0.076 [-0.087, -0.065] (s192, 9/9); dense -0.007 [-0.021, +0.008] (5/9); accuracy unchanged | E; the exploratory table (Table I) is X (tuned on validation) | 3 segments x 3 code draws per input |
| L6 | A uniform teaching pulse hurts prediction | Revision | `lm_mech.py --pulse` | `lm/s192_biopulse_20k_*` | 40.2 % / 3.450 | X (one run) | 1 run |
| L7 | The brain's advantage shrinks with more data | Revision | `lm_controls.py` (A2) | `lm/night/lm_controls.json` | -0.32 at 5k ... -0.01 at 100k | X (one segment, one code draw) | 1 |
| L8 | A Kneser-Ney 5-gram beats every brain model | What does the brain provide? | `lm_explain.py`; `uctf bench` | `lm_limits/lm_explain_real.json`; `lm/stats/` | 2.47-2.68 vs >= 3.10; KN-5 minus connectome -0.442 [95% CI -0.469, -0.416] (segment 0, block bootstrap) | E | 3 segments |
| L9 | Adding the brain to a 5-gram makes it worse | same | `lm_explain.py` | `lm_limits/lm_explain_real.json` | KC +0.03 to +0.06, all +0.10 to +0.15 | E | 3 segments |
| L10 | The KC code holds about four characters | same | `lm_explain.py`; `uctf bench` | `lm_limits/lm_explain_real.json`; `lm/stats/` | k = 2: 57-60 % (KC), 83.9 % [82.7, 85.0] (all features); k = 3: 35-38 % / 49.7 % [47.5, 51.7] | E | 3 segments |
| L11 | Degree-preserving rewired connectomes do as well as the real one | same; UCTF check | `lm_explain_conn.py`; `uctf bench --match-activity` | `lm_limits/lm_explain_conn.json`; `repro/` | within 0.03 BPC at the same gain; at matched activity 3.126 / 3.120 vs 3.122 (the full-rewiring match is imperfect: 1.4 % vs 0.2 % active over the run) | E | 3 segments x 1 rewiring each; 1 rewiring seed at matched activity |
| L12 | A GRU reaches 2.34 BPC at 100k | same | `lm_gru.py` | `lm_limits/lm_gru.json` | 2.64-2.98 at 20k, 2.34 at 100k | E (baseline, lightly tuned) | 3 segments at 20k, 1 at 100k |
| L13 | A content-dependent teacher lets the MB learn online (synthetic task) | Write gate | `mb_online.py` | `lm_limits/mb_online_v2_s*.json` | balanced accuracy 26 % (15-36 %) vs 7.8 % random teacher, chance 6.7 %; pre-stated criterion 1 met after a decoder correction | E (with a post-hoc correction, disclosed) | 5 seeds |
| L14 | On real text the online memory adds nothing to a 5-gram | Write gate | `lm_online_eval.py` | `lm_limits/lm_online_eval.json` | +0.003 to +0.018 (content) vs +0.005 to +0.019 (random); pre-stated criterion 2 failed | E (a negative result) | 3 segments |
| L15 | Retraining the bundled GPF-1 with `gpf train` reproduces it | Playground | `gpf train brain` | `repro/` notes | features equal to float32 rounding; clipped readout identical (3.122) | E (method check) | 1 |
| L16 | The worm connectome shows the fly's pattern | UCTF | `uctf bench` | `uctf/examples/celegans/README.md` | 3.135-3.160 vs context 3.205, KN-5 2.680; k = 2: 67.4 % [66.1, 68.5] | E; the operating point was chosen by a pre-stated activity criterion, but by hand; whether UCTF generalises further is X | 3 codes; rewiring 3 seeds x 2 kinds |
| L17 | At a fixed gain, rewiring changes activity; the fully rewired worm's advantage came from being quieter | UCTF | `uctf bench --match-activity` | same | 2-4 % vs 6 % active; matched: 0.014 / 0.033 BPC worse than real | E | 3 rewiring seeds |
