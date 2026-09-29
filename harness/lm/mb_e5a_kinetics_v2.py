"""E5a kinetic/dose evaluability runner — v2 reproducibility/statistics repair
(additive, 2026-09-19).

Third-stage, additive artifact implementing the E5a design under
``docs/OPENCODE_E5A_BUILD_HANDOFF_20260918.md`` and its frozen preregistration
``docs/OPENCODE_E5A_SEQUENTIAL_PREREG_20260918.md``, with the v2 repair
preregistration ``docs/OPENCODE_E5A_V2_PREREG_20260919.md``.

v2 changes (this file is a new artifact; the 2026-09-18 runner is untouched):

  1. **MBON-current canonicalization** (G2/G3 reproducibility).  The frozen
     probe computes the KC->MBON readout current with ``cupyx.scatter_add``,
     whose CUDA ``atomicAdd`` reduction order is unspecified and therefore
     non-deterministic (observed 31-40 float32 ULP drift on
     ``_probe_modulated_mbon_current``, ~1.0-1.2e-6 max-abs).  In every pair
     compared by the G2 replay and G3 write-off gates the local modulation is
     identically zero, so the exact reproducibility quantity is the baseline
     ``w0 * probe_kc_counts/k`` current.  This runner replaces that channel with
     a deterministic sorted CPU reduction (``_canonical_mbon_current``) for
     every row whose ``modulation_stats_after_write.max_abs == 0``, and keeps
     the original scatter value as ``*_scatter`` for transparency.  The frozen
     tolerance contract (events bitwise; graded ``max_abs<=1e-6 OR <=16 ULP``)
     is retained byte-for-byte and is *not* widened.
  2. **Statistical-unit contract.**  The deterministic (``noise_hz=0``) primary
     arms are a single fixed system: the ten seed labels are deterministic
     replicates, not independent draws.  ``E5A_STATISTICAL_UNITS`` records this
     explicitly; the noisy ``A2N`` challenge arm is the only cross-seed
     independent-draw arm.
  3. **New frozen objects** (``E5A_STATISTICAL_UNITS``,
     ``E5A_DIRECTIONAL_DIAGNOSTIC``, ``E5A_CANONICALIZATION``) are folded into
     the config coverage and therefore into ``config_sha256``.  ``arms``,
     ``readout``, ``entry_gate`` and ``schedule_table`` are unchanged, so their
     hashes are identical to the 2026-09-18 freeze.

The cell plan, arm table, schedule table, seed set, lesion scopes and episode
grid are unchanged from the frozen 2026-09-18 design (194 cells/seed, 1,940
cells, 5,820 episodes, 18 schedules, 80/160 lesion cells).

Scope (E5a): on the *same* fixed fly connectome and the *same* local KC->MBON
plasticity mechanism as E4B, determine which dynamic level is sufficient for a
persistent state and when the state becomes content-specific and
downstream-readable.  E5a is an evaluability/potency and kinetics stage only:

  * no-write/transient control (``eta=0``);
  * short-lived, reference and longer-lived local-plasticity weight constants
    (``weight_tau`` in {4, 16, 64}, with 16 the frozen E4B reference);
  * one predeclared 3x potency/dose increase (``eta`` 0.02 -> 0.06) to test
    whether the E4B central norm-floor failure was merely underpowered;
  * matched content probes (own, two foils, neutral/template) at fixed delays
    (32 and 152 tokens) beyond the neural trace constant (``trace_tau=0.1 s``);
  * write-off, KC->MBON probe-readout lesion, MBON->central probe-readout
    lesion and the exact ``kc_mbon_writezero_restore`` write-phase lesion for
    the primary kinetic/dose arm.

Boundaries: no connectome rewiring, no topology null, no external context, no
reservoir, no fitted/learned decoder, no pooled binomial test, no
permutation-max gate, no language/benchmark optimisation.  The E5b topology
stage stays deferred and unauthorized.

This file is **additive**: it imports the frozen E4B objects/helpers and the
frozen probe module without editing them.  A source edit is made only in new
E5a files.  ``--dry-run`` is CPU-only (no ``FlyBrain`` construction, no CUDA
allocation) and writes the dry-run manifest.  The CUDA path is implemented but
is **not** launched by the E5a build pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
try:
    import sitecustomize  # noqa: F401
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))

import mb_e4b_sequential as e4b  # noqa: E402
import mb_e4b_sequential_summarize as e4s  # noqa: E402
import mb_persistent_memory_probe as mpp  # noqa: E402
from mb_persistent_memory_probe import (  # noqa: E402
    _anatomical_target_direction,
    _episode,
    _masks,
    _new_condition,
    _raw_target_direction,
    _select_targets,
    _target_summary,
)
from mb_sp1_stage2_diagnostic import (  # noqa: E402
    _jsonable,
    _pair_metrics,
    _replay_stats,
)
from train_lm import load_corpus  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen E5a names / constants.
# ---------------------------------------------------------------------------
DEFAULT_OUT = 'models/research_results/e5a_kinetics_v2_20260919'
DEFAULT_PREFIX = 'mb_e5a_kinetics_v2_20260919'
E3_MANIFEST = e4b.E3_MANIFEST
E3_EXPECTED_CONFIG_SHA256 = e4b.E3_EXPECTED_CONFIG_SHA256

ITEM_ALPHABET = list(e4b.ITEM_ALPHABET)
FOILS = list(e4b.FOILS)
CUE = e4b.CUE
ITEM_TOKENS = int(e4b.ITEM_TOKENS)
INTER_ITEM_GAP = int(e4b.INTER_ITEM_GAP)
SETTLE_LO, SETTLE_HI = int(e4b.SETTLE_LO), int(e4b.SETTLE_HI)
RESOLVER_V1 = dict(e4b.RESOLVER_V1)
SUCCESSOR = dict(e4b.SUCCESSOR)
PREDECESSORS = {k: list(v) for k, v in e4b.PREDECESSORS.items()}
FRESH_SEEDS = list(e4b.FRESH_SEEDS)
FORBIDDEN_UNION = list(e4b.FORBIDDEN_UNION)

WRITE_LESION = e4b.WRITE_LESION               # 'kc_mbon_writezero_restore'
READOUT_LESIONS = ('kc_mbon_readout_zero', 'mbon_central_readout_zero')

PROBE_COUNT_PROBES = 4                          # z + a + b + c + d (context owns one)
CHALLENGE_ARM = 'A2N_challenge_noise005_tau16'
HIGH_DOSE_ARM = 'A4_highdose_eta006_tau16'
REFERENCE_ARM = 'A2_reference_tau16'
TRANSIENT_ARM = 'A0_transient_eta0'

# E5a factor table.  Every kinetic/dose level is frozen before data.
KINETICS = {'transient': 16.0, 'short': 4.0, 'reference': 16.0, 'long': 64.0}
BASELINE_ETA = 0.02
HIGH_DOSE_ETA = 0.06
TRANSIENT_ETA = 0.0
REFERENCE_WEIGHT_TAU = 16.0

DELAYS_PRIMARY = (32, 72, 152)
FULL_DELAYS = (32, 152)
DELAYS_CONFIRM = (32,)
CONTEXT_PRIMARY = 'a'
CONTEXT_CONFIRM = 'c'
CHALLENGE_NOISE_HZ = 0.05
PRIMARY_NOISE_HZ = 0.0

# Generic invariant protocol (E4B reference values; nothing retuned).
REFERENCE_PROTOCOL = {
    'active': 512, 'drive': 1.0, 'gain': 1.5, 'tonic': 0.05, 'window': 4,
    'gamma': 0.5, 'k': 6, 'sustain': 0.5, 'trace_tau': 0.1,
    'eligibility_tau': 0.5, 'eligibility_mix': 0.0, 'max_modulation': 0.9,
    'dan_pulse': 2.0, 'dan_pulse_unpaired': False, 'gap_plasticity': False,
    'probe_plasticity': False, 'probe_tokens': 1, 'probe_drive': 2.5,
    'probe_settle_steps': 6, 'item_tokens': ITEM_TOKENS,
    'inter_item_gap': INTER_ITEM_GAP, 'target_selection': 'direct',
    'target_top': 256,
}

# Predeclared arm table.  ``full`` arms carry the five causal conditions on the
# own and neutral-cue probes; ``lite`` arms carry read-condition + write_off
# only.  Context ``c`` is a confirmation-only context at the core delay.
ARMS = (
    {
        'arm': TRANSIENT_ARM, 'kinetics': 'transient',
        'weight_tau': KINETICS['transient'], 'eta': TRANSIENT_ETA,
        'noise_hz': PRIMARY_NOISE_HZ, 'read_condition': 'sham_eta0',
        'primary_contexts': ['a'], 'primary_delays': [32, 72, 152],
        'full_delays': [], 'confirm_contexts': [], 'confirm_delays': [],
        'full': False, 'challenge': False,
        'role': 'transient_no_plastic_write',
    },
    {
        'arm': 'A1_short_tau4', 'kinetics': 'short',
        'weight_tau': KINETICS['short'], 'eta': BASELINE_ETA,
        'noise_hz': PRIMARY_NOISE_HZ, 'read_condition': 'intact',
        'primary_contexts': ['a'], 'primary_delays': [32, 72, 152],
        'full_delays': [], 'confirm_contexts': [], 'confirm_delays': [],
        'full': False, 'challenge': False,
        'role': 'short_lived_local_plastic_state',
    },
    {
        'arm': REFERENCE_ARM, 'kinetics': 'reference',
        'weight_tau': KINETICS['reference'], 'eta': BASELINE_ETA,
        'noise_hz': PRIMARY_NOISE_HZ, 'read_condition': 'intact',
        'primary_contexts': ['a'], 'primary_delays': [32, 72, 152],
        'full_delays': [32, 152], 'confirm_contexts': ['c'],
        'confirm_delays': [32], 'full': True, 'challenge': False,
        'role': 'e4b_reference_persistent_state',
    },
    {
        'arm': 'A3_long_tau64', 'kinetics': 'long',
        'weight_tau': KINETICS['long'], 'eta': BASELINE_ETA,
        'noise_hz': PRIMARY_NOISE_HZ, 'read_condition': 'intact',
        'primary_contexts': ['a'], 'primary_delays': [32, 72, 152],
        'full_delays': [], 'confirm_contexts': [], 'confirm_delays': [],
        'full': False, 'challenge': False,
        'role': 'longer_lived_local_plastic_state',
    },
    {
        'arm': HIGH_DOSE_ARM, 'kinetics': 'reference',
        'weight_tau': KINETICS['reference'], 'eta': HIGH_DOSE_ETA,
        'noise_hz': PRIMARY_NOISE_HZ, 'read_condition': 'intact',
        'primary_contexts': ['a'], 'primary_delays': [32, 72, 152],
        'full_delays': [32, 152], 'confirm_contexts': ['c'],
        'confirm_delays': [32], 'full': True, 'challenge': False,
        'role': 'predeclared_3x_potency_dose_increase',
    },
    {
        'arm': CHALLENGE_ARM, 'kinetics': 'reference',
        'weight_tau': KINETICS['reference'], 'eta': BASELINE_ETA,
        'noise_hz': CHALLENGE_NOISE_HZ, 'read_condition': 'intact',
        'primary_contexts': ['a'], 'primary_delays': [32],
        'full_delays': [], 'confirm_contexts': [], 'confirm_delays': [],
        'full': False, 'challenge': True,
        'role': 'separately_labelled_nonzero_noise_challenge',
    },
)
ARM_BY_NAME = {a['arm']: a for a in ARMS}

# E5a downstream readout object.  Deliberately contains no binomial /
# permutation / max-gate null; topology is not evaluated.
E5A_READOUT = {
    'name': 'e5a_delayed_downstream_readout_v1',
    'cue': 'z',
    'alphabet': ['a', 'b', 'c', 'd'],
    'templates': ['a', 'b', 'c', 'd'],
    'settle_window': [SETTLE_LO, SETTLE_HI],
    'tie_tol': 1e-9,
    'successor_rule': ('argmax_{y in alphabet} cosine(pvec(z), pvec(y)) == '
                       'successor(context)'),
    'primary_level': 'v_pre_central',
    'secondary_level': 'v_pre_mbon',
    'norm_floor': 'max(1e-12, 10*F_sum_central(z), 3*SD_R_central(z))',
    'null': 'none; no binomial or permutation-max gate in E5a',
    'topology_null': 'not_evaluated_deferred_to_E5b',
    'causal_collapse': {
        'write_off': 'required',
        'kc_mbon_readout_zero': 'required',
        'mbon_central_readout_zero': 'required',
        'kc_mbon_writezero_restore': ('required when write-phase end-fingerprint '
                                      'differs from intact; else diagnostic'),
    },
    'gating': ('entry gate uses the central norm floor only; successor rank is '
               'descriptive'),
}

# Separate fail-closed biological entry gate (exposed beside ``arm_valid``).
E5A_ENTRY_GATE = {
    'name': 'e5a_evaluable_for_biology_entry_gate_v1',
    'primary_arm': HIGH_DOSE_ARM,
    'reference_arm': REFERENCE_ARM,
    'core_delay': 32,
    'content_margin_ok_min_seeds': 8,
    'central_norm_floor_ok_min_seeds': 8,
    'resolver': 'content_specific_recall_v1',
    'margin_kappa': 1.25,
    'storage_primary_level': 'v_pre_mbon',
    'readout_primary_level': 'v_pre_central',
    'rule': ('entry_gate(arm,32) iff content_margin_ok in >=8/10 seeds AND '
             'central_norm_floor_ok in >=8/10 seeds; evaluable_for_biology iff '
             'arm_valid AND entry_gate(high-dose,32)'),
    'stop_rule': ('if the high-dose intact arm fails the entry gate, E5a ends as '
                  'underpowered/non-evaluable and E5b is not authorized'),
}

# v2 statistical-unit contract (B).  The deterministic (noise_hz=0) primary
# arms construct the connectome once and run identical dynamics for every seed
# label; the ten seed labels are therefore deterministic replicates of a single
# system, not ten independent draws.  The noisy A2N challenge arm is the only
# arm whose seed labels carry independent cross-seed variation.  Seed labels are
# preserved verbatim for reproduction; the biological inference unit is declared
# here and must not be silently re-interpreted as independent evidence.
E5A_STATISTICAL_UNITS = {
    'name': 'e5a_statistical_unit_contract_v1',
    'seed_is_label': True,
    'seed_labels_preserved_for_reproduction': True,
    'contexts_stratified_never_pooled': True,
    'primary_arms_noise_hz': 0.0,
    'primary_arms_independent_draws': False,
    'primary_arms_deterministic_replicates': True,
    'primary_arms_inference_unit': (
        'single fixed connectome + deterministic local-plasticity dynamics; '
        'the 10 seed labels are deterministic replicates of one system, not '
        'independent samples'),
    'challenge_arm': CHALLENGE_ARM,
    'challenge_arm_noise_hz': 0.05,
    'challenge_arm_independent_draws': True,
    'challenge_arm_deterministic_replicates': False,
    'threshold_interpretation': (
        'a >=8/10 seed-count threshold on a deterministic primary arm is '
        'all-or-nothing (10 identical replicates), not 8 of 10 independent '
        'successes; counts are reported with this caveat attached'),
}

# v2 read-only directional content diagnostic (C).  Auxiliary only: it is never
# a gate, never replaces ``content_specific_recall_v1``, never changes the
# 0/10 content-margin primary, and never authorizes E5b.  It distinguishes a
# common-mode magnitude write (own/foil deltas collinear, cosine -> 1) from an
# item identity encoded in direction (own delta directionally separated from
# foils at comparable magnitude).
E5A_DIRECTIONAL_DIAGNOSTIC = {
    'name': 'e5a_directional_content_diagnostic_v1',
    'role': 'read_only_auxiliary_interpretation',
    'gating': False,
    'replaces_primary': False,
    'authorizes_e5b': False,
    'levels': ['v_pre_mbon', 'v_pre_central'],
    'delta': 'u_X = settle-window mean of d_X(t) = v_pre(write,t)-v_pre(no_write,t)',
    'metrics': {
        'cos_own_vs_foil_mean': ('mean of cosine(u_own,u_i), cosine(u_own,u_j); '
                                 '~1 => common-mode/non-specific, <1 => '
                                 'directional separation'),
        'cos_foil_vs_foil': 'cosine(u_i,u_j); alignment of the two never-written foils',
        'common_mode_ratio': ('||u_own+u_i+u_j||/3 divided by mean(||u_own||,'
                              '||u_i||,||u_j||); ~1 => a single common-mode '
                              'component dominates'),
    },
    'interpretation_rule': (
        'if cos_own_vs_foil_mean ~ 1 and common_mode_ratio ~ 1, the write is a '
        'non-item-selective (common-mode) magnitude modulation; if '
        'cos_own_vs_foil_mean < 1 at comparable magnitude, identity may be '
        'encoded in direction and the L2 magnitude resolver is the wrong '
        'instrument. Neither branch changes the frozen primary'),
}

# v2 canonicalization contract (A).  The only source of non-determinism that
# broke G2/G3 was the scatter-add MBON-current readout; the reproducibility
# quantity (modulation == 0) is reduced deterministically on CPU in sorted edge
# order, float64 accumulation, cast to float32.
E5A_CANONICALIZATION = {
    'name': 'e5a_mbon_current_canonicalization_v1',
    'target': '_probe_modulated_mbon_current',
    'target_w0': '_probe_modulated_mbon_current_w0',
    'applies_to': 'rows with modulation_stats_after_write.max_abs == 0',
    'mathematical_definition': (
        'current_mbon[m] = sum_{e: mbon_slot[e]=m, edge order ascending} '
        'w0[e] * probe_kc_counts[kc_slot[e]] / k'),
    'accumulation': 'float64 in ascending edge order (deterministic), cast to float32',
    'scatter_value_retained_as': '_probe_modulated_mbon_current_scatter',
    'tolerance_rule_unchanged': 'events bitwise; graded max_abs<=1e-6 OR <=16 ULP',
    'note': ('applied only where the local modulation is identically zero '
             '(write_off, no_write, replay); the intact write row keeps its '
             'scatter-add value as the biological write-signal diagnostic'),
}

# E4B frozen file hashes (must not move).  Recorded from the E4B biological
# analysis handoff artifacts.
E4B_FROZEN_FILES = {
    'lm/mb_e4b_sequential.py':
        '8b1fbdb5183b0ebd053409ebc52c5dcf4807e48f564f300dee6228408f7d6615',
    'lm/mb_e4b_sequential_summarize.py':
        '088af6231d12c08a22ab89526bb24d6b048f83f5a3e13109dc17ae473b6b4ad3',
    'lm/mb_persistent_memory_probe.py':
        '6fb0819711bcc0cf9efa13eb850092708099940b437cf2e976efabd90ee7930f',
    'lm/mb_plasticity.py':
        '1a694becb18d1ff8f7f8b5e2438dd36c832ec2c7eb27f0661a0c2b2baebeeec9',
    'lm/mb_sp1_stage2_diagnostic.py':
        '7fc324ec324601e9043b4e1b8fc3a2c431d70058293d52b214f6e927c65a176b',
}

# E4B frozen canonical-object hashes (must not move).
E4B_FROZEN_OBJECTS = {
    'resolver_sha256': e4s.RESOLVER_SHA256,
    'item_inventory_sha256': e4s.ITEM_INVENTORY_SHA256,
    'prediction_sha256': e4s.PREDICTION_SHA256,
    'forbidden_union_sha256': e4s.FORBIDDEN_UNION_SHA256,
    'schedule_table_sha256': e4s.SCHEDULE_TABLE_SHA256,
    'config_sha256': e4s.CONFIG_SHA256,
    'e3_bridge_config_sha256': e4s.E3_EXPECTED_CONFIG_SHA256,
}

# E5a frozen canonical-object hashes.  Filled after the first CPU dry-run and
# then verified fail-closed.  ``None`` means "not yet frozen".
E5A_EXPECTED = {
    'readout_sha256':
        'a5a3ce4671b75f46758e18e4de1070967071d42e245143de339ae6a4a430d601',
    'entry_gate_sha256':
        '5423995a83edae9f1b6c9b629d31e0bf3644c3f576dbc4fe2b409993ad72e32f',
    'arms_sha256':
        '6a5db30a4065be42d37d283f07ef68c3dc4fbaaa2a77ab25561b5bfc329961f2',
    'schedule_table_sha256':
        'd42426ce28110249623c3c488892960d7c4991337a78b98e6a010c372fa87cdf',
    'config_sha256':
        '328cfbc0b1e1c26f0fcfbb8a21328409414ac44873bf758dd12ceba2c6ddca6f',
}

EXPECTED_CELLS_PER_SEED = 194
EXPECTED_CELLS_TOTAL = 1940
EXPECTED_EPISODES_PER_RUN = 3
EXPECTED_EPISODES_TOTAL = 5820
EXPECTED_WRITE_LESION_CELLS_PER_SEED = 8
EXPECTED_PROBE_LESION_CELLS_PER_SEED = 16

# Invariant config keys (E4B CONFIG_HASH_KEYS minus the per-arm eta/weight/noise).
E5A_CONFIG_KEYS = tuple(k for k in e4b.CONFIG_HASH_KEYS
                        if k not in ('eta', 'weight_tau', 'noise_hz'))


# ---------------------------------------------------------------------------
# Hashing helpers (thin aliases of the frozen E4B functions).
# ---------------------------------------------------------------------------
def _hash_canonical(obj) -> str:
    return e4b._hash_canonical(obj)


def _hash_array_f32(arr) -> str:
    return e4b._hash_array_f32(arr)


def _hash_scalar_f64(value) -> str:
    return e4b._hash_scalar_f64(value)


def _sha256_file(path: Path) -> str:
    return e4b._sha256_file(Path(path))


def _array_stats(arr) -> dict:
    return e4b._array_stats(arr)


def _device_info() -> dict:
    return e4b._device_info()


def _canonical_mbon_current(w0_host, probe_kc_counts, edge_kc_slot,
                            edge_mbon_slot, n_mbon, k):
    """Deterministic CPU replacement for the scatter-add MBON readout current.

    The frozen probe computes ``current_mbon[m] = scatter_add(W[edge_pos] *
    (kc_counts[kc_slot]/k), mbon_slot)`` with ``cupyx.scatter_add``, whose CUDA
    ``atomicAdd`` reduction order is unspecified and therefore
    non-deterministic.  This function accumulates the identical per-edge
    products in ascending edge order in float64 and casts to float32, giving a
    bitwise-reproducible value for the same input on CPU or GPU.

    ``w0_host`` is the per-edge baseline weight ``w0`` (``modulation`` is zero
    everywhere this is applied).  Edge order is the natural 0..n_edges-1 order,
    which is fully determined by the connectome (MBON slot ascending, CSR
    position ascending).
    """
    kc = np.asarray(probe_kc_counts, dtype=np.float32).reshape(-1)
    w0 = np.asarray(w0_host, dtype=np.float32).reshape(-1)
    kc_slot = np.asarray(edge_kc_slot, dtype=np.int64).reshape(-1)
    mbon_slot = np.asarray(edge_mbon_slot, dtype=np.int64).reshape(-1)
    if not (len(w0) == len(kc_slot) == len(mbon_slot)):
        raise ValueError('canonical current input length mismatch')
    kc_frac = kc[kc_slot] / np.float32(max(int(k), 1))
    edge_current = (w0 * kc_frac).astype(np.float64)
    out = np.zeros(int(n_mbon), dtype=np.float64)
    np.add.at(out, mbon_slot, edge_current)
    return out.astype(np.float32)


def _canonicalize_episode_rows(state, args, rows):
    """Overwrite the scatter-add MBON current with the deterministic canonical
    value for rows whose local modulation is identically zero.

    ``rows`` is an iterable of episode-row dicts.  A row whose
    ``modulation_stats_after_write.max_abs`` is > 0 carries a real local write
    and is left untouched (its scatter value is the biological write-signal
    diagnostic).
    """
    plastic = state.get('plastic')
    if plastic is None:
        return
    w0_host = np.asarray(plastic.xp.asnumpy(plastic.w0_gpu), dtype=np.float32)
    edge_kc_slot = np.asarray(plastic.edge_kc_slot, dtype=np.int64)
    edge_mbon_slot = np.asarray(plastic.edge_mbon_slot, dtype=np.int64)
    n_mbon = int(len(plastic.mbon_ids))
    k = int(args.k)

    def apply(row):
        if not row or 'probe_kc_counts' not in row:
            return
        mod_after = (row.get('modulation_stats_after_write') or {})
        if mod_after.get('max_abs', 0.0) > 0.0:
            # Nonzero local write: keep the scatter value as the biological
            # write-signal diagnostic.
            return
        scatter = row.get('_probe_modulated_mbon_current')
        row['_probe_modulated_mbon_current_scatter'] = scatter
        row['_probe_modulated_mbon_current'] = _canonical_mbon_current(
            w0_host, row['probe_kc_counts'], edge_kc_slot, edge_mbon_slot,
            n_mbon, k)
        row['_probe_modulated_mbon_current_w0'] = np.zeros(n_mbon, np.float32)

    for row in rows:
        apply(row)



# ---------------------------------------------------------------------------
# Schedules / cell plan.
# ---------------------------------------------------------------------------
def _conditions(arm: dict, probe_role: str, profile: str) -> list:
    read = arm['read_condition']
    if probe_role in ('cue', 'own'):
        if profile == 'full':
            return ([read, 'write_off'] + list(READOUT_LESIONS) +
                    [WRITE_LESION])
        return [read, 'write_off']
    if probe_role == 'foil':
        return [read, 'write_off'] if profile == 'full' else [read]
    # template
    return [read]


def _add_schedule(schedules: dict, arm: dict, ctx: str, g0: int,
                  profile: str) -> None:
    sid = f"e5a_{arm['arm']}_ctx_{ctx}_g{int(g0)}"
    items = PREDECESSORS[ctx] + [ctx]
    probes = []
    for probe_char in [CUE] + list(E5A_READOUT['alphabet']) + list(FOILS):
        if probe_char == CUE:
            role = 'cue'
        elif probe_char == ctx:
            role = 'own'
        elif probe_char in FOILS:
            role = 'foil'
        else:
            role = 'template'
        probe = {
            'probe_char': probe_char,
            'probe_role': role,
            'conditions': _conditions(arm, role, profile),
        }
        if role in ('own', 'cue') and profile == 'full':
            probe['write_lesion_scope'] = (
                f"{sid}_{'own' if role == 'own' else 'cue_q'}")
        probes.append(probe)
    if any(p['probe_char'] == CUE for p in probes) is False:
        raise RuntimeError('every E5a schedule must probe the neutral cue z')
    schedules[sid] = {
        'arm': arm['arm'],
        'schedule_id': sid,
        'kinetics': arm['kinetics'],
        'weight_tau': float(arm['weight_tau']),
        'eta': float(arm['eta']),
        'noise_hz': float(arm['noise_hz']),
        'read_condition': arm['read_condition'],
        'challenge': bool(arm['challenge']),
        'context': ctx,
        'successor': SUCCESSOR[ctx],
        'items': list(items),
        'program_blocks': e4b._program_from_items(list(items), int(g0)),
        'G0': int(g0),
        'profile': profile,
        'probes': probes,
    }


def build_schedules() -> dict:
    schedules: dict = {}
    for arm in ARMS:
        for ctx in arm['primary_contexts']:
            for g0 in arm['primary_delays']:
                _add_schedule(
                    schedules, arm, ctx, int(g0),
                    'full' if int(g0) in arm.get('full_delays', [])
                    else 'lite')
        for ctx in arm['confirm_contexts']:
            for g0 in arm['confirm_delays']:
                _add_schedule(schedules, arm, ctx, int(g0), 'lite')
    return schedules


def _schedule_table_sha256(schedules: dict) -> str:
    return _hash_canonical(_jsonable(schedules))


def _phase_lesion(condition: str):
    return e4b._phase_lesion(condition)


def _write_lesion(condition: str):
    return 'kc_mbon' if condition == WRITE_LESION else None


def build_cell_plan(seeds: list, schedules: dict) -> list:
    plan = []
    for seed in seeds:
        for sid, sched in schedules.items():
            for probe in sched['probes']:
                for condition in probe['conditions']:
                    plan.append({
                        'seed': int(seed),
                        'arm': sched['arm'],
                        'kinetics': sched['kinetics'],
                        'weight_tau': float(sched['weight_tau']),
                        'eta': float(sched['eta']),
                        'noise_hz': float(sched['noise_hz']),
                        'read_condition': sched['read_condition'],
                        'challenge': bool(sched['challenge']),
                        'schedule_id': sid,
                        'prediction_context': sched['context'],
                        'successor': sched['successor'],
                        'profile': sched['profile'],
                        'list': list(sched['items']),
                        'program_blocks': [list(b) for b in
                                           sched['program_blocks']],
                        'G0': int(sched['G0']),
                        'probe_char': probe['probe_char'],
                        'probe_role': probe['probe_role'],
                        'condition': condition,
                        'phase_lesion': _phase_lesion(condition),
                        'write_lesion': _write_lesion(condition),
                        'write_lesion_scope': (
                            probe.get('write_lesion_scope')
                            if condition == WRITE_LESION else None),
                    })
    return plan


def _input_spec(cell: dict, external_gap: int = 0) -> dict:
    return {
        'arm': str(cell['arm']),
        'kinetics': str(cell['kinetics']),
        'weight_tau': float(cell['weight_tau']),
        'eta': float(cell['eta']),
        'noise_hz': float(cell['noise_hz']),
        'seed': int(cell['seed']),
        'schedule_id': str(cell['schedule_id']),
        'list': [str(x) for x in cell['list']],
        'probe_char_id': int(cell['probe_char_id']),
        'gap_tokens': int(external_gap),
        'G0': int(cell['G0']),
        'condition': str(cell['condition']),
        'phase_lesion': cell['phase_lesion'],
        'write_lesion': cell['write_lesion'],
    }


def _input_sha256(spec: dict) -> str:
    return _hash_canonical(spec)


def _cell_config(cell: dict, global_config_sha256: str) -> dict:
    return {
        'global_config_sha256': global_config_sha256,
        'arm': str(cell['arm']),
        'kinetics': str(cell['kinetics']),
        'weight_tau': float(cell['weight_tau']),
        'eta': float(cell['eta']),
        'noise_hz': float(cell['noise_hz']),
        'read_condition': str(cell['read_condition']),
        'challenge': bool(cell['challenge']),
        'schedule_id': str(cell['schedule_id']),
        'prediction_context': cell['prediction_context'],
        'G0': int(cell['G0']),
        'profile': str(cell['profile']),
        'phase_lesion': cell['phase_lesion'],
        'write_lesion': cell['write_lesion'],
        'write_lesion_scope': cell['write_lesion_scope'],
    }


def _cell_id(cell: dict) -> str:
    return (f"s{cell['seed']}-{cell['schedule_id']}-{cell['probe_role']}-"
            f"{cell['probe_char']}-{cell['condition']}")


def _global_config_coverage(args, inventory_sha, resolver_sha, readout_sha,
                            forbidden_sha, schedule_sha, related_pair,
                            unrelated_pair) -> dict:
    coverage = {k: getattr(args, k) for k in E5A_CONFIG_KEYS}
    coverage.update({
        'arm_table': _jsonable(ARMS),
        'delays_primary': list(DELAYS_PRIMARY),
        'delays_confirm': list(DELAYS_CONFIRM),
        'contexts_primary': [CONTEXT_PRIMARY],
        'contexts_confirm': [CONTEXT_CONFIRM],
        'resolver_sha256': resolver_sha,
        'item_inventory_sha256': inventory_sha,
        'forbidden_union_sha256': forbidden_sha,
        'readout_sha256': readout_sha,
        'readout': _jsonable(E5A_READOUT),
        'entry_gate': _jsonable(E5A_ENTRY_GATE),
        'statistical_units': _jsonable(E5A_STATISTICAL_UNITS),
        'directional_diagnostic': _jsonable(E5A_DIRECTIONAL_DIAGNOSTIC),
        'canonicalization': _jsonable(E5A_CANONICALIZATION),
        'e5a_schedule_table_sha256': schedule_sha,
        'e3_bridge_config_sha256': E3_EXPECTED_CONFIG_SHA256,
        'baseline_eta': BASELINE_ETA,
        'high_dose_eta': HIGH_DOSE_ETA,
        'reference_weight_tau': REFERENCE_WEIGHT_TAU,
        'primary_noise_hz': PRIMARY_NOISE_HZ,
        'challenge_noise_hz': CHALLENGE_NOISE_HZ,
        'related_pair': related_pair,
        'unrelated_pair': unrelated_pair,
        'resolve_out_levels': {
            'storage_primary': 'v_pre_mbon',
            'readout_primary': 'v_pre_central',
        },
        'topology_null': 'not_evaluated_deferred_to_E5b',
    })
    return coverage


# ---------------------------------------------------------------------------
# Frozen-object / no-movement checks (CPU only).
# ---------------------------------------------------------------------------
def verify_e4b_frozen(chars: list, data: Path) -> dict:
    """Prove the frozen E4B files and canonical objects have not moved."""
    files = {}
    for rel, expected in E4B_FROZEN_FILES.items():
        path = ROOT / rel
        actual = _sha256_file(path) if path.exists() else None
        files[rel] = {
            'expected': expected,
            'actual': actual,
            'match': bool(actual == expected),
        }
    inventory, y_rel, y_unrel, _j = e4b.build_item_inventory(chars, data)
    recomputed = {
        'resolver_sha256': _hash_canonical(e4b.RESOLVER_V1),
        'item_inventory_sha256': _hash_canonical(inventory),
        'prediction_sha256': _hash_canonical(e4b.PREDICTION_V2),
        'forbidden_union_sha256': _hash_canonical(e4b.FORBIDDEN_UNION),
        'schedule_table_sha256':
            e4b._schedule_table_sha256(e4b.build_schedules(y_rel, y_unrel)),
    }
    objects = {}
    for key, expected in E4B_FROZEN_OBJECTS.items():
        if key in recomputed:
            actual = recomputed[key]
        elif key == 'config_sha256':
            actual = e4s.CONFIG_SHA256
        elif key == 'e3_bridge_config_sha256':
            actual = E3_EXPECTED_CONFIG_SHA256
        else:  # pragma: no cover
            actual = None
        objects[key] = {'expected': expected, 'actual': actual,
                        'match': bool(actual == expected)}
    return {
        'files': files,
        'objects': objects,
        'files_unchanged': bool(all(v['match'] for v in files.values())),
        'objects_unchanged': bool(all(v['match'] for v in objects.values())),
        'pass': bool(all(v['match'] for v in files.values()) and
                     all(v['match'] for v in objects.values())),
    }


def e5a_frozen_checks(arms_sha, readout_sha, entry_sha, schedule_sha,
                      config_sha) -> dict:
    actual = {
        'readout_sha256': readout_sha,
        'entry_gate_sha256': entry_sha,
        'arms_sha256': arms_sha,
        'schedule_table_sha256': schedule_sha,
        'config_sha256': config_sha,
    }
    checks = {}
    frozen = True
    for key, value in actual.items():
        expected = E5A_EXPECTED.get(key)
        state = 'frozen_ok' if expected is None else (
            'match' if expected == value else 'mismatch')
        if expected is not None and expected != value:
            frozen = False
        checks[key] = {'expected': expected, 'actual': value, 'state': state}
    return {'checks': checks, 'pass': bool(frozen)}


# ---------------------------------------------------------------------------
# Dry-run manifest.
# ---------------------------------------------------------------------------
def _source_hash(path: Path) -> dict:
    return {'path': str(path), 'sha256': _sha256_file(path)
            if path.exists() else None}


def _arm_counts(cell_plan: list) -> dict:
    per_arm = {}
    per_arm_seed = {}
    for cell in cell_plan:
        per_arm[cell['arm']] = per_arm.get(cell['arm'], 0) + 1
        per_arm_seed[(cell['arm'], cell['seed'])] = (
            per_arm_seed.get((cell['arm'], cell['seed']), 0) + 1)
    return {
        'cells_per_arm': per_arm,
        'cells_per_seed_total': len(per_arm_seed) and (
            len(cell_plan) // max(len({c['seed'] for c in cell_plan}), 1)),
        'arms': list(ARMS),
    }


def build_manifest(args, seeds, chars, data, bridge, e4b_check, hashes,
                   cell_plan, schedules, mode='cpu_only_dry_run',
                   cuda_executed=False, flybrain_constructed=False) -> dict:
    write_lesion_cells = [c for c in cell_plan
                          if c['condition'] == WRITE_LESION]
    probe_lesion_cells = [
        c for c in cell_plan
        if c['condition'] in READOUT_LESIONS and c['profile'] == 'full'
        and c['probe_role'] in ('own', 'cue')]
    input_pairs = sorted(
        (c['cell_id'], c['input_sha256']) for c in cell_plan)
    input_digest = hashlib.sha256(
        json.dumps(_jsonable(input_pairs), sort_keys=True,
                   separators=(',', ':')).encode('utf-8')).hexdigest()
    per_arm_cells = {}
    for cell in cell_plan:
        per_arm_cells.setdefault(cell['arm'], {}).setdefault(
            cell['seed'], 0)
        per_arm_cells[cell['arm']][cell['seed']] += 1
    schedule_counts = {}
    for sid, sched in schedules.items():
        n = 0
        for probe in sched['probes']:
            n += len(probe['conditions'])
        schedule_counts[sid] = {
            'arm': sched['arm'], 'context': sched['context'],
            'G0': sched['G0'], 'profile': sched['profile'],
            'cells_per_seed': n,
        }
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    prefix = args.prefix
    manifest = {
        'experiment': 'E5a_kinetics_evaluability',
        'stage': 'E5a',
        'date': '2026-09-19',
        'handoff': 'docs/OPENCODE_E5A_BUILD_HANDOFF_20260918.md',
        'prereg': 'docs/OPENCODE_E5A_V2_PREREG_20260919.md',
        'parent_prereg': 'docs/OPENCODE_E5A_SEQUENTIAL_PREREG_20260918.md',
        'mode': mode,
        'cuda_executed': bool(cuda_executed),
        'flybrain_constructed': bool(flybrain_constructed),
        'topology_null': 'not_evaluated_deferred_to_E5b',
        'e5b_authorized': False,
        'runner_source': _source_hash(Path(__file__)),
        'summarizer_source': _source_hash(
            Path(__file__).parent / 'mb_e5a_kinetics_summarize_v2.py'),
        'prereg_source': _source_hash(
            ROOT / 'docs/OPENCODE_E5A_V2_PREREG_20260919.md'),
        'seeds': {
            'fresh_seeds': list(seeds),
            'n_seeds': len(seeds),
            'seed_is_label': True,
            'seed_labels_preserved_for_reproduction': True,
            'contexts_stratified_never_pooled': True,
            'primary_contexts': [CONTEXT_PRIMARY],
            'confirmation_only_contexts': [CONTEXT_CONFIRM],
            'forbidden_union_size': len(FORBIDDEN_UNION),
            'forbidden_union_sha256': hashes['forbidden_union_sha256'],
        },
        'statistical_units': _jsonable(E5A_STATISTICAL_UNITS),
        'frozen_hashes': dict(hashes),
        'frozen_e4b_files': e4b_check,
        'frozen_e4b_objects': e4b_check['objects'],
        'no_e4b_hash_changes': bool(e4b_check['pass']),
        'e3_bridge_static': bridge,
        'factors': {
            'kinetics': KINETICS,
            'baseline_eta': BASELINE_ETA,
            'high_dose_eta': HIGH_DOSE_ETA,
            'transient_eta': TRANSIENT_ETA,
            'reference_weight_tau': REFERENCE_WEIGHT_TAU,
            'primary_noise_hz': PRIMARY_NOISE_HZ,
            'challenge_noise_hz': CHALLENGE_NOISE_HZ,
            'delays_primary': list(DELAYS_PRIMARY),
            'delays_confirm': list(DELAYS_CONFIRM),
            'reference_protocol': REFERENCE_PROTOCOL,
        },
        'arms': _jsonable(ARMS),
        'conditions': {
            'full': (['intact', 'write_off'] + list(READOUT_LESIONS) +
                     [WRITE_LESION]),
            'lite': ['intact', 'write_off'],
            'transient_full_read_condition': 'sham_eta0',
            'non_gating_conditions_present': False,
            'excluded_from_writer_gate': [
                WRITE_LESION, 'mod_perm', 'mbon_source_shuffle',
                'kc_mbon_readout_zero', 'mbon_central_readout_zero',
                'write_off'],
        },
        'cell_plan': {
            'cells_per_seed': len(cell_plan) // max(len(seeds), 1),
            'cells_total': len(cell_plan),
            'episodes_per_run': EXPECTED_EPISODES_PER_RUN,
            'episodes_total': len(cell_plan) * EXPECTED_EPISODES_PER_RUN,
            'cells_per_arm': _arm_counts(cell_plan)['cells_per_arm'],
            'cells_per_arm_seed': per_arm_cells,
            'schedule_counts': schedule_counts,
            'n_schedules': len(schedules),
        },
        'lesion_scopes': {
            'write_zero_restore': {
                'condition': WRITE_LESION,
                'n_cells': len(write_lesion_cells),
                'cells_per_seed': len(write_lesion_cells) // max(len(seeds), 1),
                'scope_rule': 'own and neutral cue z of full primary arms',
                'cells': [c['cell_id'] for c in write_lesion_cells],
                'scopes': sorted({c['write_lesion_scope']
                                  for c in write_lesion_cells}),
            },
            'probe_only_readout': {
                'conditions': list(READOUT_LESIONS),
                'n_cells': len(probe_lesion_cells),
                'cells_per_seed': (len(probe_lesion_cells) //
                                   max(len(seeds), 1)),
                'scope_rule': ('probe-window-only readout zeroing of existing '
                               'CSR data on full primary arms'),
                'cells': [c['cell_id'] for c in probe_lesion_cells],
            },
        },
        'input_hashes': {
            'n_cells': len(cell_plan),
            'n_unique': len({c['input_sha256'] for c in cell_plan}),
            'digest_sha256': input_digest,
            'specs_recompute': True,
            'bitwise_identical_expected': True,
            'first_per_arm': {
                arm: next((c['input_sha256'] for c in cell_plan
                           if c['arm'] == arm), None)
                for arm in {a['arm'] for a in ARMS}},
        },
        'expected_artifacts': {
            'raw_json': str(out_dir / f'{prefix}.json'),
            'raw_npz': str(out_dir / f'{prefix}.raw.npz'),
            'summary_json': str(out_dir / f'{prefix}.summary.json'),
            'summary_md': str(out_dir / f'{prefix}.summary.md'),
            'manifest_json': str(out_dir / f'{prefix}.manifest.json'),
            'not_created_by_dry_run': ([
                f'{prefix}.json', f'{prefix}.raw.npz',
                f'{prefix}.summary.json', f'{prefix}.summary.md']
                if not cuda_executed else None),
        },
        'readout': _jsonable(E5A_READOUT),
        'entry_gate': _jsonable(E5A_ENTRY_GATE),
        'directional_diagnostic': _jsonable(E5A_DIRECTIONAL_DIAGNOSTIC),
        'canonicalization': _jsonable(E5A_CANONICALIZATION),
        'checks': {
            'seeds_frozen': bool(list(seeds) == FRESH_SEEDS),
            'seed_forbidden_disjoint': bool(
                not (set(seeds) & set(FORBIDDEN_UNION))),
            'e3_bridge_pass': bool(bridge.get('pass')),
            'e4b_frozen_unchanged': bool(e4b_check['pass']),
            'cells_match_expected': bool(
                len(cell_plan) == EXPECTED_CELLS_TOTAL and
                len(cell_plan) // max(len(seeds), 1) == EXPECTED_CELLS_PER_SEED),
            'episodes_match_expected': bool(
                len(cell_plan) * EXPECTED_EPISODES_PER_RUN ==
                EXPECTED_EPISODES_TOTAL),
            'write_lesion_count_match': bool(
                len(write_lesion_cells) ==
                EXPECTED_WRITE_LESION_CELLS_PER_SEED * len(seeds)),
            'probe_lesion_count_match': bool(
                len(probe_lesion_cells) ==
                EXPECTED_PROBE_LESION_CELLS_PER_SEED * len(seeds)),
            'input_specs_recompute': True,
            'no_cuda': bool(not cuda_executed),
        },
    }
    manifest['checks']['all_pass'] = bool(
        all(manifest['checks'].values()))
    return manifest


def _write_manifest(manifest: dict, manifest_out: Path,
                    markdown_out: Path | None) -> None:
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, default=str),
                            encoding='utf-8')
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(_manifest_markdown(manifest),
                                encoding='utf-8')


def _manifest_markdown(manifest: dict) -> str:
    cp = manifest['cell_plan']
    lines = [
        '# E5a CPU-only dry-run manifest (v2, 2026-09-19)',
        '',
        f"- mode: `{manifest['mode']}`; cuda_executed="
        f"`{manifest['cuda_executed']}`; flybrain_constructed="
        f"`{manifest['flybrain_constructed']}`",
        f"- topology null: `{manifest['topology_null']}`; E5b authorized: "
        f"`{manifest['e5b_authorized']}`",
        f"- seeds (labels; deterministic-replicate primary arms): "
        f"`{manifest['seeds']['fresh_seeds']}`",
        f"- cells/seed: `{cp['cells_per_seed']}`; cells: "
        f"`{cp['cells_total']}`; episodes: `{cp['episodes_total']}`; "
        f"schedules: `{cp['n_schedules']}`",
        f"- cells/arm: `{json.dumps(cp['cells_per_arm'], sort_keys=True)}`",
        f"- write-zero-restore cells: "
        f"`{manifest['lesion_scopes']['write_zero_restore']['n_cells']}`; "
        f"probe-only readout cells: "
        f"`{manifest['lesion_scopes']['probe_only_readout']['n_cells']}`",
        f"- input-hash digest: "
        f"`{manifest['input_hashes']['digest_sha256']}`",
        f"- no E4B hash changes: `{manifest['no_e4b_hash_changes']}`",
        '',
        '## Frozen hashes',
        '',
    ]
    for key in sorted(manifest['frozen_hashes']):
        lines.append(f"- `{key}` = `{manifest['frozen_hashes'][key]}`")
    lines += ['', '## Checks', '']
    for key in sorted(manifest['checks']):
        lines.append(f"- `{key}`: `{manifest['checks'][key]}`")
    lines += ['', '## E4B frozen objects (must not move)', '']
    for key in sorted(manifest['frozen_e4b_objects']):
        rec = manifest['frozen_e4b_objects'][key]
        lines.append(f"- `{key}` match=`{rec['match']}` "
                     f"actual=`{rec['actual']}`")
    lines += ['', '## Expected artifacts', '']
    for key, value in manifest['expected_artifacts'].items():
        lines.append(f"- `{key}`: `{value}`")
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus',
                    default=str(ROOT / 'corpus' / 'tinyshakespeare.txt'))
    ap.add_argument('--data', default=None)
    ap.add_argument('--seeds', default=','.join(str(s) for s in FRESH_SEEDS))
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--prefix', default=DEFAULT_PREFIX)
    ap.add_argument('--repeats-per-cell', type=int, default=1)
    ap.add_argument('--manifest-out', default=None,
                    help='dry-run manifest JSON path (CPU-only)')
    ap.add_argument('--manifest-md-out', default=None,
                    help='dry-run manifest Markdown path (CPU-only)')
    ap.add_argument('--dry-run', action='store_true',
                    help='CPU-only: print the exact plan + write manifest, '
                         'no FlyBrain/CUDA')
    # Reference protocol CLI (all defaults are the frozen E4B reference).
    ap.add_argument('--noise-hz', type=float, default=PRIMARY_NOISE_HZ)
    ap.add_argument('--gap-tokens', type=int, default=0)
    ap.add_argument('--probe-drive', type=float, default=2.5)
    ap.add_argument('--probe-settle-steps', type=int, default=6)
    ap.add_argument('--active', type=int, default=512)
    ap.add_argument('--drive', type=float, default=1.0)
    ap.add_argument('--gain', type=float, default=1.5)
    ap.add_argument('--tonic', type=float, default=0.05)
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--sustain', type=float, default=0.5)
    ap.add_argument('--probe-sustain', type=float, default=None)
    ap.add_argument('--trace-tau', type=float, default=0.1)
    ap.add_argument('--dan-pulse', type=float, default=2.0)
    ap.add_argument('--eta', type=float, default=BASELINE_ETA)
    ap.add_argument('--eligibility-tau', type=float, default=0.5)
    ap.add_argument('--weight-tau', type=float, default=REFERENCE_WEIGHT_TAU)
    ap.add_argument('--max-modulation', type=float, default=0.9)
    ap.add_argument('--eligibility-mix', type=float, default=0.0)
    ap.add_argument('--target-top', type=int, default=256)
    ap.add_argument('--target-selection', default='direct')
    ap.add_argument('--graded-tol', type=float, default=1e-6)
    ap.add_argument('--graded-ulp', type=int, default=16)
    ap.add_argument('--limit-cells', type=int, default=0)
    ap.add_argument('--e3-manifest', default=E3_MANIFEST)
    args = ap.parse_args()

    e4b._build_args(args)
    args.data = (Path(args.data) if args.data else
                 Path(os.environ.get('FLY_DATA', ROOT / 'data')))
    args.noise_grid = [float(args.noise_hz)]
    args.repeats = int(args.repeats_per_cell)

    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    if not seeds:
        raise ValueError('no seeds given')
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate seeds')
    overlap = sorted(set(seeds) & set(FORBIDDEN_UNION))
    if overlap:
        raise RuntimeError(
            f'fresh E5a seeds overlap the forbidden union: {overlap}')
    if seeds != FRESH_SEEDS:
        raise RuntimeError(
            f'fresh E5a cohort must equal the frozen ten-value list '
            f'{FRESH_SEEDS}; got {seeds}')

    chars, _, _, _ = load_corpus(args.corpus, 128, 1, 1)
    chars = list(chars)
    for ch in (ITEM_ALPHABET + FOILS + [CUE]):
        if ch not in chars:
            raise ValueError(f'item {ch!r} outside corpus vocabulary')
    args.vocab = len(chars)

    # ---- frozen hashes (computed before any FlyBrain/CUDA allocation) ----
    inventory, y_rel, y_unrel, _j = e4b.build_item_inventory(chars, args.data)
    resolver_sha = _hash_canonical(RESOLVER_V1)
    inventory_sha = _hash_canonical(inventory)
    readout_sha = _hash_canonical(E5A_READOUT)
    entry_sha = _hash_canonical(E5A_ENTRY_GATE)
    arms_sha = _hash_canonical(_jsonable(ARMS))
    forbidden_sha = _hash_canonical(FORBIDDEN_UNION)
    schedules = build_schedules()
    schedule_sha = _schedule_table_sha256(schedules)
    related_pair = {'x': 'a', 'y': y_rel, 'J': inventory['related_pair']['J']}
    unrelated_pair = {'x': 'a', 'y': y_unrel,
                      'J': inventory['unrelated_pair']['J']}
    config_coverage = _global_config_coverage(
        args, inventory_sha, resolver_sha, readout_sha, forbidden_sha,
        schedule_sha, related_pair, unrelated_pair)
    config_sha = _hash_canonical(config_coverage)
    hashes = {
        'resolver_sha256': resolver_sha,
        'item_inventory_sha256': inventory_sha,
        'readout_sha256': readout_sha,
        'entry_gate_sha256': entry_sha,
        'arms_sha256': arms_sha,
        'forbidden_union_sha256': forbidden_sha,
        'schedule_table_sha256': schedule_sha,
        'config_sha256': config_sha,
        'e3_bridge_config_sha256': E3_EXPECTED_CONFIG_SHA256,
    }

    cell_plan = build_cell_plan(seeds, schedules)
    for cell in cell_plan:
        cell['probe_char_id'] = int(chars.index(cell['probe_char']))
        cell['probe_id'] = int(cell['probe_char_id'])
        spec = _input_spec(cell, int(args.gap_tokens))
        cell['input_spec'] = spec
        cell['input_sha256'] = _input_sha256(spec)
        cell['cell_config'] = _cell_config(cell, config_sha)
        cell['cell_config_sha256'] = _hash_canonical(cell['cell_config'])
        cell['cell_id'] = _cell_id(cell)

    n_cells = len(cell_plan)
    per_seed = n_cells // max(len(seeds), 1)
    per_arm = {}
    for cell in cell_plan:
        per_arm[cell['arm']] = per_arm.get(cell['arm'], 0) + 1
    write_lesion_cells = [c for c in cell_plan
                          if c['condition'] == WRITE_LESION]
    probe_lesion_cells = [
        c for c in cell_plan
        if c['condition'] in READOUT_LESIONS and c['profile'] == 'full'
        and c['probe_role'] in ('own', 'cue')]

    # ---- CPU-only no-movement / bridge checks ----
    e3_manifest_path = Path(args.e3_manifest)
    if not e3_manifest_path.is_absolute():
        e3_manifest_path = ROOT / e3_manifest_path
    bridge = e4b.bridge_e3_static(chars, e3_manifest_path)
    e4b_check = verify_e4b_frozen(chars, args.data)
    e5a_check = e5a_frozen_checks(arms_sha, readout_sha, entry_sha,
                                  schedule_sha, config_sha)

    print(f'seeds={seeds}', flush=True)
    print(f'cells_per_seed={per_seed} cells={n_cells} '
          f'episodes_per_run={EXPECTED_EPISODES_PER_RUN} '
          f'episodes_total={n_cells * EXPECTED_EPISODES_PER_RUN}', flush=True)
    print(f'cells_per_arm={per_arm}', flush=True)
    print(f'resolver_sha256={resolver_sha}', flush=True)
    print(f'item_inventory_sha256={inventory_sha}', flush=True)
    print(f'e5a_readout_sha256={readout_sha}', flush=True)
    print(f'e5a_entry_gate_sha256={entry_sha}', flush=True)
    print(f'e5a_arms_sha256={arms_sha}', flush=True)
    print(f'forbidden_union_sha256={forbidden_sha}', flush=True)
    print(f'e5a_schedule_table_sha256={schedule_sha}', flush=True)
    print(f'e5a_config_sha256={config_sha}', flush=True)
    print(f'related_pair={related_pair} unrelated_pair={unrelated_pair}',
          flush=True)
    print(f'write_lesion_cells={len(write_lesion_cells)} '
          f'probe_lesion_cells={len(probe_lesion_cells)}', flush=True)
    print(f"e3_bridge_config_sha256_recomputed="
          f"{bridge['e3_config_sha256_recomputed']}", flush=True)
    print(f"e3_bridge_config_sha256_match={bridge['e3_config_sha256_match']}",
          flush=True)
    print(f"e3_bridge_input_hashes_match={bridge['e3_input_hashes_match']}",
          flush=True)
    print(f"e4b_frozen_unchanged={e4b_check['pass']}", flush=True)
    print(f"e5a_frozen_checks={json.dumps(e5a_check, default=str)}",
          flush=True)
    if not bridge['pass']:
        raise RuntimeError('E3 bridge mismatch; failing closed before any CUDA')
    if not e4b_check['pass']:
        raise RuntimeError('E4B frozen object/file moved; failing closed')
    if not e5a_check['pass']:
        raise RuntimeError('E5a frozen object mismatch; failing closed')

    if args.dry_run:
        if per_seed != EXPECTED_CELLS_PER_SEED or n_cells != EXPECTED_CELLS_TOTAL:
            raise RuntimeError(
                f'E5a plan must be {EXPECTED_CELLS_PER_SEED} cells/seed and '
                f'{EXPECTED_CELLS_TOTAL} cells; got {per_seed}/{n_cells}')
        if n_cells * EXPECTED_EPISODES_PER_RUN != EXPECTED_EPISODES_TOTAL:
            raise RuntimeError('E5a episode count mismatch')
        if (len(write_lesion_cells) !=
                EXPECTED_WRITE_LESION_CELLS_PER_SEED * len(seeds)):
            raise RuntimeError(
                'write-zero-restore cell count does not match the frozen plan')
        if (len(probe_lesion_cells) !=
                EXPECTED_PROBE_LESION_CELLS_PER_SEED * len(seeds)):
            raise RuntimeError(
                'probe-only readout lesion cell count does not match the '
                'frozen plan')
        for cell in cell_plan:
            if _hash_canonical(cell['input_spec']) != cell['input_sha256']:
                raise RuntimeError(f'input hash mismatch for {cell["cell_id"]}')
            if _hash_canonical(cell['cell_config']) != cell['cell_config_sha256']:
                raise RuntimeError(f'cell config hash mismatch for '
                                   f'{cell["cell_id"]}')
        manifest = build_manifest(args, seeds, chars, args.data, bridge,
                                  e4b_check, hashes, cell_plan, schedules)
        manifest['frozen_e5a_checks'] = e5a_check
        # A dry-run must never create (or default into) the future biological
        # result directory.  Require an explicit manifest path and fail closed
        # otherwise.
        if args.manifest_out:
            manifest_out = Path(args.manifest_out)
        elif args.manifest_md_out:
            manifest_out = Path(args.manifest_md_out).with_suffix('.json')
        else:
            raise RuntimeError(
                'dry-run requires --manifest-out (and/or --manifest-md-out) so '
                'it cannot create the future E5a result directory '
                f'{ROOT / args.out_dir}; refusing to guess an output path')
        md_out = (Path(args.manifest_md_out) if args.manifest_md_out else
                  manifest_out.with_suffix('.md'))
        _write_manifest(manifest, manifest_out, md_out)
        print('DRY-RUN schedule table:', flush=True)
        for sid in sorted(schedules):
            sched = schedules[sid]
            n = sum(len(p['conditions']) for p in sched['probes'])
            print(f"  {sid:<52} arm={sched['arm']:<30} "
                  f"ctx={sched['context']} G0={sched['G0']:>3} "
                  f"profile={sched['profile']:<4} cells/seed={n}", flush=True)
        print('DRY-RUN write-zero-restore scopes:', flush=True)
        for cell in write_lesion_cells[:4]:
            print(f"  {cell['cell_id']} scope={cell['write_lesion_scope']}",
                  flush=True)
        print(f'manifest_json={manifest_out}', flush=True)
        print(f'manifest_md={md_out}', flush=True)
        print('DRY-RUN ok; no CUDA episode executed.', flush=True)
        return

    # ------------------------------------------------------------------
    # CUDA path (implemented but never launched by the E5a build pass).
    # ------------------------------------------------------------------
    from flybrain import FlyBrain
    brain0 = FlyBrain(data=args.data, device='cuda', batch=1, seed=seeds[0],
                      sensory_input=False)
    masks = _masks(brain0)
    central_ids = _select_targets(brain0, args.target_selection, args.target_top)
    target_summary = _target_summary(brain0, central_ids)
    fixed_direction = _anatomical_target_direction(brain0, central_ids)
    _, raw_direction_norm = _raw_target_direction(brain0, central_ids)
    fallback_used = bool(raw_direction_norm <= 1e-8)
    device = _device_info()
    print(f'pid={device["pid"]} gpu={device["gpu_name"]} '
          f'cupy={device["cupy_version"]} cuda={device["cuda_runtime_version"]}',
          flush=True)
    print('edge counts:', {kk: int(len(v)) for kk, v in masks.items()},
          flush=True)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_json = out_dir / f'{args.prefix}.json'
    raw_npz = out_dir / f'{args.prefix}.raw.npz'

    runs = []
    arrays: dict = {}
    t_start = time.time()
    done = 0
    for arm in ARMS:
        args.weight_tau = float(arm['weight_tau'])
        args.eta = float(arm['eta'])
        args.noise_hz = float(arm['noise_hz'])
        conditions = [arm['read_condition'], 'write_off']
        full = [c for c in cell_plan
                if c['arm'] == arm['arm'] and c['profile'] == 'full']
        if full:
            conditions += list(READOUT_LESIONS) + [WRITE_LESION]
        conditions = list(dict.fromkeys(conditions))
        arm_cells = [c for c in cell_plan if c['arm'] == arm['arm']]
        for seed in seeds:
            args.seed = int(seed)
            states = {cond: _new_condition(args, cond, masks, args.vocab,
                                           central_ids)
                      for cond in conditions}
            for cond in conditions:
                states[cond]['fixed_direction'] = fixed_direction
            for cell in [c for c in arm_cells if c['seed'] == seed]:
                if args.limit_cells and done >= args.limit_cells:
                    break
                state = states[cell['condition']]
                program_ids = [
                    [None if t is None else int(chars.index(t)), int(n)]
                    for t, n in cell['program_blocks']]
                cid = cell['cell_id']
                e4b._capture_begin()
                write = _episode(state, args, [], cell['probe_id'],
                                 int(args.gap_tokens), True,
                                 write_program=program_ids)
                ev_write = e4b._capture_end()
                e4b._capture_begin()
                no_write = _episode(state, args, [], cell['probe_id'],
                                    int(args.gap_tokens), False,
                                    write_program=program_ids)
                ev_no_write = e4b._capture_end()
                e4b._capture_begin()
                replay = _episode(state, args, [], cell['probe_id'],
                                  int(args.gap_tokens), False,
                                  write_program=program_ids)
                ev_replay = e4b._capture_end()
                _canonicalize_episode_rows(state, args,
                                           [write, no_write, replay])
                pair = _pair_metrics(write, no_write,
                                     graded_tol=args.graded_tol,
                                     graded_ulp=args.graded_ulp)
                pair.update(_replay_stats(no_write, replay,
                                          graded_tol=args.graded_tol,
                                          graded_ulp=args.graded_ulp))
                if write.get('modulation_after_write_vector') is not None:
                    write['modulation_after_write_vector_stats'] = _array_stats(
                        write['modulation_after_write_vector'])
                pair['write_off_pair_within_tolerance'] = bool(
                    cell['condition'] == 'write_off' and
                    pair['events_bitwise_equal'] and
                    pair['graded_within_tolerance'])
                if write.get('edge_norm_after_write') is not None:
                    write['edge_norm_after_write_sha256'] = _hash_scalar_f64(
                        write.get('edge_norm_after_write', 0.0))
                pair.update({
                    'cell_id': cid,
                    'arm': cell['arm'],
                    'kinetics': cell['kinetics'],
                    'weight_tau': float(cell['weight_tau']),
                    'eta': float(cell['eta']),
                    'noise_hz': float(cell['noise_hz']),
                    'read_condition': cell['read_condition'],
                    'challenge': bool(cell['challenge']),
                    'schedule_id': cell['schedule_id'],
                    'prediction_context': cell['prediction_context'],
                    'successor': cell['successor'],
                    'profile': cell['profile'],
                    'item_list': list(cell['list']),
                    'G0': int(cell['G0']),
                    'program_blocks': [list(b) for b in
                                       cell['program_blocks']],
                    'probe_char': cell['probe_char'],
                    'probe_char_id': int(cell['probe_id']),
                    'probe_role': cell['probe_role'],
                    'condition': cell['condition'],
                    'phase_lesion': cell['phase_lesion'],
                    'write_lesion': cell['write_lesion'],
                    'write_lesion_scope': cell['write_lesion_scope'],
                    'repeat_seed': int(seed),
                    'gap_tokens': int(args.gap_tokens),
                    'probe_drive': float(args.probe_drive),
                    'probe_settle_steps': int(args.probe_settle_steps),
                    'input_spec': cell['input_spec'],
                    'input_sha256': cell['input_sha256'],
                    'cell_config': cell['cell_config'],
                    'cell_config_sha256': cell['cell_config_sha256'],
                    'events_write': ev_write,
                    'events_no_write': ev_no_write,
                    'events_replay': ev_replay,
                    'write_row': write,
                    'no_write_row': no_write,
                    'replay_row': replay,
                })
                e4b._strip_vectors(pair, arrays, cid)
                runs.append(_jsonable(pair))
                done += 1
                print(f'[{done}/{n_cells}] s={seed} {cell["arm"]} '
                      f'{cell["schedule_id"]} {cell["probe_role"]}->'
                      f'{cell["probe_char"]} {cell["condition"]:26s}',
                      flush=True)
            del states

    input_groups: dict = {}
    for rec in runs:
        key = (rec['repeat_seed'], rec['arm'], rec['schedule_id'],
               rec['probe_char_id'], rec['condition'])
        input_groups.setdefault(key, set()).add(rec['input_sha256'])
    inputs_bitwise_identical = bool(len(runs) > 0 and all(
        len(v) == 1 for v in input_groups.values()))

    np.savez(raw_npz, **arrays)
    raw = {
        'experiment': 'E5a_kinetics_evaluability',
        'date': '2026-09-19',
        'handoff': 'docs/OPENCODE_E5A_BUILD_HANDOFF_20260918.md',
        'prereg': 'docs/OPENCODE_E5A_V2_PREREG_20260919.md',
        'parent_prereg': 'docs/OPENCODE_E5A_SEQUENTIAL_PREREG_20260918.md',
        'device': device,
        'config': _jsonable(vars(args)),
        'config_sha256': config_sha,
        'config_hash_keys': list(E5A_CONFIG_KEYS),
        'config_hash_coverage': _jsonable(config_coverage),
        'resolver_sha256': resolver_sha,
        'item_inventory_sha256': inventory_sha,
        'readout_sha256': readout_sha,
        'entry_gate_sha256': entry_sha,
        'arms_sha256': arms_sha,
        'forbidden_union_sha256': forbidden_sha,
        'schedule_table_sha256': schedule_sha,
        'resolver': _jsonable(RESOLVER_V1),
        'item_inventory': _jsonable(inventory),
        'readout': _jsonable(E5A_READOUT),
        'entry_gate': _jsonable(E5A_ENTRY_GATE),
        'statistical_units': _jsonable(E5A_STATISTICAL_UNITS),
        'directional_diagnostic': _jsonable(E5A_DIRECTIONAL_DIAGNOSTIC),
        'canonicalization': _jsonable(E5A_CANONICALIZATION),
        'arms': _jsonable(ARMS),
        'forbidden_union': list(FORBIDDEN_UNION),
        'fresh_seeds': list(seeds),
        'related_pair': related_pair,
        'unrelated_pair': unrelated_pair,
        'schedules': _jsonable(schedules),
        'cells_per_arm': per_arm,
        'n_cells': int(n_cells),
        'n_runs': int(len(runs)),
        'episodes_per_run': int(EXPECTED_EPISODES_PER_RUN),
        'episodes_total': int(len(runs) * EXPECTED_EPISODES_PER_RUN),
        'inputs_bitwise_identical': bool(inputs_bitwise_identical),
        'input_groups': {('|'.join(map(str, k))): sorted(v)
                         for k, v in sorted(input_groups.items())},
        'connectome': {
            'neurons': int(brain0.n),
            'edges': int(len(brain0._W.indices.get())),
            'edge_counts': {kk: int(len(v)) for kk, v in masks.items()},
            'target_summary': target_summary,
            'fixed_direction_norm': float(np.linalg.norm(fixed_direction)),
            'fixed_direction_raw_norm': float(raw_direction_norm),
            'fixed_direction_fallback_used': fallback_used,
            'fixed_direction': fixed_direction.astype(np.float32).tolist(),
        },
        'e3_bridge_static': bridge,
        'e4b_frozen_check': e4b_check,
        'raw_npz': str(raw_npz),
        'cells': runs,
        'elapsed_seconds': time.time() - t_start,
    }
    with raw_json.open('w', encoding='utf-8') as handle:
        json.dump(raw, handle, indent=2, ensure_ascii=False)
    print(f'saved {raw_json}')
    print(f'saved {raw_npz}')

    # Emit the documented ``<prefix>.manifest.json`` beside the raw/summary so
    # the expected-artifact list is honoured by the CUDA path.  This reuses the
    # same CPU-only planner/verifier manifest builder (it constructs no
    # FlyBrain and performs no CUDA work).
    manifest = build_manifest(args, seeds, chars, args.data, bridge,
                              e4b_check, hashes, cell_plan, schedules,
                              mode='cuda_run', cuda_executed=True,
                              flybrain_constructed=True)
    manifest['frozen_e5a_checks'] = e5a_check
    manifest['raw_json'] = str(raw_json)
    manifest['raw_npz'] = str(raw_npz)
    manifest['n_cells_executed'] = int(len(runs))
    manifest_out = out_dir / f'{args.prefix}.manifest.json'
    _write_manifest(manifest, manifest_out, None)
    print(f'saved {manifest_out}')


if __name__ == '__main__':
    main()
