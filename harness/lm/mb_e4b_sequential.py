"""E4B sequential-capacity runner (additive, 2026-09-18).

Second-DeepSeek builder artifact implementing the final design in
``models/research_results/opencode_analysis/deepseek_e4b_design_final_20260918.md``
under ``docs/OPENCODE_SECOND_DS_E4B_BUILD_HANDOFF_20260918.md``.

This file is **additive**: it imports the frozen
``mb_persistent_memory_probe._episode`` path (with the additive
``kc_mbon_writezero_restore`` write-phase lesion bookkeeping) and the frozen
``_new_condition`` controls.  No frozen E3 file, no historical raw/summary, and
no connectome/CUDA configuration is modified.

It implements the four separated claims of the final design:

  * H-cap        sequential load ``A->B->...`` at a fixed retention interval;
  * H-forget     age / gradual forgetting with a matched passive reference;
  * H-interf     related / unrelated / repeated interference at matched dose;
  * H-pred       bounded finite-alphabet horizon-1 native template-rank
                 prediction with an exact stratified successor-label
                 exchangeability null (PREDICTION_V2, prereg amendment
                 2026-09-18); the 24 label permutations are descriptive only.

The runner only *plans* on ``--dry-run`` and ``--selfcheck-writeback`` (no CUDA
episode, no ``FlyBrain`` construction).  A CUDA run is never launched by this
design/build pass.  ``--bridge-e3`` re-derives the frozen E3 config/input hashes
statically and (only with the explicit ``--bridge-e3-cuda`` flag) can replay the
frozen E3 ``A0``/``G=32`` row; the CUDA comparison is not run here.
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

from flybrain import FlyBrain  # noqa: E402
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
from flylm import Encoder  # noqa: E402
from train_lm import load_corpus  # noqa: E402

DEFAULT_OUT = 'models/research_results/e4b_sequential_20260918'
DEFAULT_PREFIX = 'mb_e4b_sequential_20260918'
E3_MANIFEST = ('models/research_results/e3_continual_update_20260918/'
               'mb_e3_continual_update_20260918.manifest.json')
E3_EXPECTED_CONFIG_SHA256 = (
    'b42bfe5a7cb9d6d2ac0485fb49ca766bb1e9fd6ca0fe983e4894658813a0cda9')
E3_A0 = [('e', 32), (None, 8), (None, 32)]
E3_BRIDGE_SEEDS = [20261201, 20261202]

# ---------------------------------------------------------------------------
# Frozen E4B grammar / inventory constants (final design section 2.2/3.2).
# ---------------------------------------------------------------------------
ITEM_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
FOILS = ['i', 'j']
CUE = 'z'
CYCLE = ['a', 'b', 'c', 'd']
DISTRACTOR_ORDER = ['b', 'c', 'd', 'e', 'f', 'g', 'h']
ITEM_TOKENS = 32
INTER_ITEM_GAP = 8
G0_DEFAULT = 32
SETTLE_LO, SETTLE_HI = 6, 11
ENCODER_SPEC = {'active': 512, 'drive': 1.0, 'window': 4, 'gamma': 0.5,
                'seed': 3, 'position_codes': False}

RESOLVER_V1 = {
    'name': 'content_specific_recall_v1',
    'readout_level_primary': 'v_pre_mbon',
    'readout_level_secondary': 'v_pre_central',
    'settle_window': [6, 11],
    'foils': ['i', 'j'],
    'foil_count': 2,
    'margin_kappa': 1.25,
    'tau_R': 'max(10*F_sum, 3*SD_R)',
    'floor_adjacent': 'R_median < 100*F_sum',
    'criterion': 'R_c > tau_R AND R_c > max(R_i, R_j) * 1.25 AND not floor_adjacent',
    'rank_report': ('report R_c, R_i, R_j, argmax, and R_c/max(R_i,R_j) '
                    'for every probed cell'),
}

# PREDICTION_V2 (prereg amendment 2026-09-18, successor-null fix).  Replaces
# the structurally unsatisfiable ``observed > exhaustive permutation max`` null
# of PREDICTION_V1 with an exact stratified successor-label exchangeability
# test, and freezes the tie / norm-floor discipline.  The 24 successor-label
# permutations survive only as a descriptive upper-tail count, never a gate.
PREDICTION_V2 = {
    'name': 'bounded_horizon1_template_rank_v2',
    'alphabet': ['a', 'b', 'c', 'd'],
    'cycle': ['a', 'b', 'c', 'd', 'a'],
    'contexts_primary': ['a', 'c'],
    'contexts_primary_holdout': ['b', 'd'],
    'horizon': 1,
    'bounded': True,
    'open_ended': False,
    'statistic': 'cosine_template_rank',
    'successor_rule': ('argmax_{y in alphabet} cosine(d_X(z), d_X(y)) == s(X)'),
    'primary_level': 'v_pre_central',
    'secondary_level': 'v_pre_mbon',
    'settle_window': [6, 11],
    'tie_tol': 1e-9,
    'norm_floor': 'derived from v_pre_central F_sum/SD_R, frozen before data',
    'null': 'exact_stratified_successor_label_exchangeability',
    'null_model': 'hit_{X,s} ~ Bernoulli(0.25); Binomial(10,0.25) per context',
    'p_rule': ('p_X = P(Bin(10,0.25) >= h_X), one-sided upper tail; exact, '
               'seed-free'),
    'alpha_family': 0.05,
    'alpha_per_context': 0.025,
    'min_hits_per_context': 6,
    'pooled_stat': 'descriptive only; NEVER Binomial(20,0.25)',
    'seed_rule': ('seed is the sampling unit; contexts stratified; >=8/10 seeds '
                  'with per-seed accuracy > 0.25'),
    'causal_collapse': {
        'write_off': 'required',
        'kc_mbon_readout_zero': 'required',
        'mbon_central_readout_zero': 'required',
        'kc_mbon_writezero_restore': (
            'required only if write-phase end-fingerprint != intact (design 4.2 '
            'two-outcome); else diagnostic'),
    },
    'decoder_rule': ('no fitted decoder; a chance/null result falsifies and '
                     'stops the open-prediction line'),
}

# Prediction successor map for the grounded cycle a->b->c->d->a.
SUCCESSOR = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'}
PREDECESSORS = {'a': ['c', 'd'], 'b': ['d', 'a'],
                'c': ['a', 'b'], 'd': ['b', 'c']}

# Frozen base conditions (final design sections 3.7/3.8).
BASE_CONDITIONS = ('intact', 'write_off')
EXTRA_Q_CONDITIONS = ('kc_mbon_readout_zero', 'mbon_central_readout_zero',
                      'mod_perm', 'mbon_source_shuffle')
WRITE_LESION = 'kc_mbon_writezero_restore'
ALL_CONDITIONS = ('intact', 'write_off', 'kc_mbon_readout_zero',
                  'mbon_central_readout_zero', 'mod_perm',
                  'mbon_source_shuffle', WRITE_LESION)

# H-null scope amendment (2026-09-18).  No connectome-topology null is evaluated
# in E4B / PREDICTION_V2:
#   * ``mod_perm`` is a topology-preserving content<->edge-assignment control;
#   * ``mbon_source_shuffle`` is a construction-time downstream MBON-afferent-
#     identity control.
# Neither is a KC->MBON memory-topology null.  Both remain in the frozen grid
# only as explicitly NON-GATING exploratory conditions (gating=false,
# used_for_decision=false); the runner and summarizer compute no p-value,
# threshold or "effects vanish" claim from them.  A true topology-necessity test
# is deferred to a separate pre-registered stage; see
# ``docs/OPENCODE_E5_TOPOLOGY_STAGE_DEFERRED_20260918.md``.
NON_GATING_EXPLORATORY_CONDITIONS = ('mod_perm', 'mbon_source_shuffle')

WRITE_LESION_SCOPE = {
    'capacity_p4|a': 'capacity_p4_own_a',
    'prediction_ctx_a|z': 'prediction_ctx_a_cue_q',
}

# Frozen interference-arm scope (B2).  The final design section 3.6 allocates
# "lesions 2" to the interference arm but does not name the
# config/probe/condition; section 3.6's prose lists probes ``{a,y,i,j}`` while
# section 3.8's arithmetic (12 = 3x2x2) requires exactly two probes.  This
# object freezes the resolution in code, in ``schedule_table_sha256`` and in
# ``config_sha256`` so the builder cannot choose the scope after data:
#   * the interference matrix probes are exactly ``{a (old), y (new)}``; the
#     never-written foils i,j are not interference-arm cells (they only carry
#     the capacity/forgetting resolver).
#   * the two probe-only readout lesions live on the related config's old probe
#     ``a``: the old item is the retention quantity the H-interf statistic
#     measures, and ``rel`` is the maximal-overlap config where a downstream
#     readout lesion is most likely to expose relatedness-specific use.
# They are non-gating diagnostics only for the write-phase collapse; the
# required write-phase lesion remains the separate two-cell scope.
INTERFERENCE_PROBES = ('a', 'y')
INTERFERENCE_LESION = {
    'schedule_id': 'interference_rel',
    'probe_char': 'a',
    'probe_role': 'old',
    'conditions': ['kc_mbon_readout_zero', 'mbon_central_readout_zero'],
    'scope': 'interference_rel_old_a_probe_only',
    'gating': False,
    'rationale': ('probe-only readout lesions on the related config old probe a; '
                  'the old item is the H-interf retention quantity and rel is '
                  'the maximal-overlap config, so a downstream readout lesion is '
                  'the sharpest relatedness-specific diagnostic. Diagnostic only; '
                  'not used as a biological gate.'),
}

# Explicit config-hash coverage.  Every key affects dynamics or readout and is
# folded into the canonical-JSON global hash.  The list is stored in the raw so
# the reviewer can verify coverage.  Per-cell fields (schedule_id, G0,
# prediction_context, phase_lesion, write_lesion) are carried by the per-cell
# config hash and the input spec.
CONFIG_HASH_KEYS = (
    'active', 'drive', 'eligibility_mix', 'eligibility_tau', 'eta', 'gain',
    'gamma', 'gap_plasticity', 'gap_tokens', 'graded_tol', 'graded_ulp', 'k',
    'max_modulation', 'noise_hz', 'probe_drive', 'probe_plasticity',
    'probe_settle_steps', 'probe_sustain', 'probe_tokens',
    'record_fingerprints', 'record_probe_profile', 'save_block_snapshots',
    'save_modulation_after_write', 'save_modulation_preprobe',
    'save_probe_vectors', 'skip_probe_weight_writeback', 'sustain',
    'target_selection', 'target_top', 'tonic', 'trace_tau', 'weight_tau',
    'window', 'dan_pulse', 'dan_pulse_unpaired', 'mod_perm_seed',
    'item_tokens', 'inter_item_gap',
)


# ---------------------------------------------------------------------------
# Seed provenance (final design section 7.1/7.2, FIX-4).
# ---------------------------------------------------------------------------
def build_forbidden_union() -> list:
    union = set()
    union.add(20260916)
    union |= set(range(20260917, 20261010 + 1))
    union |= set(range(20261017, 20261019 + 1))
    union.add(20261021)
    union |= set(range(20261114, 20261123 + 1))
    union |= {20261131, 20261132, 20261133}
    union |= {20261142, 20261143}
    union |= set(range(20261201, 20261210 + 1))
    union |= set(range(20261217, 20261219 + 1))
    union |= set(range(20270917, 20270921 + 1))
    union |= set(range(20280917, 20280921 + 1))
    union |= {271828, 314159, 20268835, 20365645}
    union |= set(range(0, 7 + 1)) | {64, 65, 66, 123, 777, 987, 2026}
    union |= {8, 9, 11, 17, 42420}
    return sorted(int(x) for x in union)


FORBIDDEN_UNION = build_forbidden_union()
FRESH_SEEDS = list(range(20270101, 20270111))


def _hash_canonical(obj) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _hash_array_f32(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _hash_scalar_f64(value) -> str:
    import struct
    return hashlib.sha256(struct.pack('<d', float(value))).hexdigest()


def _array_stats(arr) -> dict:
    a = np.asarray(arr, dtype=np.float32)
    return {
        'sha256': _hash_array_f32(a),
        'length': int(a.size),
        'shape': list(a.shape),
        'dtype': 'float32',
        'max_abs': float(np.abs(a).max()) if a.size else 0.0,
        'l2': float(np.linalg.norm(a.astype(np.float64))) if a.size else 0.0,
        'sum': float(a.astype(np.float64).sum()) if a.size else 0.0,
        'nnz': int(np.count_nonzero(a)),
    }


def _device_info() -> dict:
    try:
        import cupy
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props['name']
        if isinstance(name, bytes):
            name = name.decode('utf-8', 'replace')
        cuda = int(cupy.cuda.runtime.runtimeGetVersion())
        cpv = str(cupy.__version__)
    except Exception as exc:  # pragma: no cover - environment dependent
        name = f'unknown ({exc})'
        cuda = None
        cpv = None
    return {
        'gpu_name': name,
        'cuda_runtime_version': cuda,
        'cupy_version': cpv,
        'numpy_version': str(np.__version__),
        'python_version': sys.version,
        'platform': platform.platform(),
        'pid': int(os.getpid()),
    }


# ---------------------------------------------------------------------------
# Frozen item inventory / related-unrelated resolver (final design section 2.2).
# The Seed-3 Encoder codebook is reproducible on CPU from the frozen PN set in
# ``data/brain.npz``; no CUDA allocation is required to freeze the inventory.
# ---------------------------------------------------------------------------
def _load_encoder_channels(data: Path) -> np.ndarray:
    meta = np.load(Path(data) / 'brain.npz', allow_pickle=True)
    ct = np.asarray(meta['cell_type']).astype(str)
    return np.flatnonzero(np.char.find(ct, 'PN') >= 0).astype(np.int64)


def build_item_inventory(chars: list, data: Path) -> dict:
    pn = _load_encoder_channels(data)
    enc = Encoder(pn, len(chars), active=ENCODER_SPEC['active'],
                  drive=ENCODER_SPEC['drive'], window=ENCODER_SPEC['window'],
                  gamma=ENCODER_SPEC['gamma'], seed=ENCODER_SPEC['seed'],
                  position_codes=ENCODER_SPEC['position_codes'])
    codes = {ch: set(enc.codes[int(chars.index(ch))].tolist())
             for ch in ITEM_ALPHABET}
    j = np.zeros((len(ITEM_ALPHABET), len(ITEM_ALPHABET)), np.float64)
    for i, x in enumerate(ITEM_ALPHABET):
        for k, y in enumerate(ITEM_ALPHABET):
            inter = len(codes[x] & codes[y])
            union = len(codes[x] | codes[y])
            j[i, k] = (inter / union) if union else 0.0
    ai = ITEM_ALPHABET.index('a')
    others = [k for k in range(len(ITEM_ALPHABET)) if k != ai]
    rel = others[int(np.argmax(j[ai, others]))]
    unrel = others[int(np.argmin(j[ai, others]))]
    y_rel = ITEM_ALPHABET[rel]
    y_unrel = ITEM_ALPHABET[unrel]
    inventory = {
        'alphabet': list(ITEM_ALPHABET),
        'foils': list(FOILS),
        'cue': CUE,
        'cycle': list(CYCLE),
        'encoder': dict(ENCODER_SPEC),
        'related_pair': {'x': 'a', 'y': y_rel, 'J': float(j[ai, rel])},
        'unrelated_pair': {'x': 'a', 'y': y_unrel, 'J': float(j[ai, unrel])},
        'J_matrix_sha256': hashlib.sha256(np.ascontiguousarray(
            j, dtype=np.float64).tobytes()).hexdigest(),
    }
    return inventory, y_rel, y_unrel, j


# ---------------------------------------------------------------------------
# Frozen schedule table (final design sections 3.2-3.8).
# ---------------------------------------------------------------------------
def _program_from_items(items: list, g0: int) -> list:
    blocks = []
    for idx, item in enumerate(items):
        blocks.append([item, ITEM_TOKENS])
        blocks.append([None, g0 if idx == len(items) - 1
                       else INTER_ITEM_GAP])
    return blocks


def build_schedules(y_rel: str, y_unrel: str) -> dict:
    schedules = {}

    def add(sid, arm, items, g0, probes, context=None, scope=None):
        schedules[sid] = {
            'arm': arm,
            'schedule_id': sid,
            'items': list(items),
            'program_blocks': _program_from_items(list(items), int(g0)),
            'G0': int(g0),
            'prediction_context': context,
            'successor': SUCCESSOR.get(context) if context else None,
            'probes': probes,
            'write_lesion_scope': scope,
        }

    # Capacity: tested item 'a' always written last; age fixed at G0=32.
    for p in (1, 2, 3, 4, 6, 8):
        items = DISTRACTOR_ORDER[:p - 1] + ['a']
        probes = [
            {'probe_char': 'a', 'probe_role': 'own',
             'conditions': list(BASE_CONDITIONS)},
            {'probe_char': 'i', 'probe_role': 'foil',
             'conditions': list(BASE_CONDITIONS)},
            {'probe_char': 'j', 'probe_role': 'foil',
             'conditions': list(BASE_CONDITIONS)},
        ]
        scope = None
        if p == 4:
            probes[0]['conditions'] = list(BASE_CONDITIONS) + [
                'kc_mbon_readout_zero', 'mbon_central_readout_zero',
                WRITE_LESION]
            scope = WRITE_LESION_SCOPE['capacity_p4|a']
        add(f'capacity_p{p}', 'capacity', items, G0_DEFAULT, probes,
            scope=scope)

    # Forgetting core: N=4, ages 152/112/72/32.
    forget_items = ['b', 'c', 'd', 'a']
    add('forgetting_core', 'forgetting', forget_items, G0_DEFAULT,
        [{'probe_char': c, 'probe_role': 'own' if c in forget_items else 'foil',
          'conditions': list(BASE_CONDITIONS)}
         for c in ['b', 'c', 'd', 'a', 'i', 'j']])

    # Forgetting gap sweep on the two endpoints a (young) and b (old).
    for g0 in (8, 128):
        add(f'forgetting_gap_g{g0}', 'forgetting',
            forget_items, g0,
            [{'probe_char': 'a', 'probe_role': 'own',
              'conditions': list(BASE_CONDITIONS)},
             {'probe_char': 'b', 'probe_role': 'own',
              'conditions': list(BASE_CONDITIONS)}])

    # Passive reference: N=1, list [a]; G0=32 is shared with capacity p=1 and
    # is therefore not duplicated here (final design section 3.5).
    for g0 in (8, 72, 112, 152, 256):
        add(f'passive_g{g0}', 'passive', ['a'], g0,
            [{'probe_char': 'a', 'probe_role': 'own',
              'conditions': list(BASE_CONDITIONS)}])

    # Interference matrix: related / unrelated / repeated at matched dose.
    inter_specs = [
        ('interference_rep', ['a', 'a'], 'a'),
        ('interference_unrel', ['a', y_unrel], y_unrel),
        ('interference_rel', ['a', y_rel], y_rel),
    ]
    if tuple(INTERFERENCE_PROBES) != ('a', 'y'):
        raise RuntimeError('frozen interference probe set must be {a, y}')
    for sid, items, new_char in inter_specs:
        probes = [
            {'probe_char': 'a', 'probe_role': 'old',
             'conditions': list(BASE_CONDITIONS)},
            {'probe_char': new_char, 'probe_role': 'new',
             'conditions': list(BASE_CONDITIONS)},
        ]
        is_lesion_cfg = sid == INTERFERENCE_LESION['schedule_id']
        if is_lesion_cfg:
            if probes[0]['probe_char'] != INTERFERENCE_LESION['probe_char']:
                raise RuntimeError('frozen interference lesion probe mismatch')
            probes[0]['conditions'] = (list(BASE_CONDITIONS) +
                                       list(INTERFERENCE_LESION['conditions']))
        add(sid, 'interference', items, G0_DEFAULT, probes)
        schedules[sid]['probe_lesion_scope'] = (
            INTERFERENCE_LESION['scope'] if is_lesion_cfg else None)
        schedules[sid]['probe_lesion_conditions'] = (
            list(INTERFERENCE_LESION['conditions']) if is_lesion_cfg else [])
    add('interference_new_only', 'interference', [y_unrel], G0_DEFAULT,
        [{'probe_char': y_unrel, 'probe_role': 'new',
          'conditions': list(BASE_CONDITIONS)}])

    # Prediction: two primary contexts X in {a, c} plus the pre-registered
    # held-out contexts {b, d} (amendment 2026-09-18, confirmation only).  Each
    # context is the two cycle-predecessors of X followed by X.
    pred_contexts = (list(PREDICTION_V2['contexts_primary']) +
                     list(PREDICTION_V2['contexts_primary_holdout']))
    for ctx in pred_contexts:
        items = PREDECESSORS[ctx] + [ctx]
        probes = []
        for probe_char in [CUE] + list(PREDICTION_V2['alphabet']):
            conditions = list(BASE_CONDITIONS)
            if probe_char == CUE:
                conditions += list(EXTRA_Q_CONDITIONS)
                if ctx == 'a':
                    conditions += [WRITE_LESION]
            probes.append({'probe_char': probe_char,
                           'probe_role': 'cue' if probe_char == CUE else 'template',
                           'conditions': conditions})
        scope = WRITE_LESION_SCOPE['prediction_ctx_a|z'] if ctx == 'a' else None
        add(f'prediction_ctx_{ctx}', 'prediction', items, G0_DEFAULT, probes,
            context=ctx, scope=scope)
    return schedules


def _schedule_table_sha256(schedules: dict) -> str:
    return _hash_canonical(_jsonable(schedules))


# ---------------------------------------------------------------------------
# Cell plan (one cell = seed x schedule x probe x condition).
# ---------------------------------------------------------------------------
def _phase_lesion(condition: str):
    if condition == 'kc_mbon_readout_zero':
        return 'kc_mbon'
    if condition == 'mbon_central_readout_zero':
        return 'mbon_central'
    return None


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
                        'schedule_id': sid,
                        'prediction_context': sched['prediction_context'],
                        'successor': sched['successor'],
                        'list': list(sched['items']),
                        'program_blocks': [list(b) for b in
                                           sched['program_blocks']],
                        'G0': int(sched['G0']),
                        'probe_char': probe['probe_char'],
                        'probe_role': probe['probe_role'],
                        'condition': condition,
                        'phase_lesion': _phase_lesion(condition),
                        'write_lesion': _write_lesion(condition),
                        'write_lesion_scope': sched['write_lesion_scope']
                        if condition == WRITE_LESION else None,
                        'probe_lesion_scope': (
                            sched.get('probe_lesion_scope')
                            if condition in sched.get(
                                'probe_lesion_conditions', []) else None),
                    })
    return plan


def _input_spec(cell: dict, external_gap: int = 0) -> dict:
    return {
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
        'schedule_id': str(cell['schedule_id']),
        'G0': int(cell['G0']),
        'prediction_context': cell['prediction_context'],
        'phase_lesion': cell['phase_lesion'],
        'write_lesion': cell['write_lesion'],
        'write_lesion_scope': cell['write_lesion_scope'],
        'probe_lesion_scope': cell.get('probe_lesion_scope'),
    }


def _cell_id(cell: dict) -> str:
    role = cell['probe_role']
    return (f"s{cell['seed']}-{cell['schedule_id']}-{role}-"
            f"{cell['probe_char']}-{cell['condition']}")


# ---------------------------------------------------------------------------
# Global config hash (E3 keys + frozen hashes + schedule/lesion coverage).
# ---------------------------------------------------------------------------
def _global_config_coverage(args, inventory_sha, resolver_sha, prediction_sha,
                            forbidden_sha, schedule_sha, related_pair,
                            unrelated_pair) -> dict:
    coverage = {k: getattr(args, k) for k in CONFIG_HASH_KEYS}
    coverage.update({
        'item_inventory_sha256': inventory_sha,
        'resolver_sha256': resolver_sha,
        'prediction_sha256': prediction_sha,
        'forbidden_union_sha256': forbidden_sha,
        'schedule_table_sha256': schedule_sha,
        'related_pair': related_pair,
        'unrelated_pair': unrelated_pair,
        'write_lesion_conditions': [WRITE_LESION],
        'write_lesion_scope': WRITE_LESION_SCOPE,
        'prediction_contexts': list(PREDICTION_V2['contexts_primary']),
        'prediction_contexts_holdout': list(
            PREDICTION_V2['contexts_primary_holdout']),
        'interference_probes': list(INTERFERENCE_PROBES),
        'interference_lesion': INTERFERENCE_LESION,
    })
    return coverage


# ---------------------------------------------------------------------------
# Frozen statistic capture wrappers (identical mechanism to E3).
# ---------------------------------------------------------------------------
_ORIG_FAST = mpp._cuda_step_no_host_copy
_ORIG_PRE_RESET = mpp._cuda_step_record_pre_reset
_CAPTURE: dict = {'active': False, 'hasher': None, 'steps': 0, 'spikes': 0,
                  'sizes_hasher': None}


def _fired_bytes(fired):
    host = (fired.get() if hasattr(fired, 'get') else np.asarray(fired))
    return np.ascontiguousarray(host, dtype=np.int64)


def _capture_fired(fired) -> None:
    if not _CAPTURE['active']:
        return
    host = _fired_bytes(fired)
    _CAPTURE['hasher'].update(host.tobytes())
    _CAPTURE['sizes_hasher'].update(int(host.size).to_bytes(8, 'little'))
    _CAPTURE['steps'] += 1
    _CAPTURE['spikes'] += int(host.size)


def _wrapped_fast(brain):
    fired = _ORIG_FAST(brain)
    _capture_fired(fired)
    return fired


def _wrapped_pre_reset(brain, record_indices=()):
    fired, v_pre = _ORIG_PRE_RESET(brain, record_indices)
    _capture_fired(fired)
    return fired, v_pre


mpp._cuda_step_no_host_copy = _wrapped_fast
mpp._cuda_step_record_pre_reset = _wrapped_pre_reset


def _capture_begin() -> None:
    _CAPTURE['hasher'] = hashlib.sha256()
    _CAPTURE['sizes_hasher'] = hashlib.sha256()
    _CAPTURE['steps'] = 0
    _CAPTURE['spikes'] = 0
    _CAPTURE['active'] = True


def _capture_end() -> dict:
    _CAPTURE['active'] = False
    return {
        'sha256': _CAPTURE['hasher'].hexdigest(),
        'step_sizes_sha256': _CAPTURE['sizes_hasher'].hexdigest(),
        'n_steps': int(_CAPTURE['steps']),
        'n_spikes': int(_CAPTURE['spikes']),
    }


def _build_args(ns) -> None:
    ns.probe_tokens = 1
    ns.probe_plasticity = False
    ns.gap_plasticity = False
    ns.mod_perm_seed = 271828
    # K1 (H-null scope amendment, 2026-09-18): the frozen probe module's
    # ``mbon_source_shuffle`` branch reads ``int(args.topology_null_seed)``
    # (``mb_persistent_memory_probe.py:374-375``); without this attribute the
    # whole E4B grid aborts with ``AttributeError`` before cell 1 while
    # ``--dry-run``/``--selfcheck-writeback`` never construct a condition.  This
    # value matches the prereg section 4 frozen ``topology_null_seed=271828`` and
    # only lets the explicitly NON-GATING ``mbon_source_shuffle`` telemetry
    # construct.  It is deliberately NOT added to ``CONFIG_HASH_KEYS``, so no
    # frozen hash (schedule_table/prediction/config/resolver/inventory) moves.
    ns.topology_null_seed = 271828
    ns.dan_pulse_unpaired = False
    ns.save_probe_vectors = True
    ns.record_probe_profile = True
    ns.record_fingerprints = True
    ns.save_modulation_after_write = True
    ns.save_modulation_preprobe = False
    ns.save_block_snapshots = True
    ns.skip_probe_weight_writeback = False
    ns.item_tokens = ITEM_TOKENS
    ns.inter_item_gap = INTER_ITEM_GAP


# ---------------------------------------------------------------------------
# Static bridge / local no-feedback checks (no CUDA).
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bridge_e3_static(chars: list, manifest_path: Path) -> dict:
    """Re-derive the frozen E3 config and A0 input hashes without CUDA."""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    coverage = manifest['config_hash_coverage']
    config_sha = _hash_canonical(coverage)
    input_hashes = {}
    cells = {c['cell_id']: c for c in manifest.get('cells', [])}
    for seed in E3_BRIDGE_SEEDS:
        cid = f's{seed}-A0-e-intact'
        reference = cells.get(cid, {}).get('input_sha256')
        program_ids = [[None if c is None else int(chars.index(c)), int(n)]
                       for c, n in E3_A0]
        spec = {
            'seed': int(seed),
            'program_chars': program_ids,
            'probe_char_id': int(chars.index('e')),
            'noise_hz': 0.05,
            'gap_tokens': 32,
            'condition': 'intact',
        }
        recomputed = _hash_canonical(spec)
        input_hashes[cid] = {
            'recomputed': recomputed,
            'reference': reference,
            'match': bool(reference is not None and recomputed == reference),
        }
    config_match = bool(config_sha == E3_EXPECTED_CONFIG_SHA256 and
                        manifest.get('config_sha256') == config_sha)
    return {
        'e3_manifest': str(manifest_path),
        'e3_config_sha256_expected': E3_EXPECTED_CONFIG_SHA256,
        'e3_config_sha256_recomputed': config_sha,
        'e3_config_sha256_match': config_match,
        'e3_input_hashes': input_hashes,
        'e3_input_hashes_match': bool(
            all(v['match'] for v in input_hashes.values())),
        'pass': bool(config_match and
                     all(v['match'] for v in input_hashes.values())),
    }


def selfcheck_writeback() -> dict:
    """Bookkeeping-only proof that ``write_back=False`` skips only the CSR line.

    Uses a tiny NumPy stand-in for the CUDA CSR matrix so the exact
    ``KCMBONPlasticity.update`` code path can be exercised without any GPU.
    """
    from mb_plasticity import KCMBONPlasticity

    def make():
        p = object.__new__(KCMBONPlasticity)
        p.xp = np
        p.mode = 'engineering'
        p.eta = np.float32(0.02)
        p.max_modulation = 0.9
        p.eligibility_mix = 0.0
        p.kc_ids = np.arange(3, dtype=np.int64)
        p.engineering_axis = np.asarray([1.0, -1.0], np.float32)
        p.edge_mbon_slot_gpu = np.asarray([0, 0, 1, 1], np.int64)
        p.edge_kc_slot_gpu = np.asarray([0, 1, 2, 0], np.int64)
        p.edge_pos_gpu = np.asarray([0, 1, 2, 3], np.int64)
        p.w0_gpu = np.asarray([0.5, 0.4, 0.3, 0.2], np.float32)
        p.modulation = np.zeros(4, np.float32)
        p.kc_eligibility = np.zeros(3, np.float32)
        p.last_gate = np.zeros(2, np.float32)
        p._elig_decay = np.float32(np.exp(-0.12 / 0.5))
        p._weight_decay = np.float32(np.exp(-0.12 / 16.0))
        brain = type('FakeBrain', (), {})()
        brain._W = type('FakeW', (), {})()
        brain._W.data = np.ones(4, np.float32)
        p.brain = brain
        return p

    kc = np.asarray([3.0, 0.0, 1.0], np.float32)
    with_wb = make()
    without_wb = make()
    with_wb.update(kc, teaching_signal=0.7, write_back=True)
    without_wb.update(kc, teaching_signal=0.7, write_back=False)
    mod_equal = bool(np.array_equal(with_wb.modulation, without_wb.modulation))
    elig_equal = bool(np.array_equal(with_wb.kc_eligibility,
                                     without_wb.kc_eligibility))
    csr_with = with_wb.brain._W.data.copy()
    csr_without = without_wb.brain._W.data.copy()
    expected = with_wb.w0_gpu * (np.float32(1.0) + with_wb.modulation)
    csr_only_difference = bool(
        np.allclose(csr_with, expected, rtol=0.0, atol=0.0) and
        np.array_equal(csr_without, np.ones(4, np.float32)) and
        not np.array_equal(csr_with, csr_without))
    # B3 boundary semantics: during the write phase the CSR stays at the
    # snapshotted baseline (the write-back is skipped), and at the write->gap
    # boundary the exact frozen plasticity readout ``w0*(1+modulation)`` is
    # materialized so the probe sees the accumulated local memory.
    phase = make()
    baseline = phase.brain._W.data.copy()
    phase.update(kc, teaching_signal=0.7, write_back=False)
    csr_during_write = phase.brain._W.data.copy()
    baseline_held_during_write = bool(
        np.array_equal(csr_during_write, baseline))
    expected_materialized = phase.w0_gpu * (np.float32(1.0) + phase.modulation)
    phase.brain._W.data[phase.edge_pos_gpu] = expected_materialized
    materialized = phase.brain._W.data.copy()
    materialized_matches_expected = bool(
        np.array_equal(materialized, expected_materialized))
    return {
        'modulation_byte_identical': mod_equal,
        'kc_eligibility_byte_identical': elig_equal,
        'csr_only_difference': csr_only_difference,
        'baseline_held_during_write': baseline_held_during_write,
        'materialized_matches_expected': materialized_matches_expected,
        'write_back_true_csr': csr_with.tolist(),
        'write_back_false_csr': csr_without.tolist(),
        'materialized_csr': materialized.tolist(),
        'pass': bool(mod_equal and elig_equal and csr_only_difference and
                     baseline_held_during_write and
                     materialized_matches_expected),
    }


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=str(ROOT / 'corpus' / 'tinyshakespeare.txt'))
    ap.add_argument('--data', default=None)
    ap.add_argument('--seeds', default=','.join(str(s) for s in FRESH_SEEDS),
                    help='comma-separated fresh seeds; default 20270101..20270110')
    ap.add_argument('--repeats-per-cell', type=int, default=1)
    ap.add_argument('--noise-hz', type=float, default=0.05)
    ap.add_argument('--gap-tokens', type=int, default=0,
                    help='external post-program silent gap; E4B program already '
                         'ends with the G0 silent block, so default 0')
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
    ap.add_argument('--eta', type=float, default=0.02)
    ap.add_argument('--eligibility-tau', type=float, default=0.5)
    ap.add_argument('--weight-tau', type=float, default=16.0)
    ap.add_argument('--max-modulation', type=float, default=0.9)
    ap.add_argument('--eligibility-mix', type=float, default=0.0)
    ap.add_argument('--target-top', type=int, default=256)
    ap.add_argument('--target-selection', default='direct')
    ap.add_argument('--graded-tol', type=float, default=1e-6)
    ap.add_argument('--graded-ulp', type=int, default=16)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--prefix', default=DEFAULT_PREFIX)
    ap.add_argument('--limit-cells', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true',
                    help='print the exact cell/run plan and exit (no CUDA)')
    ap.add_argument('--selfcheck-writeback', action='store_true',
                    help='run the local no-feedback write_back identity check')
    ap.add_argument('--bridge-e3', action='store_true',
                    help='static E3 config/input-hash bridge check')
    ap.add_argument('--bridge-e3-cuda', action='store_true',
                    help='(requires GPU) replay the frozen E3 A0 row; never run '
                         'by this build pass')
    ap.add_argument('--e3-manifest', default=E3_MANIFEST,
                    help='frozen E3 manifest used by the static bridge check')
    args = ap.parse_args()

    _build_args(args)
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
        raise RuntimeError(f'fresh E4B seeds overlap the forbidden union: {overlap}')
    if seeds != FRESH_SEEDS:
        raise RuntimeError(
            f'fresh E4B cohort must equal the frozen ten-value list '
            f'{FRESH_SEEDS}; got {seeds}')

    chars, _, _, _ = load_corpus(args.corpus, 128, 1, 1)
    chars = list(chars)
    for ch in (ITEM_ALPHABET + FOILS + [CUE]):
        if ch not in chars:
            raise ValueError(f'item {ch!r} outside corpus vocabulary')
    args.vocab = len(chars)

    # ---- frozen hashes (computed before any FlyBrain/CUDA allocation) ----
    inventory, y_rel, y_unrel, _j = build_item_inventory(chars, args.data)
    resolver_sha = _hash_canonical(RESOLVER_V1)
    inventory_sha = _hash_canonical(inventory)
    prediction_sha = _hash_canonical(PREDICTION_V2)
    forbidden_sha = _hash_canonical(FORBIDDEN_UNION)
    if inventory_sha == '' or resolver_sha == '' or prediction_sha == '':
        raise RuntimeError('frozen E4B hash missing')
    schedules = build_schedules(y_rel, y_unrel)
    schedule_sha = _schedule_table_sha256(schedules)
    related_pair = {'x': 'a', 'y': y_rel, 'J': inventory['related_pair']['J']}
    unrelated_pair = {'x': 'a', 'y': y_unrel,
                      'J': inventory['unrelated_pair']['J']}
    config_coverage = _global_config_coverage(
        args, inventory_sha, resolver_sha, prediction_sha, forbidden_sha,
        schedule_sha, related_pair, unrelated_pair)
    config_sha = _hash_canonical(config_coverage)

    cell_plan = build_cell_plan(seeds, schedules)
    for cell in cell_plan:
        cell['probe_char_id'] = int(chars.index(cell['probe_char']))
        cell['probe_id'] = int(cell['probe_char_id'])
        cell['repeats'] = int(args.repeats)
        spec = _input_spec(cell, int(args.gap_tokens))
        cell['input_spec'] = spec
        cell['input_sha256'] = _input_sha256(spec)
        cell['cell_config'] = _cell_config(cell, config_sha)
        cell['cell_config_sha256'] = _hash_canonical(cell['cell_config'])
        cell['cell_id'] = _cell_id(cell)
    n_cells = len(cell_plan)
    runs_total = n_cells * int(args.repeats)
    episodes_per_run = 3
    per_arm = {}
    for cell in cell_plan:
        per_arm[cell['arm']] = per_arm.get(cell['arm'], 0) + 1

    print(f'seeds={seeds}', flush=True)
    print(f'fresh_seeds={FRESH_SEEDS}', flush=True)
    print(f'n_forbidden_union={len(FORBIDDEN_UNION)} '
          f'forbidden_union_sha256={forbidden_sha}', flush=True)
    print(f'resolver_sha256={resolver_sha}', flush=True)
    print(f'item_inventory_sha256={inventory_sha}', flush=True)
    print(f'prediction_sha256={prediction_sha}', flush=True)
    print(f'schedule_table_sha256={schedule_sha}', flush=True)
    print(f'related_pair={related_pair} unrelated_pair={unrelated_pair}',
          flush=True)
    print(f'config_sha256={config_sha}', flush=True)
    print(f'config_hash_covers={sorted(config_coverage)}', flush=True)
    print(f'cells_per_seed={n_cells // max(len(seeds), 1)} '
          f'cells={n_cells} repeats_per_cell={args.repeats} '
          f'runs={runs_total} episodes_per_run={episodes_per_run} '
          f'episodes_total={runs_total * episodes_per_run}', flush=True)
    print(f'cells_per_arm={per_arm}', flush=True)

    write_lesion_cells = [c for c in cell_plan if c['condition'] == WRITE_LESION]
    print(f'required_write_lesion_cells={len(write_lesion_cells)}', flush=True)
    for cell in write_lesion_cells:
        print(f"  write_lesion {cell['cell_id']} scope={cell['write_lesion_scope']}",
              flush=True)

    probe_lesion_cells = [c for c in cell_plan if c.get('probe_lesion_scope')]
    print(f'frozen_probe_lesion_cells={len(probe_lesion_cells)} '
          f'expected={2 * len(seeds)}', flush=True)
    for cell in probe_lesion_cells:
        print(f"  probe_lesion {cell['cell_id']} "
              f"scope={cell['probe_lesion_scope']}", flush=True)

    e3_manifest_path = Path(args.e3_manifest)
    if not e3_manifest_path.is_absolute():
        e3_manifest_path = ROOT / e3_manifest_path
    bridge = bridge_e3_static(chars, e3_manifest_path)
    print(f"e3_bridge_config_sha256_recomputed={bridge['e3_config_sha256_recomputed']}",
          flush=True)
    print(f"e3_bridge_config_sha256_match={bridge['e3_config_sha256_match']}",
          flush=True)
    print(f"e3_bridge_input_hashes_match={bridge['e3_input_hashes_match']}",
          flush=True)
    if not bridge['pass']:
        raise RuntimeError('E3 bridge mismatch; failing closed before any CUDA')

    if args.selfcheck_writeback:
        check = selfcheck_writeback()
        print(f'selfcheck_writeback={json.dumps(check, sort_keys=True)}',
              flush=True)
        if not check['pass']:
            raise RuntimeError('write_back local identity self-check failed')

    if args.bridge_e3 and not args.bridge_e3_cuda:
        print('BRIDGE-E3 static check ok; CUDA replay requires --bridge-e3-cuda '
              'and a GPU (not run in this pass).', flush=True)
        return

    if args.dry_run:
        if n_cells // max(len(seeds), 1) != 142:
            raise RuntimeError(
                f'primary+held-out plan must be 142 cells/seed; got '
                f'{n_cells // max(len(seeds), 1)}')
        if runs_total != 1420 or runs_total * episodes_per_run != 4260:
            raise RuntimeError(
                f'primary plan must be 1,420 cells / 4,260 episodes; got '
                f'{runs_total} / {runs_total * episodes_per_run}')
        pred_cells = [c for c in cell_plan if c['arm'] == 'prediction']
        if len(pred_cells) != 57 * len(seeds):
            raise RuntimeError(
                'prediction arm must be 57 cells/seed (29 primary + 28 '
                f'held-out); got {len(pred_cells) // max(len(seeds), 1)}/seed')
        holdout_cells = [c for c in pred_cells
                         if c['prediction_context'] in
                         tuple(PREDICTION_V2['contexts_primary_holdout'])]
        if len(holdout_cells) != 28 * len(seeds):
            raise RuntimeError(
                'held-out prediction arm must be 28 cells/seed; got '
                f'{len(holdout_cells) // max(len(seeds), 1)}/seed')
        if len(write_lesion_cells) != 2 * len(seeds):
            raise RuntimeError(
                'exactly two required write-phase lesion cells per seed')
        if len(probe_lesion_cells) != 2 * len(seeds):
            raise RuntimeError(
                'exactly two frozen probe-only lesion cells per seed '
                '(interference_rel old a: kc_mbon_readout_zero, '
                'mbon_central_readout_zero)')
        if any(c['schedule_id'] != INTERFERENCE_LESION['schedule_id'] or
               c['probe_char'] != INTERFERENCE_LESION['probe_char'] or
               c['probe_role'] != INTERFERENCE_LESION['probe_role'] or
               c['condition'] not in INTERFERENCE_LESION['conditions']
               for c in probe_lesion_cells):
            raise RuntimeError('probe-only lesion scope deviates from the '
                               'frozen interference scope')
        print('DRY-RUN cell plan '
              '(seed, arm, schedule, probe_role, probe, condition, G0, '
              'phase_lesion, write_lesion, input_sha256):', flush=True)
        for cell in cell_plan:
            print(f"  {cell['seed']} {cell['arm']:<12} {cell['schedule_id']:<24} "
                  f"{cell['probe_role']:<8} {cell['probe_char']:>1} "
                  f"{cell['condition']:<26} G0={cell['G0']:>3} "
                  f"phase={str(cell['phase_lesion']):<12} "
                  f"write={str(cell['write_lesion']):<8} "
                  f"in={cell['input_sha256'][:12]} "
                  f"cfg={cell['cell_config_sha256'][:12]}", flush=True)
        print('DRY-RUN ok; no CUDA episode executed.', flush=True)
        return

    # ------------------------------------------------------------------
    # CUDA path (never launched by this build pass).
    # ------------------------------------------------------------------
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

    if args.bridge_e3_cuda:
        _run_e3_cuda_bridge(args, chars, brain0, masks, central_ids,
                            fixed_direction)
        return

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
    for seed in seeds:
        args.seed = int(seed)
        states = {cond: _new_condition(args, cond, masks, args.vocab,
                                       central_ids)
                  for cond in ALL_CONDITIONS}
        for cond in ALL_CONDITIONS:
            states[cond]['fixed_direction'] = fixed_direction
        for cell in [c for c in cell_plan if c['seed'] == seed]:
            if args.limit_cells and done >= args.limit_cells:
                break
            state = states[cell['condition']]
            program_ids = [[None if t is None else int(chars.index(t)), int(n)]
                           for t, n in cell['program_blocks']]
            cid = cell['cell_id']
            _capture_begin()
            write = _episode(state, args, [], cell['probe_id'],
                             int(args.gap_tokens), True,
                             write_program=program_ids)
            ev_write = _capture_end()
            _capture_begin()
            no_write = _episode(state, args, [], cell['probe_id'],
                                int(args.gap_tokens), False,
                                write_program=program_ids)
            ev_no_write = _capture_end()
            _capture_begin()
            replay = _episode(state, args, [], cell['probe_id'],
                              int(args.gap_tokens), False,
                              write_program=program_ids)
            ev_replay = _capture_end()
            pair = _pair_metrics(write, no_write, graded_tol=args.graded_tol,
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
            pair.update({
                'cell_id': cid,
                'arm': cell['arm'],
                'schedule_id': cell['schedule_id'],
                'prediction_context': cell['prediction_context'],
                'successor': cell['successor'],
                'item_list': list(cell['list']),
                'G0': int(cell['G0']),
                'program_blocks': [list(b) for b in cell['program_blocks']],
                'probe_char': cell['probe_char'],
                'probe_char_id': int(cell['probe_id']),
                'probe_role': cell['probe_role'],
                'condition': cell['condition'],
                'phase_lesion': cell['phase_lesion'],
                'write_lesion': cell['write_lesion'],
                'write_lesion_scope': cell['write_lesion_scope'],
                'probe_lesion_scope': cell.get('probe_lesion_scope'),
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
            write['edge_norm_after_write_sha256'] = _hash_scalar_f64(
                write.get('edge_norm_after_write', 0.0))
            _strip_vectors(pair, arrays, cid)
            runs.append(_jsonable(pair))
            done += 1
            print(f'[{done}/{n_cells}] s={seed} {cell["schedule_id"]} '
                  f'{cell["probe_role"]}->{cell["probe_char"]} '
                  f'{cell["condition"]:26s} '
                  f'in={cell["input_sha256"][:12]}', flush=True)

    input_groups: dict = {}
    for rec in runs:
        key = (rec['repeat_seed'], rec['schedule_id'], rec['probe_char_id'],
               rec['condition'])
        input_groups.setdefault(key, set()).add(rec['input_sha256'])
    inputs_bitwise_identical = bool(len(runs) > 0 and all(
        len(v) == 1 for v in input_groups.values()))

    np.savez(raw_npz, **arrays)
    raw = {
        'experiment': 'E4B_sequential_capacity',
        'date': '2026-09-18',
        'handoff': 'docs/OPENCODE_SECOND_DS_E4B_BUILD_HANDOFF_20260918.md',
        'design': ('models/research_results/opencode_analysis/'
                   'deepseek_e4b_design_final_20260918.md'),
        'prereg': 'docs/OPENCODE_E4B_SEQUENTIAL_PREREG_20260918.md',
        'device': device,
        'config': _jsonable(vars(args)),
        'config_sha256': config_sha,
        'config_hash_keys': list(CONFIG_HASH_KEYS),
        'config_hash_coverage': _jsonable(config_coverage),
        'resolver_sha256': resolver_sha,
        'item_inventory_sha256': inventory_sha,
        'prediction_sha256': prediction_sha,
        'forbidden_union_sha256': forbidden_sha,
        'schedule_table_sha256': schedule_sha,
        'resolver': _jsonable(RESOLVER_V1),
        'item_inventory': _jsonable(inventory),
        'prediction': _jsonable(PREDICTION_V2),
        'forbidden_union': list(FORBIDDEN_UNION),
        'fresh_seeds': list(seeds),
        'related_pair': related_pair,
        'unrelated_pair': unrelated_pair,
        'schedules': _jsonable(schedules),
        'cells_per_arm': per_arm,
        'n_cells': int(n_cells),
        'n_runs': int(len(runs)),
        'episodes_per_run': int(episodes_per_run),
        'episodes_total': int(len(runs) * episodes_per_run),
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
        'raw_npz': str(raw_npz),
        'cells': runs,
        'elapsed_seconds': time.time() - t_start,
    }
    with raw_json.open('w', encoding='utf-8') as handle:
        json.dump(raw, handle, indent=2, ensure_ascii=False)
    print(f'saved {raw_json}')
    print(f'saved {raw_npz}')


def _strip_vectors(cell: dict, arrays: dict, cell_id: str) -> None:
    for branch in ('write_row', 'no_write_row', 'replay_row'):
        row = cell.get(branch)
        if not isinstance(row, dict):
            continue
        vec = row.pop('modulation_after_write_vector', None)
        if vec is not None:
            arrays[f'{cell_id}|{branch}|modulation_after_write_vector'] = (
                np.asarray(vec, dtype=np.float32))
        for snap in row.get('write_block_snapshots') or []:
            snap_vec = snap.pop('modulation_vector', None)
            if snap_vec is not None:
                arrays[f'{cell_id}|{branch}|block{snap.get("block")}'
                       f'|modulation_vector'] = np.asarray(
                           snap_vec, dtype=np.float32)
        pre = row.pop('modulation_preprobe_vector', None)
        if pre is not None:
            arrays[f'{cell_id}|{branch}|modulation_preprobe_vector'] = (
                np.asarray(pre, dtype=np.float32))


def _run_e3_cuda_bridge(args, chars, brain0, masks, central_ids,
                        fixed_direction):
    """Replay the frozen E3 A0/G=32 row and compare to frozen E3 raw.

    Requires a GPU.  This build pass never calls it (no ``--bridge-e3-cuda``).
    """
    raw_path = (ROOT / 'models/research_results/e3_continual_update_20260918'
                / 'mb_e3_continual_update_20260918.json')
    e3 = json.loads(raw_path.read_text(encoding='utf-8'))
    e3_idx = {(int(c['repeat_seed']), c['program'], c['probe_char'],
               c['condition']): c
              for c in e3.get('cells', [])}
    program_ids = [[None if c is None else int(chars.index(c)), int(n)]
                   for c, n in E3_A0]
    probe_id = int(chars.index('e'))
    report = {}
    for seed in E3_BRIDGE_SEEDS:
        args.seed = int(seed)
        states = {cond: _new_condition(args, cond, masks, args.vocab,
                                       central_ids)
                  for cond in ('intact', 'write_off')}
        for state in states.values():
            state['fixed_direction'] = fixed_direction
        for cond in ('intact', 'write_off'):
            state = states[cond]
            _capture_begin()
            write = _episode(state, args, [], probe_id, 32, True,
                             write_program=program_ids)
            ev_write = _capture_end()
            _capture_begin()
            no_write = _episode(state, args, [], probe_id, 32, False,
                                write_program=program_ids)
            ev_no_write = _capture_end()
            reference = e3_idx.get((int(seed), 'A0', 'e', cond), {})
            ref_write = reference.get('write_row', {})
            ref_no_write = reference.get('no_write_row', {})
            report[f'{seed}:{cond}'] = {
                'write_events_sha256_match': bool(
                    reference and ev_write['sha256'] ==
                    (reference.get('events_write') or {}).get('sha256')),
                'no_write_events_sha256_match': bool(
                    reference and ev_no_write['sha256'] ==
                    (reference.get('events_no_write') or {}).get('sha256')),
                'write_probe_mbon_l2_close': bool(reference and abs(
                    float(write['probe_mbon_l2']) -
                    float(ref_write.get('probe_mbon_l2', np.nan))) <= 1e-5),
                'write_probe_central_l2_close': bool(reference and abs(
                    float(write['probe_central_l2']) -
                    float(ref_write.get('probe_central_l2', np.nan))) <= 1e-5),
                'no_write_probe_mbon_l2_close': bool(reference and abs(
                    float(no_write['probe_mbon_l2']) -
                    float(ref_no_write.get('probe_mbon_l2', np.nan))) <= 1e-5),
                'no_write_probe_central_l2_close': bool(reference and abs(
                    float(no_write['probe_central_l2']) -
                    float(ref_no_write.get('probe_central_l2', np.nan))) <= 1e-5),
            }
            report[f'{seed}:{cond}']['matches_frozen'] = bool(
                all(v for k, v in report[f'{seed}:{cond}'].items()))
    print(f'e3_cuda_bridge={json.dumps(report, sort_keys=True)}', flush=True)


if __name__ == '__main__':
    main()
