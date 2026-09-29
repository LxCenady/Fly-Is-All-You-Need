"""E4B sequential summarizer, manifest and pre-registered gates.

Reads the raw produced by ``lm/mb_e4b_sequential.py`` (JSON plus the binary
``.raw.npz`` of modulation vectors) and implements the frozen statistics and
gates of
``models/research_results/opencode_analysis/deepseek_e4b_design_final_20260918.md``:

  * content-specific recall resolver ``content_specific_recall_v1`` with two
    never-written foils ``i,j`` and margin ``kappa=1.25``;
  * sequential capacity / a-priori saturation (H-cap);
  * forgetting fits and the matched passive-reference decay/overwrite
    decomposition (H-forget, H-overwrite);
  * the related/unrelated/repeated interference matrix and specificity
    statistic (H-interf);
  * the bounded finite-alphabet horizon-1 template-rank prediction endpoint
    with an exact stratified successor-label exchangeability null
    (``PREDICTION_V2``, prereg amendment 2026-09-18) and the required causal
    collapses (H-pred);
  * the writer tolerance/median gate (G8) and validity gates G0-G14.

No dynamics are recomputed; the raw is read-only.

Level discipline (B7).  The final design registers two *different* primary
levels for two different claim families:

  * storage claims H-cap / H-forget / H-overwrite / H-interf: primary
    ``v_pre_mbon`` (97), secondary ``v_pre_central`` (256);
  * prediction claim H-pred: primary ``v_pre_central`` (256), secondary
    ``v_pre_mbon``.

The two levels are always reported separately and are never averaged with each
other.  Analysis gates fail closed when an arm is invalid (B4/B6).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import permutations
from pathlib import Path

import numpy as np

SETTLE_LO, SETTLE_HI = 6, 11
MARGIN_KAPPA = 1.25
FOIL_FACTOR = 1.25
FLOOR_FACTOR = 100.0

# B7: claim-specific primary/secondary levels (see module docstring).
STORAGE_PRIMARY_LEVEL = 'v_pre_mbon'
STORAGE_SECONDARY_LEVEL = 'v_pre_central'
PREDICTION_PRIMARY_LEVEL = 'v_pre_central'
PREDICTION_SECONDARY_LEVEL = 'v_pre_mbon'

DT_TOKEN = 0.02 * 6.0
LOAD_GRID = (1, 2, 3, 4, 6, 8)
FORGET_POSITIONS = {'b': 152, 'c': 112, 'd': 72, 'a': 32}
PASSIVE_AGES = {'passive_g8': 8, 'passive_g72': 72, 'passive_g112': 112,
                'passive_g152': 152, 'passive_g256': 256}
CAPACITY_ANCHOR_SID = 'capacity_p1'

# B6: age/dose-matched interference denominator.  The old item in ``[a,y]``
# has age 72 tokens; capacity ``p=1`` is age 32, so the matched passive
# reference at age 72 (N=1, list ``[a]``, G0=72) is the frozen denominator.
# Capacity ``p=1`` is reported only as the design-literal cross-check.
INTERF_DENOM_SID = 'passive_g72'
INTERF_DENOM_AGE = 72
INTERF_DENOM_ALT_SID = CAPACITY_ANCHOR_SID

ALPHABET4 = ['a', 'b', 'c', 'd']
CYCLE_SUCCESSOR = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'a'}
PREDICTION_CONTEXTS = ('a', 'c')
# PREDICTION_V2 (prereg amendment 2026-09-18): pre-registered held-out contexts
# are confirmation-only and never feed the primary decision or tuning.
PREDICTION_CONTEXTS_HOLDOUT = ('b', 'd')
WRITE_LESION = 'kc_mbon_writezero_restore'
PROBE_LESION_CONDITIONS = ('kc_mbon_readout_zero', 'mbon_central_readout_zero')

# PREDICTION_V2 frozen constants (must match the runner PREDICTION_V2 object).
# The exact stratified successor-label exchangeability test uses one
# Binomial(len(seeds), 0.25) upper tail per primary context; a pooled 20-cell
# binomial (two contexts pooled as 20 iid trials) is explicitly forbidden
# because it is pseudo-replication and anti-conservative.
TIE_TOL = 1e-9
NORM_FLOOR = 'max(10*F_sum, 3*SD_R)'
NORM_FLOOR_ABS = 1e-12
ALPHA_FAMILY = 0.05
ALPHA_PER_CONTEXT = 0.025
MIN_HITS_PER_CONTEXT = 6
NULL_MODEL = ('hit_{X,s} ~ Bernoulli(0.25); Binomial(10,0.25) per context; '
              'seed is the sampling unit')
INTERFER_CONFIGS = {'rep': 'interference_rep', 'unrel': 'interference_unrel',
                    'rel': 'interference_rel'}

# B2: frozen interference probe-only lesion scope; must match the runner.
INTERFERENCE_PROBES = ('a', 'y')
INTERFERENCE_LESION = {
    'schedule_id': 'interference_rel',
    'probe_char': 'a',
    'probe_role': 'old',
    'conditions': list(PROBE_LESION_CONDITIONS),
    'scope': 'interference_rel_old_a_probe_only',
}

FRESH_SEEDS = list(range(20270101, 20270111))

EXPECTED_NEURONS = 166700
EXPECTED_EDGES = 25088107
EXPECTED_EDGE_COUNTS = {
    'kc_mbon': 61210, 'dan_mbon': 3196, 'mbon_central': 28241,
    'mbon_dan': 2618, 'mbon_all': 39278, 'mbon_descending': 409,
    'mbon_cb_intrinsic': 38554,
}
WRITER_MAX_STATES = 2
WRITER_MODAL_MIN_FRACTION = 0.5
WRITER_TOL_MAX_ABS = 5.0e-2
WRITER_TOL_REL_L2 = 0.10

# B4 fail-closed G3 constants.
G3_ZERO_TOL = 1.0e-6
G3_SEPARATION_MIN = 2.0

# B6 uncertainty / support constants.
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 271828
INTERF_SPEC_MIN = 0.2
INTERF_NEW_MIN = 0.5
MIN_SEEDS = 8
CAPACITY_RETAIN_SEEDS = 9

# Frozen hash contract (B1 amendment A1 updates the config/schedule hashes after
# the B2 scope freeze; the resolver/inventory/prediction/forbidden hashes are
# unchanged because those canonical objects did not move).
# 2026-09-18 successor-null amendment: PREDICTION_V2 replaces PREDICTION_V1, so
# the prediction, schedule-table and config hashes are re-frozen here.
RESOLVER_SHA256 = 'a92ca3f256ce1db8d79eed1c948802d59f03d7c445d71724b5764e4de1ed66aa'
ITEM_INVENTORY_SHA256 = '5fdb937ae89aab5a68caaa47a271217153ea96568b9b9f5ec15c2f0ec2ee2761'
PREDICTION_SHA256 = '8c18f0a9791d9bd45f0cdf0d64ce46c306959ec843b8fceb15f99f6db9c12e1b'
FORBIDDEN_UNION_SHA256 = 'bc50143714352cc0e2af7c12a59856edb1d91c68fdda02a331542eb115544307'
SCHEDULE_TABLE_SHA256 = '604539dd3e022b946a0004e1b4bd575bab6d9572752872681ce6b5c2e35be5fb'
CONFIG_SHA256 = '9f631027a027df0ad3ead18a3626ca5b537c6589b69cfa7330f29d45be58ec6d'
E3_EXPECTED_CONFIG_SHA256 = (
    'b42bfe5a7cb9d6d2ac0485fb49ca766bb1e9fd6ca0fe983e4894658813a0cda9')


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_canonical(obj) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Cell index (B6/minor: keyed by probe_role so ``rep`` old/new do not collide).
# ---------------------------------------------------------------------------
_ROLE_FALLBACK = ('own', 'foil', 'old', 'new', 'cue', 'template', '')


def _index(cells: list) -> dict:
    idx = {}
    for cell in cells:
        key = (int(cell['repeat_seed']), str(cell['schedule_id']),
               str(cell.get('probe_role', '')), str(cell['probe_char']),
               str(cell['condition']))
        idx[key] = cell
    return idx


def _get(idx, seed, schedule_id, probe_char, condition, role=None):
    if role is not None:
        return idx.get((seed, schedule_id, role, probe_char, condition))
    for candidate in _ROLE_FALLBACK:
        cell = idx.get((seed, schedule_id, candidate, probe_char, condition))
        if cell is not None:
            return cell
    return None


# ---------------------------------------------------------------------------
# Basic signal statistics.
# ---------------------------------------------------------------------------
def _vec(row, key, t):
    prof = (row or {}).get('probe_profile') or {}
    arr = np.asarray(prof.get(key, []), dtype=np.float32)
    if arr.ndim != 2 or t >= arr.shape[0]:
        return None
    return arr[t]


def _p(cell, key, t):
    if not cell:
        return None
    w = _vec(cell.get('write_row'), key, t)
    nw = _vec(cell.get('no_write_row'), key, t)
    return None if (w is None or nw is None) else w - nw


def _n(cell, key, t):
    return _vec((cell or {}).get('no_write_row'), key, t)


def _settle_sum(cell, key):
    for t in range(SETTLE_LO, SETTLE_HI + 1):
        if _p(cell, key, t) is None:
            return None
    return float(sum(np.linalg.norm(_p(cell, key, t))
                     for t in range(SETTLE_LO, SETTLE_HI + 1)))


def _response_vector(cell, key):
    """Mean paired response over the settle window (fixed, no fitting)."""
    diffs = [_p(cell, key, t) for t in range(SETTLE_LO, SETTLE_HI + 1)]
    if any(d is None for d in diffs):
        return None
    return np.mean(np.asarray(diffs, dtype=np.float64), axis=0)


def _cosine(u, v):
    """Cosine similarity, or ``None`` when either vector is not evaluable.

    PREDICTION_V2 forbids a silent ``-1.0`` sentinel: a missing or low-norm
    vector must propagate as non-evaluable, never as an artificial minimum.
    """
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if (not np.isfinite(nu) or not np.isfinite(nv) or
            nu <= 0.0 or nv <= 0.0):
        return None
    return float(np.dot(u, v) / (nu * nv))


def _bootstrap_ci(values, stat=np.median, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    vals = [float(v) for v in values
            if v is not None and np.isfinite(float(v))]
    if not vals:
        return {'lo': None, 'hi': None, 'n': 0}
    arr = np.asarray(vals, np.float64)
    rng = np.random.default_rng(int(seed))
    stats = [float(stat(rng.choice(arr, size=arr.size, replace=True)))
             for _ in range(int(n))]
    return {'lo': float(np.percentile(stats, 2.5)),
            'hi': float(np.percentile(stats, 97.5)), 'n': int(arr.size)}


def level_stats(idx, seeds, schedule_id, probe_char, key, role=None):
    vals = {s: _settle_sum(_get(idx, s, schedule_id, probe_char, 'intact',
                                role), key) for s in seeds}
    f_sum = 0.0
    for t in range(SETTLE_LO, SETTLE_HI + 1):
        mx = 0.0
        for s in seeds:
            v = _p(_get(idx, s, schedule_id, probe_char, 'write_off', role),
                   key, t)
            if v is not None:
                mx = max(mx, float(np.linalg.norm(v)))
        f_sum += mx
    nw_sums = []
    for s in seeds:
        cell = _get(idx, s, schedule_id, probe_char, 'intact', role)
        total, ok = 0.0, True
        for t in range(SETTLE_LO, SETTLE_HI + 1):
            v = _n(cell, key, t)
            if v is None:
                ok = False
                break
            total += float(np.linalg.norm(v))
        if ok:
            nw_sums.append(total)
    sd = float(np.std(nw_sums, ddof=1)) if len(nw_sums) > 1 else 0.0
    tau = max(10.0 * f_sum, 3.0 * sd)
    resolved = [s for s in seeds if vals[s] is not None and vals[s] > tau]
    resolved_vals = [vals[s] for s in resolved]
    median = float(np.median(resolved_vals)) if resolved_vals else 0.0
    floor_adjacent = bool(median < FLOOR_FACTOR * f_sum)
    return {
        'level': key,
        'schedule_id': schedule_id,
        'probe_char': probe_char,
        'probe_role': role,
        'R_per_seed': {str(s): vals[s] for s in seeds},
        'tau_R': float(tau),
        'F_sum': float(f_sum),
        'SD_R': float(sd),
        'resolved_seeds': [int(s) for s in resolved],
        'resolved_count': int(len(resolved)),
        'R_median_over_F_sum': float(median / f_sum) if f_sum > 0 else None,
        'floor_adjacent': floor_adjacent,
        'n_defined': int(sum(1 for v in vals.values() if v is not None)),
    }


# ---------------------------------------------------------------------------
# Content-specific recall resolver (frozen object semantics).
# ---------------------------------------------------------------------------
def recall_resolver(idx, seeds, schedule_id, probe_char, level,
                    own_role='own', foil_role='foil'):
    own = level_stats(idx, seeds, schedule_id, probe_char, level, own_role)
    foils = {}
    for foil in ('i', 'j'):
        foils[foil] = level_stats(idx, seeds, schedule_id, foil, level,
                                  foil_role)
    out = {'level': level, 'schedule_id': schedule_id, 'probe_char': probe_char,
           'own': own, 'foils': foils, 'per_seed': {}}
    for s in seeds:
        r_c = own['R_per_seed'][str(s)]
        r_i = foils['i']['R_per_seed'][str(s)]
        r_j = foils['j']['R_per_seed'][str(s)]
        if r_c is None or r_i is None or r_j is None:
            out['per_seed'][str(s)] = {'recall_ok': False, 'reason': 'missing'}
            continue
        max_foil = max(r_i, r_j)
        criterion = (r_c > own['tau_R'] and
                     r_c > max_foil * MARGIN_KAPPA and
                     not own['floor_adjacent'])
        out['per_seed'][str(s)] = {
            'R_c': float(r_c), 'R_i': float(r_i), 'R_j': float(r_j),
            'argmax': 'own' if r_c >= max_foil else ('i' if r_i >= r_j else 'j'),
            'R_c_over_max_foil': float(r_c / max_foil) if max_foil > 0 else None,
            'recall_ok': bool(criterion),
        }
    out['n_recall_ok'] = int(sum(1 for v in out['per_seed'].values()
                                 if v.get('recall_ok')))
    return out


def _retained(idx, seeds, p, level):
    resolved = recall_resolver(idx, seeds, f'capacity_p{p}', 'a', level)
    return resolved['n_recall_ok'] >= CAPACITY_RETAIN_SEEDS, resolved


def capacity_summary(idx, seeds, level):
    retained = {}
    detail = {}
    for p in LOAD_GRID:
        ok, resolved = _retained(idx, seeds, p, level)
        retained[int(p)] = bool(ok)
        detail[str(p)] = resolved
    loads = [p for p in LOAD_GRID if retained[p]]
    n_star = max(loads) if loads else 0
    retained_fraction = {str(p): (detail[str(p)]['n_recall_ok'] / len(seeds)
                                  if seeds else 0.0) for p in LOAD_GRID}
    fraction_ci = {}
    for p in LOAD_GRID:
        oks = [1.0 if detail[str(p)]['per_seed'].get(str(s), {}).get(
            'recall_ok') else 0.0 for s in seeds]
        fraction_ci[str(p)] = _bootstrap_ci(oks, stat=np.mean)
    non_increasing = all(
        retained_fraction[str(LOAD_GRID[i])] >=
        retained_fraction[str(LOAD_GRID[i + 1])]
        for i in range(len(LOAD_GRID) - 1))
    saturation = None
    if non_increasing:
        for p in LOAD_GRID:
            if retained_fraction[str(p)] < 0.5:
                saturation = int(p) - 1
                break
    saturated_within_range = bool(
        retained_fraction[str(LOAD_GRID[-1])] >= 0.5)
    falsifier_f1 = bool(
        n_star == 1 or
        (retained_fraction['2'] < 0.9 and retained_fraction['1'] >= 0.9))
    return {
        'level': level,
        'retained': retained,
        'retained_fraction': retained_fraction,
        'retained_fraction_ci95': fraction_ci,
        'N_star': int(n_star),
        'saturation_at': saturation,
        'retained_fraction_non_increasing': bool(non_increasing),
        'saturated_within_range': saturated_within_range,
        'pass_H_cap': bool(n_star >= 2),
        'F1_capacity_falsified': falsifier_f1,
        'cells': detail,
    }


# ---------------------------------------------------------------------------
# Forgetting / passive reference (H-forget, H-overwrite).
# ---------------------------------------------------------------------------
def _fit_log_curve(points):
    pts = [(age, r) for age, r in points
           if r is not None and r > 0 and np.isfinite(r)]
    if len(pts) < 2:
        return {'n_points': len(pts), 'slope_per_token': None,
                'tau_eff_tokens': None, 'tau_eff_seconds': None,
                'half_life_tokens': None, 'r2': None}
    ages = np.asarray([p[0] for p in pts], np.float64)
    logs = np.asarray([np.log(p[1]) for p in pts], np.float64)
    slope, intercept = np.polyfit(ages, logs, 1)
    r2 = 1.0 - (np.sum((logs - (slope * ages + intercept)) ** 2) /
                max(np.sum((logs - logs.mean()) ** 2), 1e-12))
    return {
        'n_points': len(pts),
        'slope_per_token': float(slope),
        'intercept': float(intercept),
        'r2': float(r2),
        'tau_eff_tokens': float(-(1.0 / slope)) if slope < 0 else None,
        'tau_eff_seconds': (float(-(1.0 / slope)) * DT_TOKEN
                            if slope < 0 else None),
        'half_life_tokens': float(np.log(2.0) / -slope) if slope < 0 else None,
    }


def _per_seed_tau(points_by_seed):
    taus = {}
    for seed, pts in points_by_seed.items():
        fit = _fit_log_curve(pts)
        if fit['tau_eff_tokens'] is not None:
            taus[seed] = fit['tau_eff_tokens']
    return taus


def forgetting_summary(idx, seeds, level):
    reactive = {}
    reactive_points_by_seed = {str(s): [] for s in seeds}
    for probe, age in FORGET_POSITIONS.items():
        stats = level_stats(idx, seeds, 'forgetting_core', probe, level, 'own')
        vals = [stats['R_per_seed'][str(s)] for s in seeds]
        vals = [v for v in vals if v is not None]
        median = float(np.median(vals)) if vals else None
        non_floor = bool(median is not None and median > stats['tau_R'])
        reactive[probe] = {'age': int(age), 'R_median': median,
                           'non_floor': non_floor, 'level_stats': stats}
        for s in seeds:
            r = stats['R_per_seed'][str(s)]
            if r is not None and r > stats['tau_R']:
                reactive_points_by_seed[str(s)].append((int(age), r))
    passive = {'a@32': {
        'age': 32,
        'level_stats': level_stats(idx, seeds, CAPACITY_ANCHOR_SID, 'a',
                                   level, 'own')}}
    passive_points_by_seed = {str(s): [] for s in seeds}
    for sid, age in PASSIVE_AGES.items():
        stats = level_stats(idx, seeds, sid, 'a', level, 'own')
        passive[f'a@{age}'] = {'age': int(age), 'level_stats': stats}
        for s in seeds:
            r = stats['R_per_seed'][str(s)]
            if r is not None and r > stats['tau_R']:
                passive_points_by_seed[str(s)].append((int(age), r))
    for s in seeds:
        r = passive['a@32']['level_stats']['R_per_seed'][str(s)]
        if r is not None and r > passive['a@32']['level_stats']['tau_R']:
            passive_points_by_seed[str(s)].append((32, r))

    reactive_fit = _fit_log_curve(
        [(v['age'], v['R_median']) for v in reactive.values()
         if v['non_floor']])
    passive_fit = _fit_log_curve(
        [(v['age'], float(np.median([
            x for x in [v['level_stats']['R_per_seed'][str(s)]
                        for s in seeds] if x is not None])))
         for v in passive.values()
         if any(v['level_stats']['R_per_seed'][str(s)] is not None
                for s in seeds)])

    tau_reactive = _per_seed_tau(reactive_points_by_seed)
    tau_passive = _per_seed_tau(passive_points_by_seed)
    tau_reactive_ci = _bootstrap_ci(list(tau_reactive.values()))
    tau_passive_ci = _bootstrap_ci(list(tau_passive.values()))
    tau_passive_median = (float(np.median(list(tau_passive.values())))
                          if tau_passive else None)

    overwrite = {}
    for probe, info in reactive.items():
        age = info['age']
        match = passive.get(f'a@{age}')
        if match is None or not info['non_floor']:
            continue
        p_stats = match['level_stats']
        p_vals = {s: p_stats['R_per_seed'][str(s)] for s in seeds}
        per_seed_fraction = {}
        n_positive = 0
        for s in seeds:
            r_react = info['level_stats']['R_per_seed'][str(s)]
            r_pass = p_vals[s]
            if r_react is None or r_pass is None or r_pass <= p_stats['tau_R']:
                continue
            frac = 1.0 - r_react / r_pass
            per_seed_fraction[str(s)] = float(frac)
            if frac > 0:
                n_positive += 1
        frac_ci = _bootstrap_ci(list(per_seed_fraction.values()))
        overwrite[probe] = {
            'age': int(age),
            'R_reactive_median': info['R_median'],
            'R_passive_median': float(np.median([
                v for v in p_vals.values() if v is not None]))
            if p_vals else None,
            'overwrite_fraction_per_seed': per_seed_fraction,
            'overwrite_fraction_median': (
                float(np.median(list(per_seed_fraction.values())))
                if per_seed_fraction else None),
            'overwrite_fraction_ci95': frac_ci,
            'n_seeds_positive': int(n_positive),
            'n_seeds_evaluable': int(len(per_seed_fraction)),
            'above_ci_positive': bool(
                frac_ci['lo'] is not None and frac_ci['lo'] > 0.0 and
                n_positive >= MIN_SEEDS),
        }
    resolved_ages = sorted({v['age'] for v in reactive.values()
                            if v['non_floor']})
    h_forget_pass = bool(
        len(resolved_ages) >= 2 and
        reactive_fit['tau_eff_tokens'] is not None and
        tau_reactive_ci['lo'] is not None and tau_reactive_ci['lo'] > 0.0)
    half_life_rule = bool(
        tau_reactive_ci['hi'] is not None and tau_passive_median and
        tau_reactive_ci['hi'] < tau_passive_median / 1.5)
    overwrite_rule = any(v['above_ci_positive'] for v in overwrite.values())
    h_overwrite_pass = bool(half_life_rule or overwrite_rule)
    f2_no_gradualism = bool(
        not any(v['non_floor'] for probe, v in reactive.items()
                if probe != 'a'))
    f4_decay_only = bool(not h_overwrite_pass and
                         reactive_fit['tau_eff_tokens'] is not None)
    return {
        'level': level,
        'reactive': reactive,
        'passive': passive,
        'reactive_fit': reactive_fit,
        'passive_fit': passive_fit,
        'tau_reactive_per_seed': tau_reactive,
        'tau_passive_per_seed': tau_passive,
        'tau_reactive_median': (float(np.median(list(tau_reactive.values())))
                                if tau_reactive else None),
        'tau_reactive_ci95': tau_reactive_ci,
        'tau_passive_median': tau_passive_median,
        'tau_passive_ci95': tau_passive_ci,
        'overwrite_fraction': overwrite,
        'resolved_non_floor_ages': [int(a) for a in resolved_ages],
        'n_non_floor_ages': len(resolved_ages),
        'pass_H_forget': h_forget_pass,
        'pass_H_overwrite': h_overwrite_pass,
        'F2_no_gradualism': f2_no_gradualism,
        'F4_decay_only': f4_decay_only,
    }


# ---------------------------------------------------------------------------
# Interference matrix (H-interf).
# ---------------------------------------------------------------------------
def interference_summary(idx, seeds, level, y_rel, y_unrel):
    denom = level_stats(idx, seeds, INTERF_DENOM_SID, 'a', level, 'own')
    anchor = level_stats(idx, seeds, INTERF_DENOM_ALT_SID, 'a', level, 'own')
    new_alone = level_stats(idx, seeds, 'interference_new_only', y_unrel,
                            level, 'new')
    out = {'level': level, 'configs': {}, 'I': {},
           'denominator': {
               'schedule_id': INTERF_DENOM_SID, 'age': INTERF_DENOM_AGE,
               'level_stats': denom},
           'denominator_alt': {
               'schedule_id': INTERF_DENOM_ALT_SID, 'age': 32,
               'level_stats': anchor},
           'new_item_alone': {
               'schedule_id': 'interference_new_only', 'probe_char': y_unrel,
               'level_stats': new_alone},
           'frozen_probe_lesion_scope': INTERFERENCE_LESION,
           'frozen_probe_lesion_present': False,
           'frozen_probe_lesion_cells': []}
    new_chars = {'rep': 'a', 'unrel': y_unrel, 'rel': y_rel}
    for name, sid in INTERFER_CONFIGS.items():
        old = level_stats(idx, seeds, sid, 'a', level, 'old')
        new = level_stats(idx, seeds, sid, new_chars[name], level, 'new')
        out['configs'][name] = {'schedule_id': sid, 'old': old, 'new': new,
                                'new_char': new_chars[name]}
        ivals = {}
        for s in seeds:
            ro = old['R_per_seed'][str(s)]
            rb = denom['R_per_seed'][str(s)]
            if ro is None or rb is None or rb <= denom['tau_R'] or rb <= 0:
                continue
            ivals[str(s)] = float(1.0 - ro / rb)
        out['I'][name] = ivals
    # B2: verify the two frozen probe-only lesion cells exist with the exact
    # scope and conditions.
    lesion_cells = []
    for s in seeds:
        for cond in INTERFERENCE_LESION['conditions']:
            cell = _get(idx, s, INTERFERENCE_LESION['schedule_id'],
                        INTERFERENCE_LESION['probe_char'], cond,
                        INTERFERENCE_LESION['probe_role'])
            if cell is not None:
                lesion_cells.append(cell.get('cell_id'))
    expected_lesion = len(seeds) * len(INTERFERENCE_LESION['conditions'])
    out['frozen_probe_lesion_present'] = bool(
        len(lesion_cells) == expected_lesion and expected_lesion > 0)
    out['frozen_probe_lesion_cells'] = lesion_cells
    denom_seeds_non_floor = [
        s for s in seeds
        if denom['R_per_seed'][str(s)] is not None and
        denom['R_per_seed'][str(s)] > denom['tau_R'] and
        denom['R_per_seed'][str(s)] > 0]
    out['denominator_non_floor_seeds'] = [int(s) for s in denom_seeds_non_floor]
    paired = []
    large = []
    for s in seeds:
        ir = out['I']['rel'].get(str(s))
        iu = out['I']['unrel'].get(str(s))
        if ir is not None and iu is not None:
            paired.append(abs(ir - iu))
            if min(ir, iu) >= INTERF_SPEC_MIN:
                large.append(True)
    spec_support = sum(1 for x in paired if x >= INTERF_SPEC_MIN)
    # New-item retention: only unrel has a dose-matched new-only baseline.
    new_retention = {}
    for name, sid in INTERFER_CONFIGS.items():
        if name == 'rep':
            new_retention[name] = {'evaluable': False,
                                   'reason': 'repeat write, not a new item'}
            continue
        if name == 'rel':
            new_retention[name] = {
                'evaluable': False,
                'reason': 'no dose-matched [y_rel]-alone baseline in the '
                          'frozen grid'}
            continue
        ratios = {}
        for s in seeds:
            rn = out['configs'][name]['new']['R_per_seed'][str(s)]
            ra = new_alone['R_per_seed'][str(s)]
            if rn is None or ra is None or ra <= new_alone['tau_R'] or ra <= 0:
                continue
            ratios[str(s)] = float(rn / ra)
        n_ok = sum(1 for v in ratios.values() if v >= INTERF_NEW_MIN)
        new_retention[name] = {
            'evaluable': bool(ratios),
            'R_new_after_over_alone_per_seed': ratios,
            'n_seeds_ge_0.5': int(n_ok),
            'n_seeds_evaluable': int(len(ratios)),
            'pass': bool(len(ratios) > 0 and n_ok >= MIN_SEEDS)}
    out['new_item_retention'] = new_retention
    specificity_pass = bool(
        spec_support >= MIN_SEEDS and
        len(denom_seeds_non_floor) >= MIN_SEEDS)
    new_pass = bool(new_retention['unrel']['pass'])
    out['specificity'] = {
        'abs_I_rel_minus_I_unrel': [float(x) for x in paired],
        'n_seeds_ge_0.2': int(spec_support),
        'n_seeds_evaluable': int(len(paired)),
        'pass_specificity': specificity_pass,
        'non_specific_overwrite': bool(
            len(paired) > 0 and spec_support < MIN_SEEDS and
            len(large) >= MIN_SEEDS),
    }
    out['pass_H_interf'] = bool(specificity_pass and new_pass)
    out['F3_non_specific'] = bool(
        out['specificity']['non_specific_overwrite'])
    return out


# ---------------------------------------------------------------------------
# Prediction: bounded template-rank + exact stratified exchangeability null.
#
# PREDICTION_V2 (prereg amendment 2026-09-18).  The old
# ``observed > exhaustive successor-label permutation max`` gate is removed: the
# identity permutation makes that maximum >= observed by construction, so the
# gate was structurally unsatisfiable.  The 24 permutations are retained only as
# a descriptive upper-tail count.  The primary test is an exact one-sided
# ``Binomial(len(seeds), 0.25)`` upper tail per context.
# ---------------------------------------------------------------------------
def _prediction_norm_floor(idx, seeds, ctx, level):
    """Frozen evaluability floor derived from the context's floor statistics.

    ``max(10*F_sum, 3*SD_R)`` over the neutral-cue ``z`` write_off floor cells,
    exactly analogous to the storage ``tau_R`` rule.  A response vector is
    evaluable only when its norm strictly exceeds this floor.
    """
    stats = level_stats(idx, seeds, f'prediction_ctx_{ctx}', 'z', level, 'cue')
    return float(max(NORM_FLOOR_ABS, 10.0 * stats['F_sum'],
                     3.0 * stats['SD_R']))


def _context_prediction(idx, seeds, ctx, condition, level,
                        template_condition='intact'):
    """Per-seed successor prediction for one context.

    Only the cue ``q=z`` carries ``condition``; the template vectors always come
    from ``template_condition='intact'`` (amendment 2026-09-18) so a readout
    lesion cannot mechanically collapse prediction through missing templates.
    Missing / low-norm vectors are non-evaluable and count as a miss; they are
    never silently imputed.

    Returns ``(per_seed, tie_count, non_evaluable_count, norm_floor)``.
    """
    sid = f'prediction_ctx_{ctx}'
    floor = _prediction_norm_floor(idx, seeds, ctx, level)
    per_seed = {}
    tie_count = 0
    non_evaluable_count = 0
    for s in seeds:
        q = _response_vector(
            _get(idx, s, sid, 'z', condition, 'cue'), level)
        sims = None
        pred = None
        margin = None
        if (q is not None and np.isfinite(np.linalg.norm(q)) and
                float(np.linalg.norm(q)) > floor):
            candidate = {}
            for y in ALPHABET4:
                v = _response_vector(
                    _get(idx, s, sid, y, template_condition, 'template'), level)
                if (v is None or not np.isfinite(np.linalg.norm(v)) or
                        float(np.linalg.norm(v)) <= floor):
                    candidate = None
                    break
                c = _cosine(q, v)
                if c is None:
                    candidate = None
                    break
                candidate[y] = c
            sims = candidate
        if sims is not None:
            succ = CYCLE_SUCCESSOR[ctx]
            top = max(sims.values())
            second = sorted(sims.values(), reverse=True)[1]
            if (top - second) <= TIE_TOL:
                tie_count += 1
            pred = max(ALPHABET4, key=lambda y: (sims[y], -ALPHABET4.index(y)))
            others = [sims[y] for y in ALPHABET4 if y != succ]
            margin = float(sims[succ] - max(others)) if others else 0.0
            correct = bool(pred == succ and margin > TIE_TOL)
        else:
            non_evaluable_count += 1
            correct = False
        per_seed[str(s)] = {
            'evaluable': bool(sims is not None),
            'predicted': pred,
            'sims': sims,
            'margin': margin,
            'correct': bool(correct),
        }
    return per_seed, tie_count, non_evaluable_count, floor


def _pooled_accuracy(per_seed):
    vals = [v for v in per_seed.values() if v is not None]
    if not vals:
        return 0.0
    return float(np.mean([1.0 if v['correct'] else 0.0 for v in vals]))


def _seed_accuracy(per_seed):
    return {s: (None if v is None else (1.0 if v['correct'] else 0.0))
            for s, v in per_seed.items()}


def _mean_over_contexts(predictions, contexts):
    per_seed = {}
    for s in next(iter(predictions.values()), {}):
        vals = []
        for ctx in contexts:
            v = (predictions.get(ctx) or {}).get(s)
            if v is not None:
                vals.append(1.0 if v['correct'] else 0.0)
        per_seed[s] = float(np.mean(vals)) if vals else None
    return per_seed


def _exact_binomial_tail(h, n=10, p=0.25):
    """Exact one-sided upper-tail ``P(Bin(n, p) >= h)`` (no RNG, no scipy)."""
    from math import comb
    h = int(h)
    n = int(n)
    if h <= 0:
        return 1.0
    if h > n:
        return 0.0
    return float(sum(comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
                     for k in range(h, n + 1)))


def _successor_null_permutation_tail(contexts, predictions):
    """Descriptive-only upper-tail count over the 24 successor-label perms.

    Never used for a decision (PREDICTION_V2 replaces the max gate).
    """
    observed_total, observed_count = 0.0, 0
    for ctx in contexts:
        for v in (predictions.get(ctx) or {}).values():
            if v is None:
                continue
            observed_total += 1.0 if v['correct'] else 0.0
            observed_count += 1
    observed = (observed_total / observed_count) if observed_count else 0.0
    accs = []
    for perm in permutations(ALPHABET4):
        mapping = dict(zip(ALPHABET4, perm))
        total, count = 0.0, 0
        for ctx in contexts:
            for v in (predictions.get(ctx) or {}).values():
                if v is None:
                    continue
                total += (1.0 if v['predicted'] == mapping[CYCLE_SUCCESSOR[ctx]]
                          else 0.0)
                count += 1
        if count:
            accs.append(total / count)
    n_ge = sum(1 for a in accs if a >= observed - 1e-12)
    return {
        'observed_pooled_accuracy': observed,
        'n_permutations': len(accs),
        'n_ge_observed': int(n_ge),
        'fraction_ge_observed': (float(n_ge / len(accs)) if accs else None),
        'max_accuracy': float(max(accs)) if accs else None,
        'used_for_decision': False,
    }


def _edge_state_fingerprint(fp):
    """Canonical end-of-episode edge-state fingerprint (design 4.2)."""
    if not isinstance(fp, dict):
        return None
    return json.dumps({
        'weights': fp.get('weights'),
        'edge_weights': fp.get('edge_weights'),
        'edge_identity': fp.get('edge_identity'),
    }, sort_keys=True, separators=(',', ':'), default=str)


def _write_lesion_collapse_required(idx, seeds):
    """Design 4.2 two-outcome: required iff lesion end-fingerprint != intact.

    Both cells are on ``prediction_ctx_a`` probe ``z``.  When no comparable
    fingerprint exists the lesion is treated as required (fail-closed).
    """
    comparable, differing = [], []
    for s in seeds:
        les = _get(idx, s, 'prediction_ctx_a', 'z', WRITE_LESION, 'cue')
        intc = _get(idx, s, 'prediction_ctx_a', 'z', 'intact', 'cue')
        if les is None or intc is None:
            continue
        lf = _edge_state_fingerprint(
            (les.get('write_row') or {}).get('fingerprint_end'))
        inf = _edge_state_fingerprint(
            (intc.get('write_row') or {}).get('fingerprint_end'))
        if lf is None or inf is None:
            continue
        comparable.append(int(s))
        if lf != inf:
            differing.append(int(s))
    if not comparable:
        required = True
        reason = ('uncomparable write-phase end-fingerprint; fail-closed '
                  'required')
    else:
        required = bool(differing)
        reason = ('write-phase KC->MBON transmission contributes to the store'
                  if required else
                  'write-phase end-fingerprint byte-identical to intact; '
                  'diagnostic only')
    return required, {
        'comparable_seeds': comparable,
        'differing_seeds': differing,
        'required': bool(required),
        'reason': reason,
    }


def _holdout_summary(idx, seeds, level):
    """Pre-registered held-out contexts {b,d}: confirmation only, no tuning."""
    preds, hits, ps = {}, {}, {}
    evaluated = False
    for ctx in PREDICTION_CONTEXTS_HOLDOUT:
        sid = f'prediction_ctx_{ctx}'
        if not any(_get(idx, s, sid, 'z', 'intact', 'cue') is not None
                   for s in seeds):
            continue
        evaluated = True
        per_seed, _, _, _ = _context_prediction(idx, seeds, ctx, 'intact', level)
        preds[ctx] = per_seed
        h = int(sum(1 for s in seeds
                    if (per_seed.get(str(s)) or {}).get('correct')))
        hits[ctx] = h
        ps[ctx] = _exact_binomial_tail(h, n=len(seeds), p=0.25)
    if not evaluated:
        status = 'not_evaluated'
        confirmation = None
    else:
        status = 'evaluated'
        confirmation = bool(ps and
                            all(p <= ALPHA_PER_CONTEXT for p in ps.values()))
    return {
        'contexts': list(PREDICTION_CONTEXTS_HOLDOUT),
        'status': status,
        'hits_per_context': hits,
        'p_per_context': ps,
        'alpha_per_context': ALPHA_PER_CONTEXT,
        'min_hits_per_context': MIN_HITS_PER_CONTEXT,
        'confirmation_pass': confirmation,
        'used_for_tuning': False,
        'primary_decision_uses_holdout': False,
        'note': ('held-out confirmation only; never used for tuning or the '
                 'primary decision; not-evaluated is a frozen clause, not a '
                 'negative result'),
        'predictions': {ctx: preds[ctx] for ctx in preds},
    }


def prediction_summary(idx, seeds, level):
    contexts = list(PREDICTION_CONTEXTS)
    predictions = {}
    hits_per_context = {}
    p_per_context = {}
    tie_count = 0
    non_evaluable_count = 0
    norm_floor = {}
    for ctx in contexts:
        per_seed, ties, non_eval, floor = _context_prediction(
            idx, seeds, ctx, 'intact', level)
        predictions[ctx] = per_seed
        hits_per_context[ctx] = int(sum(
            1 for s in seeds
            if (per_seed.get(str(s)) or {}).get('correct')))
        p_per_context[ctx] = _exact_binomial_tail(
            hits_per_context[ctx], n=len(seeds), p=0.25)
        tie_count += ties
        non_evaluable_count += non_eval
        norm_floor[ctx] = floor
    observed = float(np.mean([_pooled_accuracy(predictions[ctx])
                              for ctx in contexts]))
    permutation_tail = _successor_null_permutation_tail(contexts, predictions)
    intact_seed_acc = _mean_over_contexts(predictions, contexts)
    boot = _bootstrap_ci([v for v in intact_seed_acc.values()
                          if v is not None], stat=np.mean)
    n_seeds_above_chance = int(sum(
        1 for v in intact_seed_acc.values()
        if v is not None and v > 0.25))

    # Primary null: exact per-context upper tail.  p_X <= 0.025 iff h_X >= 6 at
    # n=10.  The seed-level / bootstrap criteria are retained conjuncts
    # (report 3.3), never a replacement for the exact test.
    pass_null_per_context = bool(all(
        p_per_context[c] <= ALPHA_PER_CONTEXT for c in contexts))
    seed_supplement_pass = bool(
        observed > 0.25 and boot['lo'] is not None and boot['lo'] > 0.25 and
        n_seeds_above_chance >= MIN_SEEDS)
    accuracy_pass = bool(pass_null_per_context and seed_supplement_pass)

    write_required, write_detail = _write_lesion_collapse_required(idx, seeds)
    required = ['write_off'] + list(PROBE_LESION_CONDITIONS)
    if write_required:
        required.append(WRITE_LESION)
    collapses = {}
    for cond in required:
        # B5: the write-phase lesion only exists on context ``a``; never pool
        # the missing context ``c`` as a numeric zero.
        cond_contexts = ['a'] if cond == WRITE_LESION else contexts
        cond_pred = {}
        q_present = 0
        for ctx in cond_contexts:
            sid = f'prediction_ctx_{ctx}'
            q_present += sum(1 for s in seeds
                             if _get(idx, s, sid, 'z', cond, 'cue') is not None)
            cond_pred[ctx], _, _, _ = _context_prediction(
                idx, seeds, ctx, cond, level)
        expected_q = len(seeds) * len(cond_contexts)
        cond_acc = float(np.mean([_pooled_accuracy(cond_pred[ctx])
                                  for ctx in cond_contexts]))
        intact_ctx_acc = _mean_over_contexts(predictions, cond_contexts)
        cond_seed_acc = _mean_over_contexts(cond_pred, cond_contexts)
        drops = sum(
            1 for s in seeds
            if intact_ctx_acc.get(str(s)) is not None and
            cond_seed_acc.get(str(s)) is not None and
            (intact_ctx_acc[str(s)] - cond_seed_acc[str(s)]) >= 0.25)
        intact_vals = [v for v in intact_ctx_acc.values() if v is not None]
        intact_ctx_mean = float(np.mean(intact_vals)) if intact_vals else 0.0
        collapses[cond] = {
            'contexts': list(cond_contexts),
            'pooled_accuracy': cond_acc,
            'intact_pooled_accuracy': intact_ctx_mean,
            'drop': float(intact_ctx_mean - cond_acc),
            'n_seeds_dropped_ge_0.25': int(drops),
            'n_q_cells_present': int(q_present),
            'n_q_cells_expected': int(expected_q),
            'collapse': bool(q_present == expected_q and
                             cond_acc <= 0.25 and
                             (intact_ctx_mean - cond_acc) >= 0.25 and
                             drops >= MIN_SEEDS),
        }
    pass_P1 = bool(accuracy_pass and
                   all(collapses[c]['collapse'] for c in required))
    holdout = _holdout_summary(idx, seeds, level)
    return {
        'level': level,
        'contexts': contexts,
        'observed_pooled_accuracy': observed,
        'chance': 0.25,
        'hits_per_context': hits_per_context,
        'p_per_context': p_per_context,
        'null_model': NULL_MODEL,
        'alpha_family': ALPHA_FAMILY,
        'alpha_per_context': ALPHA_PER_CONTEXT,
        'min_hits_per_context': MIN_HITS_PER_CONTEXT,
        'tie_count': int(tie_count),
        'non_evaluable_count': int(non_evaluable_count),
        'tie_tol': TIE_TOL,
        'norm_floor': norm_floor,
        'norm_floor_rule': NORM_FLOOR,
        'null_permutation_tail': permutation_tail,
        'pass_P1_null_per_context': pass_null_per_context,
        'pass_P1_seed_supplement': seed_supplement_pass,
        'intact_per_seed_accuracy': intact_seed_acc,
        'accuracy_ci95': boot,
        'n_seeds_above_chance': n_seeds_above_chance,
        'pass_P1_accuracy': accuracy_pass,
        'required_collapse_conditions': list(required),
        'write_lesion_collapse_required': bool(write_required),
        'write_lesion_fingerprint': write_detail,
        'causal_collapses': collapses,
        'holdout': holdout,
        'pass_P1': pass_P1,
        'falsifier_F5': bool(not pass_P1),
        'predictions': {ctx: predictions[ctx] for ctx in contexts},
        'decoder_rule': ('no fitted decoder; a chance/null result falsifies and '
                         'stops the open-prediction line'),
    }


# ---------------------------------------------------------------------------
# Validity gates (B4: genuinely fail-closed).
# ---------------------------------------------------------------------------
def _writer_gate(cells, vectors, seeds):
    groups = {}
    for cell in cells:
        if cell.get('condition') == 'write_off':
            continue
        key = (int(cell['repeat_seed']), str(cell['schedule_id']))
        cid = cell.get('cell_id')
        vec = vectors.get(cid)
        if vec is None:
            continue
        groups.setdefault(key, {})[cid] = np.asarray(vec, np.float32)
    results = {}
    for key, vecs in sorted(groups.items()):
        hashes = {cid: _sha256_bytes(v) for cid, v in vecs.items()}
        state_shas = {}
        for h in hashes.values():
            state_shas[h] = state_shas.get(h, 0) + 1
        modal_sha = min(state_shas, key=lambda h: (-state_shas[h], h))
        modal_count = state_shas[modal_sha]
        evaluable = len(vecs) > 1
        rec = {
            'n_samples': len(vecs),
            'n_states': len(state_shas),
            'modal_fraction': modal_count / len(vecs),
            'evaluable': evaluable,
        }
        if evaluable:
            rep_cid = next(cid for cid, h in hashes.items()
                           if h == modal_sha)
            rep = vecs[rep_cid]
            denom = max(float(np.linalg.norm(rep.astype(np.float64))), 1e-12)
            worst_max_abs, worst_rel_l2 = 0.0, 0.0
            for v in vecs.values():
                diff = v.astype(np.float64) - rep.astype(np.float64)
                worst_max_abs = max(worst_max_abs,
                                    float(np.abs(diff).max()))
                worst_rel_l2 = max(worst_rel_l2,
                                   float(np.linalg.norm(diff)) / denom)
            rec['worst_max_abs'] = worst_max_abs
            rec['worst_rel_l2'] = worst_rel_l2
            rec['max_states_ok'] = bool(len(state_shas) <= WRITER_MAX_STATES)
            rec['modal_fraction_ok'] = bool(
                modal_count / len(vecs) >= WRITER_MODAL_MIN_FRACTION)
            rec['max_abs_ok'] = bool(worst_max_abs <= WRITER_TOL_MAX_ABS)
            rec['rel_l2_ok'] = bool(worst_rel_l2 <= WRITER_TOL_REL_L2)
            rec['pass'] = bool(
                rec['max_states_ok'] and rec['modal_fraction_ok'] and
                rec['max_abs_ok'] and rec['rel_l2_ok'])
        else:
            rec['pass'] = None
        results[f'{key[0]}:{key[1]}'] = rec
    evaluable = [v for v in results.values() if v['evaluable']]
    return {
        'groups': results,
        'n_groups': len(results),
        'n_evaluable': len(evaluable),
        'pass': bool(len(evaluable) >= 1 and
                     all(v['pass'] for v in evaluable)),
        'tolerances': {'max_abs': WRITER_TOL_MAX_ABS,
                       'rel_l2': WRITER_TOL_REL_L2},
    }


def _g3_floor_gate(raw, cells, idx, seeds):
    """B4: non-vacuous write_off floor gate (events, zero state, separation)."""
    write_off_cells = [c for c in cells if c.get('condition') == 'write_off']
    if not write_off_cells:
        return False, {'reason': 'no write_off evidence',
                       'n_write_off_cells': 0}
    events_ok = True
    zero_ok = True
    for cell in write_off_cells:
        if not (cell.get('events_bitwise_equal') and
                cell.get('graded_within_tolerance')):
            events_ok = False
        mod = (cell.get('write_row') or {}).get('modulation_stats_after_write') or {}
        edge_norm = (cell.get('write_row') or {}).get('edge_norm_after_write')
        for value in (mod.get('l1'), mod.get('l2'), mod.get('max_abs'),
                      edge_norm):
            if value is not None and abs(float(value)) > G3_ZERO_TOL:
                zero_ok = False
    groups = {}
    for cell in write_off_cells:
        if cell.get('probe_role') not in ('own', 'old'):
            continue
        groups[(cell.get('schedule_id'), cell.get('probe_char'),
                cell.get('probe_role'))] = True
    separations = {}
    separation_ok = bool(groups)
    for (sid, probe, role) in sorted(groups):
        stats = level_stats(idx, seeds, sid, probe, STORAGE_PRIMARY_LEVEL, role)
        ratio = stats['R_median_over_F_sum']
        separations[f'{sid}|{probe}|{role}'] = ratio
        if ratio is None or stats['F_sum'] <= 0 or ratio <= G3_SEPARATION_MIN:
            separation_ok = False
    ok = bool(events_ok and zero_ok and separation_ok)
    return ok, {'n_write_off_cells': len(write_off_cells),
                'events_ok': bool(events_ok), 'zero_state_ok': bool(zero_ok),
                'separation_ok': bool(separation_ok),
                'separation_min': G3_SEPARATION_MIN,
                'separations': separations}


def validity_gates(raw, cells, vectors, seeds, idx):
    gates = {}
    conn = raw.get('connectome') or {}
    gates['G0'] = bool(
        conn.get('neurons') == EXPECTED_NEURONS and
        conn.get('edges') == EXPECTED_EDGES and
        {k: conn.get('edge_counts', {}).get(k)
         for k in EXPECTED_EDGE_COUNTS} == EXPECTED_EDGE_COUNTS and
        not conn.get('fixed_direction_fallback_used', True))
    reset_fps = set()
    paired_v_zero = True
    for cell in cells:
        for branch in ('write_row', 'no_write_row'):
            fp = (cell.get(branch) or {}).get('fingerprint_reset')
            if fp is not None:
                reset_fps.add(json.dumps(fp, sort_keys=True, default=str))
                if float(fp.get('v_sum', 1.0)) != 0.0:
                    paired_v_zero = False
    gates['G1'] = bool(len(cells) > 0 and len(reset_fps) == 1 and
                       paired_v_zero)
    gates['G2'] = bool(cells) and all(
        cell.get('replay_events_bitwise_equal') and
        cell.get('replay_graded_within_tolerance') for cell in cells)
    g3_ok, g3_detail = _g3_floor_gate(raw, cells, idx, seeds)
    gates['G3'] = bool(g3_ok)
    gates['_G3_detail'] = g3_detail
    input_ok = bool(cells)
    for cell in cells:
        spec = cell.get('input_spec')
        sha = cell.get('input_sha256')
        if not spec or not sha or _hash_canonical(spec) != sha:
            input_ok = False
            break
    gates['G-input'] = bool(input_ok and raw.get('inputs_bitwise_identical')
                            is True)
    coverage = raw.get('config_hash_coverage')
    config_ok = bool(coverage and raw.get('config_sha256'))
    if config_ok:
        config_ok = bool(_hash_canonical(coverage) == raw.get('config_sha256'))
    gates['G-config'] = config_ok
    writer = _writer_gate(cells, vectors, seeds)
    gates['G8-evidence'] = bool(writer['pass'])
    gates['_G8_detail'] = writer
    write_lesion_cells = [c for c in cells if c.get('write_lesion')]
    g9_ok = bool(write_lesion_cells)
    for cell in write_lesion_cells:
        row = cell.get('write_row') or {}
        if not (row.get('write_lesion_applied') and
                row.get('write_lesion_bytes_identical') and
                row.get('write_lesion_baseline_restored') and
                row.get('write_lesion_materialized_matches_expected') and
                row.get('write_lesion_positions_match_plastic') and
                row.get('write_lesion_saved_sha256') and
                row.get('write_lesion_materialized_sha256')):
            g9_ok = False
            break
        reset_id = ((row.get('fingerprint_reset') or {}) or {}).get(
            'edge_identity')
        end_id = ((row.get('fingerprint_end') or {}) or {}).get('edge_identity')
        if reset_id is not None and end_id is not None and reset_id != end_id:
            g9_ok = False
            break
    gates['G9'] = bool(g9_ok)
    gates['G10'] = bool(vectors)
    gates['G11'] = bool(cells)
    forbidden = set(raw.get('forbidden_union') or [])
    gates['G12'] = bool(
        len(seeds) == len(FRESH_SEEDS) and
        sorted(int(s) for s in seeds) == sorted(FRESH_SEEDS) and
        not (set(int(s) for s in seeds) & forbidden))
    inventory = raw.get('item_inventory')
    inv_sha = raw.get('item_inventory_sha256')
    gates['G13'] = bool(inventory and inv_sha and
                        _hash_canonical(inventory) == inv_sha)
    resolver = raw.get('resolver')
    prediction = raw.get('prediction')
    bridge = raw.get('e3_bridge_static') or {}
    g14_ok = bool(
        raw.get('resolver_sha256') == RESOLVER_SHA256 and
        raw.get('item_inventory_sha256') == ITEM_INVENTORY_SHA256 and
        raw.get('prediction_sha256') == PREDICTION_SHA256 and
        raw.get('schedule_table_sha256') == SCHEDULE_TABLE_SHA256 and
        raw.get('forbidden_union_sha256') == FORBIDDEN_UNION_SHA256 and
        raw.get('config_sha256') == CONFIG_SHA256 and
        resolver and _hash_canonical(resolver) == raw.get('resolver_sha256') and
        prediction and
        _hash_canonical(prediction) == raw.get('prediction_sha256') and
        inventory and _hash_canonical(inventory) == raw.get('item_inventory_sha256') and
        bridge.get('e3_config_sha256_recomputed') == E3_EXPECTED_CONFIG_SHA256 and
        bridge.get('pass') is True)
    gates['G14'] = g14_ok
    public = {k: v for k, v in gates.items() if not k.startswith('_')}
    gates['arm_valid'] = bool(all(public.values()))
    return gates


# ---------------------------------------------------------------------------
# Writer vectors.
# ---------------------------------------------------------------------------
def _load_write_vectors(cells, npz):
    out = {}
    for cell in cells:
        cid = cell.get('cell_id')
        key = f'{cid}|write_row|modulation_after_write_vector'
        if key in npz.files:
            out[cid] = np.asarray(npz[key], dtype=np.float32)
    return out


def _arm_invalid_storage(storage):
    storage['arm_invalid'] = True
    storage['capacity']['pass_H_cap'] = False
    storage['forgetting']['pass_H_forget'] = False
    storage['forgetting']['pass_H_overwrite'] = False
    storage['interference']['pass_H_interf'] = False


def summarize(raw_path: Path, out_prefix: Path) -> dict:
    raw = json.loads(Path(raw_path).read_text(encoding='utf-8'))
    npz_path = Path(str(raw_path).replace('.json', '.raw.npz'))
    npz = np.load(npz_path, allow_pickle=True) if npz_path.exists() else {}
    cells = raw.get('cells') or []
    seeds = [int(s) for s in raw.get('fresh_seeds') or []]
    idx = _index(cells)
    vectors = _load_write_vectors(cells, npz)
    gates = validity_gates(raw, cells, vectors, seeds, idx)
    related = raw.get('related_pair') or {}
    unrelated = raw.get('unrelated_pair') or {}
    y_rel = related.get('y')
    y_unrel = unrelated.get('y')

    storage_primary = {
        'level': STORAGE_PRIMARY_LEVEL,
        'capacity': capacity_summary(idx, seeds, STORAGE_PRIMARY_LEVEL),
        'forgetting': forgetting_summary(idx, seeds, STORAGE_PRIMARY_LEVEL),
        'interference': interference_summary(
            idx, seeds, STORAGE_PRIMARY_LEVEL, y_rel, y_unrel),
    }
    storage_secondary = {
        'level': STORAGE_SECONDARY_LEVEL,
        'capacity': capacity_summary(idx, seeds, STORAGE_SECONDARY_LEVEL),
        'forgetting': forgetting_summary(idx, seeds, STORAGE_SECONDARY_LEVEL),
        'interference': interference_summary(
            idx, seeds, STORAGE_SECONDARY_LEVEL, y_rel, y_unrel),
    }
    prediction_primary = prediction_summary(idx, seeds,
                                            PREDICTION_PRIMARY_LEVEL)
    prediction_secondary = prediction_summary(idx, seeds,
                                              PREDICTION_SECONDARY_LEVEL)
    if not gates.get('arm_valid'):
        _arm_invalid_storage(storage_primary)
        _arm_invalid_storage(storage_secondary)
        prediction_primary['arm_invalid'] = True
        prediction_secondary['arm_invalid'] = True
        prediction_primary['pass_P1'] = False
        prediction_secondary['pass_P1'] = False
    summary = {
        'experiment': raw.get('experiment'),
        'raw_path': str(raw_path),
        'raw_sha256': _sha256(Path(raw_path)),
        'npz_sha256': _sha256(npz_path) if npz_path.exists() else None,
        'fresh_seeds': seeds,
        'n_cells': raw.get('n_cells'),
        'n_runs': raw.get('n_runs'),
        'episodes_total': raw.get('episodes_total'),
        'resolver_sha256': raw.get('resolver_sha256'),
        'item_inventory_sha256': raw.get('item_inventory_sha256'),
        'prediction_sha256': raw.get('prediction_sha256'),
        'forbidden_union_sha256': raw.get('forbidden_union_sha256'),
        'schedule_table_sha256': raw.get('schedule_table_sha256'),
        'config_sha256': raw.get('config_sha256'),
        'storage': {
            'primary_level': STORAGE_PRIMARY_LEVEL,
            'secondary_level': STORAGE_SECONDARY_LEVEL,
            'primary': storage_primary,
            'secondary': storage_secondary,
        },
        'prediction': {
            'primary_level': PREDICTION_PRIMARY_LEVEL,
            'secondary_level': PREDICTION_SECONDARY_LEVEL,
            'primary': prediction_primary,
            'secondary': prediction_secondary,
        },
        'gates': gates,
    }
    out_prefix = Path(out_prefix)
    out_prefix.with_suffix('.summary.json').write_text(
        json.dumps(summary, indent=2, default=str), encoding='utf-8')
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out-prefix', default=None)
    args = ap.parse_args()
    raw_path = Path(args.raw)
    out_prefix = Path(args.out_prefix) if args.out_prefix else raw_path
    summary = summarize(raw_path, out_prefix)
    store = summary['storage']['primary']
    pred = summary['prediction']['primary']
    print(json.dumps({
        'experiment': summary['experiment'],
        'gates': summary['gates'],
        'storage_primary_level': summary['storage']['primary_level'],
        'prediction_primary_level': summary['prediction']['primary_level'],
        'pass_H_cap': store['capacity']['pass_H_cap'],
        'pass_H_forget': store['forgetting']['pass_H_forget'],
        'pass_H_overwrite': store['forgetting']['pass_H_overwrite'],
        'pass_H_interf': store['interference']['pass_H_interf'],
        'pass_P1': pred['pass_P1'],
        'resolver_sha256': summary['resolver_sha256'],
        'prediction_sha256': summary['prediction_sha256'],
    }, indent=2, default=str))


if __name__ == '__main__':
    main()
