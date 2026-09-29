"""Stage-2 plumbing/noise diagnostic for the MB persistent-memory probe.

This is deliberately small and mechanism-first.  It does not fit a decoder,
change the connectome, or interpret effects.  It exercises the strict paired
floor and replay controls proposed by DeepSeek before any larger Stage-2
factorial batch is allowed to run.
"""

from __future__ import annotations

import argparse
import json
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
from mb_persistent_memory_probe import (  # noqa: E402
    _anatomical_target_direction,
    _episode,
    _masks,
    _new_condition,
    _raw_target_direction,
    _select_targets,
    _target_summary,
)
from train_lm import load_corpus  # noqa: E402


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.astype(bool).tolist()
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _arr(row, key, dtype=np.float32):
    value = row.get(key)
    if value is None:
        return np.zeros(0, dtype=dtype)
    return np.asarray(value, dtype=dtype)


# Frozen tolerance contract (see deepseek_sp2_smoke_canonical_review.md §5/§11):
# counts/events must match bitwise; graded channels must agree within
# ``graded_tol`` absolute and ``graded_ulp`` float32 ULP.  These are plumbing
# gates, not mechanism thresholds.
GRADED_KEYS = ('_probe_mbon_voltage', '_probe_central_voltage',
               '_probe_modulated_mbon_current', '_probe_modulated_mbon_current_w0')
EVENT_KEYS = ('probe_mbon_counts', 'probe_central_counts')
PROFILE_GRADED_KEYS = ('v_pre_mbon', 'v_pre_central')
PROFILE_EVENT_KEYS = ('mbon_events', 'central_events')


def _ulp_distance(a, b):
    """Max ULP distance and mismatch count between two float32 arrays.

    Negative float32 bit patterns are shifted above positives, giving the
    standard monotonic integer ordering.  Exact-equal elements are forced to
    distance 0 so +0.0/-0.0 never inflates the statistic.
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        return float('inf'), int(max(a.size, b.size))
    if a.size == 0:
        return 0.0, 0
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, ai + (1 << 32), ai)
    bi = np.where(bi < 0, bi + (1 << 32), bi)
    d = np.abs(ai - bi)
    d[a == b] = 0
    return float(d.max()), int(np.count_nonzero(d))


def _graded_stats(write, no_write):
    """Per-channel max |Δ|, max ULP, and mismatch count for graded vectors."""
    stats = {}
    for key in GRADED_KEYS:
        a, b = _arr(write, key), _arr(no_write, key)
        if a.shape != b.shape:
            stats[key] = {'max_abs': float('inf'), 'max_ulp': float('inf'),
                          'mismatch': -1}
            continue
        delta = np.abs(a - b) if a.size else np.zeros(0, np.float32)
        max_ulp, mismatch = _ulp_distance(a, b)
        stats[key] = {
            'max_abs': float(delta.max()) if delta.size else 0.0,
            'max_ulp': max_ulp,
            'mismatch': mismatch,
        }
    return stats


def _event_stats(write, no_write):
    stats = {}
    for key in EVENT_KEYS:
        a, b = _arr(write, key), _arr(no_write, key)
        stats[key] = {
            'equal': bool(a.shape == b.shape and np.array_equal(a, b)),
            'n': int(a.size),
        }
    return stats


def _within_tolerance(graded, graded_tol, graded_ulp):
    """Per-channel plumbing tolerance: pass if |Δ| <= tol OR ULP <= cap.

    The two bounds are alternatives (as proposed in the canonical-smoke review:
    "|Δ| <= 1e-6 (or <= 16 ULP)"): an absolute bound at the metric scale, plus
    a ULP guard that still catches catastrophic drift when a channel happens to
    sit near zero.  Real write effects are orders of magnitude larger and must
    fail this check, so it is only applied to floor/replay pairs.
    """
    for stats in graded.values():
        if stats['max_abs'] > graded_tol and stats['max_ulp'] > graded_ulp:
            return False
    return True


def _replay_stats(no_write, replay, graded_tol, graded_ulp):
    """Replay plumbing check: events bitwise, graded within frozen tolerance."""
    graded = _graded_stats(no_write, replay)
    events = _event_stats(no_write, replay)
    profile_graded = {}
    pg = (no_write.get('probe_profile') or {})
    pr = (replay.get('probe_profile') or {})
    for key in PROFILE_GRADED_KEYS:
        a = np.asarray(pg.get(key, []), dtype=np.float32)
        b = np.asarray(pr.get(key, []), dtype=np.float32)
        max_ulp, mismatch = _ulp_distance(a, b)
        delta = np.abs(a - b) if a.shape == b.shape and a.size else np.zeros(0, np.float32)
        profile_graded[key] = {
            'max_abs': float(delta.max()) if delta.size else 0.0,
            'max_ulp': max_ulp,
            'mismatch': mismatch,
        }
    profile_events = {}
    for key in PROFILE_EVENT_KEYS:
        a = np.asarray(pg.get(key, []), dtype=np.float32)
        b = np.asarray(pr.get(key, []), dtype=np.float32)
        profile_events[key] = {
            'equal': bool(a.shape == b.shape and np.array_equal(a, b)),
            'n': int(a.size),
        }
    events_equal = (all(v['equal'] for v in events.values()) and
                    all(v['equal'] for v in profile_events.values()))
    graded_ok = (_within_tolerance(graded, graded_tol, graded_ulp) and
                 _within_tolerance(profile_graded, graded_tol, graded_ulp))
    all_stats = {**graded, **{f'profile:{k}': v for k, v in profile_graded.items()}}
    worst_key = max(all_stats, key=lambda k: all_stats[k]['max_ulp'])
    return {
        'replay_events_bitwise_equal': bool(events_equal),
        'replay_graded_within_tolerance': bool(graded_ok),
        'replay_graded': graded,
        'replay_profile_graded': profile_graded,
        'replay_events': events,
        'replay_profile_events': profile_events,
        'replay_max_ulp': float(all_stats[worst_key]['max_ulp']),
        'replay_max_abs': float(max(v['max_abs'] for v in all_stats.values())),
        'replay_worst_key': str(worst_key),
    }


def _pair_metrics(write, no_write, graded_tol=1e-6, graded_ulp=16):
    """Compute only pre-registered raw deltas and tolerance-based integrity flags."""
    wm = _arr(write, '_probe_mbon_voltage')
    nm = _arr(no_write, '_probe_mbon_voltage')
    wc = _arr(write, '_probe_central_voltage')
    nc = _arr(no_write, '_probe_central_voltage')
    wi = _arr(write, '_probe_modulated_mbon_current')
    ni = _arr(no_write, '_probe_modulated_mbon_current')
    ww = _arr(write, '_probe_modulated_mbon_current_w0')
    nw = _arr(no_write, '_probe_modulated_mbon_current_w0')
    wmc = _arr(write, 'probe_mbon_counts')
    nmc = _arr(no_write, 'probe_mbon_counts')
    wcc = _arr(write, 'probe_central_counts')
    ncc = _arr(no_write, 'probe_central_counts')
    graded = _graded_stats(write, no_write)
    events = _event_stats(write, no_write)
    events_equal = all(v['equal'] for v in events.values())
    graded_ok = _within_tolerance(graded, graded_tol, graded_ulp)
    strict_zero = bool(
        np.array_equal(wm, nm) and np.array_equal(wc, nc) and
        np.array_equal(wi, ni) and np.array_equal(ww, nw) and
        np.array_equal(wmc, nmc) and np.array_equal(wcc, ncc))
    out = {
        'delta_mbon_voltage_l2': float(np.linalg.norm(wm - nm)),
        'delta_central_voltage_l2': float(np.linalg.norm(wc - nc)),
        'delta_m3_live_l2': float(np.linalg.norm(wi - ni)),
        'delta_m3_w0_proxy_l2': float(np.linalg.norm(ww - nw)),
        'delta_mbon_events_l2': float(np.linalg.norm(wmc - nmc)),
        'delta_central_events_l2': float(np.linalg.norm(wcc - ncc)),
        'write_probe_mbon_event_sum': float(np.sum(wmc)) if wmc.size else 0.0,
        'write_probe_central_event_sum': float(np.sum(wcc)) if wcc.size else 0.0,
        'no_write_probe_mbon_event_sum': float(np.sum(nmc)) if nmc.size else 0.0,
        'no_write_probe_central_event_sum': float(np.sum(ncc)) if ncc.size else 0.0,
        'graded': graded,
        'events': events,
        'events_bitwise_equal': bool(events_equal),
        'graded_within_tolerance': bool(graded_ok),
        'pair_strict_bitwise_zero': strict_zero,
        'paired_reset_fingerprint_equal': write.get('fingerprint_reset') == no_write.get('fingerprint_reset'),
        'paired_start_v_zero': bool(float(write.get('fingerprint_reset', {}).get('v_sum', 1.0)) == 0.0 and
                                    float(no_write.get('fingerprint_reset', {}).get('v_sum', 1.0)) == 0.0),
        'paired_edge_weight_fingerprint_equal': (
            write.get('fingerprint_reset', {}).get('edge_weights') ==
            no_write.get('fingerprint_reset', {}).get('edge_weights')),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=str(ROOT / 'corpus' / 'tinyshakespeare.txt'))
    ap.add_argument('--data', default=None)
    ap.add_argument('--write-tokens', type=int, default=8)
    ap.add_argument('--gap-tokens', type=int, default=8)
    ap.add_argument('--gap-grid', default=None,
                    help='comma-separated gap token counts; overrides --gap-tokens')
    ap.add_argument('--probe-drive', type=float, default=2.0)
    ap.add_argument('--probe-settle-steps', type=int, default=2)
    ap.add_argument('--noise-grid', default='0,0.05,0.2')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--conditions', default='intact,write_off,plastic_none,sham_eta0')
    ap.add_argument('--write-char', default='e')
    ap.add_argument('--probe-char', default='e')
    ap.add_argument('--active', type=int, default=512)
    ap.add_argument('--drive', type=float, default=1.0)
    ap.add_argument('--gain', type=float, default=1.5)
    ap.add_argument('--tonic', type=float, default=0.05)
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--sustain', type=float, default=0.5)
    ap.add_argument('--trace-tau', type=float, default=0.1)
    ap.add_argument('--dan-pulse', type=float, default=2.0)
    ap.add_argument('--eta', type=float, default=0.02)
    ap.add_argument('--eligibility-tau', type=float, default=0.5)
    ap.add_argument('--weight-tau', type=float, default=16.0)
    ap.add_argument('--max-modulation', type=float, default=0.9)
    ap.add_argument('--eligibility-mix', type=float, default=0.0)
    ap.add_argument('--target-top', type=int, default=256)
    ap.add_argument('--target-selection', default='direct')
    ap.add_argument('--seed', type=int, default=20260917)
    ap.add_argument('--graded-tol', type=float, default=1e-6,
                    help='frozen plumbing tolerance: max |Δ| for graded channels')
    ap.add_argument('--graded-ulp', type=int, default=16,
                    help='frozen plumbing tolerance: max float32 ULP for graded channels')
    ap.add_argument('--no-fail-fast', action='store_true',
                    help='do not abort on a paired-reset fingerprint mismatch')
    ap.add_argument('--out', default='models/research_results/mb_sp2_diag_20260917')
    args = ap.parse_args()

    noise_grid = [float(x.strip()) for x in args.noise_grid.split(',') if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(',') if x.strip()]
    gap_values = ([int(x) for x in args.gap_grid.split(',') if x.strip()]
                  if args.gap_grid else [int(args.gap_tokens)])
    if args.repeats <= 0 or args.write_tokens <= 0 or any(g < 0 for g in gap_values):
        raise ValueError('repeats/write-tokens must be positive and gap-tokens non-negative')
    args.data = Path(args.data) if args.data else Path(
        __import__('os').environ.get('FLY_DATA', ROOT / 'data'))

    chars, _, _, _ = load_corpus(args.corpus, max(args.write_tokens + 2, 128), 1, 1)
    if args.write_char not in chars or args.probe_char not in chars:
        raise ValueError('write/probe characters must be in the corpus vocabulary')
    args.vocab = len(chars)
    write_ids = np.full(args.write_tokens, chars.index(args.write_char), np.int64)
    probe_id = int(chars.index(args.probe_char))
    args.probe_drive = float(args.probe_drive)
    args.probe_tokens = 1
    args.probe_plasticity = False
    args.gap_plasticity = False
    args.mod_perm_seed = 271828
    args.dan_pulse_unpaired = False
    args.save_probe_vectors = True
    args.record_probe_profile = True
    args.record_fingerprints = True
    args.skip_probe_weight_writeback = False

    brain0 = FlyBrain(data=args.data, device='cuda', batch=1, seed=args.seed,
                      sensory_input=False)
    masks = _masks(brain0)
    central_ids = _select_targets(brain0, args.target_selection, args.target_top)
    target_summary = _target_summary(brain0, central_ids)
    fixed_direction = _anatomical_target_direction(brain0, central_ids)
    _, raw_direction_norm = _raw_target_direction(brain0, central_ids)
    fallback_used = bool(raw_direction_norm <= 1e-8)

    cells = []
    t0 = time.time()
    base_seed = int(args.seed)
    total = len(noise_grid) * args.repeats * len(conditions) * len(gap_values)
    done = 0
    for noise_index, noise_hz in enumerate(noise_grid):
        args.noise_hz = float(noise_hz)
        for repeat in range(args.repeats):
            args.seed = base_seed + 10000 * noise_index + int(repeat)
            states = {name: _new_condition(
                args, name, masks, args.vocab, central_ids) for name in conditions}
            for condition in conditions:
                for gap in gap_values:
                    state = states[condition]
                    write = _episode(state, args, write_ids, probe_id,
                                     gap, True)
                    no_write = _episode(state, args, write_ids, probe_id,
                                        gap, False)
                    replay = _episode(state, args, write_ids, probe_id,
                                      gap, False)
                    pair = _pair_metrics(write, no_write,
                                         graded_tol=args.graded_tol,
                                         graded_ulp=args.graded_ulp)
                    replay_stats = _replay_stats(no_write, replay,
                                                 graded_tol=args.graded_tol,
                                                 graded_ulp=args.graded_ulp)
                    pair.update(replay_stats)
                    # Tolerance contract replaces the unreachable strict-bitwise
                    # flag: write_off is a floor when events are bitwise equal and
                    # graded channels stay inside the frozen plumbing tolerance.
                    pair['write_off_pair_within_tolerance'] = bool(
                        condition == 'write_off' and
                        pair['events_bitwise_equal'] and
                        pair['graded_within_tolerance'])
                    pair.update({
                        'condition': condition,
                        'noise_hz': float(noise_hz),
                        'repeat': int(repeat),
                        'repeat_seed': int(args.seed),
                        'write_tokens': int(args.write_tokens),
                        'gap_tokens': int(gap),
                        'probe_drive': float(args.probe_drive),
                        'probe_settle_steps': int(args.probe_settle_steps),
                        'probe_lesion': state.get('probe_lesion'),
                        'write_row': write,
                        'no_write_row': no_write,
                        'replay_row': replay,
                    })
                    cells.append(_jsonable(pair))
                    done += 1
                    print(f'[{done}/{total}] noise={noise_hz:g} rep={repeat} '
                          f'gap={gap:3d} {condition:20s} '
                          f'ΔM={pair["delta_mbon_voltage_l2"]:.5g} '
                          f'ΔC={pair["delta_central_voltage_l2"]:.5g} '
                          f'evM={pair["write_probe_mbon_event_sum"]:.0f} '
                          f'evC={pair["write_probe_central_event_sum"]:.0f} '
                          f'replay_ev={pair["replay_events_bitwise_equal"]} '
                          f'replay_ulp={pair["replay_max_ulp"]:.0f}', flush=True)
                    if not pair['paired_reset_fingerprint_equal'] and not args.no_fail_fast:
                        out_path = Path(args.out)
                        if not out_path.is_absolute():
                            out_path = ROOT / out_path
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        invalid = out_path.with_suffix('.invalid.json')
                        invalid.write_text(json.dumps({
                            'reason': 'paired_reset_fingerprint_mismatch',
                            'offending_cell': cells[-1],
                            'cells_so_far': cells,
                        }, indent=2, ensure_ascii=False), encoding='utf-8')
                        raise RuntimeError(
                            'paired reset fingerprint mismatch; partial dump at '
                            f'{invalid}')

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'experiment': 'SP2_diagnostic_pairing_noise_pre_reset_profile',
        'protocol': {
            'no_new_edges': True,
            'no_rewiring': True,
            'no_learned_readout': True,
            'no_external_context': True,
            'probe_tokens': 1,
            'probe_settle_steps': int(args.probe_settle_steps),
            'profile_v_pre_before_reset': True,
            'paired_write_no_write': True,
            'replay_no_write': True,
            'base_seed': base_seed,
            'tolerance_contract': {
                'events': 'bitwise equal required',
                'graded_max_abs': float(args.graded_tol),
                'graded_max_ulp': int(args.graded_ulp),
                'note': ('frozen plumbing gate; counts/events must match exactly, '
                         'graded channels may differ by SpMV/scatter-add reduction order'),
            },
        },
        'config': _jsonable(vars(args)),
        'connectome': {
            'neurons': int(brain0.n),
            'edges': int(len(brain0._W.indices.get())),
            'edge_counts': {k: int(len(v)) for k, v in masks.items()},
            'downstream_target_ids': central_ids.tolist(),
            'target_summary': target_summary,
            'fixed_direction_norm': float(np.linalg.norm(fixed_direction)),
            'fixed_direction_raw_norm': float(raw_direction_norm),
            'fixed_direction_fallback_used': fallback_used,
            'fixed_direction': fixed_direction.astype(np.float32).tolist(),
        },
        'write_char': args.write_char,
        'probe_char': args.probe_char,
        'write_pattern_chars': ''.join(chars[int(x)] for x in write_ids),
        'dt_token_seconds': float(brain0.dt) * float(args.k),
        'cells': cells,
        'elapsed_seconds': time.time() - t0,
        'analysis_handoff': (
            'Raw cells, paired fingerprints, replay rows, v_pre profiles, and '
            'event profiles are preserved for DeepSeek. Tolerance flags follow '
            'the frozen contract in opencode_analysis/deepseek_sp2_smoke_canonical_review.md.'),
    }
    out.with_suffix('.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    out.with_suffix('.md').write_text(
        '# SP2 diagnostic\n\n'
        'Raw paired rows and pre-reset profiles are stored in the JSON artifact.\n'
        'This file is an execution manifest, not an interpretation.\n',
        encoding='utf-8')
    print(f'saved {out.with_suffix(".json")}')
    print(f'saved {out.with_suffix(".md")}')


if __name__ == '__main__':
    main()
