"""Topology-preserving probe for persistent MB synaptic memory.

This experiment asks a narrower causal question than the language trainer:
after a local KC->MBON write, does the synaptic state survive a silent gap
that is much longer than the neural activity trace, and does it change the
next MBON/central-brain response?  Only existing CSR weights are modified.
No learned readout drives the brain; the reported downstream quantities are
spike-count norms from MBONs and topology-defined MBON target neurons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# The bundled CuPy runtime may need the workspace compatibility hook before
# any first-time elementwise kernel compilation.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from flybrain import FlyBrain
from flylm import Encoder
from mb_plasticity import KCMBONPlasticity
from train_lm import cells_matching, load_corpus
from train_plastic_lm import _add_events, _cuda_step_no_host_copy, _inject
from batchroll import _cuda_step_record_pre_reset

def _edge_positions(brain, pre_mask, post_mask):
    idx = brain._W.indices.get()
    ind = brain._W.indptr.get()
    out = []
    for post in np.flatnonzero(post_mask):
        lo, hi = int(ind[post]), int(ind[post + 1])
        keep = pre_mask[idx[lo:hi]]
        if keep.any():
            out.extend((np.flatnonzero(keep) + lo).tolist())
    return np.asarray(out, np.int64)


def _zero_edges(brain, positions):
    if len(positions):
        brain._W.data[brain.xp.asarray(positions)] = np.float32(0.0)


def _masks(brain):
    ct = np.asarray(brain.cell_type).astype(str)
    ss = np.asarray(brain.superclass).astype(str)
    mb = np.char.find(ct, 'MBON') >= 0
    kc = np.char.find(ct, 'KC') >= 0
    dan = ((np.char.find(ct, 'PAM') >= 0) |
           (np.char.find(ct, 'PPL') >= 0) |
           (np.char.find(ct, 'PPM') >= 0))
    sensory = np.char.find(ss, 'sensory') >= 0
    central = ~sensory & ~kc & ~mb & ~dan
    descending = ss == 'descending_neuron'
    cb = ss == 'cb_intrinsic'
    return {
        'kc_mbon': _edge_positions(brain, kc, mb),
        'dan_mbon': _edge_positions(brain, dan, mb),
        'mbon_central': _edge_positions(brain, mb, central),
        'mbon_dan': _edge_positions(brain, mb, dan),
        # Zero only existing MBON outgoing CSR entries; no replacement path
        # is added by this broad topology-preserving lesion.
        'mbon_all': _edge_positions(brain, mb, np.ones(brain.n, dtype=bool)),
        'mbon_descending': _edge_positions(brain, mb, descending),
        'mbon_cb_intrinsic': _edge_positions(brain, mb, cb),
    }


def _rank_targets_by_input(brain, source_mask, post_mask, top=256):
    """Rank existing postsynaptic rows by absolute input from a source set."""
    ct = np.asarray(brain.cell_type).astype(str)
    ss = np.asarray(brain.superclass).astype(str)
    idx = brain._W.indices.get()
    data = brain._W.data.get()
    ind = brain._W.indptr.get()
    score = np.zeros(brain.n, np.float32)
    degree = np.zeros(brain.n, np.int32)
    for post in np.flatnonzero(post_mask):
        lo, hi = int(ind[post]), int(ind[post + 1])
        keep = source_mask[idx[lo:hi]]
        if keep.any():
            score[post] = np.abs(data[lo:hi][keep]).sum()
            degree[post] = int(keep.sum())
    ids = np.flatnonzero(degree > 0)
    ids = ids[np.argsort(-score[ids], kind='stable')]
    return ids[:min(int(top), len(ids))].astype(np.int64)


def _rank_mbon_targets(brain, top=256):
    """Select existing MBON downstream central neurons by anatomical input."""
    ct = np.asarray(brain.cell_type).astype(str)
    ss = np.asarray(brain.superclass).astype(str)
    mb = np.char.find(ct, 'MBON') >= 0
    dan = ((np.char.find(ct, 'PAM') >= 0) |
           (np.char.find(ct, 'PPL') >= 0) |
           (np.char.find(ct, 'PPM') >= 0))
    post = ~mb & ~dan & (np.char.find(ct, 'KC') < 0)
    post &= np.char.find(ss, 'sensory') < 0
    return _rank_targets_by_input(brain, mb, post, top)


def _select_targets(brain, selection, top=256):
    """Select direct, non-CB, descending, or second-order MBON targets."""
    ct = np.asarray(brain.cell_type).astype(str)
    ss = np.asarray(brain.superclass).astype(str)
    mb = np.char.find(ct, 'MBON') >= 0
    dan = ((np.char.find(ct, 'PAM') >= 0) |
           (np.char.find(ct, 'PPL') >= 0) |
           (np.char.find(ct, 'PPM') >= 0))
    post = ~mb & ~dan & (np.char.find(ct, 'KC') < 0)
    post &= np.char.find(ss, 'sensory') < 0
    if selection == 'direct':
        return _rank_targets_by_input(brain, mb, post, top)
    if selection == 'direct_non_cb':
        return _rank_targets_by_input(brain, mb, post & (ss != 'cb_intrinsic'), top)
    if selection == 'descending':
        return _rank_targets_by_input(brain, mb, post & (ss == 'descending_neuron'), top)
    if selection == 'second_order':
        direct = _rank_targets_by_input(brain, mb, post, brain.n)
        direct_mask = np.zeros(brain.n, dtype=bool)
        direct_mask[direct] = True
        # The second-order group receives existing edges from direct MBON
        # targets but has no direct MBON input of its own.
        return _rank_targets_by_input(brain, direct_mask, post & ~direct_mask, top)
    raise ValueError(f'unknown target selection {selection!r}')


def _rewire_incoming_rows(brain, nodes, seed=123):
    """Degree/weight-matched local topology null for selected target rows."""
    rng = np.random.default_rng(seed)
    W = brain._W.copy()
    indptr = W.indptr.get()
    for node in np.asarray(nodes, np.int64):
        lo, hi = int(indptr[node]), int(indptr[node + 1])
        degree = hi - lo
        if degree:
            src = rng.choice(brain.n, size=degree, replace=False).astype(np.int32)
            W.indices[lo:hi] = brain.xp.asarray(src)
    W.has_sorted_indices = False
    W.has_canonical_format = False
    return W


def _shuffle_mbon_sources(brain, nodes, seed=123):
    """Shuffle only MBON presynaptic identities within selected target rows.

    This null preserves, for every selected row, the number of MBON inputs and
    the row's existing weight multiset.  KC->MBON plastic edges and all
    non-MBON inputs are untouched, so it is a stricter control than replacing
    every incoming source with an arbitrary neuron.
    """
    rng = np.random.default_rng(seed)
    W = brain._W.copy()
    ct = np.asarray(brain.cell_type).astype(str)
    mb = np.char.find(ct, 'MBON') >= 0
    mb_ids = np.flatnonzero(mb).astype(np.int32)
    indptr = W.indptr.get()
    indices = W.indices.get()
    for node in np.asarray(nodes, np.int64):
        lo, hi = int(indptr[node]), int(indptr[node + 1])
        if hi <= lo:
            continue
        slots = np.flatnonzero(mb[indices[lo:hi]]) + lo
        if len(slots):
            # Sampling without replacement keeps the row a simple graph.
            new_src = rng.choice(mb_ids, size=len(slots), replace=False)
            W.indices[slots] = brain.xp.asarray(new_src)
    W.has_sorted_indices = False
    W.has_canonical_format = False
    return W


def _raw_target_direction(brain, ids):
    """Signed MBON input-weight aggregate per target, before normalization.

    Returns ``(raw_direction, norm)``.  Keeping the raw norm lets diagnostics
    record whether the fixed projection fell back to the uniform vector instead
    of silently reporting a normalized direction.
    """
    ct = np.asarray(brain.cell_type).astype(str)
    mb = np.char.find(ct, 'MBON') >= 0
    idx = brain._W.indices.get()
    ind = brain._W.indptr.get()
    data = brain._W.data.get()
    out = np.zeros(len(ids), np.float32)
    for j, node in enumerate(np.asarray(ids, np.int64)):
        lo, hi = int(ind[node]), int(ind[node + 1])
        keep = mb[idx[lo:hi]]
        if keep.any():
            out[j] = np.asarray(data[lo:hi][keep], np.float32).sum()
    return out, float(np.linalg.norm(out))


def _anatomical_target_direction(brain, ids):
    """Fixed signed target direction from intact MBON input weights.

    The direction is computed once from the unmodified connectome and reused
    for every write/no-write episode and lesion/null condition.  It is not fit
    to probe responses, so the scalar projection is an anatomical diagnostic,
    not a learned reservoir readout.
    """
    out, norm = _raw_target_direction(brain, ids)
    if norm <= 1e-8:
        # This fallback is deterministic and only applies to a target group
        # with no signed MBON input aggregate.
        out.fill(1.0 / max(np.sqrt(len(out)), 1.0))
    else:
        out /= np.float32(norm)
    return out


def _target_summary(brain, ids):
    ct = np.asarray(brain.cell_type).astype(str)
    ss = np.asarray(brain.superclass).astype(str)
    idx = brain._W.indices.get()
    ind = brain._W.indptr.get()
    mb = np.char.find(ct, 'MBON') >= 0
    degrees = []
    for node in np.asarray(ids, np.int64):
        lo, hi = int(ind[node]), int(ind[node + 1])
        degrees.append(int(mb[idx[lo:hi]].sum()))
    return {
        'n': int(len(ids)),
        'superclass': {str(x): int((ss[ids] == x).sum()) for x in np.unique(ss[ids])},
        'cell_type': {str(x): int((ct[ids] == x).sum()) for x in np.unique(ct[ids])},
        'direct_mbon_indegree_mean': float(np.mean(degrees)) if degrees else 0.0,
        'direct_mbon_indegree_nonzero': int(np.sum(np.asarray(degrees) > 0)),
    }


def _slots(brain, ids):
    slot = brain.xp.full(brain.n, -1, dtype=brain.xp.int32)
    slot[brain.xp.asarray(ids)] = brain.xp.arange(len(ids), dtype=brain.xp.int32)
    return slot


def _modulation_stats(plastic):
    """Small host-side audit of the persistent KC->MBON state.

    This is deliberately a diagnostic only: it does not create a readout or
    feed anything back into the brain.  Keeping the summary here lets the SP1
    probe record the state before/after a topology-preserving modulation
    permutation without serialising the full 61k-edge vector for every cell.
    """
    if plastic is None:
        return {
            'l1': 0.0, 'l2': 0.0, 'max_abs': 0.0,
            'clip_fraction': 0.0, 'n_edges': 0,
        }
    x = plastic.xp.asnumpy(plastic.modulation).astype(np.float64, copy=False)
    lim = float(plastic.max_modulation)
    return {
        'l1': float(np.abs(x).sum()),
        'l2': float(np.linalg.norm(x)),
        'max_abs': float(np.abs(x).max()) if len(x) else 0.0,
        'clip_fraction': float(np.mean(np.isclose(np.abs(x), lim, rtol=0.0, atol=1e-6))) if len(x) else 0.0,
        'n_edges': int(len(x)),
    }


def _weight_fingerprint(brain, edge_positions=None):
    """Return a compact, deterministic checksum of live CSR weights.

    This is an audit value only.  It intentionally uses reductions on the
    existing sparse data rather than copying the 25M-edge matrix to host; the
    optional Stage-2 diagnostic can therefore prove that paired episodes
    start from the same weights without adding a model pathway.
    """
    data = brain._W.data
    if edge_positions is not None:
        data = data[brain.xp.asarray(edge_positions, dtype=brain.xp.int64)]
    xp = brain.xp
    vals = xp.asarray(data, dtype=xp.float32)
    return {
        'n': int(vals.size),
        'sum': float(xp.asnumpy(xp.sum(vals))),
        'abs_sum': float(xp.asnumpy(xp.sum(xp.abs(vals)))),
        'sumsq': float(xp.asnumpy(xp.sum(vals * vals))),
        'first': float(xp.asnumpy(vals[0])) if vals.size else 0.0,
        'last': float(xp.asnumpy(vals[-1])) if vals.size else 0.0,
    }


def _active_modulation_stats(plastic, kc_counts):
    """Summarise modulation carried by edges out of active probe KCs."""
    if plastic is None:
        return {
            'active_kc_count': 0, 'active_edge_count': 0,
            'l1': 0.0, 'l2': 0.0, 'max_abs': 0.0,
        }
    kc = plastic.xp.asnumpy(kc_counts).astype(np.float32, copy=False).reshape(-1)
    active = kc > 0
    edge_kc = np.asarray(plastic.edge_kc_slot, dtype=np.int32)
    mod = plastic.xp.asnumpy(plastic.modulation).astype(np.float64, copy=False)
    vals = mod[active[edge_kc]]
    return {
        'active_kc_count': int(active.sum()),
        'active_edge_count': int(len(vals)),
        'l1': float(np.abs(vals).sum()),
        'l2': float(np.linalg.norm(vals)),
        'max_abs': float(np.abs(vals).max()) if len(vals) else 0.0,
    }


def _permute_modulation_in_place(state, seed):
    """Permute modulation values over the same KC->MBON edge slots.

    The CSR topology, baseline weights, edge signs, and modulation multiset
    are unchanged.  Only the association between an existing edge slot and a
    previously written modulation value is shuffled.  This is the SP1 null;
    it must never add, remove, or rewire a connectome edge.
    """
    plastic = state.get('plastic')
    if plastic is None:
        return _modulation_stats(None), _modulation_stats(None)
    before = _modulation_stats(plastic)
    perm = np.random.default_rng(int(seed)).permutation(plastic.n_edges)
    perm_gpu = plastic.xp.asarray(perm, dtype=plastic.xp.int64)
    plastic.modulation[...] = plastic.modulation[perm_gpu]
    plastic.brain._W.data[plastic.edge_pos_gpu] = (
        plastic.w0_gpu * (np.float32(1.0) + plastic.modulation))
    after = _modulation_stats(plastic)
    state['_last_modulation_permutation'] = {
        'seed': int(seed), 'before': before, 'after': after,
        'permutation_is_bijection': bool(len(perm) == len(np.unique(perm))),
    }
    return before, after


def _new_condition(args, condition, masks, vocab, central_ids):
    brain = FlyBrain(data=args.data, device='cuda', batch=1, seed=args.seed,
                     sensory_input=False)
    # Paired causal episodes should differ only in the local plasticity write.
    # The base FlyBrain has calibrated intrinsic noise; disabling it by default
    # prevents GPU RNG scheduling from masquerading as a write-minus-no-write
    # effect.  A nonzero value remains available as an explicit robustness run.
    brain.noise_hz = np.float32(args.noise_hz)
    brain.gain = np.float32(args.gain)
    brain.tonic = np.float32(args.tonic)
    ct = np.asarray(brain.cell_type).astype(str)
    ss = np.asarray(brain.superclass).astype(str)

    # Lesions are functional zeroing of existing CSR data only.  Constructing
    # the local plasticity object after DAN/MBON lesions makes its anatomical
    # gate reflect the lesioned graph.  KC->MBON output is re-zeroed after each
    # plastic update because its internal modulation remains measurable.
    lesion = condition if condition in (
        'dan_mbon', 'mbon_central', 'mbon_dan', 'mbon_all',
        'mbon_descending', 'mbon_cb_intrinsic') else None
    if lesion is not None:
        _zero_edges(brain, masks[lesion])
    if condition == 'topology_null':
        # Local null only: preserve each selected target row's indegree and
        # weight multiset while replacing presynaptic identities. KC->MBON
        # plastic edges are not touched, isolating downstream topology.
        brain._W = _rewire_incoming_rows(
            brain, central_ids, seed=int(args.topology_null_seed))
    if condition == 'mbon_source_shuffle':
        # Stricter null: preserve MBON input count and row weights, but shuffle
        # only MBON presynaptic identities.  Non-MBON inputs and all KC->MBON
        # plastic edges remain anatomically unchanged.
        brain._W = _shuffle_mbon_sources(
            brain, central_ids, seed=int(args.topology_null_seed))
    kc = np.flatnonzero(np.char.find(ct, 'KC') >= 0)
    mbon = np.flatnonzero(np.char.find(ct, 'MBON') >= 0)
    dan = np.flatnonzero((np.char.find(ct, 'PAM') >= 0) |
                         (np.char.find(ct, 'PPL') >= 0) |
                         (np.char.find(ct, 'PPM') >= 0))
    # ``plastic_none`` is the explicit name for the old no-object diagnostic;
    # keep ``frozen`` as a backwards-compatible alias.  ``write_off`` keeps
    # the object and its complete reset/decay path but the episode wrapper
    # disables the local update, giving a true paired floor.
    plastic_none = condition in ('frozen', 'plastic_none')
    eta = 0.0 if condition == 'sham_eta0' else args.eta
    # New Stage-2 readout-zero control: zero the existing KC->MBON slots only
    # during the probe window, keeping the write intact.  The construction-time
    # baseline-zero semantics are kept separately as ``kc_mbon_writezero`` so
    # both the "no baseline" and "no readout" nulls are available.
    if condition == 'kc_mbon_writezero':
        _zero_edges(brain, masks['kc_mbon'])
    plastic = None if plastic_none else KCMBONPlasticity(
        brain, mode='biological', eta=eta,
        eligibility_tau=args.eligibility_tau, weight_tau=args.weight_tau,
        max_modulation=args.max_modulation,
        eligibility_mix=args.eligibility_mix)
    if plastic is not None:
        plastic.reset()
    if condition == 'kc_mbon':
        _zero_edges(brain, masks['kc_mbon'])

    pn = cells_matching(brain, ('PN',))
    enc = Encoder(pn, vocab, active=args.active, drive=args.drive,
                  window=args.window, gamma=args.gamma, seed=3,
                  position_codes=False)
    brain.reset(seed=args.seed)
    if condition in ('dan_mbon', 'mbon_central', 'mbon_dan', 'mbon_all',
                     'mbon_descending', 'mbon_cb_intrinsic'):
        _zero_edges(brain, masks[condition])
    if condition == 'kc_mbon':
        _zero_edges(brain, masks['kc_mbon'])
    return {
        'brain': brain, 'enc': enc, 'plastic': plastic,
        'condition': condition,
        'kc': kc, 'mbon': mbon, 'dan': dan,
        'pam_ids': np.flatnonzero(np.char.find(ct, 'PAM') >= 0).astype(np.int64),
        'kc_slot': _slots(brain, kc), 'dan_slot': _slots(brain, dan),
        'mbon_slot': _slots(brain, mbon),
        'central_slot': _slots(brain, central_ids),
        'central_ids': central_ids,
        'trace_central': brain.xp.zeros(len(central_ids), brain.xp.float32),
        'trace_mbon': brain.xp.zeros(len(mbon), brain.xp.float32),
        'kc_counts': brain.xp.zeros(len(kc), brain.xp.float32),
        'dan_counts': brain.xp.zeros(len(dan), brain.xp.float32),
        'decay': np.float32(np.exp(-brain.dt / args.trace_tau)),
        'kc_lesion': condition in ('kc_mbon', 'kc_mbon_writezero'),
        'probe_lesion': (
            'kc_mbon' if condition == 'kc_mbon_readout_zero' else
            ('mbon_central' if condition == 'mbon_central_readout_zero' else None)),
        'edge_lesion': condition if condition in (
            'dan_mbon', 'mbon_central', 'mbon_dan', 'mbon_all',
            'mbon_descending', 'mbon_cb_intrinsic') else None,
        'masks': masks,
    }


def _reset_episode(state, args):
    brain = state['brain']
    brain.reset(seed=args.seed)
    state['trace_central'].fill(0)
    state['trace_mbon'].fill(0)
    if state['plastic'] is not None:
        state['plastic'].reset()
    if state['edge_lesion'] is not None:
        _zero_edges(brain, state['masks'][state['edge_lesion']])
    if state['kc_lesion']:
        _zero_edges(brain, state['masks']['kc_mbon'])
    state['_last_probe_profile'] = None
    state['_last_w0_current'] = np.zeros(len(state['mbon']), np.float32)
    if getattr(args, 'record_fingerprints', False):
        state['_episode_fingerprint_start'] = {
            'seed': int(args.seed),
            'v_sum': float(brain.xp.asnumpy(brain.xp.sum(brain.v))),
            'v_l2': float(brain.xp.asnumpy(brain.xp.linalg.norm(brain.v))),
            'weights': _weight_fingerprint(brain),
            'edge_weights': (_weight_fingerprint(
                brain, state['plastic'].edge_pos) if state.get('plastic') is not None else None),
            'edge_identity': (state['plastic'].edge_identity_fingerprint()
                              if state.get('plastic') is not None else None),
        }


def _advance_token(state, args, token_id, allow_plastic=True, pulse_dan=False,
                   capture_probe=False, record_profile=False, settle_steps=0,
                   skip_weight_writeback=False, inject_sustain=None):
    """Advance one biological token and return pre-update spike counts.

    ``record_profile`` is an additive Stage-2 diagnostic.  It snapshots the
    selected MBON/central voltages immediately before threshold/reset and can
    append a short silent settle window, so a graded response is not confused
    with the ordinary end-of-token post-reset voltage.  The default path is
    unchanged for the existing probes.

    ``inject_sustain`` overrides only the post-first-step injection scale for
    this token (probe-only impulse profile).  ``None`` keeps ``args.sustain``,
    so the write phase and the existing default path are untouched.
    """
    brain = state['brain']
    enc = state['enc']
    kcounts, dcounts = state['kc_counts'], state['dan_counts']
    kcounts.fill(0)
    dcounts.fill(0)
    mbon_counts = brain.xp.zeros(len(state['mbon']), brain.xp.float32)
    central_counts = brain.xp.zeros(len(state['central_ids']), brain.xp.float32)
    token_arr = np.asarray([int(token_id)], np.int64)
    settle_steps = max(0, int(settle_steps)) if record_profile else 0
    profile_mbon_v, profile_central_v = [], []
    profile_mbon_events, profile_central_events = [], []
    profile_steps = int(args.k) + settle_steps
    record_ids = np.concatenate((np.asarray(state['mbon'], np.int64),
                                 np.asarray(state['central_ids'], np.int64)))
    for step in range(profile_steps):
        if step < int(args.k):
            sustain_scale = (args.sustain if inject_sustain is None
                             else float(inject_sustain))
            _inject(brain, enc, token_arr, 0, 1.0 if step == 0 else sustain_scale)
        # Optional write-only pulse onto existing PAM neurons.  This is a
        # local dopamine-gate control, not a learned label or an added edge;
        # it makes the causal memory probe deterministic when intrinsic noise
        # is disabled.  Probes and no-write episodes never receive it.
        if pulse_dan and args.dan_pulse > 0 and step == 0:
            brain.v[brain.xp.asarray(state['pam_ids']), 0] += np.float32(args.dan_pulse)
        if record_profile:
            fired, v_pre = _cuda_step_record_pre_reset(brain, record_ids)
            v_pre = brain.xp.asnumpy(v_pre[:, 0]).astype(np.float32, copy=True)
            profile_mbon_v.append(v_pre[:len(state['mbon'])])
            profile_central_v.append(v_pre[len(state['mbon']):])
        else:
            fired = _cuda_step_no_host_copy(brain)
        state['trace_central'] *= state['decay']
        state['trace_mbon'] *= state['decay']
        _add_events(fired, brain, state['central_slot'], state['trace_central'], central_counts)
        _add_events(fired, brain, state['mbon_slot'], state['trace_mbon'], mbon_counts)
        _add_events(fired, brain, state['kc_slot'], None, kcounts)
        _add_events(fired, brain, state['dan_slot'], None, dcounts)
        if record_profile:
            step_mbon = brain.xp.zeros(len(state['mbon']), brain.xp.float32)
            step_central = brain.xp.zeros(len(state['central_ids']), brain.xp.float32)
            _add_events(fired, brain, state['mbon_slot'], None, step_mbon)
            _add_events(fired, brain, state['central_slot'], None, step_central)
            profile_mbon_events.append(brain.xp.asnumpy(step_mbon).astype(np.float32))
            profile_central_events.append(brain.xp.asnumpy(step_central).astype(np.float32))
    # Read the persistent component of the KC->MBON synaptic current before
    # the local update for this token.  This is a direct anatomical readout of
    # existing edge weights, not a learned decoder; it remains measurable even
    # when the MBON does not cross spike threshold.
    if state['plastic'] is not None:
        import cupyx
        p = state['plastic']
        # Use the just-collected KC activity rather than the persistent
        # eligibility bank: the probe asks whether an old edge state is read
        # by a new sensory presentation.  ``edge_current`` is computed from
        # the live CSR W, not from the captured w0 proxy; this makes a
        # KC->MBON readout-zero control a genuine zero-current check.
        edge_current = brain._W.data[p.edge_pos_gpu] * (
            kcounts[p.edge_kc_slot_gpu] / np.float32(max(args.k, 1)))
        edge_current_w0 = p.w0_gpu * p.modulation * (
            kcounts[p.edge_kc_slot_gpu] / np.float32(max(args.k, 1)))
        current_mbon = brain.xp.zeros(len(state['mbon']), brain.xp.float32)
        current_mbon_w0 = brain.xp.zeros(len(state['mbon']), brain.xp.float32)
        cupyx.scatter_add(current_mbon, p.edge_mbon_slot_gpu, edge_current)
        cupyx.scatter_add(current_mbon_w0, p.edge_mbon_slot_gpu, edge_current_w0)
        current_mbon = brain.xp.asnumpy(current_mbon)
        state['_last_w0_current'] = brain.xp.asnumpy(current_mbon_w0)
    else:
        current_mbon = np.zeros(len(state['mbon']), np.float32)
        state['_last_w0_current'] = np.zeros(len(state['mbon']), np.float32)
    if capture_probe:
        state['_last_probe_active_kc_mask'] = (brain.xp.asnumpy(kcounts) > 0)
        state['_last_probe_active_modulation'] = _active_modulation_stats(
            state['plastic'], kcounts)
    if state['plastic'] is not None:
        if allow_plastic:
            state['plastic'].update(kcounts, dcounts,
                                    write_back=not skip_weight_writeback)
            if state['kc_lesion']:
                _zero_edges(brain, state['masks']['kc_mbon'])
        else:
            # A passive probe reads the old synaptic state and then lets it
            # forget; it must not create a second KC×DAN write.
            state['plastic'].decay_only(write_back=not skip_weight_writeback)
            if state['kc_lesion']:
                _zero_edges(brain, state['masks']['kc_mbon'])
    # Keep the most recent raw population counts available to the mechanism
    # wrapper.  The default probe does not serialise these arrays; SP1 asks for
    # them explicitly as a first-probe audit.
    state['_last_kc_counts'] = brain.xp.asnumpy(kcounts).copy()
    state['_last_dan_counts'] = brain.xp.asnumpy(dcounts).copy()
    state['_last_mbon_counts'] = brain.xp.asnumpy(mbon_counts).copy()
    state['_last_central_counts'] = brain.xp.asnumpy(central_counts).copy()
    if record_profile:
        state['_last_probe_profile'] = {
            'v_pre_mbon': np.asarray(profile_mbon_v, np.float32),
            'v_pre_central': np.asarray(profile_central_v, np.float32),
            'mbon_events': np.asarray(profile_mbon_events, np.float32),
            'central_events': np.asarray(profile_central_events, np.float32),
            'end_mbon_voltage': brain.xp.asnumpy(
                brain.v[brain.xp.asarray(state['mbon']), 0]).astype(np.float32),
            'end_central_voltage': brain.xp.asnumpy(
                brain.v[brain.xp.asarray(state['central_ids']), 0]).astype(np.float32),
        }
    mbon_voltage = brain.xp.asnumpy(brain.v[brain.xp.asarray(state['mbon']), 0])
    central_voltage = brain.xp.asnumpy(
        brain.v[brain.xp.asarray(state['central_ids']), 0])
    return (brain.xp.asnumpy(mbon_counts), brain.xp.asnumpy(central_counts),
            mbon_voltage, central_voltage, current_mbon)


def _advance_silent(state, args, allow_plastic=True, skip_weight_writeback=False):
    """Advance one token with no PN injection; local decay still proceeds.

    ``skip_weight_writeback`` is additive (E4B) and only used by the required
    write-phase-only KC->MBON lesion: the local update/decay of ``modulation``
    is unchanged, but the CSR write-back is skipped while the lesion has the
    existing KC->MBON store zeroed.  The default ``False`` path is unchanged.
    """
    brain = state['brain']
    kcounts, dcounts = state['kc_counts'], state['dan_counts']
    kcounts.fill(0)
    dcounts.fill(0)
    for _ in range(args.k):
        fired = _cuda_step_no_host_copy(brain)
        state['trace_central'] *= state['decay']
        state['trace_mbon'] *= state['decay']
        _add_events(fired, brain, state['central_slot'], state['trace_central'])
        _add_events(fired, brain, state['mbon_slot'], state['trace_mbon'])
        _add_events(fired, brain, state['kc_slot'], None, kcounts)
        _add_events(fired, brain, state['dan_slot'], None, dcounts)
    if allow_plastic and state['plastic'] is not None:
        # Silent gaps use passive forgetting by default.  The flag is kept so
        # exploratory runs can explicitly re-enable spontaneous DAN writes.
        state['plastic'].update(kcounts, dcounts,
                                write_back=not skip_weight_writeback)
        if state['kc_lesion']:
            _zero_edges(brain, state['masks']['kc_mbon'])
    elif state['plastic'] is not None:
        state['plastic'].decay_only(write_back=not skip_weight_writeback)
        if state['kc_lesion']:
            _zero_edges(brain, state['masks']['kc_mbon'])


def _episode(state, args, write_ids, probe_id, gap_tokens, allow_plastic,
             write_program=None):
    _reset_episode(state, args)
    # ``write_off`` is the strict paired floor: it retains the same plastic
    # object, reset path and passive decay path as ``intact`` but disables all
    # local updates in both branches.  This avoids comparing a plastic object
    # against the legacy ``plastic=None`` simulator path.
    episode_write = bool(allow_plastic and state.get('condition') != 'write_off')
    # Required E4B write-phase-only, temporally restored KC->MBON lesion.  For
    # the ``kc_mbon_writezero_restore`` condition the existing KC->MBON CSR
    # bytes are snapshotted and zeroed at the start of the write phase; every
    # write-phase token skips only the CSR write-back so the local KC×DAN
    # ``modulation`` update is byte-identical to ``intact``.  The exact bytes
    # are restored at the write->gap boundary, after which the ordinary decay
    # and probe path runs with the accumulated modulation.  Default off keeps
    # the frozen E3 path unchanged.
    write_lesion = bool(state.get('condition') == 'kc_mbon_writezero_restore')
    write_lesion_positions_gpu = None
    saved_write_lesion_data = None
    write_lesion_saved_sha256 = None
    write_lesion_restored_sha256 = None
    write_lesion_baseline_restored = None
    write_lesion_materialized_sha256 = None
    write_lesion_materialized_expected_sha256 = None
    write_lesion_materialized_matches_expected = None
    write_lesion_positions_match_plastic = None
    if write_lesion and state.get('plastic') is not None:
        brain_wl = state['brain']
        wl_positions = state['masks']['kc_mbon']
        write_lesion_positions_gpu = brain_wl.xp.asarray(
            wl_positions, dtype=brain_wl.xp.int64)
        saved_write_lesion_data = brain_wl._W.data[
            write_lesion_positions_gpu].copy()
        write_lesion_saved_sha256 = hashlib.sha256(np.ascontiguousarray(
            brain_wl.xp.asnumpy(saved_write_lesion_data),
            dtype=np.float32).tobytes()).hexdigest()
        _zero_edges(brain_wl, wl_positions)
    # ``write_program`` (additive, N2 interference) is a list of
    # ``(token_id_or_None, count)`` blocks.  ``None`` keeps the existing
    # single-character write path bit-for-bit; a ``None`` token_id advances a
    # silent token with no PN injection and no DAN pulse.
    block_snapshots = []
    if write_program is None:
        program = [(int(t), 1) for t in write_ids]
    else:
        program = [(None if t is None else int(t), int(n)) for t, n in write_program]
    save_blocks = bool(getattr(args, 'save_block_snapshots', False))
    for block_index, (token_id, count) in enumerate(program):
        for _ in range(count):
            # By default the PAM pulse is paired across write/no-write episodes;
            # only the local KC×DAN update is toggled.  An explicitly unpaired
            # run is available for measuring the raw dopamine-pulse transient.
            pulse_dan = bool(token_id is not None and args.dan_pulse > 0 and
                             (episode_write or not args.dan_pulse_unpaired))
            if token_id is None:
                _advance_silent(state, args, allow_plastic=episode_write,
                                skip_weight_writeback=write_lesion)
            else:
                _advance_token(state, args, token_id,
                               allow_plastic=episode_write, pulse_dan=pulse_dan,
                               skip_weight_writeback=write_lesion)
        if save_blocks:
            snap = {'block': block_index, 'token_id': token_id, 'count': count,
                    'modulation': _modulation_stats(state['plastic'])}
            if episode_write and state['plastic'] is not None:
                snap['modulation_vector'] = np.asarray(
                    state['brain'].xp.asnumpy(state['plastic'].modulation),
                    dtype=np.float32).copy()
            block_snapshots.append(snap)
    if write_lesion and saved_write_lesion_data is not None:
        # Write->gap boundary, step (1): restore the exact bytes removed at the
        # start of the write phase.  ``modulation`` was never touched by the
        # lesion, so this is the unchanged pre-write baseline.
        state['brain']._W.data[write_lesion_positions_gpu] = (
            saved_write_lesion_data)
        write_lesion_restored_sha256 = hashlib.sha256(np.ascontiguousarray(
            state['brain'].xp.asnumpy(
                state['brain']._W.data[write_lesion_positions_gpu]),
            dtype=np.float32).tobytes()).hexdigest()
        write_lesion_baseline_restored = bool(
            write_lesion_saved_sha256 == write_lesion_restored_sha256)
        # Write->gap boundary, step (2): materialize the post-write memory
        # readout ``w0*(1+modulation)`` so the gap/probe path integrates
        # against the accumulated local memory rather than the freed baseline.
        # Without this the lesion would be a trivial baseline-vs-memory
        # confound (B3).  The materialization is recorded as its own hash and
        # checked against the exact frozen plasticity expression.
        if state['plastic'] is not None and write_lesion_baseline_restored:
            wl_plastic = state['plastic']
            write_lesion_positions_match_plastic = bool(np.array_equal(
                state['brain'].xp.asnumpy(write_lesion_positions_gpu),
                np.asarray(wl_plastic.edge_pos, dtype=np.int64)))
            if not write_lesion_positions_match_plastic:
                raise RuntimeError(
                    'write-lesion CSR positions do not match the plastic '
                    'edge positions; refusing to materialize the readout')
            expected_materialized = wl_plastic.w0_gpu * (
                np.float32(1.0) + wl_plastic.modulation)
            state['brain']._W.data[wl_plastic.edge_pos_gpu] = (
                expected_materialized)
            actual_materialized = state['brain']._W.data[
                wl_plastic.edge_pos_gpu]
            write_lesion_materialized_sha256 = hashlib.sha256(
                np.ascontiguousarray(state['brain'].xp.asnumpy(
                    actual_materialized), dtype=np.float32).tobytes()
            ).hexdigest()
            write_lesion_materialized_expected_sha256 = hashlib.sha256(
                np.ascontiguousarray(state['brain'].xp.asnumpy(
                    expected_materialized), dtype=np.float32).tobytes()
            ).hexdigest()
            write_lesion_materialized_matches_expected = bool(
                write_lesion_materialized_sha256 ==
                write_lesion_materialized_expected_sha256)
    plastic = state['plastic']
    modulation_after_write = _modulation_stats(plastic)
    # Optional SP2-C audit: keep the full KC->MBON modulation vector captured at
    # the end of the write phase (before the silent gap decays it) so offline
    # analyses can test whether different write characters produce
    # distinguishable synaptic patterns.  Opt-in; not part of the default path.
    modulation_after_write_vector = None
    if (getattr(args, 'save_modulation_after_write', False)
            and episode_write and plastic is not None):
        modulation_after_write_vector = np.asarray(
            state['brain'].xp.asnumpy(plastic.modulation), dtype=np.float32).copy()
    modulation_permutation = None
    if (episode_write and state.get('condition') == 'mod_perm'
            and plastic is not None):
        # Apply the null only after all write tokens and before the silent gap;
        # the no-write branch has an all-zero modulation and is unchanged.
        _, _ = _permute_modulation_in_place(
            state, getattr(args, 'mod_perm_seed', 271828))
        modulation_permutation = state.get('_last_modulation_permutation')
    edge_after_write = (float(state['brain'].xp.linalg.norm(plastic.modulation).get())
                        if plastic is not None else 0.0)
    for _ in range(int(gap_tokens)):
        _advance_silent(state, args,
                        allow_plastic=episode_write and args.gap_plasticity)
    pre_c = float(state['brain'].xp.linalg.norm(state['trace_central']).get())
    pre_m = float(state['brain'].xp.linalg.norm(state['trace_mbon']).get())
    pre_c_v = float(state['brain'].xp.linalg.norm(
        state['brain'].v[state['brain'].xp.asarray(state['central_ids']), 0]).get())
    pre_m_v = float(state['brain'].xp.linalg.norm(
        state['brain'].v[state['brain'].xp.asarray(state['mbon']), 0]).get())
    edge_preprobe = (float(state['brain'].xp.linalg.norm(plastic.modulation).get())
                     if plastic is not None else 0.0)
    modulation_preprobe = _modulation_stats(plastic)
    probe_records = []
    original_drive = state['enc'].drive
    state['enc'].drive = float(args.probe_drive)
    # Probe-time readout lesion: zero the existing lesion slots for the probe
    # window only.  The write state is preserved.  ``kc_mbon_readout_zero``
    # zeros the plastic KC->MBON slots and must skip the per-token weight
    # write-back (which would otherwise rewrite them); ``mbon_central_readout_zero``
    # zeros MBON->central slots that the write-back never touches, so its
    # write-back proceeds normally.  The exact bytes removed are saved and
    # restored in the ``finally`` block, so the lesion is truly probe-only.
    probe_lesion = state.get('probe_lesion')
    lesion_positions = None
    lesion_positions_gpu = None
    saved_lesion_data = None
    if probe_lesion is not None:
        brain_lesion = state['brain']
        lesion_positions = state['masks'][probe_lesion]
        lesion_positions_gpu = brain_lesion.xp.asarray(
            lesion_positions, dtype=brain_lesion.xp.int64)
        saved_lesion_data = brain_lesion._W.data[lesion_positions_gpu].copy()
        _zero_edges(brain_lesion, lesion_positions)
    probe_skip_writeback = bool(getattr(args, 'skip_probe_weight_writeback', False))
    if probe_lesion == 'kc_mbon':
        probe_skip_writeback = True
    try:
        for probe_index in range(int(args.probe_tokens)):
            probe_mbon, probe_central, probe_mbon_v, probe_central_v, probe_current = _advance_token(
                state, args, int(probe_id),
                allow_plastic=episode_write and args.probe_plasticity,
                capture_probe=True,
                record_profile=bool(getattr(args, 'record_probe_profile', False)),
                settle_steps=int(getattr(args, 'probe_settle_steps', 0)),
                skip_weight_writeback=probe_skip_writeback,
                inject_sustain=getattr(args, 'probe_sustain', None))
            probe_records.append({
                'probe_index': probe_index,
                'probe_mbon_l2': float(np.linalg.norm(probe_mbon)),
                'probe_central_l2': float(np.linalg.norm(probe_central)),
                'probe_mbon_voltage_l2': float(np.linalg.norm(probe_mbon_v)),
                'probe_central_voltage_l2': float(np.linalg.norm(probe_central_v)),
                'probe_modulated_mbon_current_l2': float(np.linalg.norm(probe_current)),
                '_probe_mbon_voltage': probe_mbon_v,
                '_probe_central_voltage': probe_central_v,
                '_probe_modulated_mbon_current': probe_current,
                '_probe_modulated_mbon_current_w0': state.get(
                    '_last_w0_current', np.zeros(len(state['mbon']), np.float32)),
                '_probe_profile': state.get('_last_probe_profile'),
            })
    finally:
        state['enc'].drive = original_drive
        if lesion_positions_gpu is not None and saved_lesion_data is not None:
            state['brain']._W.data[lesion_positions_gpu] = saved_lesion_data
        if probe_lesion == 'kc_mbon' and plastic is not None:
            # Match the ordinary end-of-probe weight state without replaying
            # the decay: the modulation was already decayed by the skipped
            # write-back.
            state['brain']._W.data[plastic.edge_pos_gpu] = (
                plastic.w0_gpu * (np.float32(1.0) + plastic.modulation))
    first = probe_records[0]
    edge_postprobe = (float(state['brain'].xp.linalg.norm(plastic.modulation).get())
                      if plastic is not None else 0.0)
    out = {
        'edge_norm_after_write': edge_after_write,
        'edge_norm_preprobe': edge_preprobe,
        'edge_norm_postprobe': edge_postprobe,
        'neural_trace_l2_preprobe': pre_c,
        'mbon_trace_l2_preprobe': pre_m,
        'neural_voltage_l2_preprobe': pre_c_v,
        'mbon_voltage_l2_preprobe': pre_m_v,
        'probe_mbon_l2': first['probe_mbon_l2'],
        'probe_central_l2': first['probe_central_l2'],
        'probe_mbon_voltage_l2': first['probe_mbon_voltage_l2'],
        'probe_central_voltage_l2': first['probe_central_voltage_l2'],
        'probe_modulated_mbon_current_l2': first['probe_modulated_mbon_current_l2'],
        'probe_mbon_mean': float(np.mean(probe_mbon)) if len(probe_mbon) else 0.0,
        'probe_central_mean': float(np.mean(probe_central)) if len(probe_central) else 0.0,
        '_probe_mbon_voltage': first['_probe_mbon_voltage'],
        '_probe_central_voltage': first['_probe_central_voltage'],
        '_probe_modulated_mbon_current': first['_probe_modulated_mbon_current'],
        '_probe_modulated_mbon_current_w0': first['_probe_modulated_mbon_current_w0'],
        '_probe_sequence': probe_records,
        'write_lesion_applied': bool(write_lesion),
        'write_lesion_saved_sha256': write_lesion_saved_sha256,
        'write_lesion_restored_sha256': write_lesion_restored_sha256,
        'write_lesion_bytes_identical': (
            bool(write_lesion_saved_sha256 == write_lesion_restored_sha256)
            if write_lesion else None),
        'write_lesion_baseline_restored': (
            write_lesion_baseline_restored if write_lesion else None),
        'write_lesion_materialized_sha256': (
            write_lesion_materialized_sha256 if write_lesion else None),
        'write_lesion_materialized_expected_sha256': (
            write_lesion_materialized_expected_sha256
            if write_lesion else None),
        'write_lesion_materialized_matches_expected': (
            write_lesion_materialized_matches_expected
            if write_lesion else None),
        'write_lesion_positions_match_plastic': (
            write_lesion_positions_match_plastic if write_lesion else None),
        'modulation_stats_after_write': modulation_after_write,
        'modulation_stats_preprobe': modulation_preprobe,
        'modulation_permutation': modulation_permutation,
        'write_block_snapshots': block_snapshots,
    }
    if modulation_after_write_vector is not None:
        out['modulation_after_write_vector'] = modulation_after_write_vector
    if getattr(args, 'record_fingerprints', False):
        out['fingerprint_reset'] = state.get('_episode_fingerprint_start')
        out['fingerprint_end'] = {
            'v_sum': float(state['brain'].xp.asnumpy(
                state['brain'].xp.sum(state['brain'].v))),
            'v_l2': float(state['brain'].xp.asnumpy(
                state['brain'].xp.linalg.norm(state['brain'].v))),
            'weights': _weight_fingerprint(state['brain']),
            'edge_weights': (_weight_fingerprint(
                state['brain'], plastic.edge_pos) if plastic is not None else None),
            'edge_identity': (plastic.edge_identity_fingerprint()
                              if plastic is not None else None),
        }
    if getattr(args, 'save_probe_vectors', False):
        out.update({
            'probe_kc_counts': state.get('_last_kc_counts', np.zeros(len(state['kc']), np.float32)),
            'probe_dan_counts': state.get('_last_dan_counts', np.zeros(len(state['dan']), np.float32)),
            'probe_mbon_counts': state.get('_last_mbon_counts', np.zeros(len(state['mbon']), np.float32)),
            'probe_central_counts': state.get('_last_central_counts', np.zeros(len(state['central_ids']), np.float32)),
            'probe_kc_mask': state.get('_last_probe_active_kc_mask', np.zeros(len(state['kc']), dtype=bool)),
            'probe_active_modulation_stats': state.get('_last_probe_active_modulation', _active_modulation_stats(None, [])),
            'probe_profile': state.get('_last_probe_profile'),
        })
    return out


def _fit_decay(rows, dt_token):
    x, y = [], []
    for gap, row in rows.items():
        val = float(row['write']['edge_norm_preprobe'])
        if val > 1e-8:
            x.append(float(gap) * dt_token)
            y.append(np.log(val))
    if len(x) < 2:
        return {'tau_seconds_fit': None, 'n_points': len(x)}
    slope, intercept = np.polyfit(np.asarray(x), np.asarray(y), 1)
    return {'tau_seconds_fit': float(-1.0 / slope) if slope < 0 else None,
            'slope_per_second': float(slope), 'intercept': float(intercept),
            'n_points': len(x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=str(ROOT / 'corpus' / 'tinyshakespeare.txt'))
    ap.add_argument('--data', default=None)
    ap.add_argument('--write-tokens', type=int, default=64)
    ap.add_argument('--write-char', default=None,
                    help='repeat this corpus character for the write; defaults to a real corpus prefix')
    ap.add_argument('--write-text', default=None,
                    help='cycle this literal corpus-character sequence for the write')
    ap.add_argument('--probe-char', default=None,
                    help='probe character; defaults to the symbol after the write prefix')
    ap.add_argument('--probe-tokens', type=int, default=1,
                    help='number of consecutive same-character probe pulses after each gap')
    ap.add_argument('--probe-drive', type=float, default=None,
                    help='sensory drive during passive probes; defaults to --drive')
    ap.add_argument('--probe-plasticity', action='store_true',
                    help='allow probe pulses to write KC×DAN plasticity (off by default)')
    ap.add_argument('--gap-plasticity', action='store_true',
                    help='allow spontaneous DAN writes during silent gaps (off by default)')
    ap.add_argument('--gap-tokens', default='0,1,2,4,8,16,32,64')
    ap.add_argument('--active', type=int, default=512)
    ap.add_argument('--drive', type=float, default=1.0)
    ap.add_argument('--gain', type=float, default=1.5)
    ap.add_argument('--tonic', type=float, default=0.05)
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--sustain', type=float, default=0.5)
    ap.add_argument('--trace-tau', type=float, default=0.1)
    ap.add_argument('--noise-hz', type=float, default=0.0,
                    help='intrinsic noise rate for paired causal episodes; 0 gives deterministic readback')
    ap.add_argument('--dan-pulse', type=float, default=0.0,
                    help='voltage pulse on existing PAM neurons; paired across controls by default')
    ap.add_argument('--dan-pulse-unpaired', action='store_true',
                    help='only pulse the write episode; default pairs PAM input in write/no-write controls')
    ap.add_argument('--eta', type=float, default=0.02)
    ap.add_argument('--eligibility-tau', type=float, default=0.5)
    ap.add_argument('--weight-tau', type=float, default=4.0)
    ap.add_argument('--max-modulation', type=float, default=0.9)
    ap.add_argument('--eligibility-mix', type=float, default=0.0)
    ap.add_argument('--target-top', type=int, default=256)
    ap.add_argument('--target-selection', choices=(
        'direct', 'direct_non_cb', 'descending', 'second_order'),
        default='direct',
        help='anatomical target group for the downstream readback probe')
    ap.add_argument('--conditions', default=None,
                    help='comma-separated condition subset; default runs all controls')
    ap.add_argument('--topology-null-seed', type=int, default=271828,
                    help='seed for the local degree/weight-matched downstream topology null')
    ap.add_argument('--mod-perm-seed', type=int, default=271828,
                    help='seed for the topology-preserving KC->MBON modulation permutation null')
    ap.add_argument('--out', default='models/research_results/mb_persistent_memory_probe')
    ap.add_argument('--seed', type=int, default=20260916)
    args = ap.parse_args()
    if args.probe_drive is None:
        args.probe_drive = args.drive
    if args.probe_drive <= 0:
        raise ValueError('--probe-drive must be positive')
    if args.noise_hz < 0:
        raise ValueError('--noise-hz must be non-negative')
    if args.dan_pulse < 0:
        raise ValueError('--dan-pulse must be non-negative')
    args.data = Path(args.data) if args.data else Path(
        __import__('os').environ.get('FLY_DATA', ROOT / 'data'))
    gaps = [int(x) for x in args.gap_tokens.split(',') if x.strip()]
    if any(x < 0 for x in gaps):
        raise ValueError('--gap-tokens must be non-negative')
    if args.write_tokens <= 0:
        raise ValueError('--write-tokens must be positive')
    if args.probe_tokens <= 0:
        raise ValueError('--probe-tokens must be positive')

    # Load enough real corpus symbols to make the write pattern and probe.
    chars, stream, _, _ = load_corpus(
        args.corpus, max(args.write_tokens + 2, 128), 1, 1)
    stream = np.asarray(stream, np.int64)
    args.vocab = len(chars)
    if args.write_text is not None and args.write_char is not None:
        raise ValueError('use only one of --write-text and --write-char')
    if args.write_text is not None:
        if not args.write_text:
            raise ValueError('--write-text must not be empty')
        missing = sorted(set(args.write_text) - set(chars))
        if missing:
            raise ValueError(f'--write-text contains characters outside the vocabulary: {missing!r}')
        pattern_ids = np.asarray([chars.index(ch) for ch in args.write_text], np.int64)
        write_ids = np.resize(pattern_ids, args.write_tokens).astype(np.int64)
    elif args.write_char is not None:
        if args.write_char not in chars:
            raise ValueError(f'--write-char {args.write_char!r} is outside the corpus vocabulary')
        write_ids = np.full(args.write_tokens, chars.index(args.write_char), np.int64)
    else:
        write_ids = np.resize(stream[:-1], args.write_tokens).astype(np.int64)
    if args.probe_char is not None:
        if args.probe_char not in chars:
            raise ValueError(f'--probe-char {args.probe_char!r} is outside the corpus vocabulary')
        probe_id = int(chars.index(args.probe_char))
    elif args.write_text is not None:
        probe_id = int(write_ids[0])
    elif args.write_char is not None:
        probe_id = int(write_ids[0])
    else:
        probe_id = int(stream[args.write_tokens % len(stream)])

    brain0 = FlyBrain(data=args.data, device='cuda', batch=1, seed=args.seed,
                      sensory_input=False)
    masks = _masks(brain0)
    central_ids = _select_targets(brain0, args.target_selection, args.target_top)
    target_summary = _target_summary(brain0, central_ids)
    dt_token = float(brain0.dt) * float(args.k)
    print('edge counts:', {k: int(len(v)) for k, v in masks.items()}, flush=True)
    print(f'write_tokens={len(write_ids)} probe={chars[probe_id]!r} '
          f'probe_tokens={args.probe_tokens} gaps={gaps} '
          f'dt_token={dt_token:.6f}s target_selection={args.target_selection} '
          f'downstream_targets={len(central_ids)}', flush=True)

    all_conditions = [
        'intact', 'frozen', 'plastic_none', 'write_off', 'sham_eta0',
        'dan_mbon', 'kc_mbon', 'kc_mbon_readout_zero', 'kc_mbon_writezero',
        'mbon_central', 'mbon_central_readout_zero', 'mbon_dan', 'mbon_descending',
        'mbon_cb_intrinsic', 'mbon_all', 'topology_null',
        'mbon_source_shuffle', 'mod_perm',
    ]
    conditions = all_conditions if args.conditions is None else [
        x.strip() for x in args.conditions.split(',') if x.strip()]
    unknown = sorted(set(conditions) - set(all_conditions))
    if unknown:
        raise ValueError(f'unknown --conditions entries: {unknown!r}')
    states = {name: _new_condition(args, name, masks, args.vocab, central_ids)
              for name in conditions}
    # Fixed anatomical projection is derived from the intact connectome once
    # and shared across all paired episodes/conditions.
    fixed_direction = _anatomical_target_direction(brain0, central_ids)
    for state in states.values():
        state['fixed_direction'] = fixed_direction
    results = {}
    t0 = time.time()
    for condition in conditions:
        state = states[condition]
        rows = {}
        for gap in gaps:
            write = _episode(state, args, write_ids, probe_id, gap, True)
            no_write = _episode(state, args, write_ids, probe_id, gap, False)
            write['delta_probe_mbon_l2'] = float(abs(write['probe_mbon_l2'] -
                                                     no_write['probe_mbon_l2']))
            write['delta_probe_central_l2'] = float(abs(write['probe_central_l2'] -
                                                        no_write['probe_central_l2']))
            write['delta_probe_mbon_voltage_l2'] = float(np.linalg.norm(
                write.pop('_probe_mbon_voltage') - no_write.pop('_probe_mbon_voltage')))
            write['delta_probe_central_voltage_l2'] = float(np.linalg.norm(
                write.pop('_probe_central_voltage') - no_write.pop('_probe_central_voltage')))
            write['delta_probe_modulated_mbon_current_l2'] = float(np.linalg.norm(
                write.pop('_probe_modulated_mbon_current') -
                no_write.pop('_probe_modulated_mbon_current')))
            write_sequence = write.pop('_probe_sequence')
            no_write_sequence = no_write.pop('_probe_sequence')
            sequence = []
            for wr, nr in zip(write_sequence, no_write_sequence):
                wr_m = wr.pop('_probe_mbon_voltage')
                nr_m = nr.pop('_probe_mbon_voltage')
                wr_c = wr.pop('_probe_central_voltage')
                nr_c = nr.pop('_probe_central_voltage')
                wr_i = wr.pop('_probe_modulated_mbon_current')
                nr_i = nr.pop('_probe_modulated_mbon_current')
                dc = wr_c - nr_c
                sequence.append({
                    'probe_index': int(wr['probe_index']),
                    'delta_probe_modulated_mbon_current_l2': float(np.linalg.norm(wr_i - nr_i)),
                    'delta_probe_mbon_voltage_l2': float(np.linalg.norm(wr_m - nr_m)),
                    'delta_probe_central_voltage_l2': float(np.linalg.norm(dc)),
                    'delta_probe_central_voltage_fixed_projection': float(
                        np.dot(state['fixed_direction'], dc)),
                })
            write['probe_sequence'] = sequence
            for metric in ('delta_probe_central_voltage_l2',
                           'delta_probe_mbon_voltage_l2',
                           'delta_probe_modulated_mbon_current_l2',
                           'delta_probe_central_voltage_fixed_projection'):
                values = np.asarray([x[metric] for x in sequence], np.float64)
                auc = 0.0
                for j in range(1, len(values)):
                    auc += 0.5 * (values[j - 1] + values[j])
                write[f'{metric}_mean'] = float(values.mean()) if len(values) else 0.0
                write[f'{metric}_auc8'] = float(auc / max(len(values), 1))
                if metric == 'delta_probe_central_voltage_fixed_projection':
                    write[f'{metric}_first'] = float(values[0]) if len(values) else 0.0
            for key in ('_probe_mbon_voltage', '_probe_central_voltage',
                        '_probe_modulated_mbon_current'):
                write.pop(key, None)
                no_write.pop(key, None)
            rows[str(gap)] = {'write': write, 'no_write': no_write}
            print(f'  {condition:13s} gap={gap:3d} '
                  f'edge={write["edge_norm_preprobe"]:.5g} '
                  f'dI={write["delta_probe_modulated_mbon_current_l2"]:.5g} '
                  f'dC_V={write["delta_probe_central_voltage_l2"]:.5g} '
                  f'dM_V={write["delta_probe_mbon_voltage_l2"]:.5g} '
                  f'max_seq_C={max((x["delta_probe_central_voltage_l2"] for x in sequence), default=0.0):.5g} '
                  f'proj_auc={write["delta_probe_central_voltage_fixed_projection_auc8"]:.5g}',
                  flush=True)
        results[condition] = rows

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'config': {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        'connectome': {
            'neurons': int(brain0.n),
            'edges': int(len(brain0._W.indices.get())),
            'edge_counts': {k: int(len(v)) for k, v in masks.items()},
            'downstream_target_ids': central_ids.tolist(),
            'target_selection': args.target_selection,
            'target_summary': target_summary,
            'fixed_projection': 'signed projection onto intact MBON-input-weight direction',
            'topology_null': 'selected target rows rewired with indegree and row-weight multiset preserved',
        },
        'write_pattern_ids': write_ids.tolist(),
        'write_pattern_chars': ''.join(chars[int(x)] for x in write_ids),
        'probe_id': probe_id,
        'probe_char': chars[probe_id],
        'dt_token_seconds': dt_token,
        'conditions': results,
        'decay_fit_intact': _fit_decay(results['intact'], dt_token),
        'elapsed_seconds': time.time() - t0,
        'interpretation': {
            'persistence': 'edge_norm_preprobe after silent gaps measures local synaptic state after neural traces decay',
            'causal_readback': 'write minus no_write probe response is a downstream effect; lesions zero only existing CSR edges',
            'topology': 'main conditions add no edges or rewiring; topology_null is a separate local degree/weight-matched control',
        },
    }
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = [
        '# Persistent MB memory probe', '',
        f'- write tokens: {len(write_ids)}; probe: `{chars[probe_id]!r}`; target selection: `{args.target_selection}`',
        f'- token interval: {dt_token:.6f} s; downstream target neurons: {len(central_ids)}',
        '',
        '| condition | gap tokens | edge norm preprobe | neural trace | Δ MBON V | Δ central V | fixed signed projection AUC8 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for condition in conditions:
        for gap in gaps:
            row = results[condition][str(gap)]['write']
            lines.append(f'| {condition} | {gap} | {row["edge_norm_preprobe"]:.6g} | '
                         f'{row["neural_trace_l2_preprobe"]:.6g} | '
                         f'{row["delta_probe_mbon_voltage_l2"]:.6g} | '
                         f'{row["delta_probe_central_voltage_l2"]:.6g} | '
                         f'{row["delta_probe_central_voltage_fixed_projection_auc8"]:.6g} |')
    lines += ['', f'- fitted intact edge-state tau: {summary["decay_fit_intact"]}']
    out.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'saved {out.with_suffix(".json")}')
    print(f'saved {out.with_suffix(".md")}')


if __name__ == '__main__':
    main()
