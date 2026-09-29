"""Batched parallel rollout: B independent token streams through one FlyBrain(batch=B).

On GPU (device="cuda") the batch shares the sparse multiply; on CPU it only saves
wall-clock latency, not throughput. Injection is written directly into brain.v.

Returns X (L-1, B, F) float32 features (F = readout neurons) and Y (L-1, B) int64 labels.
"""

import time

import numpy as np

from flybrain.reservoir import Trace

try:
    import progress as _progress
except ImportError:
    _progress = None


def _cuda_step_no_host_copy(brain):
    """Advance a CUDA ``FlyBrain`` without copying every spike list to the CPU.

    ``FlyBrain.step`` normally converts its fired indices to NumPy for the
    interactive API.  Batched LM collection only needs a DN trace, so retaining
    those indices on-device removes a host synchronisation at every brain step.
    Inputs have already been added directly to ``brain.v`` by the caller.
    """
    xp, B = brain.xp, brain.batch
    current = brain.synaptic_input(brain.fired) * brain.gain
    brain.v *= brain.decay
    brain.v += current + brain.tonic
    brain.v += (brain.rng.random((brain.n, B)) < brain.noise_hz * brain.dt) * xp.float32(brain.noise_amp)
    if brain.refractory_steps:
        brain.v[(brain.steps - brain.last_spike) <= brain.refractory_steps] = 0.0
    fired = xp.flatnonzero(brain.v >= 1.0)
    brain.v.ravel()[fired] = 0.0
    if brain.refractory_steps:
        brain.last_spike.ravel()[fired] = brain.steps
    brain.fired = fired
    brain.steps += 1
    return fired


def _cuda_step_record_pre_reset(brain, record_indices=()):
    """Advance one CUDA step and snapshot selected voltages before reset.

    The ordinary fast path intentionally zeroes every threshold-crossing
    voltage before returning.  That is efficient for rollout, but an end-of-
    token voltage read then confounds a real graded response with the
    discontinuous ``v>=1 -> 0`` event.  This additive helper keeps the old
    path untouched and returns ``(fired, v_pre)`` where ``v_pre`` is the
    selected population immediately before threshold/reset.  Only the small
    requested population is copied to host by the caller.
    """
    xp, B = brain.xp, brain.batch
    current = brain.synaptic_input(brain.fired) * brain.gain
    brain.v *= brain.decay
    brain.v += current + brain.tonic
    brain.v += (brain.rng.random((brain.n, B)) < brain.noise_hz * brain.dt) * xp.float32(brain.noise_amp)
    if brain.refractory_steps:
        brain.v[(brain.steps - brain.last_spike) <= brain.refractory_steps] = 0.0
    ids = xp.asarray(record_indices, dtype=xp.int64).reshape(-1)
    v_pre = brain.v[ids].copy() if len(record_indices) else xp.empty((0, B), dtype=xp.float32)
    fired = xp.flatnonzero(brain.v >= 1.0)
    brain.v.ravel()[fired] = 0.0
    if brain.refractory_steps:
        brain.last_spike.ravel()[fired] = brain.steps
    brain.fired = fired
    brain.steps += 1
    return fired, v_pre


def collect_batched(brain, encoder, streams, dns, k_steps=4, sustain=1.0, log=None,
                    feature_mode="trace", trace_tau=0.1, fast_cuda_trace=False):
    B = len(streams)
    if feature_mode not in ("trace", "voltage", "trace_voltage", "trace_time", "trace_time_voltage", "trace_memory", "trace_history"):
        raise ValueError("feature_mode must be trace, voltage, trace_voltage, trace_time, trace_time_voltage, trace_memory, or trace_history")
    if brain.batch != B:
        raise ValueError(f"brain.batch {brain.batch} != number of streams {B}")
    L = min(len(s) for s in streams)
    brain.reset()
    v = brain.v
    drive = encoder.drive
    codebooks = encoder.position_codes
    cuda = brain.device == "cuda"
    # The fast path keeps spike traces on-device and avoids a host synchronisation
    # on every 20-ms step.  It is opt-in because CUDA RNG scheduling can differ
    # slightly from the interactive (host-spike) API; both paths use the same
    # frozen weights and dynamics.
    fast_cuda_trace = bool(fast_cuda_trace and cuda and feature_mode in ("trace", "trace_voltage"))
    # A longer decay constant is an explicit working-memory knob.  The
    # connectome remains frozen; only the readout-side eligibility trace is
    # changed.  This is deliberately a single, tunable tau rather than the
    # noisy four-timescale bank used by the earlier negative ablation.
    trace = None if cuda and fast_cuda_trace else Trace(
        brain, idx=dns, aggregate="batch", tau=float(trace_tau))
    # Multi-timescale traces are a lightweight working-memory bank.  They use
    # the same frozen spikes and no learned state; the readout can decide which
    # decay constant is useful for a character context.
    memory_traces = ([Trace(brain, idx=dns, tau=tau, aggregate="batch")
                      for tau in (0.1, 0.3, 1.0, 3.0)]
                     if feature_mode == "trace_memory" else None)
    history_states = ([] if feature_mode == "trace_history" else None)
    temporal = feature_mode in ("trace_time", "trace_time_voltage")
    has_voltage = feature_mode in ("voltage", "trace_voltage", "trace_time_voltage")
    width = len(dns) * k_steps if temporal else len(dns) * (2 if feature_mode == "trace_voltage" else 1)
    if feature_mode == "trace_memory":
        width = len(dns) * 4
    elif feature_mode == "trace_history":
        width = len(dns) * 4
    if temporal and has_voltage:
        width += len(dns)
    X = np.empty((L - 1, B, width), np.float32)
    Y = np.empty((L - 1, B), np.int64)
    total = L - 1
    t0 = time.time()
    if cuda:
        import cupy
        voltage_idx = cupy.asarray(dns)
        if fast_cuda_trace:
            trace_gpu = cupy.zeros((len(dns), B), cupy.float32)
            dn_slot = cupy.full(brain.n, -1, cupy.int32)
            dn_slot[voltage_idx] = cupy.arange(len(dns), dtype=cupy.int32)
    for i in range(total):
        temporal_features = []
        for s in range(k_steps):
            scale = 1.0 if s == 0 else sustain
            if scale > 0:
                if cuda:
                    if encoder.window == 1:
                        idxs = [codebooks[0][int(streams[b][i])] for b in range(B)]
                        cats = np.concatenate(idxs).astype(np.int64)
                        bs = np.concatenate([np.full(len(c), b, np.int64) for b, c in enumerate(idxs)])
                        flat = cupy.asarray(cats * B + bs)
                        v.ravel()[flat] += np.float32(drive * scale)
                    else:
                        for b in range(B):
                            for lag in range(min(encoder.window, i + 1)):
                                c = codebooks[lag][int(streams[b][i - lag])]
                                # Keep the exact deterministic accumulation
                                # order used by the single-stream API. Atomic
                                # scatter-add changes threshold crossings for
                                # overlapping codes, even when its total drive
                                # is mathematically identical.
                                v[cupy.asarray(c), b] += np.float32(
                                    drive * scale * encoder.gamma**lag)
                else:
                    for b in range(B):
                        for lag in range(min(encoder.window, i + 1)):
                            v[codebooks[lag][int(streams[b][i - lag])], b] += (
                                drive * scale * encoder.gamma**lag)
            if cuda and fast_cuda_trace:
                fired = _cuda_step_no_host_copy(brain)
                if feature_mode != "voltage":
                    trace_gpu *= np.float32(np.exp(-brain.dt / float(trace_tau)))
                    rows, cols = fired // B, fired % B
                    slots = dn_slot[rows]
                    valid = slots >= 0
                    # Use scatter-add rather than advanced-index ``+=``:
                    # CuPy gathers before writing for advanced indexes, which
                    # can drop updates when the same DN fires in more than one
                    # batch lane during a step.
                    import cupyx
                    trace_flat = slots[valid] * B + cols[valid]
                    cupyx.scatter_add(trace_gpu.ravel(), trace_flat,
                                      cupy.ones(len(trace_flat), dtype=cupy.float32))
            else:
                fired = brain.step()
                observed = trace.observe(fired)
                if memory_traces is not None:
                    for mem_trace in memory_traces:
                        mem_trace.observe(fired)
                if temporal:
                    temporal_features.append(observed.T.copy())
        if temporal:
            trace_feat = np.concatenate(temporal_features, axis=1)
            if feature_mode == "trace_time":
                X[i] = trace_feat
            else:
                voltage = (brain.v[voltage_idx].get().T if cuda else brain.v[dns].T.copy())
                X[i] = np.concatenate((trace_feat, voltage), axis=1)
        elif feature_mode == "trace":
            X[i] = trace_gpu.get().T if cuda and fast_cuda_trace else trace.features().T
        elif feature_mode == "trace_memory":
            X[i] = np.concatenate([mem_trace.features() for mem_trace in memory_traces], axis=0).T
        elif feature_mode == "trace_history":
            history_states.append(trace.features())
            if len(history_states) > 4:
                history_states.pop(0)
            # Pad the start of a segment with zero states so the serialized
            # feature width stays fixed and no future token leaks in.
            while len(history_states) < 4:
                history_states.insert(0, np.zeros_like(history_states[0]))
            X[i] = np.concatenate(history_states, axis=0).T
        else:
            voltage = (brain.v[voltage_idx].get().T if cuda else brain.v[dns].T.copy())
            if feature_mode == "voltage":
                X[i] = voltage
            else:
                trace_feat = trace_gpu.get().T if cuda and fast_cuda_trace else trace.features().T
                X[i] = np.concatenate((trace_feat, voltage), axis=1)
        for b in range(B):
            Y[i, b] = streams[b][i + 1]
        if log is not None and _progress is not None and (i % 250 == 0 or i == total - 1):
            _progress.emit(log["run"], log["cfg"], phase=log.get("phase", "collect"),
                           done=i + 1, total=total, rate=(i + 1) * B / max(time.time() - t0, 1e-9))
    return X, Y
