"""Train a character readout on a locally plastic mushroom body.

This is intentionally a small CUDA-only bridge between the memory probe and
the language model.  The connectome is frozen except for existing KC->MBON
edges.  A linear/softmax readout is trained offline; the brain itself never
receives a global gradient.  ``biological`` and ``engineering`` runs are
separate command-line conditions and must not be compared as if they used the
same write signal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Install the CuPy include-path compatibility hook before importing modules
# that may compile their first GPU elementwise kernel.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from batchroll import _cuda_step_no_host_copy
from exp7_shakespeare import bpc, fit_temperature, softmax_rows
from flybrain import FlyBrain
from flylm import Encoder, fit_softmax_readout_gpu
from mb_plasticity import KCMBONPlasticity
from train_lm import cells_matching, load_corpus

ROOT = Path(__file__).parent.parent


def _add_events(fired, brain, slots_gpu, state, counts=None):
    if not fired.size:
        return
    import cupyx
    slots = slots_gpu[fired]
    # Keep this path device-side: ``bool(valid.any())`` forced a host sync on
    # every 20-ms step.  Empty indexed arrays are accepted by scatter_add, and
    # ``size`` is shape metadata rather than a device reduction.
    valid_slots = slots[slots >= 0]
    if not valid_slots.size:
        return
    # CuPy accepts a scalar value for scatter_add; avoid allocating a fresh
    # ``ones`` vector proportional to the spike count on every step.
    vals = np.float32(1.0)
    if state is not None:
        cupyx.scatter_add(state, valid_slots, vals)
    if counts is not None and counts is not state:
        cupyx.scatter_add(counts, valid_slots, vals)


def _inject(brain, enc, stream, i: int, scale: float) -> None:
    if scale <= 0:
        return
    xp = brain.xp
    # Position codes are fixed for the life of an encoder.  Upload them once
    # per brain rather than constructing a new CuPy index array for every
    # lag/step (up to 24 host->device allocations per token at k=6).
    code_cache = getattr(brain, '_lm_position_codes_gpu', None)
    if code_cache is None or len(code_cache) != len(enc.position_codes):
        code_cache = [[xp.asarray(code) for code in table]
                      for table in enc.position_codes]
        brain._lm_position_codes_gpu = code_cache
    for lag in range(min(enc.window, i + 1)):
        code = code_cache[lag][int(stream[i - lag])]
        amount = np.float32(enc.drive * scale * enc.gamma**lag)
        brain.v[code, 0] += amount


def _engineering_signal(stream, i: int, kind: str, code: np.ndarray) -> float:
    if kind == "constant":
        return 1.0
    if kind == "current_char":
        # Causal engineering control: the currently presented token is known
        # before its KC activity is written.
        return float(code[int(stream[i])])
    if kind == "next_char":
        # Kept only as an explicitly teacher-forced ablation.  It is not
        # usable during free generation and must never be called biological.
        return float(code[int(stream[i + 1])])
    raise ValueError(f"unknown engineering signal {kind!r}")


def _make_brain(args):
    brain = FlyBrain(data=args.data, device="cuda", batch=1, seed=args.seed,
                     sensory_input=False)
    brain.gain = float(args.gain)
    brain.tonic = float(args.tonic)
    return brain


def _edge_positions(brain, pre_mask, post_mask):
    """Return CSR data slots for an anatomical edge class."""
    idx = brain._W.indices.get()
    ind = brain._W.indptr.get()
    out = []
    for post in np.flatnonzero(post_mask):
        lo, hi = int(ind[post]), int(ind[post + 1])
        keep = pre_mask[idx[lo:hi]]
        if keep.any():
            out.extend((np.flatnonzero(keep) + lo).tolist())
    return np.asarray(out, np.int64)


def _lesion_positions(brain, lesion):
    """Resolve a topology-preserving functional lesion on existing CSR edges."""
    if lesion in (None, '', 'none'):
        return np.empty(0, np.int64)
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
    masks = {
        'kc_mbon': (kc, mb),
        'dan_mbon': (dan, mb),
        'mbon_central': (mb, central),
        'mbon_cb_intrinsic': (mb, cb),
        'mbon_descending': (mb, descending),
        'mbon_all': (mb, np.ones(brain.n, dtype=bool)),
    }
    if lesion not in masks:
        raise ValueError(f'unknown lesion {lesion!r}')
    return _edge_positions(brain, *masks[lesion])


def _zero_edge_positions(brain, positions):
    if len(positions):
        brain._W.data[brain.xp.asarray(positions)] = np.float32(0.0)


def _select_neural(brain, top: int, probe_path: Path | None) -> np.ndarray:
    ct = np.asarray(brain.cell_type).astype(str)
    if probe_path is not None and probe_path.exists():
        d = np.load(probe_path)
        score = np.asarray(d["memory_score"], np.float32)
        ids = np.flatnonzero(np.isfinite(score))
        ids = ids[np.argsort(-score[ids], kind="stable")]
        return ids[:min(int(top), len(ids))].astype(np.int64)
    mb = np.flatnonzero(
        (np.char.find(ct, "KC") >= 0) |
        (np.char.find(ct, "MBON") >= 0) |
        (np.char.find(ct, "PAM") >= 0) |
        (np.char.find(ct, "PPL") >= 0) |
        (np.char.find(ct, "PPM") >= 0)
    )
    return mb[:min(int(top), len(mb))].astype(np.int64)


def collect_stream(args, stream: np.ndarray, mode: str, enc: Encoder,
                   engineering_code: np.ndarray, neural_ids: np.ndarray,
                   state: dict | None = None,
                   label_stream: np.ndarray | None = None):
    """Collect causal features and next-character labels for one lane.

    Passing ``state`` continues the same brain, traces, and plastic synapses
    into the next stream.  This matters for adaptive memory: resetting before
    validation would create a train/validation distribution jump in the
    persistent KC→MBON state.
    """
    continued = state is not None
    if state is None:
        brain = _make_brain(args)
        ct = np.asarray(brain.cell_type).astype(str)
        kc = np.flatnonzero(np.char.find(ct, "KC") >= 0)
        mbon = np.flatnonzero(np.char.find(ct, "MBON") >= 0)
        dan = np.flatnonzero(
            (np.char.find(ct, "PAM") >= 0) |
            (np.char.find(ct, "PPL") >= 0) |
            (np.char.find(ct, "PPM") >= 0)
        )
        plastic = None if mode == "frozen" else KCMBONPlasticity(
            brain, mode=mode, eta=args.eta,
            eligibility_tau=args.eligibility_tau,
            weight_tau=args.weight_tau,
            max_modulation=args.max_modulation,
            eligibility_mix=float(getattr(args, 'eligibility_mix', 0.0)))
        if plastic is not None:
            plastic.reset()
        lesion = getattr(args, 'lesion', 'none')
        lesion_positions = _lesion_positions(brain, lesion)
        _zero_edge_positions(brain, lesion_positions)
        kc_slot = brain.xp.full(brain.n, -1, dtype=brain.xp.int32)
        dan_slot = brain.xp.full(brain.n, -1, dtype=brain.xp.int32)
        neural_slot = brain.xp.full(brain.n, -1, dtype=brain.xp.int32)
        kc_slot[brain.xp.asarray(kc)] = brain.xp.arange(len(kc), dtype=brain.xp.int32)
        dan_slot[brain.xp.asarray(dan)] = brain.xp.arange(len(dan), dtype=brain.xp.int32)
        neural_slot[brain.xp.asarray(neural_ids)] = brain.xp.arange(
            len(neural_ids), dtype=brain.xp.int32)
        trace = brain.xp.zeros(len(neural_ids), brain.xp.float32)
        # Reuse event counters across tokens.  The old implementation created
        # four GPU arrays inside every token/step, needlessly fragmenting the
        # allocator and lowering occupancy without changing dynamics.
        kc_counts = brain.xp.zeros(len(kc), brain.xp.float32)
        dan_counts = brain.xp.zeros(len(dan), brain.xp.float32)
        decay = np.float32(np.exp(-brain.dt / args.trace_tau))
        brain.reset(seed=args.seed)
        state = {"brain": brain, "kc": kc, "mbon": mbon, "dan": dan,
                 "plastic": plastic, "kc_slot": kc_slot, "dan_slot": dan_slot,
                 "neural_slot": neural_slot, "trace": trace, "decay": decay,
                 "kc_counts": kc_counts, "dan_counts": dan_counts,
                 "lesion": lesion, "lesion_positions": lesion_positions}
    else:
        brain = state["brain"]
        kc, mbon, dan = state["kc"], state["mbon"], state["dan"]
        plastic = state["plastic"]
        kc_slot, dan_slot = state["kc_slot"], state["dan_slot"]
        neural_slot = state["neural_slot"]
        trace, decay = state["trace"], state["decay"]
        kc_counts, dan_counts = state["kc_counts"], state["dan_counts"]
        lesion = state.get("lesion", "none")
        lesion_positions = state.get("lesion_positions", np.empty(0, np.int64))
    n = len(stream) - 1
    elig_bins = int(getattr(args, 'eligibility_bins', 0))
    X = np.empty((n, args.edge_bins + elig_bins + len(neural_ids)), np.float32)
    if label_stream is None:
        label_stream = stream
    if len(label_stream) != len(stream):
        raise ValueError("label_stream must have the same length as stream")
    y = np.asarray(label_stream[1:], np.int64).copy()
    t0 = time.time()
    for i in range(n):
        kc_counts.fill(0)
        dan_counts.fill(0)
        for step in range(args.k):
            _inject(brain, enc, stream, i, 1.0 if step == 0 else args.sustain)
            fired = _cuda_step_no_host_copy(brain)
            trace *= decay
            _add_events(fired, brain, neural_slot, trace)
            _add_events(fired, brain, kc_slot, None, kc_counts)
            _add_events(fired, brain, dan_slot, None, dan_counts)
        if plastic is None:
            edge = np.zeros(args.edge_bins, np.float32)
            elig = np.zeros(elig_bins, np.float32)
        else:
            signal = None if mode == "biological" else _engineering_signal(
                stream, i, args.engineering_signal, engineering_code)
            plastic.update(kc_counts, dan_counts, teaching_signal=signal)
            # Local plasticity can modulate KC->MBON edges after the update;
            # re-zero only the requested output lesion so it remains a true
            # functional ablation throughout the stream.
            if lesion != 'none':
                _zero_edge_positions(brain, lesion_positions)
            # The default is an edge-level count sketch.  ``mbon`` and
            # ``kc_mbon`` are optional anatomy-aligned controls: they expose
            # the same local KC->MBON modulation after summing over existing
            # MBON outputs, without adding units, edges, or a learned
            # projection.  This separates hash-collision loss from a genuine
            # lack of information in the local state.
            memory_feature = getattr(args, 'memory_feature', 'edge')
            if memory_feature == 'edge':
                edge = plastic.edge_state_features(args.edge_hash_bins)[:args.edge_bins]
            elif memory_feature in ('mbon', 'kc_mbon'):
                kc_state, mbon_state = plastic.state_features()
                raw = (mbon_state if memory_feature == 'mbon' else
                       np.concatenate((kc_state, mbon_state), axis=0))
                edge = np.zeros(args.edge_bins, np.float32)
                edge[:min(args.edge_bins, len(raw))] = raw[:args.edge_bins]
            else:
                raise ValueError(f'unknown memory_feature {memory_feature!r}')
            if elig_bins:
                elig = plastic.eligibility_state_features(args.eligibility_hash_bins)[:elig_bins]
            else:
                elig = np.empty(0, np.float32)
        X[i, :args.edge_bins] = edge
        if elig_bins:
            X[i, args.edge_bins:args.edge_bins + elig_bins] = elig
        X[i, args.edge_bins + elig_bins:] = brain.xp.asnumpy(trace)
        if (i + 1) % 500 == 0 or i == n - 1:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  {mode:11s} {i + 1:6d}/{n} tokens | {rate:7.1f} tokens/s",
                  flush=True)
    return X, y, {"kc": int(len(kc)), "mbon": int(len(mbon)),
                  "dan": int(len(dan)), "plastic_edges": int(plastic.n_edges if plastic else 0),
                  "neural_features": int(len(neural_ids)),
                  "continued_state": bool(continued)}, state


def score(Xtr, ytr, Xva, yva, vocab: int, args):
    W, mu, sd, Ym = fit_softmax_readout_gpu(
        Xtr, ytr, vocab, l2=args.softmax_l2, epochs=args.softmax_epochs,
        batch_size=args.softmax_batch, lr=args.softmax_lr, seed=3)
    logv = ((Xva - mu) / sd) @ W + Ym
    temp = fit_temperature(logv, yva)
    p = softmax_rows(logv / temp)
    pred = p.argmax(1)
    metrics = {
        "val_acc": float(np.mean(pred == yva)),
        "bpc": float(bpc(p, yva)),
        "bpc_t1": float(bpc(softmax_rows(logv), yva)),
        "temperature": float(temp),
    }
    return metrics, (W, mu, sd, Ym, temp)


def context_hot(stream: np.ndarray, vocab: int, order: int, buckets: int) -> np.ndarray:
    """Causal current-context one-hot/hash features for a shared LM control."""
    n = len(stream) - 1
    if order <= 0:
        return np.zeros((n, 0), np.float32)
    width = vocab ** order if order <= 2 else int(buckets)
    if width <= 0:
        raise ValueError("context bucket count must be positive")
    hot = np.zeros((n, width), np.float32)
    for i in range(n):
        idx = 0
        for lag in range(order - 1, -1, -1):
            idx = idx * vocab + int(stream[i - lag] if i >= lag else 0)
        if order >= 3:
            idx %= width
        hot[i, idx] = 1.0
    return hot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus" / "tinyshakespeare.txt"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--mode", choices=("frozen", "biological", "engineering"), default="biological")
    ap.add_argument("--train-chars", type=int, default=6000)
    ap.add_argument("--val-chars", type=int, default=2000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sustain", type=float, default=0.5)
    ap.add_argument("--active", type=int, default=512)
    ap.add_argument("--drive", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--tonic", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--position-codes", action="store_true")
    ap.add_argument("--trace-tau", type=float, default=0.1)
    ap.add_argument("--neural-top", type=int, default=512)
    ap.add_argument("--probe", default="models/research_results/mb_memory_neuron_probe.npz")
    ap.add_argument("--edge-bins", type=int, default=4096)
    ap.add_argument("--edge-hash-bins", type=int, default=8192)
    ap.add_argument("--memory-feature", choices=("edge", "mbon", "kc_mbon"),
                    default="edge",
                    help="fixed local MB state summary for memory heads; no new edges")
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--eligibility-tau", type=float, default=0.5)
    ap.add_argument("--weight-tau", type=float, default=4.0)
    ap.add_argument("--max-modulation", type=float, default=0.9)
    ap.add_argument("--eligibility-mix", type=float, default=0.0,
                    help="fraction of KC eligibility used for delayed DAN writes")
    ap.add_argument("--engineering-signal", choices=("constant", "current_char", "next_char"), default="current_char")
    ap.add_argument("--lesion", choices=("none", "kc_mbon", "dan_mbon",
                                          "mbon_central", "mbon_cb_intrinsic",
                                          "mbon_descending", "mbon_all"),
                    default="none",
                    help="topology-preserving functional lesion for causal controls")
    ap.add_argument("--softmax-l2", type=float, default=1e-4)
    ap.add_argument("--softmax-epochs", type=int, default=25)
    ap.add_argument("--softmax-batch", type=int, default=1024)
    ap.add_argument("--softmax-lr", type=float, default=0.03)
    ap.add_argument("--skip-order", type=int, default=0,
                    help="append a shared causal context control (1/2 exact, >=3 hashed)")
    ap.add_argument("--skip-buckets", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", default="models/plastic_lm_biological.npz")
    args = ap.parse_args()
    if args.data is None:
        import os
        args.data = Path(os.environ.get("FLY_DATA", ROOT / "data"))
    else:
        args.data = Path(args.data)
    chars, tr, va, _ = load_corpus(args.corpus, args.train_chars, args.val_chars, 1)
    vocab = len(chars)
    brain0 = _make_brain(args)
    pn = cells_matching(brain0, ("PN",))
    enc = Encoder(pn, vocab, active=args.active, drive=args.drive,
                  window=args.window, gamma=args.gamma, seed=3,
                  position_codes=args.position_codes)
    probe = Path(args.probe)
    if not probe.is_absolute():
        probe = ROOT / probe
    neural_ids = _select_neural(brain0, args.neural_top, probe if probe.exists() else None)
    rng = np.random.default_rng(args.seed)
    engineering_code = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=vocab)
    print(f"mode={args.mode} vocab={vocab} train={len(tr)} val={len(va)} "
          f"edge={args.edge_bins} neural={len(neural_ids)}", flush=True)
    t0 = time.time()
    Xtr, ytr, pop_tr, runner = collect_stream(
        args, tr, args.mode, enc, engineering_code, neural_ids)
    # Continue the same dynamical/plastic state into validation.  This is the
    # causal online setting; no validation label is used by the brain.
    Xva, yva, pop_va, runner = collect_stream(
        args, va, args.mode, enc, engineering_code, neural_ids, state=runner)
    if args.skip_order:
        if args.skip_order < 1 or args.skip_order > 5:
            raise ValueError("--skip-order must be between 1 and 5")
        Htr = context_hot(tr, vocab, args.skip_order, args.skip_buckets)
        Hva = context_hot(va, vocab, args.skip_order, args.skip_buckets)
        Xtr = np.concatenate((Xtr, Htr), axis=1)
        Xva = np.concatenate((Xva, Hva), axis=1)
        print(f"context control appended: order={args.skip_order} width={Htr.shape[1]}",
              flush=True)
    metrics, model = score(Xtr, ytr, Xva, yva, vocab, args)
    W, mu, sd, Ym, temp = model
    metrics.update({"mode": args.mode, "train_samples": int(len(Xtr)),
                    "val_samples": int(len(Xva)), "seconds": time.time() - t0,
                    "population_train": pop_tr, "population_val": pop_va,
                    "config": {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in vars(args).items()}})
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, W=W, mu=mu, sd=sd, Ym=Ym, temp=np.float32(temp),
             chars=np.asarray(chars), neural_ids=neural_ids,
             edge_bins=np.int32(args.edge_bins), edge_hash_bins=np.int32(args.edge_hash_bins),
             mode=np.asarray(args.mode), engineering_signal=np.asarray(args.engineering_signal))
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2), flush=True)
    print(f"model saved: {out}", flush=True)


if __name__ == "__main__":
    main()
