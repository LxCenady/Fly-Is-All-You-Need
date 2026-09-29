"""Estimate intrinsic sequence-memory capacity of the frozen fly reservoir.

The language model currently reads only a few hundred central neurons.  This
experiment widens that electrode array and measures how much delayed input can
be linearly recovered from the fly state.  It uses an iid character stream so
that the score cannot be explained by Shakespeare's n-gram statistics.  The
brain is never trained: only CUDA trace accumulation and a ridge probe are
used.

The feature array is collected once for a nested, probe-ranked central
population (up to ``--max-neurons``), then several prefixes are fitted offline.
For each prefix the script reports held-out character recall at lag 0..N and a
chance-normalized discrete memory score (sum of excess accuracies).  The
population order starts with the language-selective virtual-electrode score
and therefore directly tests whether adding apparently redundant central
neurons buys additional memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from batchroll import _cuda_step_no_host_copy
from flybrain import FlyBrain
from flylm import DETECTORS, Encoder
from probe_language_pathways import load_stream

ROOT = Path(__file__).parent.parent


def _inject_cuda(brain, encoder, streams, i: int, scale: float) -> None:
    """Inject one causal token window directly into GPU voltages."""
    if scale <= 0:
        return
    xp = brain.xp
    for b, stream in enumerate(streams):
        for lag in range(min(encoder.window, i + 1)):
            code = encoder.position_codes[lag][int(stream[i - lag])]
            amount = np.float32(encoder.drive * scale * encoder.gamma**lag)
            brain.v[xp.asarray(code), b] += amount


def _rank_central(brain, probe_path: Path | None, seed: int) -> tuple[np.ndarray, dict]:
    """Return a nested central-neuron order and its composition summary."""
    supers = np.asarray(brain.superclass).astype(str)
    central = (supers != "visual_projection") & (np.char.find(supers, "sensory") < 0)
    ids = np.flatnonzero(central)
    scores = None
    if probe_path is not None:
        d = np.load(probe_path)
        scores = np.asarray(d["target_score"], np.float32)
        if len(scores) != brain.n:
            raise ValueError(f"probe neuron count {len(scores)} != brain size {brain.n}")
        good = np.isfinite(scores[ids])
        ids = ids[good]
        order = ids[np.argsort(-scores[ids], kind="stable")]
        order_name = "probe_target_score_descending"
    else:
        order = np.random.default_rng(seed).permutation(ids)
        order_name = "random_central"
    # Keep the most useful metadata in the report; the full order is saved too.
    groups = {}
    for x in np.unique(supers[order]):
        groups[str(x)] = int(np.sum(supers[order] == x))
    return np.asarray(order, np.int64), {"order": order_name, "central_total": int(len(order)),
                                        "superclass_total": groups}


def collect_states(brain, encoder, streams, neurons, k_steps: int, sustain: float,
                    seed: int, mmap_path: Path, trace_tau: float = 0.1) -> np.memmap:
    """Collect selected trace states without copying the 166k-neuron state."""
    if brain.device != "cuda":
        raise ValueError("memory capacity collection currently requires CUDA")
    xp = brain.xp
    B = len(streams)
    L = min(len(s) for s in streams)
    total = max(L - 1, 0)
    neurons = np.asarray(neurons, np.int64)
    slot = np.full(brain.n, -1, np.int32)
    slot[neurons] = np.arange(len(neurons), dtype=np.int32)
    slot_gpu = xp.asarray(slot)
    trace = xp.zeros((len(neurons), B), xp.float32)
    if trace_tau <= 0:
        raise ValueError("trace_tau must be positive")
    trace_decay = np.float32(np.exp(-brain.dt / float(trace_tau)))
    mem = np.memmap(mmap_path, mode="w+", dtype=np.float32,
                    shape=(total * B, len(neurons)))
    ones = xp.ones(1, dtype=xp.float32)
    import cupyx

    brain.reset(seed=seed)
    t0 = time.time()
    row = 0
    for i in range(total):
        for step in range(k_steps):
            _inject_cuda(brain, encoder, streams, i, 1.0 if step == 0 else sustain)
            fired = _cuda_step_no_host_copy(brain)
            trace *= trace_decay
            if fired.size:
                rows = fired // B
                cols = fired - rows * B
                slots = slot_gpu[rows]
                valid = slots >= 0
                if bool(valid.any()):
                    flat = slots[valid] * B + cols[valid]
                    # Each fired row/column is unique, but scatter_add keeps
                    # this correct if a future simulator emits duplicates.
                    cupyx.scatter_add(trace.ravel(), flat, xp.ones(flat.size, xp.float32))
        mem[row:row + B] = trace.T.get()
        row += B
        if (i + 1) % 500 == 0 or i == total - 1:
            rate = (i + 1) * B / max(time.time() - t0, 1e-9)
            print(f"  memory rollout {i + 1:6d}/{total} tokens | {rate:7.1f} samples/s",
                  flush=True)
    mem.flush()
    return mem


def delayed_labels(streams, max_lag: int) -> np.ndarray:
    """Labels [current, previous-1, ...] aligned with flattened states."""
    n = min(len(s) for s in streams) - 1
    y = np.empty((n * len(streams), max_lag + 1), np.int64)
    row = 0
    for i in range(n):
        for s in streams:
            for lag in range(max_lag + 1):
                y[row, lag] = int(s[i - lag]) if i >= lag else -1
            row += 1
    return y


def fit_discrete_capacity(X: np.ndarray, y: np.ndarray, vocab: int,
                          sizes: list[int], lam: float, split: float) -> dict:
    """Fit all delayed one-hot probes for each nested electrode size on CUDA."""
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CuPy is required") from exc
    n_rows, max_n = X.shape
    # The first ``max_lag`` states do not have a complete delayed history.
    # Exclude them before splitting; otherwise -1 would silently index the last
    # vocabulary class in the one-hot table.
    eligible = np.flatnonzero(np.all(y >= 0, axis=1))
    if len(eligible) < 2:
        raise ValueError("not enough rows with a complete delayed history")
    cut = max(1, min(len(eligible) - 1, int(len(eligible) * split)))
    train = eligible[:cut]
    test = eligible[cut:]
    lags = y.shape[1]
    out = {"sizes": {}, "split_row": int(cut), "chance": float(1.0 / vocab)}
    for requested in sizes:
        n = min(int(requested), max_n)
        if n < 1:
            continue
        print(f"fit memory probes: {n} neurons", flush=True)
        xtr = cp.asarray(np.asarray(X[train, :n]), dtype=cp.float32)
        xte = cp.asarray(np.asarray(X[test, :n]), dtype=cp.float32)
        mu = xtr.mean(axis=0)
        sd = xtr.std(axis=0) + np.float32(1e-6)
        ztr = (xtr - mu) / sd
        zte = (xte - mu) / sd
        zmean = ztr.mean(axis=0)
        ztrc = ztr - zmean
        # All delayed probes share the same covariance factorization.
        A = ztrc.T @ ztrc + np.float32(lam * len(train)) * cp.eye(n, dtype=cp.float32)
        ytr = np.asarray(y[train])
        yte = np.asarray(y[test])
        Y = np.zeros((len(train), lags * vocab), np.float32)
        Ym = np.zeros(lags * vocab, np.float32)
        for lag in range(lags):
            Y[:, lag * vocab:(lag + 1) * vocab] = np.eye(vocab, dtype=np.float32)[ytr[:, lag]]
            Ym[lag * vocab:(lag + 1) * vocab] = Y[:, lag * vocab:(lag + 1) * vocab].mean(0)
        Bmat = ztrc.T @ (cp.asarray(Y) - cp.asarray(Ym))
        W = cp.linalg.solve(A, Bmat)
        pred_logits = zte @ W + cp.asarray(Ym)
        pred = cp.asnumpy(pred_logits).reshape(len(test), lags, vocab).argmax(2)
        acc = []
        for lag in range(lags):
            valid = yte[:, lag] >= 0
            acc.append(float(np.mean(pred[valid, lag] == yte[valid, lag])))
        chance = 1.0 / vocab
        excess = [(a - chance) / (1.0 - chance) for a in acc]
        out["sizes"][str(n)] = {
            "neurons": n,
            "accuracy_by_lag": acc,
            "chance_normalized_excess_by_lag": excess,
            "discrete_memory_capacity": float(np.sum(np.maximum(excess, 0.0))),
        }
        del xtr, xte, ztr, zte, ztrc, A, Bmat, W, pred_logits
        cp.get_default_memory_pool().free_all_blocks()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chars", type=int, default=5000,
                    help="iid character tokens per lane")
    ap.add_argument("--vocab", type=int, default=65)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sustain", type=float, default=0.5)
    ap.add_argument("--active", type=int, default=512)
    ap.add_argument("--drive", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--trace-tau", type=float, default=0.1,
                    help="decay time constant for the measured memory trace (seconds)")
    ap.add_argument("--position-codes", action="store_true")
    ap.add_argument("--max-neurons", type=int, default=4096)
    ap.add_argument("--sizes", default="128,256,512,1024,2048,4096")
    ap.add_argument("--max-lag", type=int, default=12)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--probe", default="models/research_results/probe_language_pathways_fullinput.npz",
                    help="probe NPZ for nested target-score ranking; empty = random central")
    ap.add_argument("--out", default="models/research_results/memory_capacity_probe.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.data is None:
        import os
        data = Path(os.environ.get("FLY_DATA", ROOT / "data"))
    else:
        data = Path(args.data)
    brain = FlyBrain(data=data, device=args.device, batch=args.batch, seed=args.seed)
    channels = np.concatenate([brain.cells(DETECTORS), brain.cells(["visual_projection"]),
                               brain.cells(["vnc_sensory"])])
    enc = Encoder(channels, args.vocab, active=args.active, drive=args.drive,
                  window=args.window, gamma=args.gamma, seed=3,
                  position_codes=args.position_codes)
    # iid tokens make delayed recall a genuine memory test rather than an
    # n-gram lookup.  Streams are independent lanes of the same frozen brain.
    streams = [np.ascontiguousarray(rng.integers(args.vocab, size=args.chars), dtype=np.int64)
               for _ in range(args.batch)]
    probe = Path(args.probe) if args.probe else None
    if probe is not None and not probe.is_absolute():
        probe = ROOT / probe
    order, summary = _rank_central(brain, probe if probe and probe.exists() else None, args.seed)
    max_n = min(max(int(args.max_neurons), 1), len(order))
    order = order[:max_n]
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    sizes = sorted(set(max(1, min(x, max_n)) for x in sizes))
    labels = delayed_labels(streams, args.max_lag)
    if len(labels) != args.batch * (args.chars - 1):
        raise RuntimeError("label/state length mismatch")
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="memory_capacity_", suffix=".dat",
                                     dir=str(out_path.parent), delete=False) as tmp:
        mmap_path = Path(tmp.name)
    features = None
    try:
        features = collect_states(brain, enc, streams, order, args.k, args.sustain,
                                  args.seed, mmap_path, args.trace_tau)
        fit = fit_discrete_capacity(features, labels, args.vocab, sizes, args.lam, args.split)
    finally:
        # Windows keeps a mapped file locked until every memmap object is
        # released, so explicitly drop it before removing the temporary data.
        if features is not None:
            features.flush()
            del features
        import gc
        gc.collect()
        try:
            mmap_path.unlink()
        except FileNotFoundError:
            pass
    result = {
        "config": vars(args),
        "population": summary,
        "selected_neurons": int(max_n),
        "sizes": fit["sizes"],
        "split_row": fit["split_row"],
        "chance": fit["chance"],
        "note": "iid character memory; lag 0=current, positive=previous token",
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = out_path.with_suffix(".md")
    lines = ["# Intrinsic sequence-memory capacity", "",
             "The frozen fly was driven by iid character codes; only the linear probes were fitted.",
             "Accuracy is held-out recall, with lag 0=current and positive lags=previous characters.", "",
             "| electrode neurons | " + " | ".join(f"lag {i}" for i in range(args.max_lag + 1)) + " | capacity |",
             "|---:|" + "---:|" * (args.max_lag + 1) + "---:|"]
    for size in sizes:
        row = result["sizes"][str(size)]
        vals = " | ".join(f"{x:.3f}" for x in row["accuracy_by_lag"])
        lines.append(f"| {size} | {vals} | {row['discrete_memory_capacity']:.2f} |")
    lines += ["", "The capacity column sums chance-normalized positive delayed-recall excess.",
              "The nested order is the central, probe-ranked population; increasing it tests redundancy directly."]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(out_path.with_suffix(".neurons.npy"), order)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
