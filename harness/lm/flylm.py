"""Reservoir language-model plumbing on the frozen fly connectome.

tokens -> sparse channel codes (Encoder) -> FlyBrain (frozen) -> DN spike traces
-> linear readout -> next-token prediction.
"""

from __future__ import annotations

import time

import numpy as np

from flybrain.reservoir import Trace

try:
    import progress as _progress
except ImportError:
    _progress = None

DETECTORS = ["LC4", "LPLC1", "LPLC2", "LC10a", "sensory_ascending"]


class Encoder:
    """Token id -> sparse subset of input channels; optional fading context window."""

    def __init__(self, channels, vocab: int, active: int = 32, drive: float = 0.9,
                 window: int = 1, gamma: float = 0.7, seed: int = 0,
                 position_codes: bool = False):
        rng = np.random.default_rng(seed)
        self.channels = np.asarray(channels)
        self.active = active
        self.drive = drive
        self.window = window
        self.gamma = gamma
        self.codes = [
            np.sort(self.channels[rng.choice(len(self.channels), size=active, replace=False)])
            for _ in range(vocab)
        ]
        # Position-specific codes let the reservoir distinguish "ab" from
        # "ba" when a causal window is injected.  The default reuses the
        # original codebook for backwards-compatible single-token experiments.
        self.position_codes = [self.codes]
        if position_codes:
            self.position_codes += [
                [np.sort(self.channels[rng.choice(len(self.channels), size=active, replace=False)])
                 for _ in range(vocab)]
                for _ in range(1, window)
            ]
        else:
            self.position_codes *= window

    def inject(self, tokens: np.ndarray, i: int, scale: float = 1.0):
        pairs = []
        for j in range(self.window):
            if i - j < 0:
                break
            pairs.append((self.position_codes[j][int(tokens[i - j])],
                          self.drive * self.gamma**j * scale))
        return pairs


def collect(brain, encoder: Encoder, tokens: np.ndarray, dns: np.ndarray, k_steps: int = 2,
            sustain: float = 0.0, discard: int = 0, keep_discard: bool = False, log: dict | None = None,
            labels: np.ndarray | None = None, sample: str = "end"):
    """Run the brain over a token stream; return (features, labels) at token boundaries.

    labels: target per position (default: next token). sample: "end" takes the trace
    after the token's steps, "start" right before them (uncontaminated memory)."""
    trace = Trace(brain, idx=dns)
    brain.reset()
    X, y, keep = [], [], []
    total = len(tokens) - 1
    t0 = time.time()

    def target(i):
        return int(labels[i]) if labels is not None else int(tokens[i + 1])

    for i in range(len(tokens) - 1):
        start = trace.features()
        feat = None
        for s in range(k_steps):
            if s == 0:
                inject = encoder.inject(tokens, i)
            elif sustain > 0:
                inject = encoder.inject(tokens, i, scale=sustain)
            else:
                inject = ()
            fired = brain.step(inject=inject)
            feat = trace.observe(fired)
        f = start if sample == "start" else feat
        if i >= discard:
            X.append(f)
            y.append(target(i))
        elif keep_discard:
            keep.append((f, target(i)))
        if log is not None and _progress is not None and (i % 250 == 0 or i == total - 1):
            _progress.emit(log["run"], log["cfg"], phase=log.get("phase", "collect"),
                           done=i + 1, total=total, rate=(i + 1) / max(time.time() - t0, 1e-9))
    return np.asarray(X, np.float32), np.asarray(y, np.int64), keep


def fit_readout(X: np.ndarray, y: np.ndarray, vocab: int, lam: float = 1.0):
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Z = (X - mu) / sd
    Y = np.eye(vocab, dtype=np.float32)[y]
    Ym = Y.mean(0)
    Zc = Z - Z.mean(0)
    W = np.linalg.solve(Zc.T @ Zc + lam * len(Z) * np.eye(Z.shape[1], dtype=np.float32),
                        Zc.T @ (Y - Ym))
    return W.astype(np.float32), mu, sd, Ym.astype(np.float32)


def fit_softmax_readout_gpu(X: np.ndarray, y: np.ndarray, vocab: int,
                            l2: float = 1e-4, epochs: int = 40,
                            batch_size: int = 2048, lr: float = 0.03,
                            seed: int = 0, label_smoothing: float = 0.0):
    """Fit a calibrated multiclass linear readout on CUDA with Adam.

    Ridge is a useful quick probe, but its independent one-vs-all targets are
    not a probability model.  For language-model BPC we need a categorical
    likelihood, so this keeps the frozen-reservoir setup and replaces only the
    readout objective with softmax cross-entropy.  The returned tuple matches
    :func:`fit_readout` and can use the same ``logits`` and serialization code.
    """
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - exercised on GPU hosts
        raise RuntimeError("softmax readout requires CuPy / CUDA") from exc

    if len(X) != len(y) or len(X) == 0:
        raise ValueError("X and y must be non-empty and have matching length")
    if np.any((y < 0) | (y >= vocab)):
        raise ValueError("labels must lie in [0, vocab)")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must lie in [0, 1)")

    mu = X.mean(0, dtype=np.float64).astype(np.float32)
    sd = (X.std(0, dtype=np.float64) + 1e-6).astype(np.float32)
    # Keep the frozen-brain features on the host by default, but use a full
    # device copy when it fits comfortably.  The old streaming path copied a
    # minibatch and its permutation back and forth for every Adam step.  On
    # this 8-GB card the current 20k x ~9k fused matrix is <1 GB, so keeping it
    # resident substantially raises GEMM occupancy without touching the brain
    # dynamics or the readout objective.
    n, features = X.shape
    full_gpu = False
    try:
        free_bytes, _ = cp.cuda.Device().mem_info
        full_gpu = (n * features * 4 <= 500_000_000 and
                    n * features * 4 * 1.35 < int(free_bytes))
    except Exception:
        pass
    # Keep the original CuPy permutation stream so the streaming implementation
    # is numerically comparable with the pre-streaming full-matrix trainer.
    rng = cp.random.RandomState(seed)

    # The empirical unigram prior is a stable initial intercept.  The learned
    # weights then represent the incremental evidence supplied by the brain.
    counts = cp.asarray(np.bincount(y, minlength=vocab), dtype=cp.float32)
    bias = cp.log((counts + 1.0) / (n + vocab))
    W = cp.zeros((features, vocab), dtype=cp.float32)
    mW, vW = cp.zeros_like(W), cp.zeros_like(W)
    mb, vb = cp.zeros_like(bias), cp.zeros_like(bias)
    beta1, beta2 = np.float32(0.9), np.float32(0.999)
    one = np.float32(1.0)
    step = 0
    if full_gpu:
        try:
            Zfull = cp.asarray((X - mu) / sd, dtype=cp.float32)
            yfull = cp.asarray(y, dtype=cp.int32)
        except Exception as exc:
            # A host-pinned allocation can fail even when ``mem_info`` says
            # the device copy would fit (Windows driver limits are separate
            # from free VRAM).  Fall back to the bounded streaming path rather
            # than aborting an otherwise valid brain rollout.
            print(f"full-GPU readout fallback: {type(exc).__name__}", flush=True)
            full_gpu = False
            Zfull = yfull = None
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    else:
        Zfull = yfull = None

    for epoch in range(epochs):
        order = rng.permutation(n)
        for lo in range(0, n, batch_size):
            if full_gpu:
                ids = order[lo:lo + batch_size]
                z = Zfull[ids]
                target = yfull[ids]
            else:
                ids = cp.asnumpy(order[lo:lo + batch_size])
                z = cp.asarray((X[ids] - mu) / sd, dtype=cp.float32)
                target = cp.asarray(y[ids], dtype=cp.int32)
            scores = z @ W + bias
            scores -= scores.max(axis=1, keepdims=True)
            probs = cp.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)
            if label_smoothing:
                # Gradient of cross entropy against a softened target:
                # (1-eps)*one_hot + eps/vocab.  This prevents a tiny
                # teacher-forced feature error from becoming a probability-1
                # closed-loop attractor during free generation.
                probs -= np.float32(label_smoothing / vocab)
                probs[cp.arange(len(ids)), target] -= np.float32(1.0 - label_smoothing)
            else:
                probs[cp.arange(len(ids)), target] -= one
            probs /= np.float32(len(ids))
            grad_w = z.T @ probs + np.float32(l2) * W
            grad_b = probs.sum(axis=0)

            step += 1
            mW = beta1 * mW + (one - beta1) * grad_w
            vW = beta2 * vW + (one - beta2) * (grad_w * grad_w)
            mb = beta1 * mb + (one - beta1) * grad_b
            vb = beta2 * vb + (one - beta2) * (grad_b * grad_b)
            corr1, corr2 = one - beta1 ** step, one - beta2 ** step
            W -= np.float32(lr) * (mW / corr1) / (cp.sqrt(vW / corr2) + 1e-8)
            bias -= np.float32(lr) * (mb / corr1) / (cp.sqrt(vb / corr2) + 1e-8)

    return cp.asnumpy(W), mu, sd, cp.asnumpy(bias)


def select_token_features(X: np.ndarray, tokens: np.ndarray, vocab: int, limit: int) -> np.ndarray:
    """Return the most input-token-selective reservoir features.

    This is a training-only one-way ANOVA-style signal-to-noise score: between
    current-token variance divided by within-token variance.  It deliberately
    uses the *input* token rather than the next-token target, so feature choice
    cannot peek at validation labels or directly optimise the language loss.
    """
    if len(X) != len(tokens):
        raise ValueError("features and current-token labels must have matching lengths")
    if not 0 < limit <= X.shape[1]:
        raise ValueError("feature limit must lie in [1, number of features]")
    tokens = np.asarray(tokens, np.int64)
    if np.any((tokens < 0) | (tokens >= vocab)):
        raise ValueError("token ids must lie in [0, vocab)")

    counts = np.bincount(tokens, minlength=vocab).astype(np.float64)
    sums = np.zeros((vocab, X.shape[1]), np.float64)
    sq_sums = np.zeros_like(sums)
    np.add.at(sums, tokens, X)
    np.add.at(sq_sums, tokens, X * X)
    safe_counts = np.maximum(counts, 1.0)[:, None]
    means = sums / safe_counts
    overall = X.mean(0, dtype=np.float64)
    between = (counts[:, None] * (means - overall) ** 2).sum(0)
    within = (sq_sums - sums * means).sum(0)
    within /= max(len(X) - int((counts > 0).sum()), 1)
    score = between / (within + 1e-8)
    keep = np.argpartition(score, -limit)[-limit:]
    return np.sort(keep.astype(np.int64))


def expand_selected_readout(W: np.ndarray, mu: np.ndarray, sd: np.ndarray, bias: np.ndarray,
                            selected: np.ndarray, total_features: int):
    """Embed a readout trained on a feature subset back into the full space."""
    full_W = np.zeros((total_features, W.shape[1]), np.float32)
    full_mu = np.zeros(total_features, np.float32)
    full_sd = np.ones(total_features, np.float32)
    full_W[selected] = W
    full_mu[selected] = mu
    full_sd[selected] = sd
    return full_W, full_mu, full_sd, bias


def logits(X: np.ndarray, W, mu, sd, Ym) -> np.ndarray:
    return ((X - mu) / sd) @ W + Ym


def accuracy(X: np.ndarray, y: np.ndarray, W, mu, sd, Ym) -> float:
    return float((logits(X, W, mu, sd, Ym).argmax(1) == y).mean())
