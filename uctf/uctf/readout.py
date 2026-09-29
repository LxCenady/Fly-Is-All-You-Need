"""The linear readout: softmax over standardised neuron features plus a hashed context table.

Rows: features after character i predict character i+1.  Standardisation statistics come from
the rows being fitted only; the temperature is chosen on the last 10% of the training rows by a
model fitted on the first 90%, then the readout is refitted on all training rows.  The context
table alone (no features) is the baseline every connectome must beat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LOG2 = math.log(2)


@dataclass
class ReadoutConfig:
    l2: float = 1e-2
    epochs: int = 15
    lr: float = 0.01
    buckets: int = 32768
    context_order: int = 3
    seed: int = 0


def ctx_hash(hist, order, V, buckets):
    """Hash of the last `order` characters (0 before the start), as in lm_mech.ctx_ids."""
    h = 0
    for lag in range(order - 1, -1, -1):
        h = h * V + (hist[-1 - lag] if len(hist) > lag else 0)
    return h % buckets


def context_ids(ids, order, V, buckets) -> np.ndarray:
    """Context-table row of every feature row (the characters up to and including i)."""
    return np.asarray([ctx_hash(list(ids[max(0, i - order + 1):i + 1]), order, V, buckets)
                       for i in range(len(ids) - 1)], np.int64)


def fit_readout(X, cx, y, V, cfg, clip):
    """Softmax readout: standardised features (clipped at +-clip) + a hashed context table
    initialised from smoothed counts; Adam, minibatches of 1024.  A numpy port of lm_mech.fit
    (mechanism/export_lm_model.py).  X may have zero columns (context table only)."""
    n = len(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)
    if clip:
        np.clip(Z, -clip, clip, out=Z)
    W = np.zeros((X.shape[1], V), np.float32)
    prior = (np.bincount(y, minlength=V) + 1) / (n + V)
    b = np.log(prior).astype(np.float32)
    C = np.zeros((cfg.buckets, V)); np.add.at(C, (cx, y), 1.0)
    E = (np.log((C + 2.0 * prior) / (C.sum(1, keepdims=True) + 2.0)) - np.log(prior)).astype(np.float32)
    params = [W, E, b]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    rng = np.random.RandomState(cfg.seed); step = 0
    for _ in range(cfg.epochs):
        order = rng.permutation(n)
        for lo in range(0, n, 1024):
            idx = order[lo:lo + 1024]
            s = b + E[cx[idx]] + Z[idx] @ W
            s -= s.max(1, keepdims=True)
            p = np.exp(s); p /= p.sum(1, keepdims=True)
            p[np.arange(len(idx)), y[idx]] -= 1; p /= len(idx)
            gE = np.zeros_like(E); np.add.at(gE, cx[idx], p)
            grads = [Z[idx].T @ p + cfg.l2 * W, gE, p.sum(0)]
            step += 1
            for j, (P, G) in enumerate(zip(params, grads)):
                m[j] = 0.9 * m[j] + 0.1 * G; v[j] = 0.999 * v[j] + 0.001 * G * G
                P -= cfg.lr * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "W": W, "E": E, "b": b, "clip": clip}


def readout_scores(M, X, cx):
    Z = (X - M["mu"]) / M["sd"]
    if M["clip"]:
        Z = np.clip(Z, -M["clip"], M["clip"])
    return M["b"] + M["E"][cx] + Z @ M["W"]


def score(s, y, T):
    """(accuracy, bits per character) of scores s at temperature T."""
    s = s / T
    s = s - s.max(1, keepdims=True)
    lp = s - np.log(np.exp(s).sum(1, keepdims=True))
    return float((s.argmax(1) == y).mean()), float(-lp[np.arange(len(y)), y].mean() / LOG2)


def fit_calibrated(X, cx, y, n_tr, V, cfg, clip):
    """Temperature chosen on the last 10% of the training rows, then refit on all of them.
    Returns (model, temperature, validation accuracy, validation BPC); validation = rows n_tr on."""
    cut = int(n_tr * 0.9)
    M = fit_readout(X[:cut], cx[:cut], y[:cut], V, cfg, clip)
    s = readout_scores(M, X[cut:n_tr], cx[cut:n_tr])
    Ts = np.linspace(0.6, 2.0, 29)
    T = float(Ts[int(np.argmin([score(s, y[cut:n_tr], t)[1] for t in Ts]))])
    M = fit_readout(X[:n_tr], cx[:n_tr], y[:n_tr], V, cfg, clip)
    acc, bpc = score(readout_scores(M, X[n_tr:], cx[n_tr:]), y[n_tr:], T)
    return M, T, acc, bpc
