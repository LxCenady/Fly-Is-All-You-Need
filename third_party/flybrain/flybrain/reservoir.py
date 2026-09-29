"""Generic reservoir computing on the fly connectome.

    input -> encoder -> fly brain (frozen) -> trace -> trained readout -> output

`flybrain.FlyBrain` is never trained: its weights come straight from the connectome.
This module is everything task-agnostic around it:

  * `Trace` turns the neurons that fire each step into a decaying feature vector,
    for any neuron population you name (a cell type, a `brain.groups[...]` set,
    or your own index array).
  * `run` steps the brain over a sequence of inputs and collects a `Trace`,
    so the whole loop is one call from a notebook.
  * `Readout` fits a linear (ridge) or logistic PCA readout from that activity to
    your own labels, model-selected by cross-validation, and predicts on new
    activity.

Nothing here knows about SSH Fighter, or any other task. `sshfighter/reservoir.py`
is a worked example of using this module for one game; `flyreservoir_example.py`
at the repo root is a minimal one with synthetic data, runnable without any game
or recordings.

Encoders (turning task input into neuron drive) are necessarily task-specific --
you write them by picking `brain.cells([...types], side=...)` and passing
`(idx, amount)` pairs to `brain.step(inject=...)`. `flybrain/eyes.py` is a worked
example of a visual encoder for SSH Fighter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

__all__ = ["Trace", "run", "bases_for", "project", "fit_logistic", "fit_ridge", "auc", "folds", "Readout"]


# ---- trace: turns spikes into features ---------------------------------------------------

class Trace:
    """Exponentially decaying spike trace of a neuron population.

    Pick the population by `types` (a list of MaleCNS cell types, via `brain.cells`),
    `group` (a name from `brain.groups`), or your own `idx` array. `side` restricts
    `types` to "L" or "R" (see `FlyBrain.cells`).

    With a batched brain (`batch > 1`, several flies stepped together), `aggregate`
    controls what `features()` returns: "mean" (default) is one vector, the flies'
    average trace -- the natural choice when the flies vote on one shared decision;
    "batch" keeps one column per fly, shape (n_neurons, batch), for independent
    per-fly readouts.
    """

    def __init__(self, brain, types: list[str] | None = None, group: str | None = None,
                 idx: np.ndarray | None = None, side: str | None = None, tau: float = 0.1,
                 aggregate: str = "mean"):
        if sum(x is not None for x in (types, group, idx)) != 1:
            raise ValueError("pass exactly one of types, group, idx")
        if idx is None:
            idx = brain.groups[group] if group is not None else brain.cells(types, side=side)
        if aggregate not in ("mean", "batch"):
            raise ValueError('aggregate must be "mean" or "batch"')
        self.idx = np.asarray(idx)
        self.slot = np.full(brain.n, -1, np.int64)
        self.slot[self.idx] = np.arange(len(self.idx))
        self.aggregate = aggregate
        self.batch = brain.batch
        width = len(self.idx)
        self.trace = np.zeros(width if aggregate == "mean" else (width, self.batch), np.float32)
        self.decay = np.float32(np.exp(-brain.dt / tau))

    def observe(self, fired) -> np.ndarray:
        """`fired`: whatever `FlyBrain.step()` returned -- one array of spike indices
        (batch 1), or a list of them, one per fly. Returns `features()`."""
        self.trace *= self.decay
        flies = fired if isinstance(fired, list) else [fired]
        if self.aggregate == "mean":
            for f in flies:
                slots = self.slot[f]
                self.trace[slots[slots >= 0]] += 1.0 / len(flies)
        else:
            for b, f in enumerate(flies):
                slots = self.slot[f]
                self.trace[slots[slots >= 0], b] += 1.0
        return self.features()

    def features(self) -> np.ndarray:
        return self.trace.copy()

    def reset(self) -> None:
        self.trace[...] = 0


def run(brain, steps: int, encode=None, trace: Trace | None = None, eye_drive=None) -> np.ndarray:
    """Step the brain `steps` times and collect activity, so a whole task can run
    from one call: `activity = flybrain.run(brain, len(inputs), encode=...)`.

    `encode(t)`, if given, returns the `inject` list for step `t` (see
    `FlyBrain.step`) -- typically `encoder.inject(...)` for some task-specific
    encoder built on `brain.cells(...)`. `eye_drive`, if given, is either an
    array (same every step) or `eye_drive(t)`. `trace` defaults to a `Trace`
    over every neuron; pass your own to watch a specific population.

    Returns activity stacked over time: `(steps, n_features)`, or
    `(steps, n_features, batch)` if `trace.aggregate == "batch"`.
    """
    if trace is None:
        trace = Trace(brain, idx=np.arange(brain.n))
    out = []
    for t in range(steps):
        inject = encode(t) if encode is not None else ()
        drive = eye_drive(t) if callable(eye_drive) else eye_drive
        fired = brain.step(eye_drive=drive, inject=inject)
        out.append(trace.observe(fired))
    return np.stack(out)


# ---- PCA basis shared by training and inference ------------------------------------------

def bases_for(X: np.ndarray, ks) -> dict:
    """For each k in `ks`: (mean, projection, scale) of the top-k principal
    components of X's columns. k=None just standardises every column (no PCA)."""
    mu = X.mean(0)
    out, vt = {}, None
    for k in ks:
        if k is None:
            P = np.eye(X.shape[1], dtype=np.float32)
        else:
            if vt is None:
                vt = np.linalg.svd(X - mu, full_matrices=False)[2]
            P = vt[:min(k, len(vt))].T.astype(np.float32)
        sd = ((X - mu) @ P).std(0) + 1e-6
        out[k] = (mu, P, sd)
    return out


def project(basis, X: np.ndarray) -> np.ndarray:
    mu, P, sd = basis
    return ((X - mu) @ P) / sd


# ---- fitting: linear (ridge) and logistic readouts ----------------------------------------

def fit_logistic(Z: np.ndarray, y: np.ndarray, lam: float):
    """L2-regularised logistic regression, y in {0, 1}. Returns (weights, bias)."""
    def objective(w):
        z = Z @ w[:-1] + w[-1]
        loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w[:-1] @ w[:-1]
        r = (expit(z) - y) / len(y)
        return loss, np.append(Z.T @ r + lam * w[:-1], r.sum())

    w = minimize(objective, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B").x
    return w[:-1], w[-1]


def fit_ridge(Z: np.ndarray, y: np.ndarray, lam: float):
    """Ridge regression. y is (n,) for one output or (n, outputs) for several
    (e.g. one column per class, fit jointly). Returns (weights, bias): weights
    is (features,) or (outputs, features) to match y; bias matches in the same way."""
    single = y.ndim == 1
    y2 = y[:, None] if single else y
    zm, ym = Z.mean(0), y2.mean(0)
    Zc = Z - zm
    W = np.linalg.solve(Zc.T @ Zc + lam * len(y2) * np.eye(Z.shape[1]), Zc.T @ (y2 - ym))
    b = ym - zm @ W
    return (W[:, 0], float(b[0])) if single else (W.T, b)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve. nan if y has only one class."""
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = np.empty(len(s))
    ranks[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def folds(n: int, groups: np.ndarray | None = None, k: int = 5):
    """Cross-validation splits over n samples, as (train_idx, test_idx) pairs.
    With `groups` (one label per sample, e.g. a match or recording id), this is
    leave-one-group-out. Without it, plain k-fold."""
    if groups is not None:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            test = np.flatnonzero(groups == g)
            train = np.flatnonzero(groups != g)
            if len(train):
                yield train, test
    else:
        idx = np.arange(n)
        for part in np.array_split(idx, min(k, n)):
            if len(part) == 0 or len(part) == n:
                continue
            test = part
            train = np.setdiff1d(idx, test, assume_unique=True)
            yield train, test


class Readout:
    """A PCA + linear readout: activity -> label, with the PCA rank and L2
    strength picked by cross-validation.

    `kind="ridge"` regresses onto a continuous or multi-column target (mean
    squared error). `kind="logistic"` classifies a binary target in {0, 1}
    (held-out AUC). Both compress activity to its top principal components
    first (see `bases_for`) -- this is what makes it a *reservoir* readout: the
    brain does the nonlinear mixing, the readout only has to be linear on that.
    """

    def __init__(self, kind: str, basis, w, b, cv_score: float, components: int | None, lam: float):
        self.kind, self.basis, self.w, self.b = kind, basis, w, b
        self.cv_score, self.components, self.lam = cv_score, components, lam

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, kind: str = "ridge", groups: np.ndarray | None = None,
            components=(5, 20, 60), lambdas=(1e-2, 1e-1, 1.0, 10.0), verbose: bool = False) -> "Readout":
        """X: (n_samples, n_features) activity, e.g. from `Trace.features()` stacked
        over time. y: (n_samples,) labels -- 0/1 for `kind="logistic"`, numeric
        (or (n_samples, outputs)) for `kind="ridge"`. `groups`, if given, makes
        cross-validation leave-one-group-out (e.g. one group per recorded episode)
        instead of plain k-fold, so the score isn't inflated by correlated samples
        from the same episode landing in both train and test.

        Every (k, lambda) in the grid is scored by cross-validated held-out AUC
        (logistic) or negative MSE (ridge); the best is refit on all of X, y.
        """
        if kind not in ("ridge", "logistic"):
            raise ValueError('kind must be "ridge" or "logistic"')
        fit_fn = fit_logistic if kind == "logistic" else fit_ridge
        grid = [(k, lam) for k in components for lam in lambdas]
        scores = {g: [] for g in grid}
        for train, test in folds(len(X), groups):
            Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
            if kind == "logistic" and (ytr.min() == ytr.max() or yte.min() == yte.max()):
                continue
            bases = bases_for(Xtr, components)
            for k, lam in grid:
                w, b = fit_fn(project(bases[k], Xtr), ytr, lam)
                pred = project(bases[k], Xte) @ w.T + b if w.ndim > 1 else project(bases[k], Xte) @ w + b
                score = auc(yte, expit(pred)) if kind == "logistic" else -np.mean((pred - yte) ** 2)
                if not np.isnan(score):
                    scores[(k, lam)].append(score)
        avg = {g: (float(np.mean(s)) if s else float("-inf")) for g, s in scores.items()}
        k, lam = max(avg, key=avg.get)
        basis = bases_for(X, [k])[k]
        w, b = fit_fn(project(basis, X), y, lam)
        if verbose:
            print(f"readout: {k} components, lambda {lam:g}, "
                  f"cross-validated {'AUC' if kind == 'logistic' else 'neg-MSE'} {avg[(k, lam)]:.3f}")
        return cls(kind, basis, w, b, avg[(k, lam)], k, lam)

    def predict(self, x: np.ndarray) -> np.ndarray | float:
        """x: one sample (n_features,) or a batch (n_samples, n_features). Returns
        the raw ridge output, or the logistic probability, in the same shape."""
        single = x.ndim == 1
        z = project(self.basis, x[None] if single else x)
        raw = z @ self.w.T + self.b if self.w.ndim > 1 else z @ self.w + self.b
        out = expit(raw) if self.kind == "logistic" else raw
        return out[0] if single else out

    def save(self, path: str | Path) -> None:
        np.savez(path, kind=self.kind, mu=self.basis[0], P=self.basis[1], sd=self.basis[2],
                 w=self.w, b=np.asarray(self.b), cv_score=self.cv_score,
                 components=self.components if self.components is not None else -1, lam=self.lam)

    @classmethod
    def load(cls, path: str | Path) -> "Readout":
        d = np.load(path, allow_pickle=False)
        basis = (d["mu"], d["P"], d["sd"])
        b = d["b"]
        b = float(b) if b.ndim == 0 else b
        components = int(d["components"])
        return cls(str(d["kind"]), basis, d["w"], b, float(d["cv_score"]),
                    None if components < 0 else components, float(d["lam"]))
