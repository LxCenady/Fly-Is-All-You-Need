"""Is the larva's worse-than-context BPC overfitting? One job: refit the readout on cached
features (no simulation) under variations, reporting train and validation BPC.

Predictions if it is overfitting (stated before running):
  1. train BPC far below validation BPC with the full features;
  2. validation BPC improves with fewer readout neurons or a stronger L2 penalty;
  3. the best variant gets below the context table alone (3.205).
If no variant beats the context table, the features carry no usable signal (not overfitting).
Control: rows shuffled (each character gets another character's brain state).
Exploratory diagnosis only; nothing here is reported as a model result.

    python overfit_check.py FEATURES.npy [FEATURES.npy ...] > overfit.json
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "uctf"))
from uctf.plugins.readouts import (ReadoutConfig, context_ids, fit_calibrated,  # noqa: E402
                                   readout_scores, score)
from uctf.plugins.tasks import load_split  # noqa: E402

DATA = REPO / "playground/gpf/data"


def evaluate(X, cx, y, ntr, V, cfg, clip=3.0):
    M, T, acc, bpc = fit_calibrated(X, cx, y, ntr, V, cfg, clip if X.shape[1] else None)
    tr = score(readout_scores(M, X[:ntr], cx[:ntr]), y[:ntr], T)[1]
    return {"train_bpc": round(float(tr), 4), "val_bpc": round(float(bpc), 4), "features": int(X.shape[1])}


def columns(neurons, n):
    """Feature columns of these readout neurons (spikes | voltage | trace blocks)."""
    return np.concatenate([neurons, n + neurons, 2 * n + neurons])


def main():
    sp = load_split(str(DATA / "tinyshakespeare.txt"), 20001, 5000, 0, str(DATA / "vocab.json"), 256)
    V, ids, n_tr = len(sp.chars), sp.ids, sp.n_tr
    y, ntr = ids[1:], n_tr - 1
    cfg = ReadoutConfig()
    cx = context_ids(ids, cfg.context_order, V, cfg.buckets)
    out = {"context_only": evaluate(np.zeros((len(y), 0), np.float32), cx, y, ntr, V, cfg)}
    print("context only", out["context_only"], file=sys.stderr, flush=True)
    for path in sys.argv[1:]:
        X = np.load(path)
        dims = json.loads(Path(path).with_suffix(".json").read_text())
        n = dims["spikes"]
        res = {}

        def run(name, Xv, c=cfg):
            res[name] = evaluate(Xv, cx, y, ntr, V, c)
            print(Path(path).parent.name, name, res[name], file=sys.stderr, flush=True)

        run("full", X)
        for m in (50, 150, 500, 1500):
            for d in range(3):
                pick = np.sort(np.random.default_rng(100 + d).choice(n, m, replace=False))
                run(f"neurons_{m}_draw{d}", X[:, columns(pick, n)])
        for l2 in (0.1, 1.0, 10.0):
            run(f"l2_{l2}", X, replace(cfg, l2=l2))
        for k, name in enumerate(("spikes", "voltage", "trace")):
            run(f"only_{name}", X[:, k * n:(k + 1) * n])
        rng = np.random.default_rng(7)
        Xs = X.copy()
        Xs[:ntr] = Xs[:ntr][rng.permutation(ntr)]
        Xs[ntr:] = Xs[ntr:][rng.permutation(len(Xs) - ntr)]
        run("shuffled_rows", Xs)
        out[str(path)] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
