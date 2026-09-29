"""UCTF invariants, on a small random network (CPU, seconds).

    python -m pytest uctf/tests        or        python uctf/tests/test_uctf.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uctf import controls, run  # noqa: E402
from uctf.data import Connectome, canonical  # noqa: E402
from uctf.readout import ReadoutConfig, context_ids, fit_calibrated  # noqa: E402
from uctf.substrate import LIFSubstrate  # noqa: E402


def network(n=300, seed=0, gap=True):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, 6000); post = rng.integers(0, n, 6000)
    keep = pre != post
    w = rng.uniform(0.02, 0.2, keep.sum()).astype(np.float32) * np.where(pre[keep] % 7 == 0, -1, 1)
    W = canonical(sparse.coo_matrix((w, (post[keep], pre[keep])), shape=(n, n)))
    G = None
    if gap:
        a = rng.integers(0, n, 400); b = rng.integers(0, n, 400); k = a != b
        E = sparse.coo_matrix((np.ones(k.sum(), np.float32), (a[k], b[k])), shape=(n, n)).tocsr()
        G = canonical(E.maximum(E.T))
    ann = {"cell_type": np.array(["IN" if i < 60 else "HID" if i < 240 else "OUT" for i in range(n)])}
    return Connectome(W, ann, None, "test", G)


SPEC = {"name": "test", "connectome": {"loader": "folder", "path": "unused"},
        "populations": {"IN": {"col": "cell_type", "equals": "IN"}, "HID": {"col": "cell_type", "equals": "HID"},
                        "OUT": {"col": "cell_type", "equals": "OUT"}},
        "neuron": {"gain": 1.0, "tonic": 0.05, "gap_gain": 0.3},
        "input": {"population": "IN", "active": 10, "drive": 1.5},
        "readout": [{"name": "hid", "population": "HID", "feature": "counts"},
                    {"name": "out_v", "population": "OUT", "feature": "voltage"}],
        "classes": {"populations": ["IN", "HID", "OUT"]}}


def test_rewire_preserves_degrees_and_outputs():
    cx = network()
    lab = cx.column("cell_type")
    for classes in (None, lab):
        R, info = controls.rewire(cx, seed=1, classes=classes)
        # every presynaptic neuron sends the same total (signed) weight
        assert np.allclose(np.asarray(cx.W.sum(0)).ravel(), np.asarray(R.W.sum(0)).ravel(), atol=1e-5)
        # every postsynaptic neuron keeps its number of input edges (fewer only where duplicates merged)
        assert (np.diff(R.W.indptr) <= np.diff(cx.W.indptr)).all()
        assert info["edges"] - info["edges_after_merge"] == (np.diff(cx.W.indptr) - np.diff(R.W.indptr)).sum()
        # gap junctions: symmetric, same degree of every neuron, same weights
        assert abs(R.G - R.G.T).max() == 0
        assert (np.diff(R.G.indptr) == np.diff(cx.G.indptr)).all()
        assert np.isclose(R.G.sum(), cx.G.sum())
        if classes is not None:                          # blocks of (pre class, post class) keep their edge counts
            for a in "IN", "HID", "OUT":
                for b in "IN", "HID", "OUT":
                    m0 = cx.W[np.ix_(lab == b, lab == a)]; m1 = R.W[np.ix_(lab == b, lab == a)]
                    assert np.isclose(m0.sum(), m1.sum(), atol=1e-4)


def test_simulation_deterministic_and_control_keeps_readout_cells():
    cx = network()
    a = LIFSubstrate(5, SPEC, device="cpu", connectome=cx, log=lambda m: None)
    b = LIFSubstrate(5, SPEC, device="cpu", connectome=cx, log=lambda m: None)
    seq = [0, 3, 1, 4, 2, 2, 0, 1]
    xa = np.stack([a.step_token(t) for t in seq]); xb = np.stack([b.step_token(t) for t in seq])
    assert np.array_equal(xa, xb)
    a.reset()
    assert np.array_equal(np.stack([a.step_token(t) for t in seq]), xa)
    c = LIFSubstrate(5, SPEC, {"rewire": "full", "seed": 0}, device="cpu", connectome=cx, log=lambda m: None)
    assert all(np.array_equal(r1[1], r2[1]) for r1, r2 in zip(a.readouts, c.readouts))
    assert c.W.nnz <= a.W.nnz and not np.array_equal(c.W.indices, a.W.indices)


def test_feature_key_tracks_simulation_not_display():
    ids = np.arange(50) % 5
    base = run.brain_spec(dict(SPEC))
    k0 = run.feature_key(ids, 5, base)
    view = json.loads(json.dumps(base)); view["spec"]["view"] = {"title": "x"}
    assert run.feature_key(ids, 5, view) == k0
    gain = json.loads(json.dumps(base)); gain["spec"]["neuron"]["gain"] = 1.1
    assert run.feature_key(ids, 5, gain) != k0
    ctl = run.brain_spec(dict(SPEC), control="rewire-full")
    assert run.feature_key(ids, 5, ctl) != k0


def test_readout_rows_are_aligned():
    """Features that contain the next character give near-zero BPC; features that contain only
    the current one cannot beat the context table.  Guards the row/target alignment."""
    rng = np.random.default_rng(0)
    V, n = 8, 3000
    ids = rng.integers(0, V, n)
    y = ids[1:]
    cfg = ReadoutConfig(epochs=60)                         # enough Adam steps to learn a one-hot map
    cx = context_ids(ids, 3, V, cfg.buckets)
    ntr = 2400
    leak = np.eye(V, dtype=np.float32)[y]                  # row i holds the target of row i
    assert fit_calibrated(leak, cx, y, ntr, V, cfg, 3.0)[3] < 0.5
    cur = np.eye(V, dtype=np.float32)[ids[:-1]]            # row i holds character i (already in the table)
    base = fit_calibrated(np.zeros((n - 1, 0), np.float32), cx, y, ntr, V, cfg, None)[3]
    assert fit_calibrated(cur, cx, y, ntr, V, cfg, 3.0)[3] > base - 0.05


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
