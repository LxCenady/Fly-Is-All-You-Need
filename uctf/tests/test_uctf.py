"""UCTF invariants on a small random network (CPU, seconds).

    python -m pytest uctf/tests        or        python uctf/tests/test_uctf.py
"""
import copy
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uctf import run  # noqa: E402
from uctf.core import registry  # noqa: E402
from uctf.core.connectome import (  # noqa: E402
    Connectome, Layer, canonical, load_folder)
from uctf.core.network import Network  # noqa: E402
from uctf.core.select import Selector  # noqa: E402
from uctf.core.spec import load_spec, migrate_v1  # noqa: E402
from uctf.plugins.readouts import (  # noqa: E402
    ReadoutConfig, context_ids, fit_calibrated)

MAX_LINE = 88


def network(n=300, seed=0):
    rng = np.random.default_rng(seed)
    pre, post = rng.integers(0, n, 6000), rng.integers(0, n, 6000)
    keep = pre != post
    sign = np.where(pre[keep] % 7 == 0, -1, 1)
    w = rng.uniform(0.02, 0.2, keep.sum()).astype(np.float32) * sign
    shape = (n, n)
    W = canonical(sparse.coo_matrix((w, (post[keep], pre[keep])), shape))
    a, b = rng.integers(0, n, 400), rng.integers(0, n, 400)
    k = a != b
    ones = np.ones(k.sum(), np.float32)
    E = sparse.coo_matrix((ones, (a[k], b[k])), shape).tocsr()
    G = canonical(E.maximum(E.T))
    types = ["IN" if i < 60 else "HID" if i < 240 else "OUT"
             for i in range(n)]
    layers = {"chemical": Layer("chemical", W),
              "electrical": Layer("electrical", G)}
    return Connectome({"cell_type": np.array(types)}, layers, None,
                      {"name": "test"})


def pop(name):
    return {"col": "cell_type", "equals": name}


SPEC = migrate_v1({
    "name": "test", "connectome": {"loader": "folder", "path": "unused"},
    "populations": {k: pop(k) for k in ("IN", "HID", "OUT")},
    "neuron": {"gain": 1.0, "tonic": 0.05, "gap_gain": 0.3},
    "input": {"population": "IN", "active": 10, "drive": 1.5},
    "readout": [
        {"name": "hid", "population": "HID", "feature": "counts"},
        {"name": "out_v", "population": "OUT", "feature": "voltage"}],
    "classes": {"populations": ["IN", "HID", "OUT"]}})


def make(controls=(), cx=None):
    return Network(SPEC, 5, controls, device="cpu",
                   connectome=cx or network(), log=lambda m: None)


def test_rewire_preserves_degrees_and_outputs():
    cx = network()
    lab = cx.column("cell_type")
    rewire = registry.get("transform", "rewire")
    methods = {"chemical": "permute_presynaptic",
               "electrical": "double_edge_swap"}
    sel = Selector(cx, SPEC["populations"])
    W, G = cx.layer("chemical").matrix, cx.layer("electrical").matrix
    for classes in (None, SPEC["controls"]["rewire-class"][0]["params"]
                    ["classes"]):
        R, info = rewire(cx, sel, methods, seed=1, classes=classes)
        RW, RG = R.layer("chemical").matrix, R.layer("electrical").matrix
        # every presynaptic neuron sends the same total (signed) weight
        assert np.allclose(np.asarray(W.sum(0)).ravel(),
                           np.asarray(RW.sum(0)).ravel(), atol=1e-5)
        # inputs per neuron kept; fewer only where duplicates merged
        lost = np.diff(W.indptr) - np.diff(RW.indptr)
        assert (lost >= 0).all()
        chem = info["chemical"]
        assert chem["edges"] - chem["edges_after_merge"] == lost.sum()
        # gap junctions: symmetric, same degree per neuron, same weights
        assert abs(RG - RG.T).max() == 0
        assert (np.diff(RG.indptr) == np.diff(G.indptr)).all()
        assert np.isclose(RG.sum(), G.sum())
        if classes is None:
            continue
        for a in ("IN", "HID", "OUT"):             # class blocks keep weight
            for b in ("IN", "HID", "OUT"):
                block = np.ix_(lab == b, lab == a)
                assert np.isclose(W[block].sum(), RW[block].sum(), atol=1e-4)


def test_simulation_deterministic_and_control_keeps_readout_cells():
    cx = network()
    a, b = make(cx=cx), make(cx=cx)
    seq = [0, 3, 1, 4, 2, 2, 0, 1]
    xa = np.stack([a.step_token(t) for t in seq])
    assert np.array_equal(xa, np.stack([b.step_token(t) for t in seq]))
    a.reset()
    assert np.array_equal(np.stack([a.step_token(t) for t in seq]), xa)
    ctl = run.control_steps(SPEC, "rewire-full", 0)
    c = make(ctl, cx)
    for fa, fc in zip(a.features, c.features):
        assert np.array_equal(fa.ids, fc.ids)
    Wa, Wc = a.cx.layer("chemical").matrix, c.cx.layer("chemical").matrix
    assert Wc.nnz <= Wa.nnz
    assert not np.array_equal(Wc.indices, Wa.indices)


def test_respec_changes_dynamics_only():
    net = make()
    x0 = net.step_token(1)
    net.respec("synapses.chemical.params.gain", 3.0)
    assert net.spec["synapses"][0]["params"]["gain"] == 3.0
    assert not np.array_equal(net.step_token(1), x0)
    try:
        net.respec("populations.IN.equals", "HID")
    except ValueError:
        pass
    else:
        raise AssertionError("respec must refuse wiring changes")


def test_feature_key_tracks_simulation_not_display():
    ids = np.arange(50) % 5
    base = run.brain_spec(SPEC)
    k0 = run.feature_key(ids, 5, base)
    view = copy.deepcopy(base)
    view["spec"]["view"] = {"title": "x"}
    assert run.feature_key(ids, 5, view) == k0
    gain = run.brain_spec(
        SPEC, overrides=["synapses.chemical.params.gain=1.1"])
    assert run.feature_key(ids, 5, gain) != k0
    ctl = run.brain_spec(SPEC, control="rewire-full")
    assert run.feature_key(ids, 5, ctl) != k0


def test_overrides_are_strict_and_accept_v1_paths():
    from uctf.core.spec import validate
    b = run.brain_spec(SPEC, overrides=["input.code_seed=7", "neuron.tau=0.2"])
    assert b["spec"]["input"]["params"]["seed"] == 7
    assert b["spec"]["neuron"]["params"]["tau"] == 0.2
    for bad in ("input.code_sed=7", "synapses.chemcal.params.gain=1",
                "neuron.model_x=1"):
        try:
            run.brain_spec(SPEC, overrides=[bad])
        except KeyError:
            continue
        raise AssertionError(f"override {bad!r} was accepted")
    typo = copy.deepcopy(SPEC)
    typo["input"]["code_seed"] = 5
    try:
        validate(typo)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown spec key was accepted")


def test_v1_brain_spec_loads_as_before():
    """Brain specs saved by UCTF 0.1 give the same features."""
    old = {"spec": {"name": "test", "connectome": {"path": "unused"},
                    "populations": {k: pop(k) for k in ("IN", "HID", "OUT")},
                    "neuron": {"gain": 1.0, "gap_gain": 0.3},
                    "input": {"population": "IN", "active": 10},
                    "readout": [{"name": "hid", "population": "HID",
                                 "feature": "counts"}],
                    "classes": {"populations": ["IN", "HID"]}},
           "control": {"rewire": "class", "seed": 2}}
    spec = load_spec(old["spec"])
    ctl = run.control_steps(spec, "rewire-class", 2)
    assert ctl == run.brain_spec(spec, control="rewire-class",
                                 control_seed=2)["controls"]
    cx = network()
    a = Network(spec, 5, ctl, "cpu", cx, log=lambda m: None)
    from uctf.core.spec import migrate_control_v1
    b = Network(spec, 5, migrate_control_v1(old["control"], spec), "cpu",
                cx, log=lambda m: None)
    assert np.array_equal(a.step_token(2), b.step_token(2))


def test_folder_round_trip():
    import tempfile
    cx = network()
    lay = cx.layer("chemical")
    nt = np.where(np.arange(lay.matrix.nnz) % 3, "ach", "gaba")
    cx = cx.with_layer("chemical", Layer("chemical", lay.matrix,
                                         {"transmitter": nt}))
    with tempfile.TemporaryDirectory() as d:
        back = load_folder(cx.save(Path(d) / "c"))
    for name in cx.layers:
        a, b = cx.layer(name).matrix, back.layer(name).matrix
        assert (a != b).nnz == 0
    edges = back.layer("chemical").edges
    assert (edges["transmitter"] == nt).all()


def test_transmitter_split_matches_one_layer():
    """by_transmitter with the same params for every label is the plain
    chemical synapse; a different gain for one label changes the run."""
    cx = network()
    nt = np.where(np.arange(cx.n) % 3, "ACh", "GABA")
    cx.neurons["nt"] = nt
    spec = copy.deepcopy(SPEC)
    spec["transforms"] = [{"transform": "edge_labels", "params": {
        "layer": "chemical", "column": "nt", "name": "transmitter"}}]
    same = {"default": {"gain": 1.0}}
    spec["synapses"][0] = {"layer": "chemical", "model": "by_transmitter",
                           "params": same}
    spec.pop("activity_match")          # its param (the gain) is gone
    seq = [0, 3, 1, 4, 2]
    a = make(cx=cx)
    b = Network(spec, 5, (), "cpu", cx, log=lambda m: None)
    xa = np.stack([a.step_token(t) for t in seq])
    xb = np.stack([b.step_token(t) for t in seq])
    assert np.allclose(xa, xb, atol=1e-5)
    b.respec("synapses.chemical.params.table", {"gaba": {"gain": 3.0}})
    xc = np.stack([b.step_token(t) for t in seq])
    assert not np.allclose(xa, xc)


def test_readout_rows_are_aligned():
    """Features holding the next token give near-zero BPC; features holding
    only the current one cannot beat the context table. Guards the
    row/target alignment."""
    rng = np.random.default_rng(0)
    V, n, ntr = 8, 3000, 2400
    ids = rng.integers(0, V, n)
    y = ids[1:]
    cfg = ReadoutConfig(epochs=60)
    cx = context_ids(ids, 3, V, cfg.buckets)
    leak = np.eye(V, dtype=np.float32)[y]
    assert fit_calibrated(leak, cx, y, ntr, V, cfg, 3.0)[3] < 0.5
    cur = np.eye(V, dtype=np.float32)[ids[:-1]]
    none = np.zeros((n - 1, 0), np.float32)
    base = fit_calibrated(none, cx, y, ntr, V, cfg, None)[3]
    assert fit_calibrated(cur, cx, y, ntr, V, cfg, 3.0)[3] > base - 0.05


def test_every_builtin_plugin_registers():
    got = registry.available()
    assert set(got) == set(registry.KINDS)
    assert all(got[k] for k in registry.KINDS)
    assert not registry.skipped()


def test_lines_are_short():
    """The kernel and plugins keep lines readable."""
    long = []
    for p in sorted((ROOT / "uctf").rglob("*.py")):
        for i, line in enumerate(p.read_text("utf-8").splitlines(), 1):
            if len(line) > MAX_LINE:
                long.append(f"{p.relative_to(ROOT)}:{i}")
    assert not long, f"lines over {MAX_LINE}: {long}"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
