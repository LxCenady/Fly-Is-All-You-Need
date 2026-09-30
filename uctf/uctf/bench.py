"""The standard comparison for a connectome language model.

    python -m uctf bench --substrate SPEC --data text.txt
        [--controls rewire-full,rewire-class] [--match-activity]

On one train/validation split of the text (task "next_char") it reports,
in bits per character (lower is better):

    unigram, Kneser-Ney 3/5-gram   how far simple statistics get
    context table only             the readout's hashed context table alone
    connectome                     the same readout plus the brain's features
    each control                   the same, on a control of the wiring
                                   (a named entry of the spec's "controls")

and a memory span (probe "memory_span"): how well a ridge decoder recovers
the token k steps back from each readout group, for k = 0..--span.

--match-activity also runs each control with the spec's
"activity_match.param" set so that the neurons of
"activity_match.readouts" are as active as with the real wiring, found
by bisection on the first --match-chars training characters only.

The protocol of paper_lm/ (Sections VI-VII). Results: <out>/report.json
and <out>/report.md; simulated features are cached in <out>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import run
from .core import registry
from .core.spec import get_path, override


# ---------------------------------------------------------------- activity
def activity_ids(net) -> np.ndarray:
    """The neurons whose activity is matched (spec "activity_match")."""
    names = net.spec["activity_match"]["readouts"]
    pops = [r["population"] for r in net.spec["readout"]
            if r["name"] in names]
    if not pops:
        raise ValueError("activity_match.readouts names no readout")
    return np.unique(np.concatenate([net.sel.ids(p) for p in pops]))


def active_fraction(net, ids, n) -> float:
    """Mean fraction of `ids` that fire during a token, first n tokens."""
    targets = activity_ids(net)
    net.reset()
    net.record_activity = True
    out = []
    for t in ids[:n]:
        net.step_token(int(t))
        out.append(np.isin(targets, net.last_fired["ids"]).mean())
    net.record_activity = False
    return float(np.mean(out))


def match_activity(bspec, ids, V, device, target, n=500, iters=14):
    """(value, activity): the activity_match param at which the network is
    as active as `target`; with target None, the activity at the spec's own
    value."""
    from .core.network import Network
    spec = bspec["spec"]
    param = spec["activity_match"]["param"]
    net = Network(spec, V, bspec["controls"], device, log=lambda m: None)
    v0 = float(get_path(spec, param))
    if target is None:
        return v0, active_fraction(net, ids, n)

    def at(value):
        net.respec(param, float(value))
        return active_fraction(net, ids, n)

    lo, hi = np.log(v0 / 8), np.log(v0 * 8)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if at(np.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    value = float(np.exp((lo + hi) / 2))
    return value, at(value)


def count_fraction(X, dims, spec, rows) -> float | None:
    """Mean fraction of neurons in the spike-count readouts that fire."""
    counts = {r["name"] for r in spec["readout"] if r["feature"] == "counts"}
    lo, parts = 0, []
    for name, d in dims.items():
        if name in counts:
            parts.append((X[rows, lo:lo + d] > 0).mean(1))
        lo += d
    return float(np.mean(np.concatenate(parts))) if parts else None


# ---------------------------------------------------------------- bench
def features(bspec, split, a, out, say):
    V = len(split.chars)
    key = run.feature_key(split.ids, V, bspec)[:16]
    cache, dims_file = out / f"features_{key}.npy", out / f"features_{key}.json"
    if cache.exists() and dims_file.exists():
        say(f"reusing {cache.name}")
        return np.load(cache), json.loads(dims_file.read_text("utf-8"))
    X, dims = run.simulate(split.ids, V, bspec, a.device, with_dims=True)
    np.save(cache, X)
    dims_file.write_text(json.dumps(dims), encoding="utf-8")
    return X, dims


def variants(a) -> list:
    """(label, control, seed, activity-matched) for every run."""
    out = [("connectome", "none", 0, False)]
    controls = [c.strip() for c in a.controls.split(",")]
    seeds = [int(s) for s in a.control_seeds.split(",")]
    for c in controls:
        if not c or c == "none":
            continue
        for s in seeds:
            out.append((f"{c} (seed {s})", c, s, False))
            if a.match_activity:
                label = f"{c} (seed {s}, activity-matched)"
                out.append((label, c, s, True))
    return out


def bench(a):
    say = run.log_flush
    task = registry.get("task", "next_char")(
        path=a.data, train_chars=a.train_chars, val_chars=a.val_chars,
        offset=a.offset, vocab=a.vocab, max_vocab=a.max_vocab)
    try:
        split = task.split()
    except ValueError as e:
        sys.exit(str(e))
    V, n_tr = len(split.chars), split.n_tr
    out = Path(a.out or f"bench-{Path(str(a.substrate)).stem}")
    out.mkdir(parents=True, exist_ok=True)
    readout = registry.get("readout", "softmax_context")(clip=3.0)
    probe = registry.get("probe", "memory_span")(span=a.span)
    rep = {"data": Path(a.data).name, "train_chars": n_tr,
           "val_chars": len(split.ids) - n_tr, "chars": V,
           "substrate": str(a.substrate), "baselines": {}, "variants": []}
    say(f"text: {len(split.ids):,} characters ({n_tr:,} train), "
        f"{V} distinct; baselines...")
    base = rep["baselines"]
    base["unigram"] = registry.get("baseline", "unigram")().evaluate(
        split)["val_bpc"]
    for order in (3, 5):
        kn = registry.get("baseline", "kneser_ney")(order=order)
        base[f"kn{order}"] = kn.evaluate(split)["val_bpc"]
    empty = np.zeros((len(split.ids) - 1, 0), np.float32)
    base["context_only"] = readout.evaluate(empty, split, False)["val_bpc"]
    target = None
    for label, control, seed, matched in variants(a):
        bspec = run.brain_spec(a.substrate, overrides=a.spec_set,
                               control=control, control_seed=seed)
        spec, info = bspec["spec"], {}
        if a.match_activity and control == "none":
            target = match_activity(bspec, split.ids, V, a.device, None,
                                    a.match_chars)[1]
            info["activity_first_chars"] = target
        if matched:
            value, got = match_activity(bspec, split.ids, V, a.device,
                                        target, a.match_chars)
            param = spec["activity_match"]["param"]
            override(spec, param, value)
            info.update(param=param, value=value, target=target,
                        activity_first_chars=got)
            say(f"{label}: {param} = {value:.4f} gives activity "
                f"{got:.3f} (real {target:.3f})")
        t0 = time.time()
        X, dims = features(bspec, split, a, out, say)
        res = readout.evaluate(X, split)
        span = probe.run(X, split, dims)
        rep["memory_majority"] = span["majority"]
        rep["variants"].append({
            "name": label, "control": control, "seed": seed,
            "val_bpc": res["val_bpc"], "val_acc": res["val_acc"],
            "active_fraction": count_fraction(X, dims, spec,
                                              slice(0, n_tr - 1)),
            "memory_span": span["memory_span"],
            "seconds": round(time.time() - t0), **info})
        accs = " ".join(f"{v:.2f}" for v in span["memory_span"]["all"])
        say(f"{label}: {res['val_bpc']:.3f} BPC; memory span {accs}")
        text = json.dumps(rep, indent=1)
        (out / "report.json").write_text(text, encoding="utf-8")
    write_markdown(rep, out / "report.md")
    print((out / "report.md").read_text(encoding="utf-8"))


def write_markdown(rep, path):
    b = rep["baselines"]
    L = [f"# Connectome LM benchmark: {rep['substrate']}", "",
         f"{rep['data']}: {rep['train_chars']:,} training and "
         f"{rep['val_chars']:,} validation characters, "
         f"{rep['chars']} distinct.", "",
         "| Model | Validation BPC | Readout neurons active per character |",
         "|---|---|---|",
         f"| unigram | {b['unigram']:.3f} | |",
         f"| Kneser-Ney 3-gram | {b['kn3']:.3f} | |",
         f"| Kneser-Ney 5-gram | {b['kn5']:.3f} | |",
         f"| context table only (no brain) | {b['context_only']:.3f} | |"]
    for v in rep["variants"]:
        act = v.get("active_fraction")
        act = f"{act:.1%}" if act is not None else ""
        L.append(f"| context table + {v['name']} | {v['val_bpc']:.3f} "
                 f"| {act} |")
    k = len(rep["variants"][0]["memory_span"]["all"]) if rep["variants"] else 0
    majority = " | ".join(f"{m:.2f}" for m in rep.get("memory_majority", []))
    L += ["", "Memory span: accuracy of decoding the character k steps "
          "back (ridge, validation rows).", "",
          "| Features | " + " | ".join(f"k={i}" for i in range(k)) + " |",
          "|---|" + "---|" * k,
          f"| (most frequent character) | {majority} |"]
    for v in rep["variants"]:
        for g, accs in v["memory_span"].items():
            row = " | ".join(f"{x:.2f}" for x in accs)
            L.append(f"| {v['name']}: {g} | {row} |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def parser():
    ap = argparse.ArgumentParser(
        prog="uctf bench", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    add = ap.add_argument
    add("--data", required=True)
    add("--substrate", default="malecns-v1",
        help="spec: built-in name or JSON path")
    add("--controls", default="rewire-full,rewire-class",
        help="comma-separated names from the spec's controls, or none")
    add("--control-seeds", default="0")
    add("--train-chars", type=int, default=20000)
    add("--val-chars", type=int, default=5000)
    add("--offset", type=int, default=0)
    add("--vocab")
    add("--max-vocab", type=int, default=256)
    add("--span", type=int, default=8, help="memory span: tokens back")
    add("--spec-set", action="append", metavar="PATH=VALUE")
    add("--device", default="auto")
    add("--out", help="output folder (default ./bench-<substrate>)")
    add("--match-activity", action="store_true")
    add("--match-chars", type=int, default=500)
    return ap


def main(argv):
    bench(parser().parse_args(argv))
