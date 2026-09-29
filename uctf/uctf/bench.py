"""The standard comparison for a connectome language model.

    python -m uctf bench --substrate malecns-v1 --data text.txt [--controls rewire-full,rewire-class]

On one train/validation split of the text it reports, in bits per character (lower is better):
    unigram, Kneser-Ney 3/5-gram               how far simple statistics get
    context table only                          the readout's hashed 3-character table, no brain
    connectome                                  the same readout plus the brain's features
    each control                                the same, on a degree-preserving rewired copy
and a memory span: how well a ridge decoder recovers the character k steps back from each
readout group (and from all features), for k = 0..--span.

This is the protocol of paper_lm/ (Sections VI-VII).  Questions it answers for a new connectome:
does the brain add anything over the context table, does it come near a smoothed n-gram, how many
characters does it remember, and does any of that depend on the real wiring?
Results: <out>/report.json and <out>/report.md; simulated features are cached in <out>/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys

from . import run
from .baselines import kn_bpc, unigram_bpc
from .readout import ReadoutConfig, context_ids, fit_calibrated
from .text import load_split


def ridge_span(X, ids, n_tr, span, lam=10.0, clip=3.0):
    """Decoding accuracy of the character k back (k = 0..span) on validation rows, from
    standardised, clipped features (train statistics), ridge with an intercept (as lm_explain)."""
    x_ids = ids[:-1]
    ntr = n_tr - 1
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Z = np.clip((X - mu) / sd, -clip, clip).astype(np.float32)
    Ztr = np.c_[Z[:ntr], np.ones(ntr, np.float32)]
    Zva = np.c_[Z[ntr:], np.ones(len(Z) - ntr, np.float32)]
    from scipy.linalg import cho_factor, cho_solve
    A = cho_factor(Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1], dtype=np.float32))    # factor once for all k
    V = int(ids.max()) + 1
    acc, majority = [], []
    for k in range(span + 1):
        tgt = np.concatenate([np.zeros(k, np.int64), x_ids[:len(x_ids) - k]]) if k else x_ids.copy()
        Y = np.eye(V, dtype=np.float32)[tgt]
        Wt = cho_solve(A, Ztr.T @ Y[:ntr])
        acc.append(float(((Zva @ Wt).argmax(1) == tgt[ntr:]).mean()))
        majority.append(float(np.bincount(tgt[ntr:], minlength=V).max() / len(tgt[ntr:])))
    return acc, majority


def active_fraction(X, dims, rows):
    """Mean fraction of neurons in the spike-count readout groups that fire during a character."""
    lo, parts = 0, []
    for nm, d in dims.items():
        if nm in COUNT_GROUPS:
            parts.append((X[rows, lo:lo + d] > 0).mean(1))
        lo += d
    return float(np.mean(np.concatenate(parts))) if parts else None


COUNT_GROUPS: set = set()


def match_gain(bspec, ids, V, device, target, n=500, iters=14):
    """Gain at which the (rewired) substrate's readout neurons are as active as the real one's,
    found by bisection on the first n training characters only (no prediction results used)."""
    rt = run.make_substrate(V, bspec, device=device, log=lambda m: None)
    xp = rt.xp
    ro = np.unique(np.concatenate([xp.asnumpy(i) if xp is not np else i for _, i, f, _ in rt.readouts
                                   if f == "counts"]))

    def frac(g):
        rt.gain = np.float32(g); rt.reset(); rt.record_activity = True
        out = []
        for t in ids[:n]:
            rt.step_token(int(t))
            out.append(float(np.isin(ro, rt.last_activity["ids"]).mean()))
        return float(np.mean(out))

    g0 = float(bspec["spec"]["neuron"].get("gain", 1.5))
    if target is None:                                   # just measure at the spec's gain
        return g0, frac(g0)
    lo, hi = np.log(g0 / 8), np.log(g0 * 8)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if frac(np.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    g = float(np.exp((lo + hi) / 2))
    return g, frac(g)


def main(argv):
    ap = argparse.ArgumentParser(prog="uctf bench", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--substrate", default="malecns-v1", help="connectome spec (built-in name or JSON)")
    ap.add_argument("--controls", default="rewire-full,rewire-class",
                    help="comma-separated: rewire-full, rewire-class, or none")
    ap.add_argument("--control-seeds", default="0", help="comma-separated rewiring seeds")
    ap.add_argument("--train-chars", type=int, default=20000)
    ap.add_argument("--val-chars", type=int, default=5000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--vocab")
    ap.add_argument("--max-vocab", type=int, default=256)
    ap.add_argument("--span", type=int, default=8, help="memory span: characters back to decode (8)")
    ap.add_argument("--spec-set", action="append", metavar="KEY=VALUE")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", help="output folder (default ./bench-<substrate>)")
    ap.add_argument("--match-activity", action="store_true",
                    help="also run each control at the gain where its readout neurons are as active as the "
                         "real connectome's (set on the first --match-chars training characters)")
    ap.add_argument("--match-chars", type=int, default=500)
    a = ap.parse_args(argv)
    cfg = ReadoutConfig()                 # the readout of gpf train brain: clipped features, L2 0.01
    clip = 3.0
    try:
        sp = load_split(a.data, a.train_chars, a.val_chars, a.offset, a.vocab, a.max_vocab)
    except ValueError as e:
        sys.exit(str(e))
    chars, ids, n_tr = sp.chars, sp.ids, sp.n_tr
    V, y, ntr = len(chars), ids[1:], n_tr - 1
    out = Path(a.out or f"bench-{Path(str(a.substrate)).stem}"); out.mkdir(parents=True, exist_ok=True)
    cx = context_ids(ids, cfg.context_order, V, cfg.buckets)
    rep = {"data": Path(a.data).name, "train_chars": n_tr, "val_chars": len(ids) - n_tr, "chars": len(chars),
           "substrate": str(a.substrate), "baselines": {}, "variants": []}
    say = run.log_flush
    say(f"text: {len(ids):,} characters ({n_tr:,} train), {V} distinct; baselines...")
    rep["baselines"]["unigram"] = unigram_bpc(ids, n_tr, V)
    for order in (3, 5):
        rep["baselines"][f"kn{order}"] = kn_bpc(order, ids, n_tr, V)[0]
    rep["baselines"]["context_only"] = fit_calibrated(np.zeros((len(y), 0), np.float32), cx, y, ntr, V, cfg, None)[3]
    variants = [("connectome", "none", 0, False)]
    for c in [c.strip() for c in a.controls.split(",") if c.strip() and c.strip() != "none"]:
        for s in [int(v) for v in a.control_seeds.split(",")]:
            variants.append((f"{c} (seed {s})", c, s, False))
            if a.match_activity:
                variants.append((f"{c} (seed {s}, activity-matched)", c, s, True))
    target = None
    for label, control, seed, matched in variants:
        bspec = run.brain_spec(a.substrate, overrides=a.spec_set, control=control, control_seed=seed)
        COUNT_GROUPS.update(r["name"] for r in bspec.get("spec", {}).get("readout", []) if r["feature"] == "counts")
        info = {}
        if a.match_activity and control == "none":
            target = match_gain(bspec, ids, V, a.device, None, a.match_chars)[1]
            info["activity_first_chars"] = target
        if matched:
            g, got = match_gain(bspec, ids, V, a.device, target, a.match_chars)
            bspec["spec"]["neuron"]["gain"] = g
            info.update(gain=g, activity_first_chars=got, target=target)
            say(f"{label}: gain {g:.4f} gives activity {got:.3f} (real {target:.3f})")
        key = run.feature_key(ids, V, bspec)[:16]
        cache = out / f"features_{key}.npy"
        t0 = time.time()
        dims_file = out / f"features_{key}.json"
        if cache.exists() and dims_file.exists():
            X = np.load(cache); dims = json.loads(dims_file.read_text(encoding="utf-8"))
            say(f"{label}: reusing {cache.name}")
        else:
            X, dims = run.simulate(ids, V, bspec, a.device, with_dims=True)
            np.save(cache, X); dims_file.write_text(json.dumps(dims), encoding="utf-8")
        _, _, acc, bpc = fit_calibrated(X, cx, y, ntr, V, cfg, clip)
        span, lo = {}, 0
        for nm, d in dims.items():                        # readout groups, in feature order
            if d:
                span[nm] = ridge_span(X[:, lo:lo + d], ids, n_tr, a.span)[0]
            lo += d
        span_all, majority = ridge_span(X, ids, n_tr, a.span)
        span["all"] = span_all
        rep["memory_majority"] = majority
        rep["variants"].append({"name": label, "control": control, "seed": seed, "val_bpc": bpc, "val_acc": acc,
                                "active_fraction": active_fraction(X, dims, slice(0, ntr)),
                                "memory_span": span, "seconds": round(time.time() - t0), **info})
        say(f"{label}: {bpc:.3f} BPC; memory span (all features) "
            + " ".join(f"{v:.2f}" for v in span_all))
        (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    write_markdown(rep, out / "report.md")
    print((out / "report.md").read_text(encoding="utf-8"))


def write_markdown(rep, path):
    b = rep["baselines"]
    L = [f"# Connectome LM benchmark: {rep['substrate']}", "",
         f"{rep['data']}: {rep['train_chars']:,} training and {rep['val_chars']:,} validation characters, "
         f"{rep['chars']} distinct.", "", "| Model | Validation BPC | Readout neurons active per character |",
         "|---|---|---|", f"| unigram | {b['unigram']:.3f} | |", f"| Kneser-Ney 3-gram | {b['kn3']:.3f} | |",
         f"| Kneser-Ney 5-gram | {b['kn5']:.3f} | |", f"| context table only (no brain) | {b['context_only']:.3f} | |"]
    for v in rep["variants"]:
        act = v.get("active_fraction")
        L.append(f"| context table + {v['name']} | {v['val_bpc']:.3f} | "
                 + (f"{act:.1%}" if act is not None else "") + " |")
    k = len(rep["variants"][0]["memory_span"]["all"]) if rep["variants"] else 0
    L += ["", "Memory span: accuracy of decoding the character k steps back (ridge, validation rows).", "",
          "| Features | " + " | ".join(f"k={i}" for i in range(k)) + " |", "|---|" + "---|" * k,
          "| (most frequent character) | " + " | ".join(f"{m:.2f}" for m in rep.get("memory_majority", [])) + " |"]
    for v in rep["variants"]:
        for g, accs in v["memory_span"].items():
            L.append(f"| {v['name']}: {g} | " + " | ".join(f"{x:.2f}" for x in accs) + " |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
