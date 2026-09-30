"""Confidence intervals for the language-model results.

    python repro/lm_stats.py [--bench-fly DIR] [--bench-worm DIR] [--out repro/LM_STATS.md]

1. Held-out check (LM Section VI-B; results/lm/night/select_*.json): change in validation BPC
   when the brain is added to the context table, 9 cases per input code (3 text segments x 3
   PN-code draws). Reported: mean, SD, 95% t-interval over the 9 cases, a cluster bootstrap
   over the 3 segments, and an exact two-sided sign test.
2. Per-character comparisons on one split (segment 0, 20k/5k), from the simulated features that
   `python -m uctf bench` caches: context table only, context table + connectome (UCTF
   readout), Kneser-Ney 5-gram. For each pair of models, the BPC difference with a 95% block
   bootstrap interval (blocks of 250 characters, 10,000 resamples). The per-character losses are
   saved to results/lm/stats/ so the intervals can be recomputed without a GPU.
3. Memory span (k = 2, 3) with block-bootstrap intervals over validation rows.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uctf"))
B, BLOCK = 10_000, 250


def t_interval(x):
    from scipy import stats
    x = np.asarray(x, float)
    m, s = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, len(x) - 1) * s / math.sqrt(len(x))
    return m, s, m - h, m + h


def sign_test(x):
    from scipy import stats
    x = np.asarray(x)
    k, n = int((x < 0).sum()), int((x != 0).sum())
    return k, n, float(stats.binomtest(k, n, 0.5).pvalue)


def cluster_boot(values, clusters, rng):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    ids = np.unique(clusters)
    means = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids))
        means.append(np.concatenate([values[clusters == c] for c in pick]).mean())
    return np.percentile(means, [2.5, 97.5])


def block_boot_diff(a, b, rng):
    """Mean of a - b (bits per character) with a 95% block-bootstrap interval."""
    d = np.asarray(a) - np.asarray(b)
    nb = len(d) // BLOCK
    blocks = d[:nb * BLOCK].reshape(nb, BLOCK).mean(1)
    boots = blocks[rng.integers(0, nb, (B, nb))].mean(1)
    return float(d.mean()), *np.percentile(boots, [2.5, 97.5]).tolist()


def heldout(lines, res):
    rng = np.random.default_rng(0)
    lines += ["## 1. Held-out check: brain + context table vs context table alone", "",
              "Change in validation BPC (negative = the brain helps); 9 cases = 3 text segments x 3 PN-code draws.", "",
              "| Input | mean | SD | 95% t-interval | 95% cluster bootstrap (segments) | better | sign test p |",
              "|---|---|---|---|---|---|---|"]
    for tag, lab in (("s160", "sparse 160"), ("s192", "sparse 192"), ("d512", "dense 512")):
        rs = json.loads((ROOT / "results/lm/night" / f"select_{tag}.json").read_text(encoding="utf-8"))
        d = [r["brain+context"]["val_bpc"] - r["context_only"]["val_bpc"] for r in rs]
        m, s, lo, hi = t_interval(d)
        clo, chi = cluster_boot(d, [r["offset"] for r in rs], rng)
        k, n, p = sign_test(d)
        res[f"heldout_{tag}"] = {"mean": m, "sd": s, "t95": [lo, hi], "cluster95": [clo, chi], "better": k, "n": n, "p_sign": p}
        lines.append(f"| {lab} | {m:+.3f} | {s:.3f} | [{lo:+.3f}, {hi:+.3f}] | [{clo:+.3f}, {chi:+.3f}] | {k}/{n} | {p:.4f} |")
    lines += ["", "With only 3 segments, the cluster bootstrap is coarse; the t-interval treats the 9 cases as independent.", ""]


def per_char(bench_dir: Path, label: str, lines, res, vocab):
    """Per-character losses for context only / + connectome / KN-5 on the cached bench split."""
    from uctf.baselines import KneserNey
    from uctf.readout import ReadoutConfig, context_ids, fit_calibrated, readout_scores
    from uctf.text import load_split
    rep = json.loads((bench_dir / "report.json").read_text(encoding="utf-8"))
    sp = load_split(ROOT / "playground/gpf/data/tinyshakespeare.txt", rep["train_chars"], rep["val_chars"], 0, vocab)
    ids, n_tr, V = sp.ids, sp.n_tr, len(sp.chars)
    y, ntr = ids[1:], n_tr - 1
    cfg = ReadoutConfig()
    cx = context_ids(ids, 3, V, cfg.buckets)
    # the real connectome is the first variant uctf bench simulates, so its cache is the oldest
    feats = sorted(bench_dir.glob("features_*.npy"), key=lambda p: p.stat().st_mtime)[0]
    X = np.load(feats)

    def losses(Xf, clip):
        M, T, _, _ = fit_calibrated(Xf, cx, y, ntr, V, cfg, clip)
        s = readout_scores(M, Xf[ntr:], cx[ntr:]) / T
        s = s - s.max(1, keepdims=True)
        lp = s - np.log(np.exp(s).sum(1, keepdims=True))
        return -lp[np.arange(len(y) - ntr), y[ntr:]] / math.log(2)

    ctx = losses(X[:, :0], None)
    brain = losses(X, 3.0)
    kn = KneserNey(5, V).fit([int(t) for t in ids[:n_tr]])
    hist = [int(t) for t in ids[max(0, n_tr - 5):n_tr]]; knl = []
    for t in ids[n_tr:]:
        knl.append(-kn.logprobs(hist)[t] / math.log(2)); hist.append(int(t))
    knl = np.asarray(knl)
    out = ROOT / "results/lm/stats"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"perchar_{label}.npz", context_only=ctx.astype(np.float32),
                        connectome=brain.astype(np.float32), kn5=knl.astype(np.float32), val_ids=ids[n_tr:].astype(np.int16))
    rng = np.random.default_rng(0)
    rows = [("connectome - context only", brain, ctx), ("KN-5 - connectome", knl, brain), ("KN-5 - context only", knl, ctx)]
    lines += [f"### {label}", "", f"Segment 0, {n_tr:,} training / {len(ids) - n_tr:,} validation characters; features `{feats.name}`.", "",
              "| Difference | mean BPC | 95% block bootstrap |", "|---|---|---|"]
    res[f"perchar_{label}"] = {"bpc": {"context_only": float(ctx.mean()), "connectome": float(brain.mean()), "kn5": float(knl.mean())}}
    for name, a, b in rows:
        m, lo, hi = block_boot_diff(a, b, rng)
        res[f"perchar_{label}"][name] = [m, lo, hi]
        lines.append(f"| {name} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines.append(f"\nBPC: context only {ctx.mean():.3f}, + connectome {brain.mean():.3f}, KN-5 {knl.mean():.3f}.\n")
    # memory span with block bootstrap over validation rows (all features, ridge as in uctf bench)
    acc = ridge_span_rows(X, ids, n_tr, (2, 3))
    for k, hits in acc.items():
        nb = len(hits) // BLOCK
        bl = hits[:nb * BLOCK].reshape(nb, BLOCK).mean(1)
        boots = bl[rng.integers(0, nb, (B, nb))].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[f"perchar_{label}"][f"memory_k{k}"] = [float(hits.mean()), float(lo), float(hi)]
        lines.append(f"Memory span, character {k} back decoded: {hits.mean():.3f} [95% CI {lo:.3f}, {hi:.3f}]")
    lines.append("")


def ridge_span_rows(X, ids, n_tr, ks, lam=10.0, clip=3.0):
    """Per-validation-row hit (0/1) of the ridge decoder for the character k back (as uctf.bench)."""
    from scipy.linalg import cho_factor, cho_solve
    x_ids = ids[:-1]; ntr = n_tr - 1
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Z = np.clip((X - mu) / sd, -clip, clip).astype(np.float32)
    Ztr = np.c_[Z[:ntr], np.ones(ntr, np.float32)]; Zva = np.c_[Z[ntr:], np.ones(len(Z) - ntr, np.float32)]
    A = cho_factor(Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1], dtype=np.float32))
    V = int(ids.max()) + 1
    out = {}
    for k in ks:
        tgt = np.concatenate([np.zeros(k, np.int64), x_ids[:len(x_ids) - k]])
        Y = np.eye(V, dtype=np.float32)[tgt]
        Wt = cho_solve(A, Ztr.T @ Y[:ntr])
        out[k] = ((Zva @ Wt).argmax(1) == tgt[ntr:]).astype(float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-fly")
    ap.add_argument("--bench-worm")
    ap.add_argument("--out", default=str(ROOT / "repro" / "LM_STATS.md"))
    a = ap.parse_args()
    res, lines = {}, ["# Language model: confidence intervals", "", "Generated by repro/lm_stats.py.", ""]
    heldout(lines, res)
    vocab = ROOT / "playground/gpf/data/vocab.json"
    if a.bench_fly or a.bench_worm:
        lines += ["## 2-3. Per-character comparisons and memory span (block bootstrap)", ""]
    if a.bench_fly:
        per_char(Path(a.bench_fly), "fly_malecns", lines, res, vocab)
    if a.bench_worm:
        per_char(Path(a.bench_worm), "worm_cook2019", lines, res, vocab)
    text = "\n".join(lines) + "\n"
    Path(a.out).write_text(text, encoding="utf-8")
    (ROOT / "results/lm/stats").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/lm/stats/lm_stats.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
