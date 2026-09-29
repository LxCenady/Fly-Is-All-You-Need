"""Which brain features carry the sparse-input gain, and is KC hurt by standardisation?
CPU only. Caches: s160 seed -1 at offsets 0/300k/600k. L2 5e-3 (the usual pick).
Transforms: 'std' = lm_mech standardisation (sd + 1e-6); 'clip' = standardise then clip
|z| <= 3 with no re-standardisation."""
import json
import sys
import numpy as np
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm_mech as L
import m1_core as C

NTR, NVAL, L2, NB = 20000, 5000, 5e-3, 32768
_, chars = C.protocol_args(); vocab = len(chars)
text = L.CORPUS.read_text(encoding="utf-8")


def fit(Z, cx, y, seed=0):
    n = len(y)
    W = np.zeros((Z.shape[1], vocab), np.float32) if Z is not None else None
    cnt = np.bincount(y, minlength=vocab); prior = (cnt + 1) / (n + vocab)
    b = np.log(prior).astype(np.float32)
    Cm = np.zeros((NB, vocab)); np.add.at(Cm, (cx, y), 1.0)
    E = (np.log((Cm + 2 * prior) / (Cm.sum(1, keepdims=True) + 2)) - np.log(prior)).astype(np.float32)
    params = [p for p in (W, E, b) if p is not None]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    rng = np.random.RandomState(seed); step = 0
    for _ in range(15):
        order = rng.permutation(n)
        for lo in range(0, n, 1024):
            idx = order[lo:lo + 1024]
            s = b + E[cx[idx]] + (Z[idx] @ W if W is not None else 0)
            s = s - s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
            p[np.arange(len(idx)), y[idx]] -= 1; p /= len(idx)
            gE = np.zeros_like(E); np.add.at(gE, cx[idx], p)
            grads = ([Z[idx].T @ p + L2 * W] if W is not None else []) + [gE, p.sum(0)]
            step += 1
            for j, (P, G) in enumerate(zip(params, grads)):
                m[j] = 0.9 * m[j] + 0.1 * G; v[j] = 0.999 * v[j] + 0.001 * G * G
                P -= 0.01 * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
    return W, E, b


def logits(M, Z, cx):
    W, E, b = M
    return b + E[cx] + (Z @ W if W is not None else 0)


def transform(X, lo, hi, mode):
    mu, sd = X[lo:hi].mean(0), X[lo:hi].std(0) + 1e-6
    Z = (X - mu) / sd
    return np.clip(Z, -3, 3).astype(np.float32) if mode == "clip" else Z.astype(np.float32)


def evaluate(X, mode, y, cx):
    cut = int(NTR * 0.9)
    Zh = None if X is None else transform(X, 0, cut, mode)
    M = fit(None if Zh is None else Zh[:cut], cx[:cut], y[:cut])
    s = logits(M, None if Zh is None else Zh[cut:NTR], cx[cut:NTR])
    Ts = np.linspace(0.6, 2.0, 29)
    T = float(Ts[int(np.argmin([L.score(s, y[cut:NTR], t)[1] for t in Ts]))])
    Z = None if X is None else transform(X, 0, NTR, mode)
    M = fit(None if Z is None else Z[:NTR], cx[:NTR], y[:NTR])
    return L.score(logits(M, None if Z is None else Z[NTR:], cx[NTR:]), y[NTR:], T)


SETS = [("ctx_only", None, "std"), ("mbon_v", ["mbon_v"], "std"), ("central", ["central"], "std"),
        ("mbon_v+central", ["mbon_v", "central"], "std"), ("kc_std", ["kc"], "std"),
        ("kc_clip", ["kc"], "clip"), ("all_clip", ["kc", "mbon_v", "central"], "clip")]
res = {}
for off in (0, 300000, 600000):
    ids = np.asarray([chars.index(c) for c in text[off: off + NTR + NVAL + 1]], np.int64)
    y = ids[1:]; cx = L.ctx_ids(ids[:-1], 3, NB, vocab)
    z = np.load(__import__("paths").OUT / "night" / f"s160_o{off}_c-1.npz")
    for name, fs, mode in SETS:
        X = None if fs is None else np.concatenate([z[f] for f in fs], 1)
        acc, bpc = evaluate(X, mode, y, cx)
        res.setdefault(name, []).append((acc, bpc))
        print(off, name, round(acc, 4), round(bpc, 4), flush=True)
base = np.array([b for _, b in res["ctx_only"]])
print("\nmean delta BPC vs ctx_only (per-offset deltas):")
for name in res:
    d = np.array([b for _, b in res[name]]) - base
    print(f"  {name:16s} {d.mean():+.3f}  {np.round(d, 3).tolist()}")
import paths  # noqa: E402
json.dump(res, open(paths.out("night", "kc_ablate.json"), "w"), indent=1)
