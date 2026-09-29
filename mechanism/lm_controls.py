"""LM controls (CPU only).

A1  Random-feature control: is the sparse brain's gain over the n-gram specific to the
    connectome, or does any random expansion of the recent characters give it?
      onehot4   one-hot of the current and 3 previous characters (260 dims)
      randexp   random KC-like expansion of the recent history: 8 previous characters,
                one-hot, weighted 0.5**lag (520 inputs) -> 4064 units, each summing 7
                random inputs; the top 10% units per character are set to 1
    Compared with the brain's KC counts, same readout (z clipped at +-3, L2 5e-3, context
    head, temperature on the last 10% of train), 3 segments, 20k/5k.
A2  Learning curve on the 100k cache (segment 0): train on the first N characters,
    validate on the same 20k characters (100k-120k) for N = 5k..100k.
Usage: lm_controls.py OUT.json
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\苍蝇。\mechanism")
import lm_mech as L  # noqa: E402
import m1_core as C  # noqa: E402

NB, L2 = 32768, 5e-3
_, chars = C.protocol_args(); V = len(chars)
TEXT = L.CORPUS.read_text(encoding="utf-8")
NIGHT = r"E:\mechanism_20260928\night"


def fit(Z, cx, y, seed=0):
    n = len(y)
    W = np.zeros((Z.shape[1], V), np.float32) if Z is not None else None
    cnt = np.bincount(y, minlength=V); prior = (cnt + 1) / (n + V)
    b = np.log(prior).astype(np.float32)
    Cm = np.zeros((NB, V)); np.add.at(Cm, (cx, y), 1.0)
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


def zclip(X, lo, hi):
    mu, sd = X[lo:hi].mean(0), X[lo:hi].std(0) + 1e-6
    Z = X.astype(np.float32, copy=True); Z -= mu.astype(np.float32); Z /= sd.astype(np.float32)
    np.clip(Z, -3, 3, out=Z)
    return Z


def evaluate(X, y, cx, ntr, v0, v1):
    """Temperature on the last 10% of the first ntr rows, then refit on ntr, score [v0, v1)."""
    cut = int(ntr * 0.9)
    Zh = None if X is None else zclip(X, 0, cut)
    M = fit(None if Zh is None else Zh[:cut], cx[:cut], y[:cut])
    s = logits(M, None if Zh is None else Zh[cut:ntr], cx[cut:ntr])
    Ts = np.linspace(0.6, 2.0, 29)
    T = float(Ts[int(np.argmin([L.score(s, y[cut:ntr], t)[1] for t in Ts]))])
    del Zh
    Z = None if X is None else zclip(X, 0, ntr)
    M = fit(None if Z is None else Z[:ntr], cx[:ntr], y[:ntr])
    acc, bpc = L.score(logits(M, None if Z is None else Z[v0:v1], cx[v0:v1]), y[v0:v1], T)
    return {"acc": acc, "bpc": bpc, "T": T}


def onehot_lags(ids, lags, decay=1.0):
    X = np.zeros((len(ids), V * lags), np.float32)
    for lag in range(lags):
        src = np.concatenate([np.full(lag, -1), ids[:len(ids) - lag]]) if lag else ids
        ok = src >= 0
        X[np.flatnonzero(ok), lag * V + src[ok]] = decay ** lag
    return X


_RNG = np.random.default_rng(7)
_PROJ = np.stack([_RNG.choice(V * 8, size=7, replace=False) for _ in range(4064)])   # 4064 x 7


def randexp(ids):
    H = onehot_lags(ids, 8, 0.5)                       # n x 520
    out = np.zeros((len(ids), len(_PROJ)), np.float32)
    k = int(0.10 * len(_PROJ))
    for lo in range(0, len(ids), 2000):                 # chunked: n x 4064 x 7 is too large
        A = H[lo:lo + 2000][:, _PROJ].sum(2)
        thr = np.partition(A, -k, axis=1)[:, -k][:, None]
        out[lo:lo + 2000] = A >= np.maximum(thr, 1e-9)
    return out


def data(off, n):
    ids = np.asarray([chars.index(c) for c in TEXT[off: off + n + 1]], np.int64)
    return ids[:-1], ids[1:], L.ctx_ids(ids[:-1], 3, NB, V)


def main(out):
    res = {"A1": {}, "A2": {}}
    save = lambda: open(out, "w").write(json.dumps(res, indent=1))
    t0 = time.time()
    for off in (0, 300000, 600000):                                   # A1
        x_ids, y, cx = data(off, 25000)
        z = np.load(rf"{NIGHT}\s160_o{off}_c-1.npz")
        for name, X in (("ctx_only", None), ("brain_kc", z["kc"]), ("onehot4", onehot_lags(x_ids, 4)),
                        ("randexp", randexp(x_ids))):
            r = evaluate(X, y, cx, 20000, 20000, 25000)
            res["A1"].setdefault(name, []).append({"offset": off, **r}); save()
            print(f"A1 {off} {name} {r['acc']:.4f} {r['bpc']:.4f}  {time.time() - t0:.0f}s", flush=True)
    x_ids, y, cx = data(0, 120000)                                    # A2
    for N in (5000, 10000, 20000, 50000, 100000):
        for name in ("ctx_only", "brain_kc", "randexp"):
            if name == "ctx_only":
                X = None
            elif name == "brain_kc":
                X = np.load(rf"{NIGHT}\s160_100k.npz")["kc"]
            else:
                X = randexp(x_ids)
            r = evaluate(X, y, cx, N, 100000, 120000)
            del X
            res["A2"].setdefault(name, []).append({"train": N, **r}); save()
            print(f"A2 N={N} {name} {r['acc']:.4f} {r['bpc']:.4f}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
