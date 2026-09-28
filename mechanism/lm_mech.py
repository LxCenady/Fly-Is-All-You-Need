"""Character LM on the post-fix simulator, with readouts chosen by the mechanism study.

Runs the same PN -> brain protocol as the mechanism experiments (m1_core:
canonical CSR, validated state handling), one continuous token stream, and
fits a softmax next-character readout.  Mechanism-informed options:

  --active / --scale   PN code size and drive (sparse KC regime at 160-192 x1.5)
  --code glom          bilateral glomerular symbol codes
  --mode biological    KC->MBON plasticity on (natural DAN activity, no pulse)
  features (comma list):
     kc        KC spike counts of the token (4064; the sparse cue code)
     mbon_v    MBON pre-reset voltage, mean over the token's 6 steps (97):
               the subthreshold channel that spike-gated readouts lose
     mbon_spk  MBON spike counts (97)
     central   256 MBON-target central neurons: trace + mean pre-reset voltage
     retr      MB retrieval current sum_k dW_km * c_k / 6 read by the current
               token's KCs before its own write (97); zero in frozen mode
  --lags 0,1,2         also stack the non-KC groups of the previous tokens
  --skip 3             hashed order-3 character context (learned embedding per
                       bucket; equivalent to the one-hot head of train_lm.py)
Train = corpus[0:N], val = corpus[N:N+M]; temperature is fitted on the last 10%
of the training rows only.  Reports accuracy, BPC, unigram/bigram baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
import mb_persistent_memory_probe as pmp  # noqa: E402

CORPUS = Path(r"D:\flybrain_lm_cuda\corpus\tinyshakespeare.txt")


def collect(st, args, ids, bio, feats, pulse=False):
    rows = {f: [] for f in feats}
    t0 = time.time()
    dan_spikes = 0.0
    for i, t in enumerate(ids):
        pmp._advance_token(st, args, int(t), allow_plastic=bio, pulse_dan=pulse,
                           record_profile=True, settle_steps=0)
        prof = st["_last_probe_profile"]
        if "kc" in feats:
            rows["kc"].append(st["_last_kc_counts"].astype(np.float32))
        if "mbon_v" in feats:
            rows["mbon_v"].append(prof["v_pre_mbon"].mean(0))
        if "mbon_spk" in feats:
            rows["mbon_spk"].append(st["_last_mbon_counts"].astype(np.float32))
        if "central" in feats:
            rows["central"].append(np.concatenate((st["trace_central"].get(),
                                                   prof["v_pre_central"].mean(0))).astype(np.float32))
        if "retr" in feats:
            rows["retr"].append(np.asarray(st["_last_w0_current"], np.float32))
        dan_spikes += float(st["_last_dan_counts"].sum())
        if (i + 1) % 1000 == 0:
            p = st["plastic"]
            print(f"  {i + 1}/{len(ids)} tokens {(i + 1) / (time.time() - t0):.1f} tok/s "
                  f"dan_spikes/token {dan_spikes / (i + 1):.2f} "
                  f"|mod|max {float(abs(p.modulation).max()):.4f}", flush=True)
    return {f: np.asarray(v, np.float32) for f, v in rows.items()}


def stack_lags(F, lags, feats):
    parts = []
    for f in feats:
        X = F[f]
        for lag in (lags if f != "kc" else [0]):
            if lag == 0:
                parts.append(X)
            else:
                Y = np.zeros_like(X); Y[lag:] = X[:-lag]
                parts.append(Y)
    return np.concatenate(parts, 1) if parts else None


def ctx_ids(ids, order, buckets, vocab):
    out = np.zeros(len(ids), np.int64)
    if order == 0:
        return out
    for i in range(len(ids)):
        h = 0
        for lag in range(order - 1, -1, -1):
            h = h * vocab + (int(ids[i - lag]) if i >= lag else 0)
        out[i] = h % buckets
    return out


def fit(X, cx, y, vocab, buckets, l2, epochs, lr, seed=0):
    import cupy as cp
    n = len(y)
    use_x = X is not None
    if use_x:
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = cp.asarray((X - mu) / sd, cp.float32)
        W = cp.zeros((X.shape[1], vocab), cp.float32)
    else:
        mu = sd = None
        W = None
    cnt = np.bincount(y, minlength=vocab)
    prior = (cnt + 1) / (n + vocab)
    b = cp.asarray(np.log(prior), cp.float32)
    # Context table initialised from smoothed counts on the fitting rows only:
    # E[c, y] = log P(y | c) - log P(y), Dirichlet prior alpha * P(y).  A rare
    # bucket otherwise receives too few Adam steps to move from zero.
    E0 = np.zeros((max(buckets, 1), vocab), np.float64)
    if buckets > 1:
        C = np.zeros((buckets, vocab)); np.add.at(C, (cx, y), 1.0)
        alpha = 2.0
        E0 = np.log((C + alpha * prior) / (C.sum(1, keepdims=True) + alpha)) - np.log(prior)
    E = cp.asarray(E0, cp.float32)
    Y = cp.asarray(y, cp.int32); CX = cp.asarray(cx, cp.int64)
    params = [p for p in (W, E, b) if p is not None]
    m = [cp.zeros_like(p) for p in params]; v = [cp.zeros_like(p) for p in params]
    rng = cp.random.RandomState(seed); step = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for lo in range(0, n, 1024):
            idx = order[lo:lo + 1024]
            s = b + E[CX[idx]]
            if use_x:
                s = s + Z[idx] @ W
            s -= s.max(1, keepdims=True)
            p = cp.exp(s); p /= p.sum(1, keepdims=True)
            p[cp.arange(len(idx)), Y[idx]] -= 1; p /= len(idx)
            gE = cp.zeros_like(E); cp.add.at(gE, CX[idx], p)
            grads = []
            if use_x:
                grads.append(Z[idx].T @ p + l2 * W)
            grads += [gE, p.sum(0)]           # no L2 on the count-initialised table
            step += 1
            for j, (P, G) in enumerate(zip(params, grads)):
                m[j] = 0.9 * m[j] + 0.1 * G; v[j] = 0.999 * v[j] + 0.001 * G * G
                P -= lr * (m[j] / (1 - 0.9 ** step)) / (cp.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
    return {"mu": mu, "sd": sd, "W": None if W is None else W.get(), "E": E.get(), "b": b.get()}


def logits(M, X, cx):
    s = M["b"] + M["E"][cx]
    if M["W"] is not None:
        s = s + ((X - M["mu"]) / M["sd"]) @ M["W"]
    return s


def score(s, y, T=1.0):
    s = s / T
    s = s - s.max(1, keepdims=True)
    lp = s - np.log(np.exp(s).sum(1, keepdims=True))
    return float((s.argmax(1) == y).mean()), float(-lp[np.arange(len(y)), y].mean() / np.log(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--active", type=int, default=512)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--code", default="random")
    ap.add_argument("--mode", default="frozen", choices=("frozen", "biological"))
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--weight-tau", type=float, default=16.0)
    ap.add_argument("--features", default="central")
    ap.add_argument("--lags", default="0")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--buckets", type=int, default=32768)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--cache", default="", help="npz path to save/load collected features")
    ap.add_argument("--offset", type=int, default=0, help="corpus start (text segment)")
    ap.add_argument("--code-seed", type=int, default=-1,
                    help="re-draw the random PN symbol codes with this seed (-1: default codes)")
    ap.add_argument("--pulse", action="store_true",
                    help="teacher: PAM pulse on every token (biological mode writes each token)")
    a = ap.parse_args()
    t0 = time.time()
    feats = [f for f in a.features.split(",") if f and f != "none"]
    lags = [int(x) for x in a.lags.split(",")]
    args, chars = C.protocol_args(eta=a.eta, weight_tau=a.weight_tau)
    text = CORPUS.read_text(encoding="utf-8")
    vocab = len(chars)
    ids = np.asarray([chars.index(c) for c in text[a.offset: a.offset + a.train + a.val + 1]], np.int64)
    F = {}
    if feats:
        if a.cache and Path(a.cache).exists():
            z = np.load(a.cache)
            F = {f: z[f] for f in feats}
        else:
            args.active, args.drive, args.probe_drive = a.active, 1.0 * a.scale, 2.5 * a.scale
            st = C.build(args)
            st["enc"].drive = args.drive
            if a.code == "glom":
                from m3_pnkc import set_glom_codes
                set_glom_codes(st, a.active, 11 if a.code_seed < 0 else a.code_seed)
            elif a.code_seed >= 0:
                enc = st["enc"]
                r = np.random.default_rng(a.code_seed)
                ch = np.asarray(enc.channels)
                enc.codes = [np.sort(ch[r.choice(len(ch), size=a.active, replace=False)])
                             for _ in range(len(enc.codes))]
                enc.position_codes = [enc.codes] * enc.window
                st["brain"]._lm_position_codes_gpu = None
            C.reset(st, args)
            F = collect(st, args, ids[:-1], a.mode == "biological", feats, a.pulse)
            if a.cache:
                np.savez(a.cache, **F)
    X = stack_lags(F, lags, feats) if feats else None
    y = ids[1:]
    cx = ctx_ids(ids[:-1], a.skip, a.buckets, vocab) if a.skip else np.zeros(len(y), np.int64)
    ntr = a.train
    cut = int(ntr * 0.9)
    nb = a.buckets if a.skip else 1
    Xtr = None if X is None else X[:ntr]
    M = fit(None if X is None else X[:cut], cx[:cut], y[:cut], vocab, nb, a.l2, a.epochs, a.lr)
    s_hold = logits(M, None if X is None else X[cut:ntr], cx[cut:ntr])
    Ts = np.linspace(0.6, 2.0, 29)
    T = float(Ts[int(np.argmin([score(s_hold, y[cut:ntr], t)[1] for t in Ts]))])
    M = fit(Xtr, cx[:ntr], y[:ntr], vocab, nb, a.l2, a.epochs, a.lr)
    acc, bpc = score(logits(M, None if X is None else X[ntr:], cx[ntr:]), y[ntr:], T)
    tr_acc, _ = score(logits(M, Xtr, cx[:ntr]), y[:ntr], T)
    uni = np.bincount(y[:ntr], minlength=vocab) + 1.0; uni /= uni.sum()
    big = np.ones((vocab, vocab)); np.add.at(big, (ids[:ntr], y[:ntr]), 1); big /= big.sum(1, keepdims=True)
    yv, pv = y[ntr:], ids[ntr:-1]
    res = {"val_acc": acc, "val_bpc": bpc, "train_acc": tr_acc, "temperature": T,
           "unigram_bpc": float(-np.log2(uni[yv]).mean()), "bigram_bpc": float(-np.log2(big[pv, yv]).mean()),
           "bigram_acc": float((big[pv].argmax(1) == yv).mean()),
           "n_features": 0 if X is None else int(X.shape[1]), "seconds": round(time.time() - t0, 1),
           "config": vars(a)}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    main()
