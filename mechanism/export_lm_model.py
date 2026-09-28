"""Export a trained readout for the revised LM (CPU only; numpy port of lm_mech.fit).

Model: sparse PN codes (160 active, drive x1.5), frozen brain, all features
(KC counts, MBON voltage and spikes, 256 central neurons) + hashed order-3
context head, L2 1e-2 (fixed in advance, not chosen on validation), 20k/5k.
Features come from the overnight cache; the brain simulator is needed to
compute features for new text.
Usage: export_lm_model.py CACHE.npz OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lm_mech as L  # noqa: E402

FEATS = ["kc", "mbon_v", "mbon_spk", "central"]
NTR, NVAL, L2, EPOCHS, LR, NB = 20000, 5000, 1e-2, 15, 0.01, 32768


def fit_np(X, cx, y, vocab, seed=0):
    n = len(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)
    W = np.zeros((X.shape[1], vocab), np.float32)
    cnt = np.bincount(y, minlength=vocab)
    prior = (cnt + 1) / (n + vocab)
    b = np.log(prior).astype(np.float32)
    C = np.zeros((NB, vocab)); np.add.at(C, (cx, y), 1.0)
    E = (np.log((C + 2.0 * prior) / (C.sum(1, keepdims=True) + 2.0)) - np.log(prior)).astype(np.float32)
    params = [W, E, b]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    rng = np.random.RandomState(seed); step = 0
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for lo in range(0, n, 1024):
            idx = order[lo:lo + 1024]
            s = b + E[cx[idx]] + Z[idx] @ W
            s -= s.max(1, keepdims=True)
            p = np.exp(s); p /= p.sum(1, keepdims=True)
            p[np.arange(len(idx)), y[idx]] -= 1; p /= len(idx)
            gE = np.zeros_like(E); np.add.at(gE, cx[idx], p)
            grads = [Z[idx].T @ p + L2 * W, gE, p.sum(0)]
            step += 1
            for j, (P, G) in enumerate(zip(params, grads)):
                m[j] = 0.9 * m[j] + 0.1 * G; v[j] = 0.999 * v[j] + 0.001 * G * G
                P -= LR * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
    return {"mu": mu, "sd": sd, "W": W, "E": E, "b": b}


def main():
    cache, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    import m1_core as C
    _, chars = C.protocol_args()
    vocab = len(chars)
    text = L.CORPUS.read_text(encoding="utf-8")
    ids = np.asarray([chars.index(c) for c in text[: NTR + NVAL + 1]], np.int64)
    y = ids[1:]
    cx = L.ctx_ids(ids[:-1], 3, NB, vocab)
    z = np.load(cache)
    X = np.concatenate([z[f] for f in FEATS], 1)
    cut = int(NTR * 0.9)
    M = fit_np(X[:cut], cx[:cut], y[:cut], vocab)
    s = L.logits(M, X[cut:NTR], cx[cut:NTR])
    Ts = np.linspace(0.6, 2.0, 29)
    T = float(Ts[int(np.argmin([L.score(s, y[cut:NTR], t)[1] for t in Ts]))])
    M = fit_np(X[:NTR], cx[:NTR], y[:NTR], vocab)
    acc, bpc = L.score(L.logits(M, X[NTR:], cx[NTR:]), y[NTR:], T)
    tr_acc, _ = L.score(L.logits(M, X[:NTR], cx[:NTR]), y[:NTR], T)
    np.savez_compressed(out / "readout.npz", W=M["W"], E=M["E"].astype(np.float16), b=M["b"],
                        mu=M["mu"].astype(np.float32), sd=M["sd"].astype(np.float32))
    meta = {"val_acc": acc, "val_bpc": bpc, "train_acc": tr_acc, "temperature": T,
            "chars": "".join(chars), "features": FEATS,
            "feature_dims": {f: int(z[f].shape[1]) for f in FEATS},
            "brain": {"pn_active": 160, "drive_scale": 1.5, "code": "random", "code_seed": -1,
                      "mode": "frozen", "simulator": "flybrain LIF, MaleCNS v1.0, protocol of mechanism/m1_core.py"},
            "readout": {"l2": L2, "epochs": EPOCHS, "lr": LR, "context_order": 3, "buckets": NB,
                        "E_dtype": "float16", "fit": "numpy port of lm_mech.fit (CPU)"},
            "data": {"corpus": "TinyShakespeare", "train": [0, NTR], "val": [NTR, NTR + NVAL]},
            "source_cache": cache.name}
    (out / "model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("val_acc", "val_bpc", "train_acc", "temperature")}))


if __name__ == "__main__":
    main()
