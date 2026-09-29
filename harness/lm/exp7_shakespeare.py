"""Experiment 7: TinyShakespeare char-level pilot with the validated interface.

a256 (det+vp), K=2, sustained drive, DN trace at token end, ridge one-hot readout.
Reports accuracy and bits-per-char vs unigram/bigram, then samples 300 chars.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import progress
from flylm import DETECTORS, Encoder, accuracy, collect, fit_readout
from flybrain import FlyBrain
from flybrain.reservoir import Trace

RUN = "exp7"
CORPUS = Path(__file__).parent.parent / "corpus" / "tinyshakespeare.txt"
TRAIN_LEN, VAL_LEN, SEL_LEN = 20000, 5000, 2000
K_STEPS, ACTIVE, DRIVE, SUSTAIN = 2, 256, 1.0, 1.0
LAMS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def softmax_rows(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ngram_baselines(train_ids, val_ids, vocab):
    uni = (np.bincount(train_ids, minlength=vocab) + 1.0)
    uni /= uni.sum()
    big = np.ones((vocab, vocab))
    np.add.at(big, (train_ids[:-1], train_ids[1:]), 1.0)
    big /= big.sum(1, keepdims=True)
    p_uni = uni[val_ids[1:]]
    p_bi = big[val_ids[:-1], val_ids[1:]]
    pred_uni = np.full(len(val_ids) - 1, int(uni.argmax()))
    pred_bi = big.argmax(1)[val_ids[:-1]]
    return p_uni, p_bi, pred_uni, pred_bi


def bpc(p, y):
    q = p if p.ndim == 1 else p[np.arange(len(y)), y]
    return float(-np.mean(np.log2(np.clip(q, 1e-12, None))))


def fit_temperature(logits, y):
    best = (1e9, 1.0)
    for t in np.linspace(0.05, 3.0, 60):
        p = softmax_rows(logits / t)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if nll < best[0]:
            best = (nll, t)
    return best[1]


def generate(brain, enc, dns, mu, sd, W, Ym, temp, starts, itos, n=300, seed=0):
    rng = np.random.default_rng(seed)
    trace = Trace(brain, idx=dns)
    brain.reset()
    hist = []
    for c in starts:
        hist.append(int(c))
        for s in range(K_STEPS):
            scale = 1.0 if s == 0 else SUSTAIN
            fired = brain.step(inject=enc.inject(np.asarray(hist), len(hist) - 1, scale=scale))
            trace.observe(fired)
    out = []
    for _ in range(n):
        for s in range(K_STEPS):
            scale = 1.0 if s == 0 else SUSTAIN
            fired = brain.step(inject=enc.inject(np.asarray(hist), len(hist) - 1, scale=scale))
            feat = trace.observe(fired)
        p = softmax_rows((((feat - mu) / sd) @ W + Ym)[None] / temp)[0]
        c = int(rng.choice(len(p), p=p))
        hist.append(c)
        out.append(itos[c])
    return "".join(out)


def main():
    text = CORPUS.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    ids = np.array([stoi[c] for c in text], np.int64)
    print(f"corpus {len(ids)} chars, vocab {vocab}", flush=True)

    brain = FlyBrain(device="auto", seed=64)
    dns = brain.cells(["descending_neuron"])
    channels = np.concatenate([brain.cells(DETECTORS), brain.cells(["visual_projection"])])
    enc = Encoder(channels, vocab, active=ACTIVE, drive=DRIVE, window=1, seed=3)

    tr = ids[:TRAIN_LEN]
    va = ids[TRAIN_LEN:TRAIN_LEN + VAL_LEN]
    t0 = time.time()
    cache = Path(__file__).parent / "cache_exp7.npz"
    if cache.exists():
        d = np.load(cache)
        Xtr, ytr, Xva, yva = d["Xtr"], d["ytr"], d["Xva"], d["yva"]
        print(f"loaded cache ({cache.name})", flush=True)
    else:
        progress.emit(RUN, "shakespeare", phase="train", done=0, total=TRAIN_LEN - 1, rate=0.0)
        Xtr, ytr, _ = collect(brain, enc, tr, dns, k_steps=K_STEPS, sustain=SUSTAIN,
                              log={"run": RUN, "cfg": "shakespeare", "phase": "train"})
        Xva, yva, _ = collect(brain, enc, va, dns, k_steps=K_STEPS, sustain=SUSTAIN,
                              log={"run": RUN, "cfg": "shakespeare", "phase": "val"})
        np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)

    cut = len(Xtr) - SEL_LEN
    best = None
    for lam in LAMS:
        W, mu, sd, Ym = fit_readout(Xtr[:cut], ytr[:cut], vocab, lam=lam)
        a = accuracy(Xtr[cut:], ytr[cut:], W, mu, sd, Ym)
        if best is None or a > best[0]:
            best = (a, lam)
    W, mu, sd, Ym = fit_readout(Xtr, ytr, vocab, lam=best[1])
    acc_tr = accuracy(Xtr, ytr, W, mu, sd, Ym)
    acc_va = accuracy(Xva, yva, W, mu, sd, Ym)

    logt = ((Xtr[cut:] - mu) / sd) @ W + Ym
    temp = fit_temperature(logt, ytr[cut:])
    logv = ((Xva - mu) / sd) @ W + Ym
    pv = softmax_rows(logv / temp)
    bpc_val = bpc(pv, yva)
    p_uni, p_bi, pred_uni, pred_bi = ngram_baselines(tr, va, vocab)
    acc_uni = float((pred_uni == yva).mean())
    acc_bi = float((pred_bi == yva).mean())
    bpc_uni, bpc_bi = bpc(p_uni, yva), bpc(p_bi, yva)
    elapsed = time.time() - t0

    metrics = {"val_acc": acc_va, "train_acc": acc_tr, "bpc": bpc_val,
               "bpc_unigram": bpc_uni, "bpc_bigram": bpc_bi, "lam": best[1]}
    progress.emit(RUN, "shakespeare", status="done", metrics=metrics, time=elapsed)
    np.savez(Path(__file__).parent / "readout_exp7.npz", W=W, mu=mu, sd=sd, Ym=Ym,
             temp=temp, chars=np.array(chars))
    with open(Path(__file__).parent / "exp7_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"val acc {acc_va:.3f} (train {acc_tr:.3f}) | unigram {acc_uni:.3f} | bigram {acc_bi:.3f}", flush=True)
    print(f"BPC: fly {bpc_val:.3f} | unigram {bpc_uni:.3f} | bigram {bpc_bi:.3f} | temp {temp:.2f}", flush=True)

    text_out = generate(brain, enc, dns, mu, sd, W, Ym, temp, va[-120:], itos, n=300)
    print("generated sample:\n" + "-" * 60, flush=True)
    print(text_out, flush=True)
    print("-" * 60, flush=True)


if __name__ == "__main__":
    main()
