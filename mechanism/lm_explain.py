"""What does the sparse brain add?  Stronger baselines and a memory profile (CPU).

KN   Interpolated modified Kneser-Ney character n-gram (orders 2..7).  For stacking, training
     rows get KN log-probabilities from 5-fold cross-fitting (counts from the other folds);
     validation rows use counts from all training rows.
STACK  Readout logits = logP_KN(order n) + b + Z W   (Z: clipped-z features, L2 5e-3,
     temperature on the last 10% of train).  Compared: KN alone, KN + brain KC, KN + onehot4,
     KN + onehot4 + brain KC.
MEM  Memory profile: ridge decoders from brain features at t to the character at t-k,
     k = 0..12; validation accuracy vs the majority-class rate.
Usage: lm_explain.py OUT.json CACHE_TAG [CACHE_TAG ...]   (tags like s160_o0_c-1 in night/,
       or conn/class_o0 etc.; the offset is parsed from the tag)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm_controls as K  # noqa: E402

import paths  # noqa: E402
E = paths.OUT
NTR, NVAL = 20000, 5000


# ---------------- Kneser-Ney ----------------
class KN:
    def __init__(self, order, V):
        self.n, self.V = order, V

    def fit(self, seq):
        n = self.n
        self.c = [defaultdict(lambda: defaultdict(int)) for _ in range(n + 1)]   # c[k][ctx][w]
        for i in range(len(seq)):
            if seq[i] < 0:                                   # -1 = segment break, never a target
                continue
            for k in range(1, n + 1):
                if i - k + 1 < 0 or (k > 1 and seq[i - k + 1] < 0):
                    break                                    # contexts never span a break
                ctx = tuple(seq[i - k + 1:i]); self.c[k][ctx][seq[i]] += 1
        # continuation counts for lower orders: number of distinct left extensions
        self.cc = [defaultdict(lambda: defaultdict(int)) for _ in range(n + 1)]
        for k in range(2, n + 1):
            for ctx, d in self.c[k].items():
                for w in d:
                    self.cc[k - 1][ctx[1:]][w] += 1
        self.D = []
        for k in range(n + 1):
            src = self.c[k] if k == n else self.cc[k]
            nr = np.zeros(5)
            for d in src.values():
                for v in d.values():
                    if v <= 4:
                        nr[v] += 1
            Y = nr[1] / max(nr[1] + 2 * nr[2], 1)
            self.D.append([0.0] + [max(min(r - (r + 1) * Y * nr[r + 1] / max(nr[r], 1), r), 0.0)
                                   for r in (1, 2, 3)])
        return self

    def _p(self, k, ctx, top):
        if k == 0:
            return np.full(self.V, 1.0 / self.V)
        src = self.c[k] if top else self.cc[k]
        d = src.get(ctx[len(ctx) - (k - 1):] if k > 1 else ())
        lower = self._p(k - 1, ctx, False)
        if not d:
            return lower
        tot = sum(d.values()); D = self.D[k]
        p = np.zeros(self.V); nr = [0, 0, 0, 0]
        for w, v in d.items():
            dv = D[min(v, 3)]; p[w] = max(v - dv, 0) / tot; nr[min(v, 3)] += 1
        gamma = (D[1] * nr[1] + D[2] * nr[2] + D[3] * nr[3]) / tot
        return p + gamma * lower

    def logprob_rows(self, hist, rows):
        out = np.zeros((len(rows), self.V), np.float32)
        for j, i in enumerate(rows):
            ctx = tuple(hist[max(0, i - self.n + 2):i + 1])     # last n-1 chars up to i (predict i+1)
            k = min(self.n, len(ctx) + 1)
            out[j] = np.log(np.maximum(self._p(k, ctx, True), 1e-12))
        return out


def kn_offsets(ids, order):
    """ids: chars 0..NTR+NVAL; row t predicts ids[t+1] from ids[..t]."""
    y = ids[1:]
    off = np.zeros((NTR + NVAL, K.V), np.float32)
    folds = np.array_split(np.arange(NTR), 5)
    for f in folds:                                          # cross-fitted training rows
        keep = np.setdiff1d(np.arange(NTR), f)
        m = KN(order, K.V)
        # fit on the other folds as separate segments (no cross-boundary contexts)
        m.c = None
        segs = np.split(keep, np.flatnonzero(np.diff(keep) > 1) + 1)
        seq = []
        for s in segs:
            seq += list(ids[s[0]:s[-1] + 2]) + [-1] * order     # -1 pads break contexts
        m.fit([x for x in seq])
        off[f] = m.logprob_rows(ids, f)
    m = KN(order, K.V).fit(list(ids[:NTR + 1]))
    off[NTR:] = m.logprob_rows(ids, np.arange(NTR, NTR + NVAL))
    return off, y


def stack_eval(off, Z, y, l2=5e-3):
    """logits = off + b + Z W; temperature on last 10% of train; returns (acc, bpc)."""
    def fit(rows):
        n = len(rows)
        W = np.zeros((Z.shape[1], K.V), np.float32) if Z is not None else None
        b = np.zeros(K.V, np.float32)
        params = [p for p in (W, b) if p is not None]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        rng = np.random.RandomState(0); step = 0
        for _ in range(15):
            order = rng.permutation(n)
            for lo in range(0, n, 1024):
                idx = rows[order[lo:lo + 1024]]
                s = off[idx] + b + (Z[idx] @ W if W is not None else 0)
                s = s - s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
                p[np.arange(len(idx)), y[idx]] -= 1; p /= len(idx)
                grads = ([Z[idx].T @ p + l2 * W] if W is not None else []) + [p.sum(0)]
                step += 1
                for j, (P, G) in enumerate(zip(params, grads)):
                    m[j] = 0.9 * m[j] + 0.1 * G; v[j] = 0.999 * v[j] + 0.001 * G * G
                    P -= 0.01 * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
        return W, b

    def lg(M, rows):
        W, b = M
        return off[rows] + b + (Z[rows] @ W if W is not None else 0)

    cut = int(NTR * 0.9)
    M = fit(np.arange(cut))
    s = lg(M, np.arange(cut, NTR)); Ts = np.linspace(0.6, 2.0, 29)
    T = float(Ts[int(np.argmin([K.L.score(s, y[cut:NTR], t)[1] for t in Ts]))])
    M = fit(np.arange(NTR))
    return K.L.score(lg(M, np.arange(NTR, NTR + NVAL)), y[NTR:NTR + NVAL], T)


def ridge_decode(Z, target, lam=10.0):
    Y = np.eye(K.V, dtype=np.float32)[target]
    Ztr = np.c_[Z[:NTR], np.ones(NTR, np.float32)]
    A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1], dtype=np.float32)
    Wt = np.linalg.solve(A, Ztr.T @ Y[:NTR])
    pred = (np.c_[Z[NTR:], np.ones(NVAL, np.float32)] @ Wt).argmax(1)
    return float((pred == target[NTR:]).mean())


def main(out, tags):
    res = {}
    for tag in tags:
        off_chars = int(re.search(r"_o(\d+)", tag).group(1))
        ids = np.asarray([K.chars.index(c) for c in K.TEXT[off_chars: off_chars + NTR + NVAL + 1]], np.int64)
        x_ids = ids[:-1]
        z = np.load(E / (tag + ".npz") if "/" in tag else E / "night" / (tag + ".npz"))
        kc = K.zclip(z["kc"], 0, NTR)
        oh4 = K.zclip(K.onehot_lags(x_ids, 4), 0, NTR)
        r = {"kc_active_frac": float((z["kc"][:NTR] > 0).mean())}
        for order in (3, 5, 7):                              # KN alone and stacked
            off, y = kn_offsets(ids, order)
            for name, Zs in (("kn", None), ("kn+kc", kc), ("kn+onehot4", oh4),
                             ("kn+onehot4+kc", np.concatenate([oh4, kc], 1))):
                acc, bpc = stack_eval(off, Zs, y)
                r[f"{name}_{order}"] = {"acc": acc, "bpc": bpc}
                print(tag, order, name, round(acc, 4), round(bpc, 4), flush=True)
        maj = []
        for k in range(13):                                   # memory profile
            tgt = np.concatenate([np.full(k, -1), x_ids[:len(x_ids) - k]]) if k else x_ids.copy()
            tgt[tgt < 0] = 0
            r.setdefault("mem_kc", []).append(ridge_decode(kc, tgt))
            maj.append(float(np.bincount(tgt[NTR:], minlength=K.V).max() / NVAL))
        r["mem_majority"] = maj
        print(tag, "mem", [round(a, 3) for a in r["mem_kc"]], flush=True)
        res[tag] = r
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
