"""Criterion 2 of the online-MB pilot (CPU): does the content-taught mushroom body add
information beyond a Kneser-Ney 5-gram, and beyond a simple online cache?

Features stacked on the cross-fitted KN-5 log-probabilities (readout of lm_explain.stack_eval,
clipped z, L2 5e-3; temperature on the last 10% of train):
  cacheH   log-probs of a causal, exponentially decayed character cache (context = last 2
           characters, half-life H characters, add-beta smoothing towards the unigram), built
           online over the whole stream (train and validation) from the past only
  mb_<cond>  memory features of the online-MB run: memory current on the 97 MBONs and its
           decomposition into the K class gate patterns (mb_online_text.py)
Usage: lm_online_eval.py OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\苍蝇。\mechanism")
import lm_controls as K  # noqa: E402
import lm_explain as X  # noqa: E402

D = Path(r"E:\mechanism_20260928\mbtext")
NTR, NVAL = 20000, 5000


def cache_features(ids, half_life, beta=1.0):
    """Row t (predicting ids[t+1]) uses counts of (ids[s-1], ids[s]) -> ids[s+1] for s < t only."""
    d = 0.5 ** (1.0 / half_life)
    uni = np.bincount(ids[:NTR], minlength=K.V) + 1.0; uni /= uni.sum()
    table, last = {}, {}
    out = np.zeros((NTR + NVAL, K.V), np.float32)
    for t in range(NTR + NVAL):
        ctx = (int(ids[t - 1]) if t > 0 else -1, int(ids[t]))
        if ctx in table:
            table[ctx] *= d ** (t - last[ctx]); last[ctx] = t
            c = table[ctx]
        else:
            c = np.zeros(K.V)
        out[t] = np.log((c + beta * uni) / (c.sum() + beta))
        # after predicting row t we learn ids[t+1]; update the context of row t
        if t + 1 < len(ids):
            if ctx not in table:
                table[ctx] = np.zeros(K.V); last[ctx] = t
            table[ctx][ids[t + 1]] += 1.0
    return out


def main(out):
    res = {}
    for off in (0, 300000, 600000):
        ids = np.asarray([K.chars.index(c) for c in K.TEXT[off: off + NTR + NVAL + 1]], np.int64)
        kn5, y = X.kn_offsets(ids, 5)
        feats = {f"cache{h}": K.zclip(cache_features(ids, h), 0, NTR) for h in (100, 1000)}
        for cond in ("content", "random"):
            f = D / f"{cond}_o{off}.npz"
            if f.exists():
                z = np.load(f)
                feats[f"mb_{cond}"] = K.zclip(np.concatenate([z["I"], z["coef"]], 1)[:NTR + NVAL], 0, NTR)
        r = {"kn5": X.stack_eval(kn5, None, y)}
        for name, Z in feats.items():
            r[f"kn5+{name}"] = X.stack_eval(kn5, Z, y)
        for cond in ("content", "random"):
            if f"mb_{cond}" in feats:
                r[f"kn5+cache100+mb_{cond}"] = X.stack_eval(
                    kn5, np.concatenate([feats["cache100"], feats[f"mb_{cond}"]], 1), y)
        res[off] = {k: {"acc": v[0], "bpc": v[1]} for k, v in r.items()}
        print(off, {k: round(v[1], 4) for k, v in r.items()}, flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
