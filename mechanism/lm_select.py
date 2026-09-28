"""Honest readout selection for cached LM runs.

For each cache (one brain configuration, one corpus segment, one code seed):
  candidates = feature set {all, kc, nokc} x L2 {5e-3, 1e-2, 3e-2}, context head on
  selection  = fit on the first 90% of the training rows, BPC on the last 10%
  report     = refit the chosen candidate on all training rows, evaluate once on
               validation; also the context-only model and the brain-only model
               (chosen feature set and L2, no context head) on the same split.
The validation split is never used for choosing.
Usage: lm_select.py OUT.json CACHE1:OFFSET:TRAIN:VAL [CACHE2:...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lm_mech as L  # noqa: E402

SETS = {"all": ["kc", "mbon_v", "mbon_spk", "central"], "kc": ["kc"],
        "nokc": ["mbon_v", "mbon_spk", "central"]}
L2S = [5e-3, 1e-2, 3e-2]


def run(cache, offset, ntr, nval):
    import m1_core as C
    _, chars = C.protocol_args()
    text = L.CORPUS.read_text(encoding="utf-8")
    vocab = len(chars)
    ids = np.asarray([chars.index(c) for c in text[offset: offset + ntr + nval + 1]], np.int64)
    y = ids[1:]
    cx = L.ctx_ids(ids[:-1], 3, 32768, vocab)
    z = np.load(cache)
    cut = int(ntr * 0.9)

    def evaluate(X, skip, l2, tr_end, ev_lo, ev_hi, T=None):
        nb = 32768 if skip else 1
        c = cx if skip else np.zeros_like(cx)
        M = L.fit(None if X is None else X[:tr_end], c[:tr_end], y[:tr_end], vocab, nb, l2, 15, 0.01)
        s = L.logits(M, None if X is None else X[ev_lo:ev_hi], c[ev_lo:ev_hi])
        if T is None:
            return L.score(s, y[ev_lo:ev_hi])
        return L.score(s, y[ev_lo:ev_hi], T)

    def temp(X, skip, l2):
        nb = 32768 if skip else 1
        c = cx if skip else np.zeros_like(cx)
        M = L.fit(None if X is None else X[:cut], c[:cut], y[:cut], vocab, nb, l2, 15, 0.01)
        s = L.logits(M, None if X is None else X[cut:ntr], c[cut:ntr])
        Ts = np.linspace(0.6, 2.0, 29)
        return float(Ts[int(np.argmin([L.score(s, y[cut:ntr], t)[1] for t in Ts]))])

    cands = []
    for name, fs in SETS.items():
        X = np.concatenate([z[f] for f in fs], 1)
        for l2 in L2S:
            _, bpc_hold = evaluate(X, True, l2, cut, cut, ntr)
            cands.append((bpc_hold, name, l2))
    cands.sort()
    _, best_set, best_l2 = cands[0]
    X = np.concatenate([z[f] for f in SETS[best_set]], 1)
    out = {"cache": cache, "offset": offset, "train": ntr, "val": nval,
           "selected": {"features": best_set, "l2": best_l2},
           "holdout_ranking": [{"bpc": round(b, 4), "features": n, "l2": l} for b, n, l in cands]}
    for label, XX, skip in (("brain+context", X, True), ("brain_only", X, False),
                            ("context_only", None, True)):
        T = temp(XX, skip, best_l2)
        acc, bpc = evaluate(XX, skip, best_l2, ntr, ntr, ntr + nval, T)
        out[label] = {"val_acc": acc, "val_bpc": bpc, "temperature": T}
    return out


if __name__ == "__main__":
    res = []
    for spec in sys.argv[2:]:
        cache, off, ntr, nval = spec.rsplit(":", 3)
        r = run(cache, int(off), int(ntr), int(nval))
        res.append(r)
        Path(sys.argv[1]).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(Path(cache).name, r["selected"], {k: r[k] for k in ("brain+context", "brain_only", "context_only")},
              flush=True)
