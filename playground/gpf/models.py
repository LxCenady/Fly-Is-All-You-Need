"""Character models for the playground.  KN and GRU need only numpy (+ torch for the GRU);
the connectome model (GPF-1, gpf/brain.py) needs flybrain with GPU support and an NVIDIA GPU."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
CORPUS = DATA / "tinyshakespeare.txt"
VOCAB = list(json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))["chars"])
V = len(VOCAB)
IDX = {c: i for i, c in enumerate(VOCAB)}


def encode(text: str) -> list[int]:
    return [IDX[c] for c in text if c in IDX]


class CharModel:
    name = "base"

    def reset(self):
        raise NotImplementedError

    def feed(self, t: int) -> np.ndarray:
        """Consume one character id; return logits for the next character."""
        raise NotImplementedError


# ------------------------------------------------------------------ Kneser-Ney
class KNModel(CharModel):
    """Interpolated modified Kneser-Ney character n-gram."""

    def __init__(self, order: int = 7, train_chars: int = 1_000_000, log=print):
        self.n = order
        self.name = f"Kneser-Ney {order}-gram ({train_chars:,} chars)"
        t0 = time.time()
        self._fit(encode(CORPUS.read_text(encoding="utf-8")[:train_chars]))
        log(f"{self.name} fitted in {time.time() - t0:.0f}s")
        self.reset()

    def _fit(self, seq):
        n = self.n
        c = [defaultdict(lambda: defaultdict(int)) for _ in range(n + 1)]
        for i in range(len(seq)):
            for k in range(1, n + 1):
                if i - k + 1 < 0:
                    break
                c[k][tuple(seq[i - k + 1:i])][seq[i]] += 1
        cc = [defaultdict(lambda: defaultdict(int)) for _ in range(n + 1)]
        for k in range(2, n + 1):
            for ctx, d in c[k].items():
                for w in d:
                    cc[k - 1][ctx[1:]][w] += 1
        self.c, self.cc, self.D = c, cc, []
        for k in range(n + 1):
            nr = np.zeros(5)
            for d in (c[k] if k == n else cc[k]).values():
                for v in d.values():
                    if v <= 4:
                        nr[v] += 1
            Y = nr[1] / max(nr[1] + 2 * nr[2], 1)
            self.D.append([0.0] + [max(min(r - (r + 1) * Y * nr[r + 1] / max(nr[r], 1), r), 0.0)
                                   for r in (1, 2, 3)])

    def _p(self, k, ctx, top):
        if k == 0:
            return np.full(V, 1.0 / V)
        d = (self.c[k] if top else self.cc[k]).get(ctx[len(ctx) - (k - 1):] if k > 1 else ())
        lower = self._p(k - 1, ctx, False)
        if not d:
            return lower
        tot = sum(d.values()); D = self.D[k]; p = np.zeros(V); nr = [0, 0, 0, 0]
        for w, v in d.items():
            p[w] = max(v - D[min(v, 3)], 0) / tot; nr[min(v, 3)] += 1
        return p + (D[1] * nr[1] + D[2] * nr[2] + D[3] * nr[3]) / tot * lower

    def reset(self):
        self.hist = []

    def feed(self, t):
        self.hist.append(int(t))
        ctx = tuple(self.hist[-(self.n - 1):]) if self.n > 1 else ()
        return np.log(np.maximum(self._p(min(self.n, len(ctx) + 1), ctx, True), 1e-12))


# ------------------------------------------------------------------ GRU
class GRUModel(CharModel):
    """1-layer GRU (hidden 256) trained with torch; inference in numpy, same equations as
    torch.nn.GRU:  r = s(W_ir x + b_ir + W_hr h + b_hr),  z = s(W_iz x + b_iz + W_hz h + b_hz),
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn)),  h' = (1 - z) * n + z * h."""

    def __init__(self, weights: str = "gru_1000000.npz", log=print):
        w = np.load(DATA / weights)
        self.emb = w["emb_weight"].astype(np.float64)
        self.Wih, self.Whh = w["rnn_weight_ih_l0"].astype(np.float64), w["rnn_weight_hh_l0"].astype(np.float64)
        self.bih, self.bhh = w["rnn_bias_ih_l0"].astype(np.float64), w["rnn_bias_hh_l0"].astype(np.float64)
        self.Wo, self.bo = w["out_weight"].astype(np.float64), w["out_bias"].astype(np.float64)
        self.H = self.Whh.shape[1]
        self.name = "GRU (hidden 256, trained on 1M chars)"
        log(f"{self.name} loaded")
        self.reset()

    def reset(self):
        self.h = np.zeros(self.H)

    def feed(self, t):
        H, x, h = self.H, self.emb[int(t)], self.h
        gi, gh = self.Wih @ x + self.bih, self.Whh @ h + self.bhh
        sig = lambda a: 1.0 / (1.0 + np.exp(-a))
        r = sig(gi[:H] + gh[:H]); z = sig(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = (1.0 - z) * n + z * h
        return self.Wo @ self.h + self.bo


# ------------------------------------------------------------------ connectome
class BrainModel(CharModel):
    """GPF-1: the MaleCNS v1.0 connectome simulated one character at a time (gpf/brain.py,
    needs flybrain with GPU support) + the trained linear readout bundled in data/."""

    def __init__(self, readout: str = "brain_readout_20k", log=print):
        from . import brain as fly
        if not fly.available():
            raise RuntimeError("GPF-1 needs flybrain with GPU support (CuPy, CUDA 12) and an NVIDIA GPU")
        rd = DATA / readout
        meta = json.loads((rd / "model.json").read_text(encoding="utf-8"))
        d = np.load(rd / "readout.npz")
        self.W, self.E, self.b = d["W"], d["E"].astype(np.float32), d["b"]
        self.mu, self.sd = d["mu"], d["sd"]
        self.Tcal, self.clip = float(meta["temperature"]), meta.get("clip")
        assert meta["chars"] == "".join(VOCAB), "vocabulary mismatch"
        b = meta["brain"]
        log("building the connectome simulation (166,700 neurons; the first run downloads ~260 MB)...")
        self.rt = fly.FlyRuntime(V, {"active": b["pn_active"], "drive": 1.0 * b["drive_scale"]})
        self.name = "GPF-1 (fly connectome)"
        log(f"{self.name} ready")
        self.reset()

    def reset(self):
        self.rt.reset()
        self.hist = []

    def feed(self, t):
        x = self.rt.step_token(int(t))
        z = (x - self.mu) / self.sd
        if self.clip:
            z = np.clip(z, -self.clip, self.clip)
        self.hist.append(int(t))
        h = 0
        for lag in (2, 1, 0):                          # hash of the last 3 characters (lm_mech.ctx_ids)
            h = h * V + (self.hist[-1 - lag] if len(self.hist) > lag else 0)
        return (self.b + self.E[h % self.E.shape[0]] + z @ self.W) / self.Tcal


MODELS = {
    "kn7": ("Kneser-Ney 7-gram, 1M chars", lambda log: KNModel(7, 1_000_000, log)),
    "kn5-20k": ("Kneser-Ney 5-gram, 20k chars (same data as the brain)", lambda log: KNModel(5, 20_000, log)),
    "gru": ("GRU, 1M chars", lambda log: GRUModel(log=log)),
    "brain": ("GPF-1 (fly connectome; needs flybrain + NVIDIA GPU)", lambda log: BrainModel(log=log)),
}


# ------------------------------------------------------------------ sampling
def sample(logits, temp, topk, rng):
    s = np.asarray(logits, np.float64) / max(temp, 1e-6)
    if topk:
        s = np.where(s >= np.sort(s)[-topk], s, -np.inf)
    p = np.exp(s - s.max()); p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate(model, prompt, n, temp=0.8, topk=0, seed=0, on_char=None, stop=lambda: False):
    """Reset, inject the prompt character by character, then write n characters."""
    rng = np.random.default_rng(seed)
    model.reset()
    ids = encode(prompt) or [IDX["\n"]]
    for t in ids:
        logits = model.feed(t)
    out = []
    for _ in range(n):
        if stop():
            break
        t = sample(logits, temp, topk, rng)
        out.append(VOCAB[t])
        if on_char:
            on_char(VOCAB[t])
        logits = model.feed(t)
    return "".join(out)
