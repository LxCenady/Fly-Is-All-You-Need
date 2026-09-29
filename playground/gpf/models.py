"""Character models for the playground.  KN and GRU need only numpy (torch only to train a GRU);
the connectome model (GPF-1, gpf/brain.py) needs flybrain with GPU support and an NVIDIA GPU.

Built-in models live in gpf/data/.  Models you train yourself (python -m gpf train ..., see
gpf/train.py) live in ~/.gpf/models/<name>/ (or $GPF_MODELS) and are listed after the
built-ins as "user:<name>".  Every model carries its own character set."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
CORPUS = DATA / "tinyshakespeare.txt"
VOCAB = list(json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))["chars"])


def models_dir() -> Path:
    return Path(os.environ.get("GPF_MODELS") or Path.home() / ".gpf" / "models")


class CharModel:
    name = "base"
    chars: list[str] = VOCAB

    def set_chars(self, chars):
        self.chars = list(chars)
        self.idx = {c: i for i, c in enumerate(self.chars)}
        self.V = len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.idx[c] for c in text if c in self.idx]

    def reset(self):
        raise NotImplementedError

    def feed(self, t: int) -> np.ndarray:
        """Consume one character id; return logits for the next character."""
        raise NotImplementedError


# ------------------------------------------------------------------ Kneser-Ney
class KNModel(CharModel):
    """Interpolated modified Kneser-Ney character n-gram (uctf.baselines), fitted when loaded."""

    def __init__(self, order: int = 7, text: str | None = None, chars=None, name=None, log=print):
        from uctf.baselines import KneserNey
        self.n = order
        self.set_chars(chars or VOCAB)
        text = CORPUS.read_text(encoding="utf-8")[:1_000_000] if text is None else text
        self.name = name or f"Kneser-Ney {order}-gram ({len(text):,} chars)"
        t0 = time.time()
        self.kn = KneserNey(order, self.V).fit(self.encode(text))
        log(f"{self.name} fitted in {time.time() - t0:.0f}s")
        self.reset()

    def reset(self):
        self.hist = []

    def feed(self, t):
        self.hist.append(int(t))
        return self.kn.logprobs(self.hist)


# ------------------------------------------------------------------ GRU
class GRUModel(CharModel):
    """1-layer GRU trained with torch; inference in numpy, same equations as torch.nn.GRU:
    r = s(W_ir x + b_ir + W_hr h + b_hr),  z = s(W_iz x + b_iz + W_hz h + b_hz),
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn)),  h' = (1 - z) * n + z * h."""

    def __init__(self, weights=DATA / "gru_1000000.npz", chars=None, name=None, log=print):
        w = np.load(weights)
        self.set_chars(chars or VOCAB)
        self.emb = w["emb_weight"].astype(np.float64)
        self.Wih, self.Whh = w["rnn_weight_ih_l0"].astype(np.float64), w["rnn_weight_hh_l0"].astype(np.float64)
        self.bih, self.bhh = w["rnn_bias_ih_l0"].astype(np.float64), w["rnn_bias_hh_l0"].astype(np.float64)
        self.Wo, self.bo = w["out_weight"].astype(np.float64), w["out_bias"].astype(np.float64)
        self.H = self.Whh.shape[1]
        assert self.emb.shape[0] == self.V, "vocabulary mismatch"
        self.name = name or f"GRU (hidden {self.H}, trained on 1M chars)"
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
def ctx_hash(hist, order, V, buckets):
    from uctf.readout import ctx_hash as h
    return h(hist, order, V, buckets)


class BrainModel(CharModel):
    """GPF-1: the MaleCNS v1.0 connectome simulated one character at a time (gpf/brain.py,
    needs flybrain with GPU support) + a trained linear readout (model.json + readout.npz)."""

    def __init__(self, readout=DATA / "brain_readout_20k", name=None, log=print):
        from . import brain as fly
        if not fly.available():
            raise RuntimeError("GPF-1 needs flybrain with GPU support (CuPy, CUDA 12) and an NVIDIA GPU")
        rd = Path(readout)
        meta = json.loads((rd / "model.json").read_text(encoding="utf-8"))
        d = np.load(rd / "readout.npz")
        self.W, self.E, self.b = d["W"], d["E"].astype(np.float32), d["b"]
        self.mu, self.sd = d["mu"], d["sd"]
        self.Tcal, self.clip = float(meta["temperature"]), meta.get("clip")
        self.set_chars(meta["chars"])
        ro = meta.get("readout", {})
        self.order, self.buckets = int(ro.get("context_order", 3)), int(ro.get("buckets", self.E.shape[0]))
        b = meta["brain"]
        log("building the connectome simulation (166,700 neurons; the first run downloads ~260 MB)...")
        self.rt = fly.make_substrate(self.V, b)
        self.name = name or "GPF-1 (fly connectome)"
        log(f"{self.name} ready")
        self.reset()

    def reset(self):
        self.rt.reset()
        self.hist = []

    # activity view for the web UI
    def map_payload(self):
        return self.rt.map_payload()

    def record_activity(self, on=True):
        self.rt.record_activity = bool(on)

    def activity(self):
        return self.rt.last_activity

    def feed(self, t):
        x = self.rt.step_token(int(t))
        z = (x - self.mu) / self.sd
        if self.clip:
            z = np.clip(z, -self.clip, self.clip)
        self.hist.append(int(t))
        h = ctx_hash(self.hist, self.order, self.V, self.buckets)
        return (self.b + self.E[h] + z @ self.W) / self.Tcal


# ------------------------------------------------------------------ registry
# key -> (label, factory(log), description, needs the GPU brain)
BUILTIN = {
    "brain": ("GPF-1 (fly connectome; needs flybrain + NVIDIA GPU)", lambda log: BrainModel(log=log),
              "166,700 simulated neurons per character, GPU. Pretrained by evolution.", True),
    "kn7": ("Kneser-Ney 7-gram, 1M chars", lambda log: KNModel(7, log=log),
            "Counts of 7-character sequences in 1M characters. Instant.", False),
    "kn5-20k": ("Kneser-Ney 5-gram, 20k chars (same data as the brain)",
                lambda log: KNModel(5, CORPUS.read_text(encoding="utf-8")[:20_000], log=log),
                "Trained on the same 20k characters as the fly. Fair comparison.", False),
    "gru": ("GRU, 1M chars", lambda log: GRUModel(log=log),
            "A small trained recurrent network, 1M characters. The best writer here.", False),
}
MODELS: dict = {}


def load_user_model(d: Path, log=print) -> CharModel:
    meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
    kind, name = meta["gpf_type"], meta.get("name", d.name)
    if kind == "kn":
        text = (d / "corpus.txt").read_text(encoding="utf-8")
        return KNModel(meta["order"], text, meta["chars"], name, log)
    if kind == "gru":
        return GRUModel(d / "gru.npz", meta["chars"], name, log)
    if kind == "brain":
        return BrainModel(d, name, log)
    raise ValueError(f"unknown model type {kind!r} in {d}")


def user_models() -> dict:
    out = {}
    root = models_dir()
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if (p / "model.json").is_file()):
        try:
            meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
            kind = meta["gpf_type"]
        except (ValueError, KeyError, OSError):
            continue
        m = meta.get("metrics", {})
        kind_name = {"kn": f"{meta.get('order', '')}-gram", "gru": "GRU", "brain": "connectome"}.get(kind, kind)
        desc = (f"Your {kind_name} model, "
                f"{meta.get('data', {}).get('train_chars', 0):,} training chars of "
                f"{meta.get('data', {}).get('corpus', '?')}"
                + (f"; validation {m['val_bpc']:.2f} BPC" if "val_bpc" in m else "") + ".")
        out["user:" + d.name] = (meta.get("name", d.name), (lambda log, d=d: load_user_model(d, log)),
                                 desc, kind == "brain")
    return out


def refresh_models() -> dict:
    """Rebuild MODELS in place (built-ins first, then the user's models)."""
    MODELS.clear(); MODELS.update(BUILTIN); MODELS.update(user_models())
    return MODELS


refresh_models()


# ------------------------------------------------------------------ sampling
def sample(logits, temp, topk, rng):
    s = np.asarray(logits, np.float64) / max(temp, 1e-6)
    if topk:
        s = np.where(s >= np.sort(s)[-topk], s, -np.inf)
    p = np.exp(s - s.max()); p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate(model, prompt, n, temp=0.8, topk=0, seed=0, on_char=None, stop=lambda: False, on_feed=None):
    """Reset, inject the prompt character by character, then write n characters.  Characters
    outside the model's character set are skipped.  on_char(c): each written character.
    on_feed(c, phase): after the model has taken in a character, phase "read" (prompt) or
    "write" (its own output)."""
    rng = np.random.default_rng(seed)
    chars = model.chars
    model.reset()
    ids = model.encode(prompt) or [model.idx.get("\n", 0)]
    for t in ids:
        if stop():
            return ""
        logits = model.feed(t)
        if on_feed:
            on_feed(chars[t], "read")
    out = []
    for _ in range(n):
        if stop():
            break
        t = sample(logits, temp, topk, rng)
        out.append(chars[t])
        if on_char:
            on_char(chars[t])
        logits = model.feed(t)
        if on_feed:
            on_feed(chars[t], "write")
    return "".join(out)
