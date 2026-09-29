"""Train your own GPF model on any text file.

    python -m gpf train kn    --data my.txt --name my-ngram  [--order 5]
    python -m gpf train gru   --data my.txt --name my-gru    [--hidden 256 --max-steps 20000]
    python -m gpf train brain --data my.txt --name my-fly    [--substrate malecns-v1 --train-chars 20000]

The text is split into a training part and a validation part (the characters that follow it).
The model is saved to ~/.gpf/models/<name>/ (or $GPF_MODELS/<name>/) and appears in the web UI,
the terminal UI and --cli as "user:<name>".  Every run prints validation bits per character
(BPC, lower is better) for the new model next to simple baselines on the same split.

  kn     Kneser-Ney character n-gram.  numpy only; seconds to minutes.
  gru    1-layer GRU.  Needs torch (uses the GPU if torch sees one).
  brain  A connectome as a frozen reservoir: a simulated nervous system turns each character
         into neuron features; only a linear readout (plus a hashed 3-character context table)
         is trained.  --substrate picks the connectome spec (gpf/connectome/specs/ or your own
         JSON); the default, malecns-v1, is the whole fly CNS with GPF-1's recipe (paper_lm/,
         Section VI; flybrain + NVIDIA GPU, ~40-60 characters/s).  --control rewire-full or
         rewire-class trains on a degree-preserving rewired copy instead: the null model that
         asks whether the real wiring matters.  For the whole comparison use  python -m gpf bench.

Options can also come from a JSON file: --config params.json (command-line flags win).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .models import GRUModel, KNModel, ctx_hash, models_dir

DEFAULTS = {
    "common": {"train_chars": None, "val_chars": None, "offset": 0, "max_vocab": 256, "vocab": None, "seed": 0},
    "kn": {"order": 5},
    "gru": {"embed": 32, "hidden": 256, "seq": 100, "batch": 32, "lr": 3e-3, "max_steps": 20000,
            "eval_every": 100, "patience": 8, "device": "auto"},
    "brain": {"train_chars": 20000, "substrate": "malecns-v1", "pn_active": None, "drive": None,
              "control": "none", "control_seed": 0, "spec_set": None, "device": "auto", "l2": 1e-2, "epochs": 15, "lr": 0.01, "context_order": 3,
              "buckets": 32768, "clip": 3.0, "keep_features": False},
}
LOG2 = math.log(2)


# ------------------------------------------------------------------ data
def load_split(a):
    """Read the file, pick the training and validation characters, build the character set."""
    text = Path(a.data).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    text = text[a.offset:]
    if a.train_chars is None:
        n_val = a.val_chars or max(1000, min(len(text) // 10, 100_000))
        n_tr = len(text) - n_val
    else:
        n_tr = a.train_chars
        n_val = a.val_chars or max(1000, n_tr // 4 if a.kind == "brain" else n_tr // 10)
    if n_tr < 1000 or n_tr + n_val > len(text):
        sys.exit(f"not enough text: {len(text):,} characters after offset {a.offset}, "
                 f"need {n_tr:,} for training + {n_val:,} for validation")
    part = text[:n_tr + n_val]
    freq = Counter(part)
    if a.vocab:                                  # a fixed character set: vocab.json or any text file
        src = Path(a.vocab).read_text(encoding="utf-8")
        try:
            chars = list(json.loads(src)["chars"])
        except (ValueError, KeyError, TypeError):
            chars = sorted(set(src))
    else:
        chars = sorted(c for c, _ in freq.most_common(a.max_vocab))
    dropped = sum(v for c, v in freq.items() if c not in set(chars))
    idx = {c: i for i, c in enumerate(chars)}
    ids = np.asarray([idx[c] for c in part if c in idx], np.int64)
    n_tr = n_tr - sum(1 for c in part[:n_tr] if c not in idx)
    print(f"text: {len(part):,} characters ({n_tr:,} train, {len(ids) - n_tr:,} validation), "
          f"{len(chars)} distinct" + (f"; {dropped:,} characters outside the character set dropped"
                                      if dropped else ""), flush=True)
    return part, chars, ids, n_tr


def unigram_bpc(ids, n_tr, V):
    p = (np.bincount(ids[:n_tr], minlength=V) + 1.0) / (n_tr + V)
    return float(-np.log2(p[ids[n_tr:]]).mean())


def kn_bpc(order, chars, ids, n_tr, log=lambda m: None):
    """Validation BPC and accuracy of a Kneser-Ney model fitted on the training characters."""
    text = "".join(chars[i] for i in ids[:n_tr])
    m = KNModel(order, text, chars, log=log)
    m.reset()
    for t in ids[max(0, n_tr - order):n_tr]:       # context: the end of the training text
        logits = m.feed(t)
    lp, hit = 0.0, 0
    for t in ids[n_tr:]:
        lp += logits[t]; hit += int(np.argmax(logits) == t)
        logits = m.feed(t)
    n = len(ids) - n_tr
    return -lp / n / LOG2, hit / n


def save(a, kind, chars, part, n_tr, metrics, extra):
    out = Path(a.models_dir or models_dir()) / a.name
    if out.exists() and not a.overwrite:
        sys.exit(f"{out} exists; pick another --name or add --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"gpf_type": kind, "name": a.name, "chars": "".join(chars), "metrics": metrics,
            "data": {"corpus": Path(a.data).name, "offset": a.offset, "train_chars": n_tr,
                     "val_chars": len(part) - n_tr,
                     "sha256": hashlib.sha256(part.encode("utf-8")).hexdigest()},
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta.update(extra)
    return out, meta


def write_meta(out, meta):
    (out / "model.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def report(a, metrics):
    print("\nvalidation bits per character (lower is better):")
    for k, v in metrics.items():
        if k.endswith("bpc"):
            print(f"  {k:<28} {v:.3f}")
    gpf = "gpf" if getattr(sys, "frozen", False) else "python -m gpf"
    print(f"\nsaved as user:{a.name}.  Try it:\n"
          f"  {gpf} --web            (pick \"{a.name}\" under Your models)\n"
          f"  {gpf} --cli --model user:{a.name} --prompt \"...\"")


# ------------------------------------------------------------------ Kneser-Ney
def train_kn(a):
    part, chars, ids, n_tr = load_split(a)
    t0 = time.time()
    bpc, acc = kn_bpc(a.order, chars, ids, n_tr, log=print)
    metrics = {"val_bpc": bpc, "val_acc": acc, "unigram_val_bpc": unigram_bpc(ids, n_tr, len(chars))}
    out, meta = save(a, "kn", chars, part, n_tr, metrics, {"order": a.order})
    (out / "corpus.txt").write_text("".join(chars[i] for i in ids[:n_tr]), encoding="utf-8")
    write_meta(out, meta)
    print(f"fitted and evaluated in {time.time() - t0:.0f}s")
    report(a, metrics)


# ------------------------------------------------------------------ GRU
def train_gru(a):
    try:
        import torch
    except ImportError:
        sys.exit("training a GRU needs torch:  pip install torch")
    part, chars, ids, n_tr = load_split(a)
    V = len(chars)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

    class GRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(V, a.embed); self.rnn = torch.nn.GRU(a.embed, a.hidden, batch_first=True)
            self.out = torch.nn.Linear(a.hidden, V)

        def forward(self, x, h=None):
            o, h = self.rnn(self.emb(x), h)
            return self.out(o), h

    def bpc_over(model, lo, hi, warm=1000):
        """Mean BPC predicting ids[lo+1..hi] (targets), hidden state warmed up on the
        `warm` characters before lo."""
        model.eval(); h = None; tot, n = 0.0, 0
        with torch.no_grad():
            for s in range(max(0, lo - warm), hi, 2000):
                e = min(s + 2000, hi)
                x = torch.tensor(ids[s:e], device=dev)[None]; yt = torch.tensor(ids[s + 1:e + 1], device=dev)[None]
                lg, h = model(x, h)
                lp = torch.log_softmax(lg, -1).gather(-1, yt[..., None])[0, :, 0]
                m = torch.arange(s, e, device=dev) >= lo
                tot += float(-lp[m].sum()); n += int(m.sum())
        model.train()
        return tot / max(n, 1) / LOG2

    cut = int(n_tr * 0.9)                      # early stopping on the last 10% of the training text
    if cut < a.seq + 2:
        sys.exit("training text too short for --seq")
    model = GRU().to(dev); opt = torch.optim.Adam(model.parameters(), a.lr)
    print(f"GRU: embedding {a.embed}, hidden {a.hidden}, {sum(p.numel() for p in model.parameters()):,} "
          f"parameters, device {dev}", flush=True)
    best, best_state, bad, step, t0 = 1e9, None, 0, 0, time.time()
    while bad < a.patience and step < a.max_steps:
        st = rng.integers(0, cut - a.seq - 1, a.batch)
        x = torch.tensor(np.stack([ids[s:s + a.seq] for s in st]), device=dev)
        y = torch.tensor(np.stack([ids[s + 1:s + a.seq + 1] for s in st]), device=dev)
        lg, _ = model(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1
        if step % a.eval_every == 0:
            hb = bpc_over(model, cut, n_tr - 1)
            if hb < best - 1e-4:
                best, bad = hb, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"  step {step:>6}  train loss {loss.item() / LOG2:.3f}  held-out {hb:.3f} BPC"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    val = bpc_over(model, n_tr - 1, len(ids) - 1)
    metrics = {"val_bpc": val, "heldout_bpc": best, "steps": step,
               "unigram_val_bpc": unigram_bpc(ids, n_tr, V)}
    if not a.no_baseline:
        print("fitting a Kneser-Ney 5-gram on the same split for comparison...", flush=True)
        metrics["kn5_val_bpc"] = kn_bpc(5, chars, ids, n_tr)[0]
    out, meta = save(a, "gru", chars, part, n_tr, metrics,
                     {"gru": {k: getattr(a, k) for k in DEFAULTS["gru"]}})
    w = {k.replace(".", "_"): v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    np.savez_compressed(out / "gru.npz", **w)
    write_meta(out, meta)
    GRUModel(out / "gru.npz", chars, a.name, log=lambda m: None)          # check it loads
    report(a, metrics)


# ------------------------------------------------------------------ connectome
def fit_readout(X, cx, y, V, a, clip):
    """Softmax readout: standardised brain features (clipped at +-clip) + a hashed context table
    initialised from smoothed counts; Adam, minibatches of 1024.  A numpy port of lm_mech.fit
    (mechanism/export_lm_model.py).  X may have zero columns (context table only)."""
    n = len(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)
    if clip:
        np.clip(Z, -clip, clip, out=Z)
    W = np.zeros((X.shape[1], V), np.float32)
    prior = (np.bincount(y, minlength=V) + 1) / (n + V)
    b = np.log(prior).astype(np.float32)
    C = np.zeros((a.buckets, V)); np.add.at(C, (cx, y), 1.0)
    E = (np.log((C + 2.0 * prior) / (C.sum(1, keepdims=True) + 2.0)) - np.log(prior)).astype(np.float32)
    params = [W, E, b]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    rng = np.random.RandomState(a.seed); step = 0
    for _ in range(a.epochs):
        order = rng.permutation(n)
        for lo in range(0, n, 1024):
            idx = order[lo:lo + 1024]
            s = b + E[cx[idx]] + Z[idx] @ W
            s -= s.max(1, keepdims=True)
            p = np.exp(s); p /= p.sum(1, keepdims=True)
            p[np.arange(len(idx)), y[idx]] -= 1; p /= len(idx)
            gE = np.zeros_like(E); np.add.at(gE, cx[idx], p)
            grads = [Z[idx].T @ p + a.l2 * W, gE, p.sum(0)]
            step += 1
            for j, (P, G) in enumerate(zip(params, grads)):
                m[j] = 0.9 * m[j] + 0.1 * G; v[j] = 0.999 * v[j] + 0.001 * G * G
                P -= a.lr * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "W": W, "E": E, "b": b, "clip": clip}


def readout_scores(M, X, cx):
    Z = (X - M["mu"]) / M["sd"]
    if M["clip"]:
        Z = np.clip(Z, -M["clip"], M["clip"])
    return M["b"] + M["E"][cx] + Z @ M["W"]


def score(s, y, T):
    s = s / T
    s = s - s.max(1, keepdims=True)
    lp = s - np.log(np.exp(s).sum(1, keepdims=True))
    return float((s.argmax(1) == y).mean()), float(-lp[np.arange(len(y)), y].mean() / LOG2)


def fit_calibrated(X, cx, y, n_tr, V, a, clip):
    """Temperature chosen on the last 10% of the training rows, then refit on all of them."""
    cut = int(n_tr * 0.9)
    M = fit_readout(X[:cut], cx[:cut], y[:cut], V, a, clip)
    s = readout_scores(M, X[cut:n_tr], cx[cut:n_tr])
    Ts = np.linspace(0.6, 2.0, 29)
    T = float(Ts[int(np.argmin([score(s, y[cut:n_tr], t)[1] for t in Ts]))])
    M = fit_readout(X[:n_tr], cx[:n_tr], y[:n_tr], V, a, clip)
    acc, bpc = score(readout_scores(M, X[n_tr:], cx[n_tr:]), y[n_tr:], T)
    return M, T, acc, bpc


def set_path(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def brain_spec(a):
    """What the model is trained on.  A connectome spec (built-in name or JSON path, with
    --pn-active/--drive/--spec-set overrides and an optional rewiring control), or the legacy
    GPF-1 runtime."""
    if a.substrate == "flybrain-malecns-v1":
        return {"substrate": a.substrate, "pn_active": a.pn_active or 160, "drive_scale": a.drive or 1.5,
                "encoder_seed": 3}
    from .connectome import load_spec
    spec = json.loads(json.dumps(load_spec(a.substrate)))               # a private copy
    if a.pn_active:
        spec["input"]["active"] = a.pn_active
    if a.drive:
        spec["input"]["drive"] = a.drive
    for item in a.spec_set or []:
        key, _, val = item.partition("=")
        try:
            val = json.loads(val)
        except ValueError:
            pass
        set_path(spec, key.strip(), val)
    control = None
    if a.control and a.control != "none":
        control = {"rewire": {"rewire-full": "full", "rewire-class": "class"}[a.control], "seed": a.control_seed}
    return {"spec": spec, "control": control, "substrate": spec.get("name", "custom")}


def feature_key(ids, V, bspec) -> str:
    """Identifies simulated features: the text, the character set and everything in the substrate
    spec that changes the simulation (not its description or brain-view settings)."""
    b = json.loads(json.dumps(bspec))
    for k in ("view", "description"):
        b.get("spec", {}).pop(k, None)
    return hashlib.sha256(ids.tobytes() + json.dumps([V, b], sort_keys=True).encode()).hexdigest()


def brain_features(ids, V, a, cache: Path | None):
    """Run the frozen connectome over the text (one continuous stream) and collect the
    features of every character.  Reuses a matching cache from an earlier --keep-features run."""
    key = feature_key(ids, V, brain_spec(a))
    if cache is not None and cache.exists():
        z = np.load(cache)
        if str(z["key"]) == key:
            print("reusing the simulated features from features.npz", flush=True)
            return z["X"].astype(np.float32), key
    return simulate(ids, V, brain_spec(a), a.device or "auto"), key


def log_flush(msg):
    print(msg, flush=True)


def simulate(ids, V, bspec: dict, device="auto", log=log_flush, with_dims=False):
    """Run the substrate over the text as one continuous stream; row i = features after
    character i (used to predict character i+1).  with_dims: also return the size of each
    readout group, in feature order."""
    from . import brain as fly
    ctl = bspec.get("control")
    log(f"building the substrate ({bspec.get('substrate')}" + (f", {ctl['rewire']} rewiring, seed {ctl['seed']}"
                                                              if ctl else "") + ")...")
    try:
        rt = fly.make_substrate(V, bspec, device=device)
    except ImportError as e:
        sys.exit(f"this substrate needs a package that is missing ({e}).  The MaleCNS connectome needs\n"
                 "  pip install \"flybrain[gpu]==0.1.0\"   (and an NVIDIA GPU)")
    rt.reset()
    n = len(ids) - 1
    X = None; t0 = time.time()
    for i in range(n):
        x = rt.step_token(int(ids[i]))
        if X is None:
            X = np.zeros((n, len(x)), np.float32)
        X[i] = x
        if (i + 1) % 1000 == 0 or i + 1 == n:
            r = (i + 1) / (time.time() - t0)
            log(f"  simulated {i + 1:,}/{n:,} characters  ({r:.0f}/s, {(n - i - 1) / r / 60:.1f} min left)")
    if with_dims:
        dims = getattr(rt, "readout_sizes", None) or {"features": X.shape[1]}
        return X, dict(dims)
    return X


def train_brain(a):
    part, chars, ids, n_tr = load_split(a)
    V = len(chars)
    target = Path(a.models_dir or models_dir()) / a.name
    if target.exists() and not a.overwrite:
        sys.exit(f"{target} exists; pick another --name or add --overwrite")
    cache = target / "features.npz"
    X, key = brain_features(ids, V, a, cache if a.keep_features or cache.exists() else None)
    y = ids[1:]
    cx = np.asarray([ctx_hash(list(ids[max(0, i - a.context_order + 1):i + 1]), a.context_order, V, a.buckets)
                     for i in range(len(ids) - 1)], np.int64)
    ntr = n_tr - 1                                   # rows: features at character i predict i+1
    print("fitting the readout (CPU)...", flush=True)
    M, T, acc, bpc = fit_calibrated(X, cx, y, ntr, V, a, a.clip or None)
    metrics = {"val_bpc": bpc, "val_acc": acc}
    print("fitting the same readout without the brain (context table only) for comparison...", flush=True)
    metrics["context_only_val_bpc"] = fit_calibrated(X[:, :0], cx, y, ntr, V, a, None)[3]
    metrics["unigram_val_bpc"] = unigram_bpc(ids, n_tr, V)
    if not a.no_baseline:
        metrics["kn5_val_bpc"] = kn_bpc(5, chars, ids, n_tr)[0]
    tr_acc = score(readout_scores(M, X[:ntr], cx[:ntr]), y[:ntr], T)[0]
    metrics["train_acc"] = tr_acc
    out, meta = save(a, "brain", chars, part, n_tr, metrics, {
        "temperature": T, "clip": a.clip or None, "val_bpc": bpc, "val_acc": acc,
        "features": ([r["name"] for r in brain_spec(a)["spec"]["readout"]] if a.substrate != "flybrain-malecns-v1"
                     else ["kc", "mbon_v", "mbon_spk", "central"]),
        "brain": dict(brain_spec(a), mode="frozen"),
        "readout": {"l2": a.l2, "epochs": a.epochs, "lr": a.lr, "context_order": a.context_order,
                    "buckets": a.buckets, "E_dtype": "float16", "fit": "gpf/train.py fit_readout (CPU)"}})
    np.savez_compressed(out / "readout.npz", W=M["W"], E=M["E"].astype(np.float16), b=M["b"],
                        mu=M["mu"], sd=M["sd"])
    if a.keep_features:
        np.savez(cache, X=X, key=key)
    elif cache.exists():
        cache.unlink()
    write_meta(out, meta)
    report(a, metrics)
    if metrics["val_bpc"] > metrics["context_only_val_bpc"]:
        print("note: on this split the brain features did not beat the context table alone.")


# ------------------------------------------------------------------ command line
def parser():
    ap = argparse.ArgumentParser(prog="gpf train", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("kind", choices=["kn", "gru", "brain"], help="model type")
    ap.add_argument("--data", required=True, help="training text (UTF-8 file)")
    ap.add_argument("--name", required=True, help="model name (folder under the models directory)")
    ap.add_argument("--config", help="JSON file with any of the options below")
    ap.add_argument("--models-dir", help="where to save (default ~/.gpf/models or $GPF_MODELS)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing model of the same name")
    ap.add_argument("--no-baseline", action="store_true", help="skip the Kneser-Ney 5-gram comparison")
    g = ap.add_argument_group("data")
    g.add_argument("--train-chars", type=int, help="training characters (default: all but the "
                                                    "validation part; brain: 20000)")
    g.add_argument("--val-chars", type=int, help="validation characters that follow the training part "
                                                  "(default 10%%, brain 25%% of training)")
    g.add_argument("--offset", type=int, help="skip this many characters at the start")
    g.add_argument("--max-vocab", type=int, help="keep the most frequent N distinct characters (256)")
    g.add_argument("--vocab", help="fixed character set instead: a vocab.json ({\"chars\": ...}) or a text "
                                   "file whose distinct characters are used; others are dropped")
    g.add_argument("--seed", type=int)
    g = ap.add_argument_group("kn")
    g.add_argument("--order", type=int, help="n-gram order (5)")
    g = ap.add_argument_group("gru")
    for k in ("embed", "hidden", "seq", "batch", "max_steps", "eval_every", "patience"):
        g.add_argument("--" + k.replace("_", "-"), type=int)
    g.add_argument("--lr", type=float, help="learning rate (gru 3e-3, brain readout 0.01)")
    g.add_argument("--device", help="auto, cpu or cuda")
    g = ap.add_argument_group("brain")
    g.add_argument("--substrate", help="connectome spec: a built-in name (malecns-v1) or a JSON file; "
                                       "see gpf/connectome/ (flybrain-malecns-v1 = the legacy GPF-1 runtime)")
    g.add_argument("--spec-set", action="append", metavar="KEY=VALUE",
                   help="override one spec entry, e.g. --spec-set neuron.gain=1.8 (repeatable)")
    g.add_argument("--control", choices=["none", "rewire-full", "rewire-class"],
                   help="train on a degree-preserving rewired copy of the connectome instead (null model)")
    g.add_argument("--control-seed", type=int, help="seed of the rewiring (0)")
    g.add_argument("--pn-active", type=int, help="input neurons driven per character (spec default; "
                                                 "MaleCNS: 160 of 675 projection neurons)")
    g.add_argument("--drive", type=float, help="input drive (spec default; MaleCNS 1.5)")
    g.add_argument("--l2", type=float, help="readout L2 penalty (0.01)")
    g.add_argument("--epochs", type=int, help="readout epochs (15)")
    g.add_argument("--context-order", type=int, help="characters hashed into the context table (3; 0 = none)")
    g.add_argument("--buckets", type=int, help="context-table size (32768)")
    g.add_argument("--clip", type=float, help="clip standardised features at +-clip (3; 0 = off)")
    g.add_argument("--keep-features", action="store_true", default=None,
                   help="save the simulated features so a rerun with other readout options is instant")
    return ap


def main(argv):
    ap = parser()
    a = ap.parse_args(argv)
    conf = json.loads(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
    merged = dict(DEFAULTS["common"]); merged.update(DEFAULTS[a.kind])      # defaults < config < flags
    merged.update({k.replace("-", "_"): v for k, v in conf.items()})
    for k, v in merged.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    a.name = a.name.strip().replace("/", "-").replace("\\", "-")
    {"kn": train_kn, "gru": train_gru, "brain": train_brain}[a.kind](a)
