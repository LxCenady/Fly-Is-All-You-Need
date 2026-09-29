"""Scalable char-level LM trainer on the frozen fly connectome.

Batched parallel rollout (see batchroll.py) -> ridge readout -> metrics + generation.

Examples (CPU smoke test):
  python lm/train_lm.py --device cpu --batch 4 --train-chars 4000 --val-chars 2000 --k 4

On a GPU platform:
  python lm/train_lm.py --device cuda --batch 32 --train-chars 1000000 --val-chars 50000 \
      --k 4 --active 256 --out models/shake_1m.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import progress
from batchroll import collect_batched
from exp7_shakespeare import bpc, fit_temperature, softmax_rows
from flylm import (DETECTORS, Encoder, accuracy, expand_selected_readout, fit_readout,
                   fit_softmax_readout_gpu, select_token_features)

from flybrain import FlyBrain

ROOT = Path(__file__).parent.parent


def cells_matching(brain, patterns):
    """Select annotated cell types by substring, preserving connectome order."""
    types = np.asarray(brain.cell_type).astype(str)
    mask = np.zeros(len(types), bool)
    for pattern in patterns:
        mask |= np.char.find(types, str(pattern)) >= 0
    return np.flatnonzero(mask)


def load_corpus(path, train_chars, val_chars, batch, offset=0):
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], np.int64)
    tr = ids[offset:offset + train_chars]
    va = ids[offset + train_chars:offset + train_chars + val_chars]
    streams = [np.ascontiguousarray(s) for s in np.array_split(tr, batch)]
    return chars, tr, va, streams


def collect_segmented(brain, enc, streams, dns, k, sustain, seg, run, sample_cap=None,
                      feature_mode="trace", trace_tau=0.1, fast_cuda_trace=False):
    B = len(streams)
    L = min(len(s) for s in streams)
    n_seg = max(1, (L - 1) // seg)
    Xs, Ys = [], []
    done = 0
    t0 = time.time()
    for si in range(n_seg):
        lo = si * seg
        hi = min(L, lo + seg) if si < n_seg - 1 else L
        part = [s[lo:hi] for s in streams]
        if len(part[0]) < 2:
            break
        X, Y = collect_batched(brain, enc, part, dns, k_steps=k, sustain=sustain,
                               log={"run": run, "cfg": f"seg{si + 1}/{n_seg}", "phase": "train"},
                               feature_mode=feature_mode, trace_tau=trace_tau,
                               fast_cuda_trace=fast_cuda_trace)
        Xs.append(X.reshape(-1, X.shape[2]))
        Ys.append(Y.reshape(-1))
        done += X.shape[0]
        if sample_cap and done >= sample_cap:
            break
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    if sample_cap and len(X) > sample_cap:
        sel = np.linspace(0, len(X) - 1, sample_cap).astype(np.int64)
        X, Y = X[sel], Y[sel]
    print(f"collected {len(X)} samples in {time.time() - t0:.0f}s "
          f"({len(X) / max(time.time() - t0, 1e-9):.0f} samples/s)", flush=True)
    return X, Y


def collect_val(brain, enc, va, dns, k, sustain, batch, feature_mode="trace", trace_tau=0.1,
                fast_cuda_trace=False):
    streams = [np.ascontiguousarray(s) for s in np.array_split(va, batch)]
    X, Y = collect_batched(brain, enc, streams, dns, k_steps=k, sustain=sustain,
                           feature_mode=feature_mode, trace_tau=trace_tau,
                           fast_cuda_trace=fast_cuda_trace)
    return X.reshape(-1, X.shape[2]), Y.reshape(-1), streams


def generate(brain, enc, dns, chars, mu, sd, W, Ym, temp, starts, k, sustain,
             n=300, seed=0, feature_mode="trace", decode="sample", top_k=8,
             bigram=None, bigram_mix=0.0, skip_bigram=False, skip_order=0,
             skip_buckets=8192, trace_tau=0.1):
    from flybrain.reservoir import Trace
    itos = {i: c for i, c in enumerate(chars)}
    rng = np.random.default_rng(seed)
    trace = Trace(brain, idx=dns, tau=float(trace_tau))
    multiscale = feature_mode == "trace_memory"
    state_history = [] if feature_mode == "trace_history" else None
    memory_traces = ([Trace(brain, idx=dns, tau=tau)
                      for tau in (0.1, 0.3, 1.0, 3.0)]
                     if multiscale else None)
    brain.reset()
    hist = []
    for c in starts:
        hist.append(int(c))
        for s in range(k):
            scale = 1.0 if s == 0 else sustain
            fired = brain.step(inject=enc.inject(np.asarray(hist), len(hist) - 1, scale=scale))
            trace.observe(fired)
            if memory_traces is not None:
                for memory_trace in memory_traces:
                    memory_trace.observe(fired)
        if state_history is not None:
            state_history.append(trace.features())
            if len(state_history) > 4:
                state_history.pop(0)
    out = []
    temporal = feature_mode in ("trace_time", "trace_time_voltage")
    multiscale = feature_mode == "trace_memory"
    has_voltage = feature_mode in ("voltage", "trace_voltage", "trace_time_voltage")
    for _ in range(n):
        temporal_features = []
        for s in range(k):
            scale = 1.0 if s == 0 else sustain
            fired = brain.step(inject=enc.inject(np.asarray(hist), len(hist) - 1, scale=scale))
            observed = trace.observe(fired)
            if memory_traces is not None:
                for memory_trace in memory_traces:
                    memory_trace.observe(fired)
            if temporal:
                temporal_features.append(observed.copy())
        trace_feat = np.concatenate(temporal_features) if temporal else trace.features()
        if feature_mode == "trace_history":
            state_history.append(trace.features())
            if len(state_history) > 4:
                state_history.pop(0)
            while len(state_history) < 4:
                state_history.insert(0, np.zeros_like(state_history[0]))
            feat = np.concatenate(state_history)
        elif multiscale:
            feat = np.concatenate([memory_trace.features() for memory_trace in memory_traces])
        elif feature_mode in ("trace", "trace_time"):
            feat = trace_feat
        else:
            voltage = brain.v[brain.xp.asarray(dns)]
            voltage = voltage.get() if brain.device == "cuda" else voltage.copy()
            # ``generate`` reuses the batched training brain.  Trace uses the
            # default mean aggregation across lanes, so collapse voltage in
            # the same way before concatenating; otherwise batch>1 produces a
            # (features, batch) array and breaks the readout.
            if voltage.ndim > 1:
                voltage = voltage.mean(axis=1)
            feat = voltage if feature_mode == "voltage" else np.concatenate((trace_feat, voltage))
        readout_feat = feat
        order = int(skip_order or (1 if skip_bigram else 0))
        if order:
            if len(hist) < order:
                ctx = [0] * (order - len(hist)) + hist
            else:
                ctx = hist[-order:]
            idx = 0
            for token in ctx:
                idx = idx * len(chars) + int(token)
            width = len(chars) ** order if order <= 2 else int(skip_buckets)
            if order > 2:
                idx %= width
            one_hot = np.zeros(width, np.float32)
            one_hot[idx] = 1.0
            readout_feat = np.concatenate((feat, one_hot))
        p = softmax_rows((((readout_feat - mu) / sd) @ W + Ym)[None] / temp)[0]
        if bigram is not None and bigram_mix > 0:
            # A small Markov skip head supplies local character legality while
            # the fly readout remains the dominant signal.  This is useful for
            # readable sampling, but is kept explicit so pure-brain metrics are
            # never confused with the hybrid decoder.
            q = np.asarray(bigram[int(hist[-1])], np.float64)
            p = (1.0 - bigram_mix) * p + bigram_mix * q
            p /= p.sum()
        if decode == "greedy":
            c = int(np.argmax(p))
        elif decode == "topk":
            kk = min(max(int(top_k), 1), len(p))
            ids = np.argpartition(p, -kk)[-kk:]
            q = p[ids]
            q /= q.sum()
            c = int(rng.choice(ids, p=q))
        else:
            c = int(rng.choice(len(p), p=p))
        hist.append(c)
        out.append(itos[c])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus" / "tinyshakespeare.txt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--train-chars", type=int, default=4000)
    ap.add_argument("--val-chars", type=int, default=2000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--sustain", type=float, default=1.0)
    ap.add_argument("--active", type=int, default=256)
    ap.add_argument("--drive", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=None,
                    help="override frozen LIF synaptic gain (useful for sparse MB dynamics)")
    ap.add_argument("--tonic", type=float, default=None,
                    help="override frozen LIF tonic drive")
    ap.add_argument("--window", type=int, default=1,
                    help="causal token window injected into the frozen brain")
    ap.add_argument("--gamma", type=float, default=0.7,
                    help="per-token decay within --window")
    ap.add_argument("--position-codes", action="store_true",
                    help="use independent input codes for each causal window position")
    ap.add_argument("--include-vnc-sensory", action="store_true",
                    help="add the VNC sensory input population used by the high-SNR code probe")
    ap.add_argument("--input-path", choices=("visual", "olfactory", "olfactory_pn", "mixed"), default="visual",
                    help="input route: visual-like detectors, ORNs, olfactory projection neurons, or a mix")
    ap.add_argument("--memory-readout", action="store_true",
                    help="read the annotated mushroom-body memory circuit (KC/MBON/DAN/APL/DPM/hDelta)")
    ap.add_argument("--features", choices=("trace", "voltage", "trace_voltage", "trace_time", "trace_time_voltage", "trace_memory", "trace_history"), default="trace",
                    help="read traces, per-step trace timing, membrane voltage, or combinations")
    ap.add_argument("--trace-tau", type=float, default=0.1,
                    help="decay time constant (seconds) for the readout spike trace; larger values retain working-memory context")
    ap.add_argument("--fast-cuda-trace", action="store_true",
                    help="keep trace accumulation on GPU (faster, with tiny CUDA RNG scheduling differences)")
    ap.add_argument("--readout-types", default="descending_neuron",
                    help="comma-separated cell types/superclasses for the readout")
    ap.add_argument("--readout", choices=("ridge", "softmax"), default="ridge",
                    help="ridge probe or CUDA-trained categorical softmax readout")
    ap.add_argument("--softmax-l2", type=float, default=1e-4)
    ap.add_argument("--softmax-epochs", type=int, default=40)
    ap.add_argument("--softmax-batch", type=int, default=2048)
    ap.add_argument("--softmax-lr", type=float, default=0.03)
    ap.add_argument("--select-features", type=int, default=0,
                    help="keep this many current-token-selective reservoir features (0 keeps all)")
    ap.add_argument("--probe-select", default="",
                    help="probe NPZ stem; replace the readout population with its target-selective neurons")
    ap.add_argument("--probe-score", default="target_score",
                    help="array name inside --probe-select used for ranking (e.g. memory_score)")
    ap.add_argument("--probe-top", type=int, default=512,
                    help="number of neurons to keep from --probe-select")
    ap.add_argument("--probe-central", action="store_true",
                    help="with --probe-select, exclude visual/input and all *sensory* superclasses")
    ap.add_argument("--skip-bigram", action="store_true",
                    help="append current-token one-hot features to the brain readout")
    ap.add_argument("--skip-order", type=int, default=0,
                    help="append context features for this order (1/2 exact, >=3 hashed history)")
    ap.add_argument("--skip-buckets", type=int, default=8192,
                    help="hash bucket count when --skip-order is 3")
    ap.add_argument("--seg", type=int, default=20000)
    ap.add_argument("--sample-cap", type=int, default=0)
    ap.add_argument("--gen", type=int, default=300)
    ap.add_argument("--decode", choices=("sample", "greedy", "topk"), default="sample",
                    help="generation decoder: stochastic sample, argmax, or restricted top-k")
    ap.add_argument("--top-k", type=int, default=8,
                    help="candidate count for --decode topk")
    ap.add_argument("--bigram-mix", type=float, default=0.0,
                    help="optional local bigram prior mixed only into generation")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    chars, tr, va, streams = load_corpus(args.corpus, args.train_chars, args.val_chars, args.batch)
    vocab = len(chars)
    print(f"corpus vocab {vocab}, train {len(tr)}, val {len(va)}, batch {args.batch}, K {args.k}",
          flush=True)

    # Olfactory receptor neurons are the natural entry point to the
    # Kenyon-cell/MBON memory circuit.  Disable recurrent sensory-input
    # synapses in that mode so the simulator's known ORN tonic loop cannot
    # swamp the injected code; ORN output synapses remain intact.
    brain = FlyBrain(device=args.device, batch=args.batch, seed=64,
                     sensory_input=(args.input_path == "visual"))
    if args.gain is not None:
        brain.gain = float(args.gain)
    if args.tonic is not None:
        brain.tonic = float(args.tonic)
    readout_types = [x.strip() for x in args.readout_types.split(",") if x.strip()]
    if not readout_types:
        raise ValueError("--readout-types must name at least one population")
    if args.memory_readout:
        memory_patterns = ("KC", "MBON", "PAM", "PPL", "PPM", "APL", "DPM", "hDelta")
        dns = cells_matching(brain, memory_patterns)
        readout_label = "mushroom-body memory circuit"
    else:
        dns = brain.cells(readout_types)
        readout_label = ",".join(readout_types)
    if not len(dns):
        raise ValueError(f"no cells found for readout types: {readout_types}")
    if args.input_path == "olfactory":
        channel_pools = [cells_matching(brain, ("ORN",))]
    elif args.input_path == "olfactory_pn":
        channel_pools = [cells_matching(brain, ("PN",))]
    elif args.input_path == "mixed":
        channel_pools = [brain.cells(DETECTORS), brain.cells(["visual_projection"]),
                         cells_matching(brain, ("ORN",))]
        if args.include_vnc_sensory:
            channel_pools.append(brain.cells(["vnc_sensory"]))
    else:
        channel_pools = [brain.cells(DETECTORS), brain.cells(["visual_projection"])]
        if args.include_vnc_sensory:
            channel_pools.append(brain.cells(["vnc_sensory"]))
    channels = np.concatenate(channel_pools)
    print(f"readout {len(dns)} cells ({readout_label}) | encoder entries {len(channels)}"
          f" | input path {args.input_path}", flush=True)
    enc = Encoder(channels, vocab, active=args.active, drive=args.drive, window=args.window,
                  gamma=args.gamma, seed=3, position_codes=args.position_codes)

    # A probe-selected readout keeps the connectome frozen but focuses the
    # lightweight linear head on neurons that actually modulated with the
    # next-character condition.  The score is computed offline by the probe;
    # no validation labels or gradient ever enter the brain.
    probe_scores = None
    if args.probe_select:
        probe_path = Path(args.probe_select)
        if probe_path.suffix != ".npz":
            probe_path = probe_path.with_suffix(".npz")
        if not probe_path.exists():
            raise FileNotFoundError(f"probe file not found: {probe_path}")
        with np.load(probe_path) as probe:
            if args.probe_score not in probe.files:
                raise KeyError(f"probe array not found: {args.probe_score}")
            probe_scores = np.asarray(probe[args.probe_score], np.float32)
        if len(probe_scores) != brain.n:
            raise ValueError(f"probe neuron count {len(probe_scores)} != brain size {brain.n}")
        candidate = np.ones(brain.n, bool)
        if args.probe_central:
            supers = np.asarray(brain.superclass).astype(str)
            candidate = (supers != "visual_projection") & (np.char.find(supers, "sensory") < 0)
        candidate &= np.isfinite(probe_scores)
        ids = np.flatnonzero(candidate)
        keep_n = min(max(int(args.probe_top), 1), len(ids))
        ids = ids[np.argpartition(probe_scores[ids], -keep_n)[-keep_n:]]
        ids = ids[np.argsort(-probe_scores[ids])]
        dns = np.asarray(ids, np.int64)
        print(f"probe-selected readout: {len(dns)} neurons from {probe_path}"
              f" ({'central only' if args.probe_central else 'all populations'})", flush=True)

    t0 = time.time()
    if args.trace_tau <= 0:
        raise ValueError("--trace-tau must be positive")
    Xtr, ytr = collect_segmented(brain, enc, streams, dns, args.k, args.sustain, args.seg,
                                 "train_lm", args.sample_cap or None, args.features,
                                 args.trace_tau, args.fast_cuda_trace)
    Xva, yva, va_streams = collect_val(brain, enc, va, dns, args.k, args.sustain, args.batch,
                                        args.features, args.trace_tau, args.fast_cuda_trace)

    # A small identity skip lets the linear head retain exact current-token
    # information that a random sparse encoder can otherwise smear out.  The
    # frozen fly reservoir still supplies the remaining features and temporal
    # mixing; this is an explicit hybrid ablation, not part of pure-brain scores.
    skip_order = int(args.skip_order or (1 if args.skip_bigram else 0))
    if skip_order < 0 or skip_order > 5:
        raise ValueError("--skip-order must be between 0 and 5")
    if skip_order == 3 and args.skip_buckets < 64:
        raise ValueError("--skip-buckets must be at least 64")
    if skip_order:
        if args.select_features or args.sample_cap:
            raise ValueError("--skip-bigram is incompatible with --select-features/--sample-cap")
        def context_hot(stream_list):
            n = min(len(s) for s in stream_list)
            width = vocab ** skip_order if skip_order <= 2 else args.skip_buckets
            hot = np.zeros(((n - 1) * len(stream_list), width), np.float32)
            row = 0
            for i in range(n - 1):
                for stream in stream_list:
                    idx = 0
                    for lag in range(skip_order - 1, -1, -1):
                        token = int(stream[i - lag]) if i >= lag else 0
                        idx = idx * vocab + token
                    if skip_order > 2:
                        idx %= width
                    hot[row, idx] = 1.0
                    row += 1
            return hot
        Htr = context_hot(streams)
        Hva = context_hot(va_streams)
        Xtr = np.concatenate((Xtr, Htr), axis=1)
        Xva = np.concatenate((Xva, Hva), axis=1)
        print(f"identity context skip appended: +{Htr.shape[1]} features", flush=True)

    selected = None
    Xtr_fit = Xtr
    if args.select_features:
        n_stream = min(len(s) for s in streams)
        current_tokens = np.stack([s[:n_stream - 1] for s in streams], axis=1).reshape(-1)
        if len(current_tokens) != len(Xtr):
            raise ValueError("--select-features currently requires one unsampled training segment")
        selected = select_token_features(Xtr, current_tokens, vocab, args.select_features)
        Xtr_fit = Xtr[:, selected]
        print(f"selected {len(selected)}/{Xtr.shape[1]} input-token-selective features", flush=True)

    if args.readout == "ridge":
        cut = max(1, len(Xtr_fit) - max(2000, len(Xtr_fit) // 10))
        best = None
        for lam in (0.1, 1.0, 10.0, 100.0):
            W, mu, sd, Ym = fit_readout(Xtr_fit[:cut], ytr[:cut], vocab, lam=lam)
            a = accuracy(Xtr_fit[cut:], ytr[cut:], W, mu, sd, Ym)
            if best is None or a > best[0]:
                best = (a, lam)
        W, mu, sd, Ym = fit_readout(Xtr_fit, ytr, vocab, lam=best[1])
    else:
        if brain.device != "cuda":
            raise ValueError("--readout softmax requires --device cuda")
        W, mu, sd, Ym = fit_softmax_readout_gpu(
            Xtr_fit, ytr, vocab, l2=args.softmax_l2, epochs=args.softmax_epochs,
            batch_size=args.softmax_batch, lr=args.softmax_lr, seed=3)
        best = (float("nan"), args.softmax_l2)
    if selected is not None:
        W, mu, sd, Ym = expand_selected_readout(W, mu, sd, Ym, selected, Xtr.shape[1])
    acc_tr = accuracy(Xtr, ytr, W, mu, sd, Ym)
    acc_va = accuracy(Xva, yva, W, mu, sd, Ym)

    logv = ((Xva - mu) / sd) @ W + Ym
    temp = fit_temperature(logv, yva)
    bpc_fly = bpc(softmax_rows(logv / temp), yva)
    bpc_t1 = bpc(softmax_rows(logv), yva)
    n = min(len(s) for s in va_streams)
    prv = np.stack([s[:n - 1] for s in va_streams], axis=1).reshape(-1)
    uni = (np.bincount(tr, minlength=vocab) + 1.0)
    uni /= uni.sum()
    big = np.ones((vocab, vocab))
    np.add.at(big, (tr[:-1], tr[1:]), 1.0)
    big /= big.sum(1, keepdims=True)
    p_uni = uni[yva]
    p_bi = big[prv, yva]
    pred_uni = np.full_like(yva, int(uni.argmax()))
    pred_bi = big.argmax(1)[prv]
    metrics = {
        "val_acc": acc_va, "train_acc": acc_tr, "bpc": bpc_fly, "bpc_t1": bpc_t1,
        "bpc_unigram": bpc(p_uni, yva), "bpc_bigram": bpc(p_bi, yva),
        "acc_unigram": float((pred_uni == yva).mean()),
        "acc_bigram": float((pred_bi == yva).mean()),
        "lam": best[1], "temp": temp, "samples": int(len(Xtr)),
        "seconds": time.time() - t0,
        "config": vars(args),
    }
    progress.emit("train_lm", str(args.out or "run"), status="done", metrics=metrics,
                  time=metrics["seconds"])
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2),
          flush=True)

    out = Path(args.out) if args.out else Path(__file__).parent / "models" / "readout_last.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, W=W, mu=mu, sd=sd, Ym=Ym, temp=temp, chars=np.array(chars),
             feature_neurons=np.asarray(dns, np.int64),
             probe_scores=(probe_scores if probe_scores is not None else np.empty(0, np.float32)),
             input_path=np.array(args.input_path), memory_readout=np.array(bool(args.memory_readout)),
             gain=np.array(float(brain.gain), np.float32), tonic=np.array(float(brain.tonic), np.float32),
             trace_tau=np.array(float(args.trace_tau), np.float32),
             window=np.array(int(args.window), np.int32), gamma=np.array(float(args.gamma), np.float32),
             active=np.array(int(args.active), np.int32), sustain=np.array(float(args.sustain), np.float32),
             features=np.array(args.features), fast_cuda_trace=np.array(bool(args.fast_cuda_trace)),
             k_steps=np.array(int(args.k), np.int32), skip_order=np.array(int(skip_order), np.int32),
             skip_buckets=np.array(int(args.skip_buckets), np.int32))
    with open(out.with_suffix(".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"model saved: {out}", flush=True)

    bigram = None
    if args.bigram_mix > 0:
        bigram = np.ones((vocab, vocab), np.float64)
        np.add.at(bigram, (tr[:-1], tr[1:]), 1.0)
        bigram /= bigram.sum(1, keepdims=True)
    if args.gen:
        text_out = generate(brain, enc, dns, chars, mu, sd, W, Ym, temp, va[-120:],
                            args.k, args.sustain, n=args.gen, feature_mode=args.features,
                            decode=args.decode, top_k=args.top_k,
                            bigram=bigram, bigram_mix=args.bigram_mix,
                            skip_bigram=args.skip_bigram, skip_order=skip_order,
                            skip_buckets=args.skip_buckets, trace_tau=args.trace_tau)
        print("generated sample:\n" + "-" * 60, flush=True)
        print(text_out, flush=True)
        print("-" * 60, flush=True)


if __name__ == "__main__":
    main()
