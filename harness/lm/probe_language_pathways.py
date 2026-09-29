"""Virtual-electrode probe for language-like processing in the frozen fly brain.

The probe does not train the connectome.  It presents a character stream through
the same sparse Encoder used by the language model, records spike events from all
166k neurons, and asks which neurons are selective for the current/next character.
The resulting electrodes are then connected to their immediate parents/children
in the MaleCNS graph.  An optional small stimulation experiment estimates whether
the most selective electrodes can move the descending-neuron population.

Example (CUDA, from the ASCII junction so CuPy NVRTC can compile):
  python lm/probe_language_pathways.py --device cuda --chars 5000 --batch 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
from batchroll import _cuda_step_no_host_copy
from flylm import DETECTORS, Encoder
from flybrain import FlyBrain

ROOT = Path(__file__).parent.parent


def load_stream(path: str | Path, chars: int, batch: int, offset: int = 0):
    """Return character vocabulary, ids, and equally sized independent streams."""
    text = Path(path).read_text(encoding="utf-8")
    alphabet = sorted(set(text))
    stoi = {c: i for i, c in enumerate(alphabet)}
    ids = np.asarray([stoi[c] for c in text], np.int64)
    ids = ids[offset:offset + chars]
    streams = [np.ascontiguousarray(s) for s in np.array_split(ids, batch)]
    return alphabet, ids, streams


def _batch_inject(brain, encoder: Encoder, streams, i: int, scale: float):
    """Apply each lane's causal sparse code directly to brain.v."""
    if scale <= 0:
        return
    xp = brain.xp
    cuda = brain.device == "cuda"
    for b, stream in enumerate(streams):
        for lag in range(min(encoder.window, i + 1)):
            code = encoder.position_codes[lag][int(stream[i - lag])]
            amount = np.float32(encoder.drive * scale * encoder.gamma**lag)
            if cuda:
                brain.v[xp.asarray(code), b] += amount
            else:
                brain.v[code, b] += amount


def _events(brain):
    """Advance one step and return one NumPy spike-index array per lane."""
    B = brain.batch
    if brain.device == "cuda":
        flat = _cuda_step_no_host_copy(brain).get()
        if B == 1:
            return [flat]
        rows, cols = np.divmod(flat, B)
        order = np.argsort(cols, kind="stable")
        return list(np.split(rows[order], np.cumsum(np.bincount(cols, minlength=B))[:-1]))
    fired = brain.step()
    return [np.asarray(fired)] if B == 1 else [np.asarray(x) for x in fired]


def collect_probe(brain, encoder: Encoder, streams, vocab: int, k_steps: int,
                  sustain: float):
    """Collect current/target character-conditioned spike counts for every neuron."""
    B = len(streams)
    L = min(len(s) for s in streams)
    total = max(0, L - 1)
    n = brain.n
    current = np.zeros((vocab, n), np.int32)
    target = np.zeros((vocab, n), np.int32)
    depth = np.zeros((k_steps, n), np.int64)
    total_counts = np.zeros(n, np.int64)
    samples = np.zeros(vocab, np.int64)
    brain.reset(seed=64)
    t0 = time.time()
    for i in range(total):
        cur = np.asarray([s[i] for s in streams], np.int64)
        nxt = np.asarray([s[i + 1] for s in streams], np.int64)
        samples += np.bincount(cur, minlength=vocab)
        for step in range(k_steps):
            _batch_inject(brain, encoder, streams, i,
                          1.0 if step == 0 else sustain)
            lane_events = _events(brain)
            for b, rows in enumerate(lane_events):
                if len(rows) == 0:
                    continue
                # np.add.at intentionally preserves duplicate spikes in one step.
                np.add.at(current[cur[b]], rows, 1)
                np.add.at(target[nxt[b]], rows, 1)
                np.add.at(depth[step], rows, 1)
                np.add.at(total_counts, rows, 1)
        if (i + 1) % 250 == 0 or i == total - 1:
            rate = (i + 1) * B / max(time.time() - t0, 1e-9)
            print(f"  probe {i + 1:6d}/{total} tokens | {rate:7.1f} samples/s", flush=True)
    return current, target, total_counts, samples, depth


def selectivity(counts: np.ndarray, samples: np.ndarray, total_counts: np.ndarray):
    """ANOVA-like between-character selectivity, scaled by the neuron's activity."""
    denom = np.maximum(samples, 1)[:, None].astype(np.float64)
    means = counts.astype(np.float64) / denom
    total_samples = max(int(samples.sum()), 1)
    overall = total_counts.astype(np.float64) / total_samples
    between = ((means - overall[None, :]) ** 2 * samples[:, None]).sum(axis=0)
    # A rate-normalized score favors reproducible modulation over very loud cells.
    score = between / (overall + 1e-3)
    return score.astype(np.float32), means.astype(np.float32), overall.astype(np.float32)


def _name(arr, idx: int, default: str = "unknown") -> str:
    if arr is None:
        return default
    value = arr[idx]
    return str(value.item() if hasattr(value, "item") else value)


def metadata_summary(indices: np.ndarray, brain: FlyBrain):
    supers = Counter(_name(brain.superclass, int(i)) for i in indices)
    types = Counter(_name(brain.cell_type, int(i)) for i in indices)
    sides = Counter(_name(brain.side, int(i)) for i in indices)
    return {"superclass": dict(supers), "cell_type": dict(types), "side": dict(sides)}


def top_records(score, means, overall, brain: FlyBrain, top_n: int, label: str,
                candidate_mask: np.ndarray | None = None):
    active_mask = overall > 0
    if candidate_mask is not None:
        active_mask &= np.asarray(candidate_mask, bool)
    active = np.flatnonzero(active_mask)
    if len(active) == 0:
        return []
    take = min(top_n, len(active))
    order = active[np.argpartition(score[active], -take)[-take:]]
    order = order[np.argsort(-score[order])]
    records = []
    for idx in order:
        rates = means[:, idx]
        hi = int(np.argmax(rates))
        lo = int(np.argmin(rates))
        records.append({
            "neuron": int(idx),
            "score": float(score[idx]),
            "baseline_rate_events_per_sample": float(overall[idx]),
            "preferred_token": hi,
            "preferred_rate": float(rates[hi]),
            "min_token": lo,
            "min_rate": float(rates[lo]),
            "cell_type": _name(brain.cell_type, int(idx)),
            "superclass": _name(brain.superclass, int(idx)),
            "side": _name(brain.side, int(idx)),
            "electrode": label,
        })
    return records


def add_graph_context(records, brain: FlyBrain, data_dir: Path, target_score: np.ndarray,
                      input_channels: np.ndarray, parent_top: int = 5, child_top: int = 5):
    """Attach high-scoring direct parents/children to the electrode records."""
    try:
        W = sparse.load_npz(data_dir / "weights.npz").tocsr()
    except Exception as exc:
        print(f"  graph context unavailable: {exc}")
        return
    input_set = np.zeros(brain.n, bool)
    input_set[input_channels] = True
    for rec in records:
        idx = rec["neuron"]
        row = W.getrow(idx)
        parents = row.indices
        pdata = row.data
        porder = np.argsort(-target_score[parents]) if len(parents) else np.empty(0, np.int64)
        porder = porder[:parent_top]
        rec["input_parent_count"] = int(input_set[parents].sum())
        rec["parents"] = [
            {"neuron": int(parents[j]), "weight": float(pdata[j]),
             "score": float(target_score[parents[j]]),
             "cell_type": _name(brain.cell_type, int(parents[j])),
             "superclass": _name(brain.superclass, int(parents[j]))}
            for j in porder
        ]
        lo, hi = int(brain.indptr[idx]), int(brain.indptr[idx + 1])
        children = brain.indices[lo:hi]
        corder = np.argsort(-target_score[children]) if len(children) else np.empty(0, np.int64)
        corder = corder[:child_top]
        rec["children"] = [
            {"neuron": int(children[j]), "weight": float(brain.weights[lo + j]),
             "score": float(target_score[children[j]]),
             "cell_type": _name(brain.cell_type, int(children[j])),
             "superclass": _name(brain.superclass, int(children[j]))}
            for j in corder
        ]


def run_causal(brain, encoder, streams, dns, nodes, k_steps, sustain, drive):
    """Stimulation electrode: compare DN spikes with/without a voltage pulse."""
    B = len(streams)
    L = min(len(s) for s in streams)
    total = max(0, L - 1)
    dn_slot = np.full(brain.n, -1, np.int64)
    dn_slot[dns] = np.arange(len(dns), dtype=np.int64)

    def run(node=None):
        brain.reset(seed=777)
        out = np.zeros(len(dns), np.float64)
        for i in range(total):
            for step in range(k_steps):
                _batch_inject(brain, encoder, streams, i,
                              1.0 if step == 0 else sustain)
                if node is not None:
                    idx = brain.xp.asarray([int(node)])
                    brain.v[idx, :] += np.float32(drive)
                for rows in _events(brain):
                    slots = dn_slot[rows]
                    valid = slots >= 0
                    if valid.any():
                        np.add.at(out, slots[valid], 1.0)
        return out / max(total * B, 1)

    base = run(None)
    effects = []
    for node in nodes:
        delta = run(int(node)) - base
        order = np.argsort(-np.abs(delta))[:5]
        effects.append({
            "neuron": int(node),
            "dn_delta_spikes_per_sample": float(delta.sum()),
            "max_abs_dn_delta": float(np.abs(delta).max()) if len(delta) else 0.0,
            "top_dn": [{"neuron": int(dns[j]), "delta": float(delta[j]),
                        "cell_type": _name(brain.cell_type, int(dns[j]))}
                       for j in order],
        })
    return effects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus" / "tinyshakespeare.txt"))
    ap.add_argument("--data", default=None, help="FlyBrain data directory (defaults to FLY_DATA)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chars", type=int, default=5000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sustain", type=float, default=0.5)
    ap.add_argument("--active", type=int, default=512)
    ap.add_argument("--drive", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--position-codes", action="store_true")
    ap.add_argument("--include-visual-projection", action="store_true",
                    help="include the visual_projection input pool used by train_lm")
    ap.add_argument("--include-vnc-sensory", action="store_true")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--graph-top", type=int, default=50)
    ap.add_argument("--causal-top", type=int, default=8)
    ap.add_argument("--causal-chars", type=int, default=300)
    ap.add_argument("--causal-drive", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "models" / "research_results" / "probe_language_pathways"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data) if args.data else Path(__import__("os").environ.get("FLY_DATA", ROOT / "data"))
    chars, ids, streams = load_stream(args.corpus, args.chars, args.batch, args.offset)
    vocab = len(chars)
    brain = FlyBrain(data=data_dir, device=args.device, batch=args.batch, seed=64)
    channel_types = DETECTORS + (["visual_projection"] if args.include_visual_projection else [])
    channel_types += (["vnc_sensory"] if args.include_vnc_sensory else [])
    channels = brain.cells(channel_types)
    if args.active > len(channels):
        raise ValueError(f"--active {args.active} exceeds {len(channels)} encoder channels")
    enc = Encoder(channels, vocab, active=args.active, drive=args.drive,
                  window=args.window, gamma=args.gamma, seed=3,
                  position_codes=args.position_codes)
    dns = brain.cells(["descending_neuron"])
    print(f"brain {brain.n:,} neurons | {len(brain.indices):,} edges | {len(dns)} DNs | "
          f"channels {len(channels):,} | batch {args.batch} | device {brain.device}", flush=True)
    print(f"stream {len(ids):,} chars, vocab {vocab}, k={args.k}, window={args.window}, "
          f"gamma={args.gamma}, position_codes={args.position_codes}", flush=True)

    current, target, total_counts, samples, depth = collect_probe(
        brain, enc, streams, vocab, args.k, args.sustain)
    cur_score, cur_means, overall = selectivity(current, samples, total_counts)
    tgt_score, tgt_means, _ = selectivity(target, samples, total_counts)
    cur_records = top_records(cur_score, cur_means, overall, brain, args.top, "current")
    tgt_records = top_records(tgt_score, tgt_means, overall, brain, args.top, "target_next")
    add_graph_context(tgt_records[:args.graph_top], brain, data_dir, tgt_score, channels)
    # The raw top list is intentionally inclusive: the encoder itself is an
    # important sanity check.  For pathway discovery we also expose a filtered
    # list that excludes the three sensory/input superclasses, plus a DN-only
    # list that shows what reaches the motor/readout interface.
    input_superclasses = ("visual_projection", "sensory_ascending", "vnc_sensory")
    superclass_names = np.asarray(brain.superclass).astype(str)
    # Some metadata uses suffixes such as vnc_sensory_tbc or cb_sensory.
    # Treat every superclass containing "sensory" as an input/relay layer.
    central_mask = (superclass_names != "visual_projection") & (np.char.find(superclass_names, "sensory") < 0)
    dn_mask = np.zeros(brain.n, bool)
    dn_mask[dns] = True
    central_records = top_records(tgt_score, tgt_means, overall, brain, args.top,
                                  "target_next_central", central_mask)
    dn_records = top_records(tgt_score, tgt_means, overall, brain, min(args.top, len(dns)),
                             "target_next_dn", dn_mask)
    add_graph_context(central_records[:args.graph_top], brain, data_dir, tgt_score, channels)

    active_neurons = np.flatnonzero(total_counts > 0)
    print(f"active neurons: {len(active_neurons):,}/{brain.n:,} "
          f"({len(active_neurons) / brain.n:.2%}); active DNs: "
          f"{int((total_counts[dns] > 0).sum())}/{len(dns)}", flush=True)
    print("target-selective electrodes (top 12):")
    for rec in tgt_records[:12]:
        print(f"  n={rec['neuron']:6d} score={rec['score']:9.2f} "
              f"rate={rec['baseline_rate_events_per_sample']:.3f} "
              f"pref={rec['preferred_token']} {chars[rec['preferred_token']]!r} "
              f"{rec['superclass']}/{rec['cell_type']} {rec['side']}")
    print("target top superclass mix:", metadata_summary(
        np.asarray([r["neuron"] for r in tgt_records[:args.graph_top]], np.int64), brain)["superclass"])
    print("target top cell-type mix:", metadata_summary(
        np.asarray([r["neuron"] for r in tgt_records[:args.graph_top]], np.int64), brain)["cell_type"])
    print("central (non-input) target electrodes (top 8):")
    for rec in central_records[:8]:
        print(f"  n={rec['neuron']:6d} score={rec['score']:9.2f} "
              f"pref={rec['preferred_token']} {chars[rec['preferred_token']]!r} "
              f"{rec['superclass']}/{rec['cell_type']} {rec['side']}")
    print("DN target electrodes (top 8):")
    for rec in dn_records[:8]:
        print(f"  n={rec['neuron']:6d} score={rec['score']:9.2f} "
              f"pref={rec['preferred_token']} {chars[rec['preferred_token']]!r} "
              f"{rec['cell_type']} {rec['side']}")

    causal = []
    if args.causal_top > 0 and (central_records or tgt_records):
        # Stimulate central candidates when possible; sensory candidates are
        # useful for validating the input interface but are not a language path.
        causal_source = central_records if central_records else tgt_records
        causal_nodes = [r["neuron"] for r in causal_source[:args.causal_top]]
        causal_streams = [np.ascontiguousarray(s[:args.causal_chars]) for s in
                          np.array_split(ids[:args.causal_chars], args.batch)]
        print(f"running stimulation electrodes on {len(causal_nodes)} nodes "
              f"({args.causal_chars} chars)...", flush=True)
        causal = run_causal(brain, enc, causal_streams, dns, causal_nodes,
                            args.k, args.sustain, args.causal_drive)
        for row in causal:
            print(f"  stimulate n={row['neuron']:6d}: total DN delta/sample "
                  f"{row['dn_delta_spikes_per_sample']:+.4f}, "
                  f"max DN {row['max_abs_dn_delta']:+.4f}")

    np.savez_compressed(out.with_suffix(".npz"),
                        current_counts=current, target_counts=target,
                        total_counts=total_counts, samples=samples, depth_counts=depth,
                        input_channels=channels, descending_neurons=dns,
                        current_score=cur_score, target_score=tgt_score,
                        vocabulary=np.asarray(chars))
    report = {
        "config": vars(args) | {"data": str(data_dir), "vocab": vocab,
                                 "neurons": brain.n, "edges": int(len(brain.indices)),
                                 "descending_neurons": len(dns),
                                 "encoder_channels": len(channels)},
        "active_neurons": int(len(active_neurons)),
        "active_fraction": float(len(active_neurons) / brain.n),
        "active_descending_neurons": int((total_counts[dns] > 0).sum()),
        "current_top": cur_records,
        "target_top": tgt_records,
        "target_central_top": central_records,
        "target_dn_top": dn_records,
        "target_top_superclass": metadata_summary(
            np.asarray([r["neuron"] for r in tgt_records[:args.graph_top]], np.int64), brain)["superclass"],
        "target_top_cell_type": metadata_summary(
            np.asarray([r["neuron"] for r in tgt_records[:args.graph_top]], np.int64), brain)["cell_type"],
        "causal_stimulation": causal,
        "notes": "Counts are spike events per token boundary over k integration steps; scores use current/next-character ANOVA modulation.",
    }
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# FlyBrain language pathway probe", "", f"- active neurons: {len(active_neurons):,}/{brain.n:,} ({len(active_neurons)/brain.n:.2%})", f"- active DNs: {int((total_counts[dns] > 0).sum())}/{len(dns)}", "", "## Top next-character electrodes", ""]
    for r in tgt_records[:args.graph_top]:
        lines.append(f"- neuron {r['neuron']} — score {r['score']:.2f}, preferred `{chars[r['preferred_token']]!r}`, {r['superclass']}/{r['cell_type']} ({r['side']}), rate {r['baseline_rate_events_per_sample']:.3f}")
        if r.get("parents"):
            p = ", ".join(f"{x['neuron']}:{x['superclass']}({x['score']:.1f})" for x in r["parents"][:3])
            lines.append(f"  - upstream: {p}; input parents={r.get('input_parent_count', 0)}")
        if r.get("children"):
            c = ", ".join(f"{x['neuron']}:{x['superclass']}({x['score']:.1f})" for x in r["children"][:3])
            lines.append(f"  - downstream: {c}")
    lines += ["", "## Central (non-input) electrodes", ""]
    for r in central_records[:args.graph_top]:
        lines.append(f"- neuron {r['neuron']} — score {r['score']:.2f}, preferred `{chars[r['preferred_token']]!r}`, {r['superclass']}/{r['cell_type']} ({r['side']})")
        if r.get("parents"):
            p = ", ".join(f"{x['neuron']}:{x['superclass']}({x['score']:.1f})" for x in r["parents"][:3])
            lines.append(f"  - upstream: {p}; input parents={r.get('input_parent_count', 0)}")
        if r.get("children"):
            c = ", ".join(f"{x['neuron']}:{x['superclass']}({x['score']:.1f})" for x in r["children"][:3])
            lines.append(f"  - downstream: {c}")
    if causal:
        lines += ["", "## Stimulation summary", ""]
        for r in causal:
            lines.append(f"- neuron {r['neuron']}: DN delta/sample {r['dn_delta_spikes_per_sample']:+.4f}, max single-DN delta {r['max_abs_dn_delta']:+.4f}")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out.with_suffix('.npz')}")
    print(f"saved {out.with_suffix('.json')}")
    print(f"saved {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
