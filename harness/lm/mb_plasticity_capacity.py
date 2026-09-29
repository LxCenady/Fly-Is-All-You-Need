"""Compare frozen and local-plastic mushroom-body sequence memory.

The only mutable synapses are existing KC->MBON edges (see ``mb_plasticity``).
The script measures two independent state channels:

* ``neural_activity``: KC + MBON spike traces after each token;
* ``weight_state``: compact KC/MBON summaries of the learned KC->MBON
  modulation, with no neural activity mixed in.

``biological`` mode gates writes with simulated PAM/PPL/PPM activity and the
actual DAN->MBON connectome.  ``engineering`` mode uses an explicit caller
teaching signal and is reported separately; it is not reinforcement learning.
The default batch is one on purpose: a plastic synapse state cannot be shared
between independent batch lanes without contaminating their memories.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from batchroll import _cuda_step_no_host_copy
from flybrain import FlyBrain
from flylm import Encoder
from memory_capacity_probe import fit_discrete_capacity
from mb_plasticity import KCMBONPlasticity
from train_lm import cells_matching

ROOT = Path(__file__).parent.parent


def _inject_pn(brain, encoder, stream, i: int, scale: float) -> None:
    if scale <= 0:
        return
    xp = brain.xp
    cache = getattr(brain, '_mb_position_codes_gpu', None)
    if cache is None or len(cache) != len(encoder.position_codes):
        cache = [[xp.asarray(code) for code in table]
                 for table in encoder.position_codes]
        brain._mb_position_codes_gpu = cache
    for lag in range(min(encoder.window, i + 1)):
        code = cache[lag][int(stream[i - lag])]
        amount = np.float32(encoder.drive * scale * encoder.gamma**lag)
        brain.v[code, 0] += amount


def _add_events(fired, brain, slots_gpu, state, counts=None):
    """Accumulate flat CUDA spike indices into a one-lane trace/count vector."""
    if not fired.size:
        return
    import cupyx
    rows = fired  # batch=1, so flat indices equal neuron rows
    slots = slots_gpu[rows]
    valid_slots = slots[slots >= 0]
    if not valid_slots.size:
        return
    vals = np.float32(1.0)
    if state is not None:
        cupyx.scatter_add(state, valid_slots, vals)
    if counts is not None and counts is not state:
        cupyx.scatter_add(counts, valid_slots, vals)


def _labels(stream: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(stream) - 1
    y = np.empty((n, max_lag + 1), np.int64)
    for i in range(n):
        for lag in range(max_lag + 1):
            y[i, lag] = int(stream[i - lag]) if i >= lag else -1
    return y


def _engineering_signal(stream: np.ndarray, i: int, vocab: int, kind: str,
                        code: np.ndarray) -> float:
    """Return a label-only DAN-like write signal for engineering ablations."""
    target = int(stream[i + 1])
    if kind == "current_char":
        # Causal control: write the token that has just driven the MB.  This
        # is useful for validating the KC->MBON write/read path itself; it
        # does not peek at the future label.
        return float(code[int(stream[i])])
    if kind == "next_char":
        return float(code[target])
    if kind == "prediction_error":
        # A deliberately transparent baseline predictor: repeat the current
        # character.  The future label is used only in engineering mode.
        return 1.0 if target != int(stream[i]) else -1.0
    if kind == "constant":
        return 1.0
    raise ValueError(f"unknown engineering signal {kind!r}")


def run_condition(args, stream: np.ndarray, mode: str,
                  engineering_code: np.ndarray | None = None):
    brain = FlyBrain(data=args.data, device="cuda", batch=1, seed=args.seed,
                     sensory_input=False)
    brain.gain = float(args.gain)
    brain.tonic = float(args.tonic)
    pn = cells_matching(brain, ("PN",))
    encoder = Encoder(pn, args.vocab, active=args.active, drive=args.drive,
                      window=args.window, gamma=args.gamma, seed=3,
                      position_codes=args.position_codes)
    kc = np.flatnonzero(np.char.find(np.asarray(brain.cell_type).astype(str), "KC") >= 0)
    mbon = np.flatnonzero(np.char.find(np.asarray(brain.cell_type).astype(str), "MBON") >= 0)
    dan = np.flatnonzero(
        (np.char.find(np.asarray(brain.cell_type).astype(str), "PAM") >= 0) |
        (np.char.find(np.asarray(brain.cell_type).astype(str), "PPL") >= 0) |
        (np.char.find(np.asarray(brain.cell_type).astype(str), "PPM") >= 0)
    )
    if mode == "frozen":
        plastic = None
    else:
        plastic = KCMBONPlasticity(
            brain, mode=mode, eta=args.eta,
            eligibility_tau=args.eligibility_tau,
            weight_tau=args.weight_tau,
            max_modulation=args.max_modulation,
            eligibility_mix=args.eligibility_mix)
    # Token-boundary traces are the same feature definition as the existing
    # frozen reservoir readout.  Weight state is recorded after the local write.
    step_decay = np.float32(np.exp(-brain.dt / args.trace_tau))
    kc_slot = brain.xp.full(brain.n, -1, dtype=brain.xp.int32)
    mbon_slot = brain.xp.full(brain.n, -1, dtype=brain.xp.int32)
    dan_slot = brain.xp.full(brain.n, -1, dtype=brain.xp.int32)
    kc_slot[brain.xp.asarray(kc)] = brain.xp.arange(len(kc), dtype=brain.xp.int32)
    mbon_slot[brain.xp.asarray(mbon)] = brain.xp.arange(len(mbon), dtype=brain.xp.int32)
    dan_slot[brain.xp.asarray(dan)] = brain.xp.arange(len(dan), dtype=brain.xp.int32)
    kc_trace = brain.xp.zeros(len(kc), brain.xp.float32)
    mbon_trace = brain.xp.zeros(len(mbon), brain.xp.float32)
    dan_dummy = brain.xp.zeros(len(dan), brain.xp.float32)
    kc_counts = brain.xp.zeros(len(kc), brain.xp.float32)
    dan_counts = brain.xp.zeros(len(dan), brain.xp.float32)
    n_rows = len(stream) - 1
    neural = np.empty((n_rows, len(kc) + len(mbon)), np.float32)
    weight = np.zeros((n_rows, len(kc) + len(mbon)), np.float32)
    edge_weight = np.zeros((n_rows, int(args.edge_bins)), np.float32)
    eligibility = np.zeros((n_rows, len(kc) + len(mbon)), np.float32)
    dan_spikes = 0.0
    gate_max = 0.0
    modulation_max = 0.0
    brain.reset(seed=args.seed)
    t0 = time.time()
    for i in range(n_rows):
        kc_counts.fill(0)
        dan_counts.fill(0)
        for step in range(args.k):
            _inject_pn(brain, encoder, stream, i, 1.0 if step == 0 else args.sustain)
            fired = _cuda_step_no_host_copy(brain)
            kc_trace *= step_decay
            mbon_trace *= step_decay
            _add_events(fired, brain, kc_slot, kc_trace, kc_counts)
            _add_events(fired, brain, mbon_slot, mbon_trace)
            _add_events(fired, brain, dan_slot, None, dan_counts)
        neural[i] = np.concatenate((brain.xp.asnumpy(kc_trace),
                                    brain.xp.asnumpy(mbon_trace)))
        if plastic is not None:
            signal = None
            if mode == "engineering":
                if engineering_code is None:
                    raise ValueError("engineering_code is required")
                signal = _engineering_signal(stream, i, args.vocab,
                                              args.engineering_signal,
                                              engineering_code)
            plastic.update(kc_counts, dan_counts, teaching_signal=signal)
            dan_spikes += float(brain.xp.asnumpy(dan_counts).sum())
            gate_max = max(gate_max, float(brain.xp.asnumpy(brain.xp.abs(plastic.last_gate)).max()))
            modulation_max = max(modulation_max, float(brain.xp.asnumpy(brain.xp.abs(plastic.modulation)).max()))
            kstate, mstate = plastic.state_features()
            weight[i] = np.concatenate((kstate, mstate))
            edge_weight[i] = plastic.edge_state_features(int(args.edge_bins))
            estate, emstate = plastic.eligibility_features()
            eligibility[i] = np.concatenate((estate, emstate))
        if (i + 1) % 250 == 0 or i == n_rows - 1:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  {mode:11s} {i + 1:6d}/{n_rows} tokens | {rate:7.1f} tokens/s",
                  flush=True)
    return neural, weight, edge_weight, eligibility, {"kc": int(len(kc)), "mbon": int(len(mbon)),
                            "dan": int(len(dan)), "pn": int(len(pn)),
                            "plastic_edges": int(plastic.n_edges if plastic else 0),
                            "dan_spikes": float(dan_spikes),
                            "gate_max": float(gate_max),
                            "modulation_max": float(modulation_max)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--device", choices=("cuda",), default="cuda")
    ap.add_argument("--chars", type=int, default=3000)
    ap.add_argument("--vocab", type=int, default=65)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sustain", type=float, default=0.5)
    ap.add_argument("--active", type=int, default=512)
    ap.add_argument("--drive", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--tonic", type=float, default=0.05)
    ap.add_argument("--trace-tau", type=float, default=0.1)
    ap.add_argument("--position-codes", action="store_true")
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--eligibility-tau", type=float, default=0.5)
    ap.add_argument("--weight-tau", type=float, default=8.0)
    ap.add_argument("--max-modulation", type=float, default=0.9)
    ap.add_argument("--edge-bins", type=int, default=8192,
                    help="fixed count-sketch width for individual KC→MBON state")
    ap.add_argument("--eligibility-mix", type=float, default=0.0,
                    help="fraction of KC eligibility used for delayed DAN writes")
    ap.add_argument("--engineering-signal", choices=("current_char", "next_char", "prediction_error", "constant"),
                    default="next_char")
    ap.add_argument("--conditions", default="frozen,biological,engineering")
    ap.add_argument("--max-lag", type=int, default=8)
    ap.add_argument("--sizes", default="128,256,512,1024,2048,4096")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", default="models/research_results/mb_plasticity_capacity.json")
    args = ap.parse_args()
    if args.data is None:
        import os
        args.data = Path(os.environ.get("FLY_DATA", ROOT / "data"))
    else:
        args.data = Path(args.data)
    rng = np.random.default_rng(args.seed)
    stream = np.asarray(rng.integers(args.vocab, size=args.chars), np.int64)
    labels = _labels(stream, args.max_lag)
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    unknown = set(conditions) - {"frozen", "biological", "engineering"}
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    # Fixed random +/- code makes next-character engineering writes explicit
    # while avoiding a learned model or a hidden global gradient.
    engineering_code = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=args.vocab)
    config = dict(vars(args))
    config["data"] = str(args.data)
    results = {"config": config, "conditions": {},
               "labels": {"rows": int(len(labels)), "chance": float(1.0 / args.vocab)}}
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    for mode in conditions:
        neural, weight, edge_weight, eligibility, pop = run_condition(args, stream, mode, engineering_code)
        sizes_clipped = sorted(set(max(1, min(int(x), neural.shape[1])) for x in sizes))
        fit_neural = fit_discrete_capacity(neural, labels, args.vocab,
                                           sizes_clipped, args.lam, args.split)
        fit_weight = fit_discrete_capacity(weight, labels, args.vocab,
                                           sizes_clipped, args.lam, args.split)
        edge_sizes = sorted(set(max(1, min(int(x), edge_weight.shape[1])) for x in sizes))
        fit_edge_weight = fit_discrete_capacity(edge_weight, labels, args.vocab,
                                                 edge_sizes, args.lam, args.split)
        fit_eligibility = fit_discrete_capacity(eligibility, labels, args.vocab,
                                                 sizes_clipped, args.lam, args.split)
        results["conditions"][mode] = {
            "population": pop,
            "neural_activity": fit_neural["sizes"],
            "weight_state": fit_weight["sizes"],
            "edge_weight_state": fit_edge_weight["sizes"],
            "eligibility_state": fit_eligibility["sizes"],
        }
        del neural, weight, edge_weight, eligibility
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    lines = ["# Frozen vs dopamine-gated KC→MBON memory", "",
             "All conditions use the same iid PN stream. Neural activity and KC→MBON weight state are decoded separately.", ""]
    for mode, result in results["conditions"].items():
        lines += [f"## {mode}", "", "### Neural activity", "",
                  "| neurons | " + " | ".join(f"lag {i}" for i in range(args.max_lag + 1)) + " | capacity |",
                  "|---:|" + "---:|" * (args.max_lag + 1) + "---:|"]
        for size, row in result["neural_activity"].items():
            vals = " | ".join(f"{x:.3f}" for x in row["accuracy_by_lag"])
            lines.append(f"| {size} | {vals} | {row['discrete_memory_capacity']:.2f} |")
        lines += ["", "### KC→MBON weight state", "",
                  "| neurons | " + " | ".join(f"lag {i}" for i in range(args.max_lag + 1)) + " | capacity |",
                  "|---:|" + "---:|" * (args.max_lag + 1) + "---:|"]
        for size, row in result["weight_state"].items():
            vals = " | ".join(f"{x:.3f}" for x in row["accuracy_by_lag"])
            lines.append(f"| {size} | {vals} | {row['discrete_memory_capacity']:.2f} |")
        lines += ["", "### KC→MBON edge weight state (8192-bin fixed sketch)", "",
                  "| bins | " + " | ".join(f"lag {i}" for i in range(args.max_lag + 1)) + " | capacity |",
                  "|---:|" + "---:|" * (args.max_lag + 1) + "---:|"]
        for size, row in result["edge_weight_state"].items():
            vals = " | ".join(f"{x:.3f}" for x in row["accuracy_by_lag"])
            lines.append(f"| {size} | {vals} | {row['discrete_memory_capacity']:.2f} |")
        lines += ["", "### KC eligibility (auxiliary, not persistent weight)", "",
                  "| neurons | " + " | ".join(f"lag {i}" for i in range(args.max_lag + 1)) + " | capacity |",
                  "|---:|" + "---:|" * (args.max_lag + 1) + "---:|"]
        for size, row in result["eligibility_state"].items():
            vals = " | ".join(f"{x:.3f}" for x in row["accuracy_by_lag"])
            lines.append(f"| {size} | {vals} | {row['discrete_memory_capacity']:.2f} |")
        lines.append("")
    lines += ["Frozen = no mutable synapses; biological = simulated PAM/PPL/PPM gate;",
              "engineering = explicit next-character/prediction-error DAN-like gate.",
              "The experiment is batch=1 so a single plastic synapse state is not shared between streams."]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
