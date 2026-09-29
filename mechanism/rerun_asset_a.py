"""Re-run capacity asset A on the post-fix (canonical CSR) simulator.

Asset A (mb_plasticity_vocab65_biological_formal.json, 2026-09-16) predates the
brain.py canonicalization fix (2026-09-17 22:56).  This script re-runs the same
config through the unmodified ``mb_plasticity_capacity.run_condition`` and adds:

* an edge-identity assertion after every plastic run (the pre-fix failure mode);
* readouts split by population: KC only, MBON only (all 97), KC+MBON (4161),
  plus the old 4096 prefix for comparison with asset A;
* capacity split by lag band: lag 0 / lags 1-3 (injected by the window-4
  encoder) / lags 4-12 (beyond the input window);
* several brain seeds with the same token stream, so frozen-vs-plastic
  differences can be compared with seed-to-seed spread.

Output: one JSON under --out.  Nothing in lm/ or models/ is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
LM = paths.HARNESS / "lm"
sys.path.insert(0, str(LM))
sys.path.insert(0, str(LM.parent))
import sitecustomize  # noqa: F401,E402  (CuPy include path for NVRTC)
import mb_plasticity_capacity as cap  # noqa: E402
from memory_capacity_probe import fit_discrete_capacity  # noqa: E402
from mb_plasticity import KCMBONPlasticity  # noqa: E402

ASSET_A = (paths.LEGACY / "research_results"
               r"\mb_plasticity_vocab65_biological_formal.json")

_instances: list = []


class _Recorded(KCMBONPlasticity):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _instances.append(self)


cap.KCMBONPlasticity = _Recorded


def _bands(row: dict) -> dict:
    ex = np.maximum(np.asarray(row["chance_normalized_excess_by_lag"]), 0.0)
    return {"capacity": float(ex.sum()), "lag0": float(ex[0]),
            "lag1_3": float(ex[1:4].sum()), "lag4_plus": float(ex[4:].sum()),
            "accuracy_by_lag": [round(float(a), 6) for a in row["accuracy_by_lag"]]}


def _fit(X, labels, vocab, lam, split):
    r = fit_discrete_capacity(X, labels, vocab, [X.shape[1]], lam, split)
    return _bands(next(iter(r["sizes"].values())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="20260916,20260917,20260918")
    ap.add_argument("--conditions", default="frozen,biological")
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    cfg = json.loads(ASSET_A.read_text(encoding="utf-8"))["config"]
    args = argparse.Namespace(**cfg)
    args.data = Path(cfg["data"])
    args.edge_bins = 8192
    args.eligibility_mix = 0.0
    stream_rng = np.random.default_rng(int(cfg["seed"]))       # asset A stream
    stream = np.asarray(stream_rng.integers(args.vocab, size=args.chars), np.int64)
    labels = cap._labels(stream, args.max_lag)

    out = {"asset_a_config": cfg, "stream_seed": int(cfg["seed"]),
           "note": "same token stream as asset A; brain seed varied", "runs": []}
    for seed in [int(s) for s in cli.seeds.split(",")]:
        for mode in cli.conditions.split(","):
            args.seed = seed
            _instances.clear()
            t0 = time.time()
            neural, weight, edge, elig, pop = cap.run_condition(args, stream, mode)
            ident = None
            if _instances:
                p = _instances[-1]
                live = p.xp.asnumpy(p.brain._W.indices[p.edge_pos_gpu]).astype(np.int64)
                ident = bool(np.array_equal(live, p.edge_pre_ids))
                if not ident:
                    raise RuntimeError("KC->MBON edge identity drifted")
            nk = pop["kc"]
            fits = {
                "neural_kc": _fit(neural[:, :nk], labels, args.vocab, args.lam, args.split),
                "neural_mbon": _fit(neural[:, nk:], labels, args.vocab, args.lam, args.split),
                "neural_all": _fit(neural, labels, args.vocab, args.lam, args.split),
                "neural_prefix4096": _fit(neural[:, :4096], labels, args.vocab, args.lam, args.split),
            }
            if mode != "frozen":
                fits.update({
                    "weight_all": _fit(weight, labels, args.vocab, args.lam, args.split),
                    "weight_mbon": _fit(weight[:, nk:], labels, args.vocab, args.lam, args.split),
                    "edge_sketch8192": _fit(edge, labels, args.vocab, args.lam, args.split),
                    "edge_prefix4096": _fit(edge[:, :4096], labels, args.vocab, args.lam, args.split),
                })
            rates = {"kc_mean_trace": float(neural[:, :nk].mean()),
                     "mbon_mean_trace": float(neural[:, nk:].mean()),
                     "mbon_silent_frac": float((neural[:, nk:].max(0) == 0).mean())}
            out["runs"].append({"seed": seed, "mode": mode, "population": pop,
                                "edge_identity_ok": ident, "activity": rates,
                                "fits": fits, "seconds": round(time.time() - t0, 1)})
            Path(cli.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"done seed={seed} mode={mode} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
