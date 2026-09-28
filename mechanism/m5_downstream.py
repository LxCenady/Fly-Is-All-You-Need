"""Why does the sparse-regime memory not reach the central readout?

Hypothesis (stated before data): LIF neurons transmit only spikes; in the
sparse regime the probe leaves MBONs subthreshold (~1/97 spiking), so the
write changes MBON voltage but not MBON spikes, and nothing reaches downstream.

Test: during the probe token only, add a constant depolarising bias b to every
MBON voltage at each step (an excitability intervention, reported separately,
not physiology).  b in {0, 0.02, 0.05, 0.08, 0.12} per step (steady offset
about 5.5 b; MBON rest ~0.28, threshold 1).  For pairs (a,c), (b,d),
probes = own item and z, measure on the common no-write fast state:
  MBON spikes in the probe (m_X vs m_0), central spikes,
  W_mbon, W_central = |r(m_X) - r(m_0)|, D_central = |r(m_X) - r(m_Y)|.
Operating points: PN 160 / 192 / 224 at x1.5, and 512 x1.0.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
import mb_persistent_memory_probe as pmp  # noqa: E402

BIAS = [0.0, 0.02, 0.05, 0.08, 0.12]   # per-step; steady offset ~ b / (1 - e^-0.2) = 5.5 b
POINTS = [(160, 1.5), (192, 1.5), (224, 1.5), (512, 1.0)]
PAIRS = [("a", "c"), ("b", "d")]
_orig = pmp._cuda_step_record_pre_reset
_state = {"bias": 0.0, "ids": None}


def _biased(brain, record_indices=()):
    if _state["bias"] and _state["ids"] is not None:
        brain.v[_state["ids"], 0] += np.float32(_state["bias"])
    return _orig(brain, record_indices)


pmp._cuda_step_record_pre_reset = _biased          # used only by probe profiles


def run_point(active, scale):
    args, chars = C.protocol_args()
    args.active, args.drive, args.probe_drive = int(active), 1.0 * scale, 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    _state["ids"] = st["brain"].xp.asarray(st["mbon"])
    tid = {c: chars.index(c) for c in "abcdz"}

    def episode(item, plastic):
        C.reset(st, args)
        C.write_item(st, args, tid[item], plastic=plastic)
        C.gap(st, args, 32)
        return C.snapshot(st)

    rows = []
    for x, y in PAIRS:
        SX, SY, S0 = episode(x, True), episode(y, True), episode(x, False)
        for pch in (x, "z"):
            for b in BIAS:
                _state["bias"] = b
                R = {}
                for k, slow in (("X", SX), ("Y", SY), ("0", "zero")):
                    C.restore(st, S0, slow)
                    R[k] = C.probe(st, args, tid[pch])
                _state["bias"] = 0.0
                rows.append({
                    "pair": [x, y], "probe": pch, "bias": b,
                    "mbon_spikes": {k: R[k]["mbon_events"] for k in R},
                    "central_spikes": {k: R[k]["central_events"] for k in R},
                    "W_mbon": float(np.linalg.norm(R["X"]["mbon"] - R["0"]["mbon"])),
                    "W_central": float(np.linalg.norm(R["X"]["central"] - R["0"]["central"])),
                    "D_central": float(np.linalg.norm(R["X"]["central"] - R["Y"]["central"])),
                })
    del st
    return {"active": active, "scale": scale, "rows": rows}


if __name__ == "__main__":
    out, t0 = [], time.time()
    for a, s in POINTS:
        out.append(run_point(a, s))
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"done {a} x{s} {time.time() - t0:.0f}s", flush=True)
