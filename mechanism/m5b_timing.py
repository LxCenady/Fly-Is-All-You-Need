"""Is downstream change explained by MBON spike count OR timing differences?

For each (point, bias, pair, probe): MBON rasters (12 steps x 97) for m_X and
m_0 on the common no-write fast state, central response change W_central.
Classify each cell:
  count_diff   total MBON spike counts differ
  timing_only  counts equal but rasters differ (a spike moved step or neuron)
  identical    rasters identical
and report W_central per class (floor ~1e-7).
Stated reading: if W_central > 1e-4 occurs only when rasters differ, downstream
transmission is fully spike-gated.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402
from m5_downstream import _state  # noqa: E402  (installs the probe-bias hook)

POINTS = [(160, 1.5), (192, 1.5), (224, 1.5), (512, 1.0)]
BIAS = [0.0, 0.02, 0.05, 0.08]
PAIRS = [("a", "c"), ("b", "d"), ("e", "g"), ("f", "h")]

out, t0 = [], time.time()
for active, scale in POINTS:
    args, chars = C.protocol_args()
    args.active, args.drive, args.probe_drive = active, 1.0 * scale, 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    _state["ids"] = st["brain"].xp.asarray(st["mbon"])
    tid = {c: chars.index(c) for c in "abcdefghz"}
    for x, y in PAIRS:
        C.reset(st, args); C.write_item(st, args, tid[x], plastic=True); C.gap(st, args, 32)
        SX = C.snapshot(st)
        C.reset(st, args); C.write_item(st, args, tid[x], plastic=False); C.gap(st, args, 32)
        S0 = C.snapshot(st)
        for pch in (x, "z"):
            for b in BIAS:
                _state["bias"] = b
                R = {}
                for k, slow in (("X", SX), ("0", "zero")):
                    C.restore(st, S0, slow)
                    R[k] = C.probe(st, args, tid[pch])
                _state["bias"] = 0.0
                rx, r0 = R["X"]["mbon_raster"], R["0"]["mbon_raster"]
                cls = ("count_diff" if rx.sum() != r0.sum() else
                       "timing_only" if not np.array_equal(rx, r0) else "identical")
                out.append({"active": active, "bias": b, "pair": x + y, "probe": pch, "class": cls,
                            "mbon_spikes": [float(rx.sum()), float(r0.sum())],
                            "raster_mismatch": int(np.abs(rx - r0).sum()),
                            "W_mbon": float(np.linalg.norm(R["X"]["mbon"] - R["0"]["mbon"])),
                            "W_central": float(np.linalg.norm(R["X"]["central"] - R["0"]["central"]))})
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"done {active} {time.time() - t0:.0f}s", flush=True)
    del st

for cls in ("count_diff", "timing_only", "identical"):
    w = np.array([r["W_central"] for r in out if r["class"] == cls])
    if len(w):
        print(f"{cls:>12}: n={len(w):3d}  W_central>1e-4 in {int((w > 1e-4).sum())}  "
              f"median {np.median(w):.2e}  max {w.max():.2e}")
