"""Post-fix memory curve: does readout forgetting just follow the preset synaptic decay?

The old persistent-memory line (tau16 / 38.3% at 15.36 s) predates the CSR fix,
and 38.3% = exp(-15.36/16) is the preset decay itself.  This re-measures the
curve with the validated snapshot interface and separates:

  mod(t)   |m_X(t)|, |m_X(t) - m_Y(t)|            synaptic state (expected exp(-t/tau))
  W(t)     |r(m_X) - r(m_0)|                      write effect at the readout
  D(t)     |r(m_X) - r(m_Y)|                      content difference at the readout
  ratio    D(t)/D(t_ref) vs mod-diff(t)/mod-diff(t_ref): linear transfer => equal

Arms: weight_tau in {4, 16, 64} s and 'no decay' (tau = 1e12 s, a mechanism
intervention, not physiology).  eta 0.02, noise 0.  Pairs (a,c), (b,d).
One long silent gap per episode; at each checkpoint: snapshot -> probe ->
restore snapshot -> continue (restore validated in m1_validate).
The m_0 reference is the same episode with plasticity off (same fast state
trajectory up to ULP), so W isolates the synaptic write.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402

CHECK = [8, 16, 32, 64, 128, 256, 512, 1024]     # tokens after write (0.12 s each)
TAUS = [4.0, 16.0, 64.0, 1e12]
PAIRS = [("a", "c"), ("b", "d")]


def main(out_path: str, active: int | None = None, scale: float | None = None):
    t0 = time.time()
    out = {"design": __doc__, "checkpoints_tokens": CHECK, "token_seconds": 0.12,
           "active": active, "scale": scale, "runs": []}
    for tau in TAUS:
        args, chars = C.protocol_args(weight_tau=tau)
        if active is not None:
            args.active, args.drive, args.probe_drive = int(active), 1.0 * scale, 2.5 * scale
        st = C.build(args)
        st["enc"].drive = args.drive
        p = st["plastic"]
        kslot, mslot = p.edge_kc_slot.astype(np.int64), p.edge_mbon_slot.astype(np.int64)
        w0 = p.w0_gpu.get().astype(np.float64)

        def current(dw, kc):
            return np.bincount(mslot, weights=dw * kc[kslot] / args.k, minlength=len(p.mbon_ids))
        tid = {ch: chars.index(ch) for ch in "abcdz"}

        def trajectory(item, plastic):
            C.reset(st, args)
            C.write_item(st, args, tid[item], plastic=plastic)
            snaps, done = {}, 0
            for g in CHECK:
                C.gap(st, args, g - done)
                done = g
                snaps[g] = C.snapshot(st)
            return snaps

        for x, y in PAIRS:
            SX, SY, S0 = trajectory(x, True), trajectory(y, True), trajectory(x, False)
            for g in CHECK:
                row = {"tau": tau, "pair": [x, y], "gap_tokens": g,
                       "t_s": round(g * 0.12, 3)}
                mx, my = SX[g]["modulation"].get(), SY[g]["modulation"].get()
                row["mod_norm_X"] = float(np.linalg.norm(mx))
                row["mod_diff"] = float(np.linalg.norm(mx - my))
                row["fast_diff_X_vs_0"] = float(np.linalg.norm(
                    (SX[g]["v"] - S0[g]["v"]).get()))
                for pch in (x, y, "z"):
                    R = {}
                    for key, slow in (("X", SX[g]), ("Y", SY[g]), ("0", "zero")):
                        C.restore(st, S0[g], slow)          # common no-write fast state
                        R[key] = C.probe(st, args, tid[pch])
                    kc = st["_last_kc_counts"].astype(np.float64)       # probe KCs (m_0 run)
                    dwx = SX[g]["w_slots"].get().astype(np.float64) - w0
                    dwy = SY[g]["w_slots"].get().astype(np.float64) - w0
                    row[f"I|{pch}|W"] = float(np.linalg.norm(current(dwx, kc)))
                    row[f"I|{pch}|D"] = float(np.linalg.norm(current(dwx - dwy, kc)))
                    row[f"spk|{pch}|X0"] = [R["X"]["mbon_events"], R["0"]["mbon_events"]]
                    for lv in ("mbon", "central"):
                        row[f"{lv}|{pch}|W"] = float(np.linalg.norm(R["X"][lv] - R["0"][lv]))
                        row[f"{lv}|{pch}|D"] = float(np.linalg.norm(R["X"][lv] - R["Y"][lv]))
                        row[f"{lv}|{pch}|ref"] = float(np.linalg.norm(R["0"][lv]))
                out["runs"].append(row)
            out["seconds"] = round(time.time() - t0, 1)
            Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"tau={tau:g} pair={x}{y} {time.time() - t0:.0f}s", flush=True)
        del st


if __name__ == "__main__":
    if len(sys.argv) > 4:
        TAUS = [float(x) for x in sys.argv[4].split(",")]
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None,
         float(sys.argv[3]) if len(sys.argv) > 3 else None)
