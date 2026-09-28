"""M1: does the memory effect ride on the synaptic state m or the fast state s?

Design (fixed before looking at any M1 data)
  pairs      (a, c) and (b, d): two write programs of equal length (32 tokens,
             DAN pulse each token), differing only in the written item
  gaps       G in {0, 2, 8, 32} silent tokens between write and probe
  probes     z (neutral cue) and both written items
  states     at the pre-probe instant of each natural episode:
               S_X      write X with plasticity          -> (m_X, s_X)
               S_X0     write X with plasticity off      -> (m_0, s_X0)
             s_zero  = silent reset state (artificial; reported separately)
  cells      r(m, s) for m in {m_X, m_Y, m_0} x s in {s_X, s_Y, s_X0, s_zero}
  levels     v_pre_mbon (97 x settle 6), v_pre_central (256 x 6)

Statistics (per pair, gap, probe, level)
  D_nat  = |r(mX,sX) - r(mY,sY)|                natural content difference
  D_m    = mean_s |r(mX,s) - r(mY,s)|, s in {sX, sY}     swap only m
  D_s    = mean_m |r(m,sX) - r(m,sY)|, m in {mX, mY}     swap only s
  I      = |r(mX,sX) - r(mX,sY) - r(mY,sX) + r(mY,sY)|   interaction
  D_m|0  = |r(mX,sX0) - r(mY,sX0)|  content via m on a common no-write fast state
  D_m|z  = |r(mX,sz)  - r(mY,sz)|   same on the artificial silent state
  W_X    = |r(mX,sX) - r(m0,sX)|    write effect (memory present vs absent)
  floor  = max natural-replay difference measured in m1_validate (per level)

Pre-stated reading (ratios to D_nat; "~0" means <= 10 x floor):
  m carries content   : D_m/D_nat ~ 1 and D_s ~ 0 and D_m|0 ~ D_m
  s carries content   : D_s/D_nat ~ 1 and D_m ~ 0
  both / interaction  : otherwise; I reported as is
  Result restricted to this model, this arm (eta .02, tau 16 s, noise 0),
  these two pairs.  No claim about content generality or real flies.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402

PAIRS = [("a", "c"), ("b", "d")]
GAPS = [0, 2, 8, 32]
LEVELS = ("mbon", "central")


def nrm(x):
    return float(np.linalg.norm(x))


def main(out_path: str, active: int | None = None, scale: float | None = None):
    t0 = time.time()
    args, chars = C.protocol_args()
    if active is not None:
        args.active, args.drive, args.probe_drive = int(active), 1.0 * scale, 2.5 * scale
    st = C.build(args)
    st["enc"].drive = args.drive
    tid = {ch: chars.index(ch) for ch in "abcdz"}
    out = {"design": __doc__, "args": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                        for k, v in vars(args).items()},
           "cells": []}

    def episode(item, plastic, g):
        C.reset(st, args)
        C.write_item(st, args, tid[item], plastic=plastic)
        C.gap(st, args, g)
        return C.snapshot(st)

    for x, y in PAIRS:
        for g in GAPS:
            S = {"X": episode(x, True, g), "Y": episode(y, True, g),
                 "X0": episode(x, False, g), "Y0": episode(y, False, g)}
            ms = {"mX": S["X"], "mY": S["Y"], "m0": "zero"}
            ss = {"sX": S["X"], "sY": S["Y"], "sX0": S["X0"], "sz": "zero"}
            dist = {"X_vs_Y": C.state_distance(S["X"], S["Y"]),
                    "X_vs_X0": C.state_distance(S["X"], S["X0"])}
            for pch in (x, y, "z"):
                R = {}
                for mk, mv in ms.items():
                    for sk, sv in ss.items():
                        C.restore(st, "zero", None)      # clear, then install
                        C.restore(st, sv, mv)
                        R[(mk, sk)] = C.probe(st, args, tid[pch])
                cell = {"pair": [x, y], "gap": g, "probe": pch,
                        "state_distance": dist, "stats": {}, "events": {}}
                for lv in LEVELS:
                    r = {k: v[lv] for k, v in R.items()}
                    s = {
                        "D_nat": nrm(r[("mX", "sX")] - r[("mY", "sY")]),
                        "D_m": 0.5 * (nrm(r[("mX", "sX")] - r[("mY", "sX")]) +
                                      nrm(r[("mX", "sY")] - r[("mY", "sY")])),
                        "D_s": 0.5 * (nrm(r[("mX", "sX")] - r[("mX", "sY")]) +
                                      nrm(r[("mY", "sX")] - r[("mY", "sY")])),
                        "I": nrm(r[("mX", "sX")] - r[("mX", "sY")] -
                                 r[("mY", "sX")] + r[("mY", "sY")]),
                        "D_m_on_sX0": nrm(r[("mX", "sX0")] - r[("mY", "sX0")]),
                        "D_m_on_szero": nrm(r[("mX", "sz")] - r[("mY", "sz")]),
                        "D_s_with_m0": nrm(r[("m0", "sX")] - r[("m0", "sY")]),
                        "W_X": nrm(r[("mX", "sX")] - r[("m0", "sX")]),
                        "W_Y": nrm(r[("mY", "sY")] - r[("m0", "sY")]),
                        "s_nowrite_effect": nrm(r[("mX", "sX")] - r[("mX", "sX0")]),
                        "ref_norm": nrm(r[("mX", "sX")]),
                    }
                    cell["stats"][lv] = s
                cell["events"] = {f"{mk}|{sk}": [v["mbon_events"], v["central_events"]]
                                  for (mk, sk), v in R.items()}
                out["cells"].append(cell)
            out["seconds"] = round(time.time() - t0, 1)
            Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"pair={x}{y} gap={g} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None,
         float(sys.argv[3]) if len(sys.argv) > 3 else None)
