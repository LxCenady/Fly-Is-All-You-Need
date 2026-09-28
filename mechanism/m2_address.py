"""M2: is the memory effect carried by WHERE the write lands or by HOW MUCH is written?

Fixed design (before data):
  state      gap G = 32 tokens after a 32-token write; M1 showed the fast state
             carries no content there, so the synaptic state is the only carrier.
             Common fast state = the matched no-write episode (s_X0).
  object     dW_X = W_X - w0 on the 61,210 KC->MBON slots (actual weight change,
             not modulation, because w0 differs across edges)
  scrambles  sigma permutes dW positions; the multiset of dW is preserved exactly
               all          : across all edges           (destroys KC and MBON address)
               within_mbon  : among edges onto one MBON  (keeps each MBON's dW
                              distribution and sum; destroys WHICH KCs were written)
               within_kc    : among edges out of one KC  (keeps which KCs were
                              written and how much; destroys WHICH MBON/compartment)
             the same sigma is applied to m_X and m_Y; 5 draws per type
  restore    reinstall the original dW after scrambling (must return to orig)
  readout    probe responses (v_pre_mbon, v_pre_central; settle window)
             effect  e(m)   = r(m) - r(m_0)
             content c(m,m') = r(m) - r(m')
             report |e_sigma|/|e|, cos(e_sigma, e), |c_sigma|/|c|, cos(c_sigma, c)

Pre-stated reading:
  address matters   : cos(c_sigma, c) << 1 or |c_sigma|/|c| << 1, restored by restore
  amount suffices   : cos ~ 1 and magnitude ratio ~ 1 under the scramble
  within_mbon keeps content but within_kc destroys it -> MBON/compartment routing
  carries it; the reverse -> KC identity (sparse code) carries it.
Limits: this model, eta .02, tau 16 s, noise 0, pairs (a,c), (b,d).
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
TYPES = ["all", "within_mbon", "within_kc"]
DRAWS = 5
G = 32


def group_perm(groups: np.ndarray, rng) -> np.ndarray:
    """sigma with groups[sigma[i]] == groups[i]; new_dW = dW[sigma]."""
    pos = np.argsort(groups, kind="stable")
    order = np.lexsort((rng.random(len(groups)), groups))
    sigma = np.empty(len(groups), np.int64)
    sigma[pos] = order
    assert np.array_equal(groups[sigma], groups)
    return sigma


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def main(out_path: str, active: int | None = None, scale: float | None = None):
    t0 = time.time()
    args, chars = C.protocol_args()
    if active is not None:                      # operating-point override
        args.active = int(active)
    if scale is not None:
        args.drive = 1.0 * float(scale)
        args.probe_drive = 2.5 * float(scale)
    st = C.build(args)
    st["enc"].drive = args.drive
    p, xp = st["plastic"], st["brain"].xp
    tid = {ch: chars.index(ch) for ch in "abcdz"}
    w0 = p.w0_gpu.get().astype(np.float64)
    groups = {"all": np.zeros(p.n_edges, np.int64),
              "within_mbon": p.edge_mbon_slot.astype(np.int64),
              "within_kc": p.edge_kc_slot.astype(np.int64)}
    out = {"design": __doc__, "active": int(args.active), "drive": float(args.drive),
           "probe_drive": float(args.probe_drive),
           "w0": {"min": float(w0.min()), "max": float(w0.max()),
                                     "n_nonpositive": int((w0 <= 0).sum())},
           "cells": []}

    def episode(item, plastic):
        C.reset(st, args)
        C.write_item(st, args, tid[item], plastic=plastic)
        C.gap(st, args, G)
        return C.snapshot(st)

    def slow_from_dw(dw):
        w = (w0 + dw).astype(np.float32)
        mod = np.where(w0 != 0, dw / np.where(w0 != 0, w0, 1), 0).astype(np.float32)
        return {"modulation": xp.asarray(mod), "w_slots": xp.asarray(w)}

    def resp(fast, slow, probe_id):
        C.restore(st, fast, slow)
        r = C.probe(st, args, probe_id)
        r["kc_active"] = st["_last_kc_counts"] > 0
        return r

    for x, y in PAIRS:
        SX, SY, S0 = episode(x, True), episode(y, True), episode(x, False)
        dwx = SX["w_slots"].get().astype(np.float64) - w0
        dwy = SY["w_slots"].get().astype(np.float64) - w0
        kc_written_x = np.zeros(len(p.kc_ids), bool)
        kc_written_x[p.edge_kc_slot[np.abs(dwx) > 0]] = True
        for pch in (x, y, "z"):
            pid = tid[pch]
            base = {k: resp(S0, s, pid) for k, s in
                    (("X", SX), ("Y", SY), ("0", "zero"))}
            again = resp(S0, SX, pid)                     # replay of orig
            active = base["0"]["kc_active"]
            cell = {"pair": [x, y], "probe": pch, "gap": G,
                    "kc": {"written_by_X": int(kc_written_x.sum()),
                           "probe_active": int(active.sum()),
                           "overlap": int((kc_written_x & active).sum())},
                    "orig": {}, "scramble": {}}
            for lv in ("mbon", "central"):
                e = base["X"][lv] - base["0"][lv]
                c = base["X"][lv] - base["Y"][lv]
                cell["orig"][lv] = {"e": float(np.linalg.norm(e)),
                                    "c": float(np.linalg.norm(c)),
                                    "replay_maxabs": float(np.max(np.abs(again[lv] - base["X"][lv])))}
            for typ in TYPES:
                rows = []
                for d in range(DRAWS):
                    rng = np.random.default_rng(1000 * d + TYPES.index(typ))
                    sg = group_perm(groups[typ], rng)
                    rx = resp(S0, slow_from_dw(dwx[sg]), pid)
                    ry = resp(S0, slow_from_dw(dwy[sg]), pid)
                    row = {"draw": d}
                    for lv in ("mbon", "central"):
                        e = base["X"][lv] - base["0"][lv]
                        c = base["X"][lv] - base["Y"][lv]
                        es = rx[lv] - base["0"][lv]
                        cs = rx[lv] - ry[lv]
                        row[lv] = {"e_ratio": float(np.linalg.norm(es) / max(np.linalg.norm(e), 1e-30)),
                                   "e_cos": cos(es, e),
                                   "c_ratio": float(np.linalg.norm(cs) / max(np.linalg.norm(c), 1e-30)),
                                   "c_cos": cos(cs, c)}
                    rows.append(row)
                rr = resp(S0, SX, pid)                    # restore after scrambles
                cell["scramble"][typ] = {
                    "draws": rows,
                    "restore_maxabs": {lv: float(np.max(np.abs(rr[lv] - base["X"][lv])))
                                       for lv in ("mbon", "central")}}
            out["cells"].append(cell)
            Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"pair={x}{y} probe={pch} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1],
         int(sys.argv[2]) if len(sys.argv) > 2 else None,
         float(sys.argv[3]) if len(sys.argv) > 3 else None)
