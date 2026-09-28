"""M1 step 1: snapshot/restore must reproduce natural continuation.

Checks (arm A2-like: eta 0.02, weight_tau 16 s, noise 0):
  floor     natural run twice from reset          -> replay floor
  restore   snapshot at pre-probe, scramble the brain with an unrelated
            episode, restore, probe                -> must equal natural
  neg_slow  restore with ONE modulation entry perturbed by 1e-3
  neg_fast  restore with ONE voltage perturbed by 1e-3
  neg_wslot restore modulation but leave stale CSR slots (the classic bug)
  w_other   every non-plastic CSR entry keeps its hash throughout
The negative controls must exceed the floor; otherwise the check is blind.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m1_core as C  # noqa: E402


def diff(a, b):
    return {lvl: float(np.max(np.abs(a[lvl] - b[lvl]))) for lvl in ("mbon", "central")} | {
        "events_equal": bool(a["mbon_events"] == b["mbon_events"] and
                             a["central_events"] == b["central_events"])}


def main(out_path: str, gap_tokens: int = 32):
    t0 = time.time()
    args, chars = C.protocol_args()
    st = C.build(args)
    A, B, P = chars.index("a"), chars.index("c"), chars.index("z")
    w_hash0 = C.w_other_hash(st)

    def natural():
        C.reset(st, args)
        C.write_item(st, args, A)
        C.gap(st, args, gap_tokens)
        snap = C.snapshot(st)
        return snap, C.probe(st, args, P)

    snap, r_nat = natural()
    reps = [natural()[1] for _ in range(4)]
    pair = [diff(x, y) for i, x in enumerate([r_nat] + reps)
            for y in ([r_nat] + reps)[i + 1:]]
    res = {"gap_tokens": gap_tokens,
           "floor": {"mbon": max(d["mbon"] for d in pair),
                     "central": max(d["central"] for d in pair),
                     "events_equal": all(d["events_equal"] for d in pair),
                     "n_pairs": len(pair)}}

    def scramble():
        C.reset(st, args)
        C.write_item(st, args, B)
        C.gap(st, args, 5)

    rest = []
    for _ in range(3):
        scramble()
        C.restore(st, snap, snap)
        r = C.probe(st, args, P)
        rest += [diff(r, x) for x in [r_nat] + reps]
    res["restore"] = {"mbon": max(d["mbon"] for d in rest),
                      "central": max(d["central"] for d in rest),
                      "events_equal": all(d["events_equal"] for d in rest),
                      "n_pairs": len(rest)}

    scramble()
    C.restore(st, snap, snap)
    p = st["plastic"]
    i = int(np.argmax(np.abs(p.modulation.get())))
    p.modulation[i] += np.float32(1e-3)
    st["brain"]._W.data[p.edge_pos_gpu] = p.w0_gpu * (np.float32(1) + p.modulation)
    res["neg_slow"] = diff(r_nat, C.probe(st, args, P)) | {"edge": i}

    scramble()
    C.restore(st, snap, snap)
    ids = st["brain"].xp.asarray(st["mbon"])
    st["brain"].v[ids, 0] += np.float32(1e-3)          # all 97 MBON voltages
    res["neg_fast"] = diff(r_nat, C.probe(st, args, P))

    scramble()
    stale = st["brain"]._W.data[p.edge_pos_gpu].copy()
    C.restore(st, snap, None)
    p.modulation[...] = snap["modulation"]           # modulation only
    st["brain"]._W.data[p.edge_pos_gpu] = stale      # CSR left from scramble
    res["neg_wslot"] = diff(r_nat, C.probe(st, args, P))

    res["w_other_hash_constant"] = bool(C.w_other_hash(st) == w_hash0)
    fl = max(res["floor"]["mbon"], res["floor"]["central"])
    rs = max(res["restore"]["mbon"], res["restore"]["central"])
    negs = {k: max(res[k]["mbon"], res[k]["central"]) for k in ("neg_slow", "neg_fast", "neg_wslot")}
    res["verdict"] = {
        # pre-set rule: restore passes if its worst diff <= 2x the worst
        # natural-replay diff (both are GPU reduction-order noise)
        "restore_to_floor_ratio": (rs / fl) if fl > 0 else None,
        "restore_within_floor": bool(rs <= 2 * fl),
        "restore_bitwise": bool(rs == 0.0),
        "negatives_detected": {k: bool(v > max(fl, rs)) for k, v in negs.items()},
        "seconds": round(time.time() - t0, 1)}
    Path(out_path).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32)
