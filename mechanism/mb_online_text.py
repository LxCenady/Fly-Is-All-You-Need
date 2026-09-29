"""Online MB memory on real text (GPU).  Criterion 2 of the online-learning pilot: does a
content-taught mushroom body add information beyond a Kneser-Ney 5-gram?

The simulation reads TinyShakespeare (segment OFFSET, 25k characters: 20k train + 5k
validation) with plasticity on and the past-trace write rule of mb_online.py.  The teacher
pulses the DAN type of the current character's class (content) or of a random class
(random).  65 characters -> K classes (calibrated DAN types), assigned round-robin in order of
frequency so that classes are balanced.  Per character the script saves the memory current
read by the current KCs (97 MBONs) and its least-squares decomposition into the classes'
delivered gate patterns (K coefficients): these are the "memory features" for the readout.
Usage: mb_online_text.py OUT.npz OFFSET {content|random} [seed]
"""
import sys
import time

import numpy as np

from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_core as C  # noqa: E402
import mb_online as O  # noqa: E402
import mb_persistent_memory_probe as pmp  # noqa: E402


def main(out, offset, cond, seed=0):
    args, chars = C.protocol_args()
    st = C.build(args)
    groups, G, ids = O.pick_groups(st)
    del st
    strength = O.calibrate(groups, ids)
    groups = [g for g in groups if strength[g] >= 1.0]
    K = len(groups)
    raw =open(C.CORPUS, encoding="utf-8").read()
    seq = raw[offset: offset + 25000]
    freq = {c: raw[:1_000_000].count(c) for c in chars}
    order = sorted(chars, key=lambda c: -freq[c])
    cls_of = {c: i % K for i, c in enumerate(order)}

    args.active, args.drive, args.probe_drive = 160, 1.5, 3.75
    st = C.build(args); st["enc"].drive = args.drive
    C.reset(st, args)
    p = st["plastic"]; p.update = O.past_trace_update(p)
    rng = np.random.default_rng(seed)
    gemp = np.zeros((K, len(p.mbon_ids)))
    I_all = np.zeros((len(seq), len(p.mbon_ids)), np.float32)
    coef_all = np.zeros((len(seq), K), np.float32)
    t0 = time.time()
    for t, ch in enumerate(seq):
        k = cls_of[ch] if cond == "content" else int(rng.integers(K))
        st["pam_ids"] = ids[groups[k]]
        pmp._advance_token(st, args, chars.index(ch), allow_plastic=True, pulse_dan=True)
        if not gemp[k].any():
            gemp[k] = p.xp.asnumpy(p.last_gate)
        I = np.asarray(st["_last_w0_current"], np.float64)
        I_all[t] = I
        known = np.flatnonzero(np.abs(gemp).sum(1) > 0)
        if len(known) and np.abs(I).max() > 0:
            coef_all[t, known] = np.linalg.lstsq(gemp[known].T, I, rcond=None)[0]
        if (t + 1) % 1000 == 0:
            print(f"  {t + 1}/{len(seq)} {(t + 1) / (time.time() - t0):.1f} tok/s", flush=True)
    np.savez(out, I=I_all, coef=coef_all, cls=np.array([cls_of[c] for c in seq]),
             groups=np.array(groups), offset=offset, cond=cond)
    print("saved", out, flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), a[3], int(a[4]) if len(a) > 4 else 0)
