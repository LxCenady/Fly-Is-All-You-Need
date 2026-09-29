"""Pilot: can the mushroom body learn next-character associations online, one exposure at a
time, if the dopaminergic teacher depends on content (a GRU-like write gate)?

Task.  A text made of a few made-up 5-letter words with disjoint letters, in random order,
separated by spaces, e.g. "qzkvb mplwe qzkvb ...".  Within a word, the current letter fully
determines the next one, but only after the word has been seen once.
Classes.  Each character is mapped to one of K DAN types with little overlap in the MBONs they
gate (greedy selection, |cos| < 0.5 between gate patterns); the class of a character = its DAN
type.
Write rule (past-trace).  When character t+1 is presented, its class's DAN type is pulsed.  The
update writes the KC eligibility trace of the previous characters (not the current one) into
KC->MBON synapses gated by that DAN type: "recent context -> class of what came next".  This is
the model's rule with eligibility_mix = 1 minus the same-token term.
Readout (no training).  After character t, the model's memory current I_m = sum_k w0 m_km c_k/6
with the current KC counts c is decomposed by least squares into the gate patterns actually
delivered by each class's DAN pulse (memory currents superpose linearly); predicted class of
character t+1 = the class with the largest coefficient.
Conditions.  content: teacher = class of the current character.  random: a random class each
token (same number and strength of writes, no content).  frozen: no writes.
Success criterion 1 (fixed in advance): content accuracy on repeated occurrences of a word is
above its first-occurrence accuracy and above the random-teacher condition.
Usage: mb_online.py OUT.json [n_words] [n_tokens] [seed]
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\苍蝇。\mechanism")
import m1_core as C  # noqa: E402
import mb_persistent_memory_probe as pmp  # noqa: E402


def past_trace_update(p):
    xp = p.xp

    def update(kc_counts, dan_counts=None, teaching_signal=None, write_back=True):
        kc = xp.asarray(kc_counts, dtype=xp.float32).reshape(-1) / np.float32(6.0)
        prev = p.kc_eligibility * p._elig_decay                 # trace of earlier tokens only
        p.kc_eligibility[...] = prev + kc
        gate = p._gate(dan_counts, teaching_signal)
        p.last_gate = gate.copy()
        p.modulation *= p._weight_decay
        p.modulation += np.float32(p.eta) * gate[p.edge_mbon_slot_gpu] * prev[p.edge_kc_slot_gpu]
        p.modulation = xp.clip(p.modulation, np.float32(-p.max_modulation), np.float32(p.max_modulation))
        if write_back:
            p.brain._W.data[p.edge_pos_gpu] = p.w0_gpu * (np.float32(1.0) + p.modulation)
    return update


def pick_groups(st, thr=0.5, min_mbons=5):
    p = st["plastic"]; A = p.xp.asnumpy(p.dan_axis)
    ct = np.asarray(st["brain"].cell_type).astype(str); dan = np.asarray(p.dan_ids)
    types = sorted(set(ct[dan]), key=lambda t: -(ct[dan] == t).sum())
    chosen, gates = [], []
    for t in types:
        g = A[:, ct[dan] == t].sum(1)
        if (np.abs(g) > 1e-9).sum() < min_mbons:
            continue
        u = g / np.linalg.norm(g)
        if all(abs(u @ v) < thr for v in gates):
            chosen.append(t); gates.append(u)
    ids = {t: dan[ct[dan] == t].astype(np.int64) for t in chosen}
    return chosen, np.array(gates), ids


def calibrate(groups, ids, active=160, scale=1.5):
    args, chars = C.protocol_args()
    args.active, args.drive, args.probe_drive = active, 1.0 * scale, 2.5 * scale
    st = C.build(args); st["enc"].drive = args.drive
    p = st["plastic"]; out = {}
    for g in groups:
        C.reset(st, args); st["pam_ids"] = ids[g]
        pmp._advance_token(st, args, chars.index("a"), allow_plastic=True, pulse_dan=True)
        out[g] = float(np.abs(p.xp.asnumpy(p.last_gate)).sum())
    del st
    return out


def make_text(chars, n_words, n_tokens, rng):
    letters = [c for c in "abcdefghijklmnopqrstuvwxyz"]
    rng.shuffle(letters)
    words = ["".join(letters[5 * i:5 * i + 5]) for i in range(n_words)]
    text, occ, pos = [], [], []
    seen = {w: 0 for w in words}
    while len(text) < n_tokens:
        w = words[rng.integers(len(words))]
        seen[w] += 1
        for j, c in enumerate(w + " "):
            text.append(c); occ.append(seen[w]); pos.append(j)
    return words, text[:n_tokens], np.array(occ[:n_tokens]), np.array(pos[:n_tokens])


def run(cond, text, cls_of, groups, G, ids, chars, active=160, scale=1.5, seed=0):
    args, _ = C.protocol_args()
    args.active, args.drive, args.probe_drive = active, 1.0 * scale, 2.5 * scale
    st = C.build(args); st["enc"].drive = args.drive
    C.reset(st, args)
    p = st["plastic"]
    p.update = past_trace_update(p)
    rng = np.random.default_rng(seed)
    K = len(groups)
    gemp = np.zeros((K, len(p.mbon_ids)))              # delivered gate pattern per class
    preds, scores_log = [], []
    for t, ch in enumerate(text):
        k = cls_of[ch] if cond == "content" else int(rng.integers(K))
        st["pam_ids"] = ids[groups[k]]
        plastic = cond != "frozen"
        pmp._advance_token(st, args, chars.index(ch), allow_plastic=plastic, pulse_dan=plastic)
        if plastic and not gemp[k].any():             # deterministic per class: record once
            gemp[k] = p.xp.asnumpy(p.last_gate)
        I = np.asarray(st["_last_w0_current"], np.float64)   # memory read by the current KCs
        known = np.flatnonzero(np.abs(gemp).sum(1) > 0)
        if len(known) and np.abs(I).max() > 0:
            coef = np.linalg.lstsq(gemp[known].T, I, rcond=None)[0]
            preds.append(int(known[np.argmax(coef)]))
        else:
            preds.append(-1)
        scores_log.append(float(np.abs(I).sum()))
    return np.array(preds), np.array(scores_log), np.abs(gemp).sum(1)


def main(out, n_words=5, n_tokens=900, seed=0):
    rng = np.random.default_rng(seed)
    args, chars = C.protocol_args()
    st = C.build(args)
    groups, G, ids = pick_groups(st)
    del st
    # Calibration (added after seed 0 failed): a DAN type whose pulse delivers a tiny gate
    # (2-cell PPL types, total gate about 0.05) gets huge least-squares coefficients from
    # noise.  Keep only types whose pulse delivers a total gate of at least 1.
    strength = calibrate(groups, ids)
    keep = [i for i, g in enumerate(groups) if strength[g] >= 1.0]
    groups, G = [groups[i] for i in keep], G[keep]
    K = len(groups)
    words, text, occ, pos = make_text(chars, n_words, n_tokens, rng)
    alphabet = sorted(set(text))
    cls_of = {c: i % K for i, c in enumerate(rng.permutation(alphabet))}
    y = np.array([cls_of[c] for c in text])
    # prediction made after token t is for token t+1; evaluate within-word transitions
    target = y[1:]; o = occ[1:]; within = (pos[1:] >= 1)          # next char is 2nd..5th letter or the space
    letters = within & (pos[1:] <= 4)                                # next char is a letter, not the space
    res = {"groups": list(map(str, groups)), "K": K, "calibration": strength, "words": words,
           "n_tokens": n_tokens, "seed": seed,
           "majority": float(np.bincount(target[within], minlength=K).max() / within.sum()),
           "majority_letters": float(np.bincount(target[letters], minlength=K).max() / letters.sum()),
           "chance_balanced": 1.0 / K}
    for cond in ("content", "random", "frozen"):
        t0 = time.time()
        pr, mag, strength = run(cond, text, cls_of, groups, G, ids, chars, seed=seed)
        pr = pr[:-1]
        r = {}
        for lab, m in (("first", within & (o == 1)), ("second", within & (o == 2)),
                       ("third_plus", within & (o >= 3))):
            r[lab] = {"n": int(m.sum()), "acc": float((pr[m] == target[m]).mean()) if m.sum() else None}
        m = within & (o >= 3)
        r["third_plus_balanced"] = float(np.mean([(pr[m][target[m] == c] == c).mean()
                                                  for c in np.unique(target[m])]))
        m = letters & (o >= 3)
        r["third_plus_letters"] = float((pr[m] == target[m]).mean())
        m = letters & (o == 1)
        r["first_letters"] = float((pr[m] == target[m]).mean())
        r["memory_current_mean"] = float(mag.mean()); r["write_strength"] = strength.round(3).tolist()
        r["seconds"] = round(time.time() - t0, 1)
        res[cond] = r
        print(cond, json.dumps(r), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], *(int(x) for x in a[2:5]))
