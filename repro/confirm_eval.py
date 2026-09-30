"""Evaluate the pre-registered confirmatory run (repro/PREREGISTRATION.md). Do not edit after
registration: the sha256 of this file is recorded in the pre-registration.

    python repro/confirm_eval.py CONFIRM_DIR > repro/CONFIRMATORY.md

CONFIRM_DIR holds the outputs of `repro/run.py repro/confirm.json CONFIRM_DIR` (mechanism, files
m2_c20x / route_c20x / teacher_g2x / down_c20x / m1x_c20x) and of the LM benches:
lm_fly_c201..203/report.json and lm_worm_c201..203/report.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import replication_stats as R  # noqa: E402


def verdict(ok):
    return "CONFIRMED" if ok else "NOT CONFIRMED"


def main(d):
    d = Path(d)
    res, _ = R.summarise(d, Path(__file__).resolve().parents[1] / "results")
    out = ["# Confirmatory run: results", "",
           "Evaluated by repro/confirm_eval.py exactly as registered in repro/PREREGISTRATION.md.", ""]

    def line(h, ok, text):
        out.append(f"- **{h}: {verdict(ok)}.** {text}")
        return ok

    # H1 (M8)
    s160, s192, s512 = res["M8_160"], res["M8_192"], res["M8_512"]
    line("H1", s160["spec"][0] > 2 and s192["spec"][0] > 2 and s160["kept"][0] < 0.6 and s192["kept"][0] < 0.6
         and s512["spec"][0] < 1.5,
         f"cue specificity 160/192/512: {s160['spec'][0]:.2f} / {s192['spec'][0]:.2f} / {s512['spec'][0]:.2f} "
         f"(need > 2, > 2, < 1.5); KC-identity shuffle keeps 160/192: {s160['kept'][0]:.2f} / {s192['kept'][0]:.2f} (need < 0.6).")
    # H2 (M10)
    a, b = res["M10_160"], res["M10_192"]
    line("H2", abs(a["d_swap"][0]) < 0.03 and abs(b["d_swap"][0]) < 0.03 and a["d_perm"][0] > 0.10 and b["d_perm"][0] > 0.10,
         f"side+subtype swap changes item cosine by {a['d_swap'][0]:+.3f} / {b['d_swap'][0]:+.3f} (need |.| < 0.03); "
         f"KC->MBON shuffle by {a['d_perm'][0]:+.3f} / {b['d_perm'][0]:+.3f} (need > +0.10).")
    # H3 (M16)
    t = res["M16"]["teacher_share"][0]
    line("H3", t > 0.6, f"teacher share of output variance {t:.2f} (need > 0.60).")
    # H4 (M19)
    p1, p0 = res["M19"]["p_central_if_mbon_changed"][0], res["M19"]["p_central_if_unchanged"][0]
    line("H4", p1 >= 0.9 and p0 <= 0.10,
         f"central changed in {p1:.2f} of cells where the MBON spike count changed (need >= 0.90) and in "
         f"{p0:.3f} where it did not (need <= 0.10).")
    # H5 (M2)
    m, mn = res["M2"]["Dm_over_Dnat_gap8plus"][0], res["M2"]["min"]
    line("H5", m >= 0.99 and mn >= 0.95, f"D_m / D_nat from 8 tokens: median {m:.4f}, minimum {mn:.4f} (need >= 0.99, >= 0.95).")

    # LM
    def bench(tag):
        reps = []
        for p in sorted(d.glob(f"lm_{tag}_c*/report.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            v = {x["name"]: x for x in r["variants"]}
            real = v["connectome"]
            cls = next(x for n, x in v.items() if n.startswith("rewire-class") and "activity-matched" in n)
            full = next((x for n, x in v.items() if n.startswith("rewire-full") and "activity-matched" in n), None)
            reps.append({"seed": p.parent.name, "ctx": r["baselines"]["context_only"], "kn5": r["baselines"]["kn5"],
                         "real": real["val_bpc"], "class": cls["val_bpc"], "full": full["val_bpc"] if full else None,
                         "k2": real["memory_span"]["all"][2], "k4": real["memory_span"]["all"][4]})
        return reps

    fly, worm = bench("fly"), bench("worm")
    for r in fly + worm:
        out.append(f"  - {r['seed']}: context {r['ctx']:.3f}, connectome {r['real']:.3f}, KN-5 {r['kn5']:.3f}, "
                   f"class-rewired (matched) {r['class']:.3f}" + (f", fully rewired (matched) {r['full']:.3f}" if r['full'] else "")
                   + f"; memory k=2 {r['k2']:.2f}, k=4 {r['k4']:.2f}")
    ok6 = len(fly) == 3 and all(r["real"] < r["ctx"] and r["kn5"] < r["real"] - 0.3 and 0.6 <= r["k2"] <= 0.95 and r["k4"] < 0.45
                                for r in fly)
    line("H6", ok6, "fly, 3 fresh codes on unseen text: connectome below context in every case, KN-5 more than 0.3 below "
                    "the connectome, memory k=2 in [0.60, 0.95] and k=4 < 0.45.")
    ok7 = len(fly) == 3 and all(abs(r["class"] - r["real"]) <= 0.05 for r in fly)
    line("H7", ok7, "fly: class-rewired connectome at matched activity within 0.05 BPC of the real one in every case.")
    ok8 = len(worm) == 3 and all(r["real"] < r["ctx"] and r["kn5"] < r["real"] - 0.3 and abs(r["class"] - r["real"]) <= 0.06
                                 and 0.5 <= r["k2"] <= 0.9 for r in worm)
    line("H8", ok8, "worm, 3 fresh codes on unseen text: connectome below context, KN-5 more than 0.3 below, class-rewired "
                    "(matched) within 0.06, memory k=2 in [0.50, 0.90].")
    out += ["", "No hypothesis was re-tested with other seeds or thresholds after these results were seen."]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
