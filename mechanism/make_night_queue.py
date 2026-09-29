"""Build tonight's queue (order 1 -> 3 -> 2) for queue_runner.py."""
import json
import os
from pathlib import Path

import paths
M = str(paths.MECH)
O = str(paths.OUT / "night")
Path(O).mkdir(parents=True, exist_ok=True)
F = "kc,mbon_v,mbon_spk,central"
CONFIGS = {"s160": ("160", "1.5"), "s192": ("192", "1.5"), "d512": ("512", "1.0")}
OFFSETS = [0, 300000, 600000]
SEEDS = [-1, 11, 23]            # -1 = default codes (encoder seed 3)
q = []

# ---- 1. LM robustness: 3 configs x 3 corpus segments x 3 code seeds, 20k/5k ----
caches = {k: [] for k in CONFIGS}
for off in OFFSETS:
    for seed in SEEDS:
        for name, (act, sc) in CONFIGS.items():
            tag = f"{name}_o{off}_c{seed}"
            cache = os.path.join(O, f"{tag}.npz")
            caches[name].append(f"{cache}:{off}:20000:5000")
            q.append({"name": f"lm1_{tag}", "args": [
                os.path.join(M, f"lm_mech.py"), "--out", os.path.join(O, f"{tag}.json"), "--train", "20000", "--val", "5000",
                "--active", act, "--scale", sc, "--mode", "frozen", "--features", F, "--l2", "1e-2",
                "--offset", str(off), "--code-seed", str(seed), "--cache", cache]})
q.append({"name": "barrier_1", "barrier": True})
for name, specs in caches.items():
    q.append({"name": f"lm1_select_{name}", "args": [os.path.join(M, f"lm_select.py"), os.path.join(O, f"select_{name}.json")] + specs})

# ---- 3. mechanism reinforcement ----
conds = "profile_type_side:3,profile_type_side:4,profile_type_side:5,kc_perm:3,kc_perm:4," \
        "profile_type:3,profile_type:4,pn_perm:3,pn_perm:4"
for pt in ("160:1.5", "192:1.5"):
    for code in ("random", "glom"):
        q.append({"name": f"mech3_route_{pt.split(':')[0]}_{code}", "args": [
            os.path.join(M, f"m3_kcmbon.py"), os.path.join(O, f"mech_route_{pt.split(':')[0]}_{code}.json"), pt, conds, code]})
q.append({"name": "mech3_m2_12pairs", "args": [
    os.path.join(M, f"m2_current.py"), os.path.join(O, f"mech_m2_12pairs.json"),
    "ac,bd,eg,fh,ik,jl,mo,np,qs,rt,uw,vx", "160:1.5,192:1.5,512:1.0"]})

# ---- 2. larger LM: 100k/20k ----
big = []
for name in ("s160", "d512"):
    act, sc = CONFIGS[name]
    cache = os.path.join(O, f"{name}_100k.npz")
    big.append(f"{cache}:0:100000:20000")
    q.append({"name": f"lm2_{name}_100k", "args": [
        os.path.join(M, f"lm_mech.py"), "--out", os.path.join(O, f"{name}_100k.json"), "--train", "100000", "--val", "20000",
        "--active", act, "--scale", sc, "--mode", "frozen", "--features", F, "--l2", "1e-2",
        "--cache", cache]})
q.append({"name": "barrier_2", "barrier": True})
for spec in big:
    q.append({"name": f"lm2_select_{Path(spec.split(':')[0] + ':' + spec.split(':')[1]).stem}",
              "args": [os.path.join(M, f"lm_select.py"), os.path.join(O, f"select_{Path(spec.rsplit(':', 3)[0]).stem}.json"), spec]})

Path(os.path.join(O, f"queue.json")).write_text(json.dumps(q, indent=1), encoding="utf-8")
print(len(q), "jobs")
for j in q:
    print(" ", j["name"])
