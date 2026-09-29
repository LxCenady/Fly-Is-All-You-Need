"""Sensitivity queue for the two emergent mechanism claims (retrieval key; side+subtype routing).
Each variant reruns the real-wiring reference in the same job."""
import json
import os
from pathlib import Path

import paths
M = str(paths.MECH)
OUT = paths.OUT / "sens"; OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = {"base": [], "gain1.2": ["MB_GAIN=1.2"], "gain1.8": ["MB_GAIN=1.8"],
            "tonic0.03": ["MB_TONIC=0.03"], "tonic0.08": ["MB_TONIC=0.08"],
            "noise2": ["MB_NOISE=2"], "noise5": ["MB_NOISE=5"]}
CONDS = "real:0,profile_type_side:3,profile_type:3,kc_perm:3"
jobs = []
for v, env in VARIANTS.items():
    pre = [os.path.join(M, f"with_env.py"), *env, "--"]
    jobs.append({"name": f"sens_{v}_route160r", "args": pre + [os.path.join(M, f"m3_kcmbon.py"), str(OUT / f"route160r_{v}.json"),
                                                              "160:1.5", CONDS, "random"]})
    jobs.append({"name": f"sens_{v}_route192g", "args": pre + [os.path.join(M, f"m3_kcmbon.py"), str(OUT / f"route192g_{v}.json"),
                                                              "192:1.5", CONDS, "glom"]})
    jobs.append({"name": f"sens_{v}_m2", "args": pre + [os.path.join(M, f"m2_current.py"), str(OUT / f"m2_{v}.json"),
                                                       "ac,bd,eg,fh,ik,jl", "160:1.5,512:1.0"]})
(OUT / "queue.json").write_text(json.dumps(jobs, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(jobs), "jobs")
