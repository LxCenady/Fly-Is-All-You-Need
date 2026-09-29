"""APL lesion / feedback queue: does APL feedback set KC sparseness in this model?
apl_off zeroes APL->KC; apl_fbX scales KC->APL (total weight ~0.9 in the connectome) by X."""
import json
import os
from pathlib import Path

import paths
M = str(paths.MECH)
OUT = paths.OUT / "apl"; OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = {"base": [], "apl_off": ["MB_LESION=apl_off"], "apl_fb100": ["MB_LESION=apl_fb100"],
            "apl_fb1000": ["MB_LESION=apl_fb1000"]}
jobs = []
for v, env in VARIANTS.items():
    pre = [os.path.join(M, f"with_env.py"), *env, "--"]
    jobs.append({"name": f"apl_{v}_m2", "args": pre + [os.path.join(M, f"m2_current.py"), str(OUT / f"m2_{v}.json"),
                                                      "ac,bd,eg,fh,ik,jl", "160:1.5,512:1.0"]})
    jobs.append({"name": f"apl_{v}_route", "args": pre + [os.path.join(M, f"m3_kcmbon.py"), str(OUT / f"route160r_{v}.json"),
                                                         "160:1.5", "real:0,profile_type_side:3,kc_perm:3", "random"]})
(OUT / "queue.json").write_text(json.dumps(jobs, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(jobs), "jobs")
