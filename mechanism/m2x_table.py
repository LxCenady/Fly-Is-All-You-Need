"""Summarise the 6-pair M2 current run: per point, distribution over pairs."""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1], encoding="utf-8"))
for pt in d:
    own, oth, z, kcid_r, kcid_c, kc_r, kc_c = [], [], [], [], [], [], []
    cells = pt["cells"]
    for c in cells:
        x, y = c["pair"]
        if c["probe"] == x:
            own.append(c["I_x_norm"])
            kcid_r.append(c["scramble"]["within_mbon"]["median"][0])
            kcid_c.append(c["scramble"]["within_mbon"]["median"][1])
            kc_r.append(c["scramble"]["within_kc"]["median"][0])
            kc_c.append(c["scramble"]["within_kc"]["median"][1])
        elif c["probe"] == y:
            oth.append(c["I_x_norm"])
        else:
            z.append(c["I_x_norm"])
    own, oth, z = map(np.array, (own, oth, z))
    spec = own / ((oth + z) / 2)
    print(f"PN {pt['active']} x{pt['scale']}  pairs={len(own)}")
    print(f"  cue specificity |I_own|/mean(|I_other|,|I_z|): per pair {np.round(spec, 2).tolist()}  median {np.median(spec):.2f}")
    print(f"  own probe, KC-identity scramble (within_mbon): e_ratio {np.round(kcid_r, 2).tolist()}  cos {np.round(kcid_c, 2).tolist()}")
    print(f"  own probe, routing scramble (within_kc):       e_ratio {np.round(kc_r, 2).tolist()}  cos {np.round(kc_c, 2).tolist()}")
