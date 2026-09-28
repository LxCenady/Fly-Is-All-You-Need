"""Summarise M2: median over 5 draws of e/c ratios and cosines."""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("w0:", d["w0"])
for lv in ("mbon", "central"):
    print(f"\n== {lv} ==  cols: |e| |c| replay | per type: e_ratio e_cos c_ratio c_cos (median of 5) restore")
    for cell in d["cells"]:
        o = cell["orig"][lv]
        head = (f"{''.join(cell['pair'])} p={cell['probe']} kc(w={cell['kc']['written_by_X']},"
                f"act={cell['kc']['probe_active']},ov={cell['kc']['overlap']}) "
                f"|e|={o['e']:.3e} |c|={o['c']:.3e} rep={o['replay_maxabs']:.1e}")
        parts = []
        for typ, s in cell["scramble"].items():
            m = {k: np.median([r[lv][k] for r in s["draws"]]) for k in ("e_ratio", "e_cos", "c_ratio", "c_cos")}
            parts.append(f"{typ}: {m['e_ratio']:.2f} {m['e_cos']:+.2f} {m['c_ratio']:.2f} {m['c_cos']:+.2f} "
                         f"rs={s['restore_maxabs'][lv]:.0e}")
        print(head + "\n     " + " | ".join(parts))
