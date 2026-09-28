"""Tabulate the post-fix asset A rerun next to the original asset A numbers."""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1], encoding="utf-8"))
keys = ["neural_prefix4096", "neural_all", "neural_kc", "neural_mbon",
        "weight_all", "weight_mbon", "edge_sketch8192", "edge_prefix4096"]
print("orig asset A (pre-fix): frozen neural@4096 1.1664 | bio neural@4096 1.1581 | weight 0.2283 | edge@4096 0.3562")
print(f"{'seed':>9} {'mode':>10} {'id_ok':>5} | " + " | ".join(f"{k}: cap(l0,l1-3,l4+)" for k in keys[:4]))
by = {}
for r in d["runs"]:
    f = r["fits"]
    by.setdefault(r["mode"], []).append(r)
    line = " | ".join(f"{f[k]['capacity']:.4f}({f[k]['lag0']:.3f},{f[k]['lag1_3']:.3f},{f[k]['lag4_plus']:.3f})"
                      for k in keys[:4])
    print(f"{r['seed']:>9} {r['mode']:>10} {str(r['edge_identity_ok']):>5} | {line}  act={r['activity']}")
    if r["mode"] != "frozen":
        print(" " * 28 + " | ".join(f"{k}={f[k]['capacity']:.4f}(l0 {f[k]['lag0']:.3f}, l4+ {f[k]['lag4_plus']:.3f})"
                                     for k in keys[4:]))
print()
for k in keys[:4]:
    fr = np.array([r["fits"][k]["capacity"] for r in by["frozen"]])
    bi = np.array([r["fits"][k]["capacity"] for r in by["biological"]])
    print(f"{k:>18}: frozen {fr.mean():.4f} (range {fr.min():.4f}-{fr.max():.4f})  "
          f"bio {bi.mean():.4f} (range {bi.min():.4f}-{bi.max():.4f})  paired diff bio-frozen {np.round(bi-fr,4).tolist()}")
