"""Tabulate the operating-point sweep."""
import glob
import json

rows = []
for f in sorted(glob.glob(str(__import__("paths").OUT / "sweep_s*.json"))):
    rows += json.load(open(f, encoding="utf-8"))
rows.sort(key=lambda r: (r["scale"], r["active"]))
print("scale active | KC frac w2 w8 w32 probe | KC Jacc w32 probe | MBON act w32 probe | "
      "KCs written | W_mbon(a) D_mbon(a) D_mbon(z) | D_central(z)")
for r in rows:
    d, m = r["density"], r["memory"]
    print(f"{r['scale']:>4} {r['active']:>4} | {d['w2']['kc_frac']:.3f} {d['w8']['kc_frac']:.3f} "
          f"{d['w32']['kc_frac']:.3f} {d['probe']['kc_frac']:.3f} | "
          f"{(d['w32']['kc_jaccard'] or float('nan')):.2f} {(d['probe']['kc_jaccard'] or float('nan')):.2f} | "
          f"{d['w32']['mbon_active']:5.1f} {d['probe']['mbon_active']:5.1f} | {m['kc_written_a']:>4} | "
          f"{m['mbon|a|W']:.2e} {m['mbon|a|D']:.2e} {m['mbon|z|D']:.2e} | {m['central|z|D']:.2e}")
