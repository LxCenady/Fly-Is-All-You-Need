import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
for pt in d:
    print(f"\n### PN {pt['active']} x{pt['scale']}  cols: bias | MBON spikes X/0 | central spikes X/0 | W_mbon W_central D_central")
    for r in pt["rows"]:
        ms, cs = r["mbon_spikes"], r["central_spikes"]
        print(f"{''.join(r['pair'])} p={r['probe']} b={r['bias']:.2f} | {ms['X']:5.0f}/{ms['0']:5.0f} | "
              f"{cs['X']:6.0f}/{cs['0']:6.0f} | {r['W_mbon']:.2e} {r['W_central']:.2e} {r['D_central']:.2e}")
