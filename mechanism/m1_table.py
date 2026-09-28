"""Print the M1 table (ratios to D_nat and to the replay floor)."""
import json
import sys

FLOOR = {"mbon": 2.384185791015625e-07, "central": 1.1920928955078125e-07}  # per element max
d = json.load(open(sys.argv[1], encoding="utf-8"))
lv = sys.argv[2] if len(sys.argv) > 2 else "mbon"
print(f"level={lv}  columns: D_nat | D_m/D_nat D_s/D_nat I/D_nat | D_m|sX0 /D_nat D_m|sz /D_nat | W_X/ref | fastdiff(v l2) | modulation l2 X-Y")
for c in d["cells"]:
    s = c["stats"][lv]
    n = s["D_nat"] or float("nan")
    sd = c["state_distance"]["X_vs_Y"]
    print(f"{''.join(c['pair'])} G={c['gap']:>2} p={c['probe']}  {s['D_nat']:.3e} | "
          f"{s['D_m']/n:6.3f} {s['D_s']/n:6.3f} {s['I']/n:6.3f} | "
          f"{s['D_m_on_sX0']/n:6.3f} {s['D_m_on_szero']/n:6.3f} | "
          f"{s['W_X']/max(s['ref_norm'],1e-30):.3e} | v {sd['v']['l2']:.3e} fired {sd['fired']['sym_diff']:>4} | mod {sd['modulation']['l2']:.3e}")
