"""Compare rerun outputs with the committed results.

    python repro/compare.py OUTDIR [--rtol 1e-3] [--atol 1e-5] [--report repro/RERUN_REPORT.md]

For every OUTDIR/<id>.manifest.json with a compare_to file, walks both JSON documents in parallel
and compares every number (lists by position, dicts by key).  Timing fields and paths are
ignored.  A number matches if |a - b| <= atol + rtol * |b|.  Structural differences (missing
keys, list lengths) are reported separately.  GPU sparse products are not bit-deterministic
(voltages differ at the last float32 bit), so exact equality is not expected for voltages;
spike counts and anything derived only from them should match exactly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {"seconds", "time", "elapsed", "t_wall", "corpus", "data", "wall_s", "runtime_s"}


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                out["structure"].append(f"{path}/{k}: only in {'rerun' if k in a else 'committed'}")
                continue
            walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["structure"].append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool):
        out["n"] += 1
        if a != b:
            out["diffs"].append((path, a, b, math.inf))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out["n"] += 1
        if (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b)):
            return
        d = abs(a - b)
        out["max_abs"] = max(out["max_abs"], d)
        if b != 0:
            out["max_rel"] = max(out["max_rel"], d / abs(b))
        if d > out["atol"] + out["rtol"] * abs(b):
            out["diffs"].append((path, a, b, d))
    elif a != b and not (isinstance(a, str) and isinstance(b, str) and ("\\" in a or "/" in a)):
        out["structure"].append(f"{path}: {str(a)[:40]!r} vs {str(b)[:40]!r}")


def compare(rerun: Path, committed: Path, rtol, atol) -> dict:
    out = {"n": 0, "diffs": [], "structure": [], "max_abs": 0.0, "max_rel": 0.0, "rtol": rtol, "atol": atol}
    walk(json.loads(rerun.read_text(encoding="utf-8")), json.loads(committed.read_text(encoding="utf-8")), "", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--report")
    a = ap.parse_args()
    rows = []
    for man in sorted(Path(a.outdir).glob("*.manifest.json")):
        m = json.loads(man.read_text(encoding="utf-8"))
        if not m.get("compare_to"):
            continue
        rerun, ref = Path(a.outdir) / m["output"], ROOT / m["compare_to"]
        if m.get("exit_code") != 0 or not rerun.exists():
            rows.append((m["id"], m.get("claims", []), "FAILED TO RUN", None)); continue
        r = compare(rerun, ref, a.rtol, a.atol)
        verdict = "match" if not r["diffs"] and not r["structure"] else "DIFFERS"
        rows.append((m["id"], m.get("claims", []), verdict, r))
    lines = ["# Rerun report", "",
             f"Tolerance: |rerun - committed| <= {a.atol} + {a.rtol} x |committed| for every number.", "",
             "| Job | Claims | Numbers compared | Verdict | Max abs. diff | Max rel. diff |", "|---|---|---|---|---|---|"]
    detail = []
    for jid, claims, verdict, r in rows:
        if r is None:
            lines.append(f"| {jid} | {', '.join(claims)} | - | {verdict} | | |"); continue
        lines.append(f"| {jid} | {', '.join(claims)} | {r['n']:,} | {verdict} | {r['max_abs']:.2e} | {r['max_rel']:.2e} |")
        if r["diffs"] or r["structure"]:
            detail.append(f"\n## {jid}\n")
            detail += [f"- structure: {s}" for s in r["structure"][:20]]
            worst = sorted(r["diffs"], key=lambda t: -t[3])[:15]
            detail += [f"- `{p}`: rerun {x} vs committed {y}" for p, x, y, _ in worst]
            if len(r["diffs"]) > 15:
                detail.append(f"- ... {len(r['diffs']) - 15} more numbers outside tolerance")
    text = "\n".join(lines + detail) + "\n"
    print(text)
    if a.report:
        Path(a.report).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
