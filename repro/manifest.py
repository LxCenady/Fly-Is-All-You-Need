"""Write results/MANIFEST.json: for every file under results/, its sha256, size and provenance.

    python repro/manifest.py [--rerun DIR ...] [--queues DIR ...]

Provenance levels (best available is recorded):
  rerun-verified    rerun with the canonical command in the pinned environment (flybrain
                    0.1.0.post1) and found to match the committed file within tolerance; the rerun's
                    manifest (commit, environment, data hashes, timing) is embedded
  command-recorded  the exact command is known from the job queue that produced it
  script-known      the producing script is known (docs/CLAIMS_EVIDENCE.md), not the exact arguments
  derived           a summary or plot input computed from other results
  unknown           none of the above (listed so that the gap is visible)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lenient_json(text):
    try:
        return json.loads(text)
    except ValueError:
        return json.loads(re.sub(r'\\(?![\\"/bfnrtu])', r"\\\\", text))


def norm_arg(a: str) -> str:
    """Absolute paths of the author's machine -> repository-relative script / {out} placeholders."""
    a = a.replace("\\", "/")
    m = re.search(r"/(mechanism|harness|uctf|playground|paper|paper_lm)/(.+)$", a)
    if m and not a.endswith((".json", ".npz")):
        return f"{m.group(1)}/{m.group(2)}"
    if re.match(r"^[A-Za-z]:/", a) and a.endswith((".json", ".npz", ".npy")):
        return "{out:" + Path(a).name + "}"
    return a


def queue_commands(dirs):
    cmds = {}
    for d in dirs:
        for q in Path(d).rglob("queue*.json"):
            if "status" in q.name:
                continue
            try:
                jobs = lenient_json(q.read_text(encoding="utf-8"))
            except Exception:                                # noqa: BLE001
                continue
            for j in jobs if isinstance(jobs, list) else []:
                args = j.get("args", [])
                outs = [a for a in args if isinstance(a, str) and a.endswith(".json") and ("--out" not in a)]
                for o in outs[:1]:
                    cmds.setdefault(Path(o).name, {"queue": str(q.relative_to(Path(d).parent)).replace("\\", "/"),
                                                   "name": j.get("name"),
                                                   "command": ["python"] + [norm_arg(a) for a in args if isinstance(a, str)],
                                                   "env": j.get("env")})
    return cmds


def claims_scripts():
    """Output name -> (claim ids, script) from docs/CLAIMS_EVIDENCE.md."""
    out = {}
    for line in (ROOT / "docs" / "CLAIMS_EVIDENCE.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7 or not re.match(r"^[MLW]\d+$", cells[1] if len(cells) > 1 else ""):
            continue
        cid, script, output = cells[1], cells[4], cells[5]
        for name in re.findall(r"`([^`]+\.(?:json|npz))`", output):
            for n in re.split(r"[{},]", name):
                out.setdefault(Path(name).name, {"claims": [], "script": script})["claims"].append(cid)
    return out


def embedded_config(p: Path):
    """The output records its own full configuration (a top-level "config" or "args" dict)."""
    if p.suffix != ".json":
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None
    for k in ("config", "args"):
        if isinstance(d, dict) and isinstance(d.get(k), dict):
            return k
    return None


KNOWN = {   # output-name prefix -> producing script (from the scripts' own docstrings and the results register)
    "m3_kcmbon": "mechanism/m3_kcmbon.py", "m6_side": "mechanism/m3_kcmbon.py", "m6_cluster": "mechanism/m3_kcmbon.py",
    "m6_glom_route": "mechanism/m3_kcmbon.py", "m3_pnkc": "mechanism/m3_pnkc.py", "kc_input_clusters": "mechanism/m6_structure.py",
    "sweep_s": "mechanism/sweep_opoint.py", "sign_m2": "mechanism/m2_current.py (MB_SIGN=depress)",
    "sign_route": "mechanism/m3_kcmbon.py (MB_SIGN=depress)", "sign_teacher": "mechanism/m7_teacher.py (MB_SIGN=depress)",
    "mb_online_v2": "mechanism/mb_online.py", "lm_online_eval": "mechanism/lm_online_eval.py",
    "exp7_metrics": "harness/lm/exp7_shakespeare.py (historical)",
}


def script_by_name(p: Path):
    for pre, s in KNOWN.items():
        if p.stem.startswith(pre):
            return s
    stem = re.sub(r"(_100k|_a\d+|_s\d.*|_g\d+|_c-?\d+.*)$", "", p.stem)
    for s in (ROOT / "mechanism" / f"{stem}.py", ROOT / "mechanism" / f"{p.stem}.py"):
        if s.exists():
            return str(s.relative_to(ROOT)).replace("\\", "/")
    return None


def rerun_matches(dirs):
    import sys
    sys.path.insert(0, str(ROOT / "repro"))
    from compare import compare
    ok = {}
    for d in dirs:
        for man in Path(d).glob("*.manifest.json"):
            m = json.loads(man.read_text(encoding="utf-8"))
            ref = m.get("compare_to")
            if not ref or m.get("exit_code") != 0 or not (Path(d) / m["output"]).exists():
                continue
            r = compare(Path(d) / m["output"], ROOT / ref, 1e-3, 1e-5)
            if not r["diffs"] and not r["structure"]:
                ok[Path(ref).name] = {"rerun_manifest": m, "numbers_compared": r["n"],
                                      "max_abs_diff": r["max_abs"], "tolerance": "1e-5 + 1e-3 relative"}
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", nargs="*", default=[])
    ap.add_argument("--queues", nargs="*", default=[])
    a = ap.parse_args()
    files = subprocess.run(["git", "ls-files", "results"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    files += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "results").rglob("*")
              if p.is_file() and str(p.relative_to(ROOT)).replace("\\", "/") not in files]
    reruns, queues, scripts = rerun_matches(a.rerun), queue_commands(a.queues), claims_scripts()
    entries, counts = {}, {}
    for f in sorted(set(files)):
        p = ROOT / f
        if not p.is_file() or p.name == "MANIFEST.json":
            continue
        e = {"sha256": sha256(p), "bytes": p.stat().st_size}
        name = p.name
        if name in reruns:
            e["provenance"] = "rerun-verified"; e.update(reruns[name])
        elif name in queues:
            e["provenance"] = "command-recorded"; e.update(queues[name])
        elif (cfg := embedded_config(p)) is not None:
            e["provenance"] = "config-recorded"; e["config_key"] = cfg
            e["script"] = "mechanism/lm_mech.py" if f.startswith("results/lm/") else scripts.get(name, {}).get("script")
        elif name in scripts:
            e["provenance"] = "script-known"; e.update(scripts[name])
        elif (s := script_by_name(p)) is not None:
            e["provenance"] = "script-known"; e["script"] = s
        elif p.suffix in (".md",) or "stats" in f or "SUMMARY" in name:
            e["provenance"] = "derived"
        else:
            e["provenance"] = "unknown"
        if name in scripts and "claims" not in e:
            e["claims"] = scripts[name]["claims"]
        counts[e["provenance"]] = counts.get(e["provenance"], 0) + 1
        entries[f] = e
    doc = {"about": __doc__.strip().splitlines()[0], "levels": __doc__.split("Provenance levels")[1].strip(),
           "counts": counts, "files": entries}
    (ROOT / "results" / "MANIFEST.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
