"""Run the canonical jobs of a job list and record a manifest for every output.

    python repro/run.py repro/headline.json OUTDIR [job_id ...]

For each job: runs `python <script> <args>` from the repository root ({out} = OUTDIR/<id>.json),
one job at a time, at below-normal priority and with at most FLY_THREADS (default 4) CPU threads,
and writes OUTDIR/<id>.manifest.json:
    command, git commit (+ dirty flag), start/end/seconds/exit code, sha256 of the output,
    environment (python, platform, flybrain version + sha256 of its brain.py, numpy/scipy/cupy,
    GPU, driver, CUDA runtime), sha256 of the environment lock (repro/requirements-lock.txt),
    sha256 of the brain data files, and the FLY_*/MB_* environment variables.
Existing outputs with a successful manifest are skipped (rerun with --force).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "repro" / "requirements-lock.txt"
_hash_cache: dict = {}


def sha256(path: Path, cache=False) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    key = (str(path), path.stat().st_size, path.stat().st_mtime)
    if cache and key in _hash_cache:
        return _hash_cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if cache:
        _hash_cache[key] = h.hexdigest()
    return h.hexdigest()


def git(*a) -> str:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return ""


def environment(python: str) -> dict:
    code = r"""
import json, platform, sys, importlib
out = {"python": sys.version.split()[0], "platform": platform.platform()}
for m in ("numpy", "scipy", "numba", "cupy", "torch", "flybrain"):
    try:
        mod = importlib.import_module(m); out[m] = getattr(mod, "__version__", "?")
    except Exception as e:
        out[m] = None
try:
    import flybrain, hashlib, pathlib
    out["flybrain_brain_py_sha256"] = hashlib.sha256((pathlib.Path(flybrain.__file__).parent / "brain.py").read_bytes()).hexdigest()
except Exception:
    pass
try:
    import cupy
    out["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
    out["cuda_driver"] = cupy.cuda.runtime.driverGetVersion()
    p = cupy.cuda.runtime.getDeviceProperties(0); out["gpu"] = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
except Exception:
    pass
print(json.dumps(out))
"""
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    try:
        env = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:                                        # noqa: BLE001
        env = {"error": r.stderr[-500:]}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        env["nvidia_smi"] = smi
    except Exception:                                        # noqa: BLE001
        pass
    return env


def data_hashes() -> dict:
    sys.path.insert(0, str(ROOT / "mechanism"))
    try:
        import paths
        d = Path(paths.DATA)
    except Exception:                                        # noqa: BLE001
        return {}
    return {f: sha256(d / f, cache=True) for f in ("brain.npz", "weights.npz")}


def main(argv):
    force = "--force" in argv
    argv = [a for a in argv if a != "--force"]
    jobs_file, outdir = Path(argv[0]), Path(argv[1])
    only = set(argv[2:])
    outdir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(jobs_file.read_text(encoding="utf-8"))
    python = os.environ.get("FLY_PYTHON") or sys.executable
    threads = os.environ.get("FLY_THREADS", "4")
    env = dict(os.environ, OMP_NUM_THREADS=threads, NUMBA_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS=threads, PYTHONIOENCODING="utf-8")
    envinfo = environment(python)
    dh = data_hashes()
    flags = 0x00004000 if os.name == "nt" else 0              # BELOW_NORMAL_PRIORITY_CLASS
    for job in spec["jobs"]:
        if only and job["id"] not in only:
            continue
        out = outdir / f"{job['id']}.json"
        man = outdir / f"{job['id']}.manifest.json"
        if not force and man.exists() and json.loads(man.read_text(encoding="utf-8")).get("exit_code") == 0:
            print(f"skip {job['id']} (done)", flush=True)
            continue
        cmd = [python, job["script"]] + [a.replace("{out}", str(out)) for a in job["args"]]
        t0 = time.time(); started = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"run  {job['id']}: {' '.join(cmd[1:])}", flush=True)
        with open(outdir / f"{job['id']}.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=flags if os.name == "nt" else 0,
                                preexec_fn=(lambda: os.nice(10)) if os.name != "nt" else None).returncode
        record = {
            "id": job["id"], "claims": job.get("claims", []), "output": out.name, "output_sha256": sha256(out),
            "command": ["python"] + cmd[1:], "cwd": "<repository root>",
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "mechanism", "harness", "uctf", "third_party")),
            "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
            "exit_code": rc, "environment": envinfo, "requirements_lock_sha256": sha256(LOCK),
            "data_sha256": dh, "threads": int(threads),
            "env_vars": {k: v for k, v in os.environ.items() if k.startswith(("MB_",))},
            "compare_to": job.get("compare"),
        }
        man.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"     {'ok' if rc == 0 else 'FAILED'} in {record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
