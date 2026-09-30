"""One entry point for reproducing the repository's results.

    python repro/make.py env        check the environment against repro/requirements-lock.txt
    python repro/make.py test       all tests: flybrain CSR identity (GPU), UCTF invariants
    python repro/make.py figures    rebuild every figure of both papers from results/ (CPU, ~1 min)
    python repro/make.py verify-figures   rebuild into a temp folder and compare with the committed PNGs
    python repro/make.py headline OUTDIR   rerun the headline experiments (GPU, ~1 h), with manifests
    python repro/make.py compare OUTDIR    compare a rerun with the committed results
    python repro/make.py manifest   rewrite results/MANIFEST.json (sha256 + provenance of every result)

Environment: see repro/ENVIRONMENT.md.  GPU jobs run one at a time at below-normal priority.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("FLY_PYTHON") or sys.executable


def run(*cmd, cwd=ROOT, env=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env)
    if r.returncode:
        sys.exit(r.returncode)


def env_check():
    """Installed versions must equal the lock; flybrain must be the patched fork."""
    want = {}
    for line in (ROOT / "repro" / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "==" in line and not line.startswith(("#", "--")):
            name, ver = line.split("==")[0], line.split("==")[1].split()[0]
            want[name.lower()] = ver
    code = "import json, importlib.metadata as m; print(json.dumps({d.metadata['Name'].lower(): d.version for d in m.distributions()}))"
    import json
    have = json.loads(subprocess.run([PY, "-c", code], capture_output=True, text=True).stdout)
    bad = {k: (v, have.get(k)) for k, v in want.items() if have.get(k) != v}
    fb = have.get("flybrain")
    print(f"{len(want)} locked packages; mismatches: {bad or 'none'}; flybrain {fb}")
    if bad or fb != "0.1.0.post1":
        sys.exit("environment differs from repro/requirements-lock.txt")


def figures(outroot=None):
    env = dict(os.environ)
    for d, script in (("paper", "make_figures.py"), ("paper_lm", "make_figures_lm.py")):
        run(PY, script, cwd=ROOT / d, env=env)


def verify_figures():
    """Rebuild, then compare pixel by pixel with the committed PNGs (restored afterwards)."""
    import numpy as np
    from PIL import Image
    tmp = Path(tempfile.mkdtemp())
    for d in ("paper", "paper_lm"):
        shutil.copytree(ROOT / d / "figures", tmp / d)
    figures()
    bad = []
    for d in ("paper", "paper_lm"):
        for f in sorted((tmp / d).glob("*.png")):
            a = np.asarray(Image.open(f).convert("RGB")); b = np.asarray(Image.open(ROOT / d / "figures" / f.name).convert("RGB"))
            ok = a.shape == b.shape and (a == b).all()
            print(f"{d}/{f.name}: {'identical' if ok else 'DIFFERENT'}")
            if not ok:
                bad.append(f.name)
    shutil.rmtree(tmp)
    if bad:
        sys.exit(f"{len(bad)} figures differ")


def main(argv):
    if not argv:
        print(__doc__); return
    cmd, rest = argv[0], argv[1:]
    if cmd == "env":
        env_check()
    elif cmd == "test":
        run(PY, "third_party/flybrain/tests/test_csr_identity.py")
        run(PY, "uctf/tests/test_uctf.py")
    elif cmd == "figures":
        figures()
    elif cmd == "verify-figures":
        verify_figures()
    elif cmd == "headline":
        run(PY, "repro/run.py", "repro/headline.json", rest[0] if rest else "outputs/headline", *rest[1:])
    elif cmd == "compare":
        run(PY, "repro/compare.py", rest[0], "--report", "repro/RERUN_REPORT.md")
    elif cmd == "manifest":
        run(PY, "repro/manifest.py")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
