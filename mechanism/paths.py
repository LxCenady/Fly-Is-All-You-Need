"""Where the research scripts find their inputs and put their outputs.

Nothing is hard-coded.  Each location comes from an environment variable, else from a
git-ignored `paths_local.json` next to this file, else from a default inside the repository:

  FLY_DATA     connectome files brain.npz / weights.npz  default: flybrain's folder, ~/fly-data
               (MB_DATA, if set, overrides it: used for the rewired-connectome controls)
  FLY_HARNESS  experiment harness (lm/ + sitecustomize)   default: <repo>/harness
  FLY_CORPUS   TinyShakespeare text                       default: <repo>/playground/gpf/data/tinyshakespeare.txt
  FLY_OUT      experiment outputs (JSON/NPZ)              default: <repo>/outputs
  FLY_LEGACY   runs from the earlier project (models/)    default: <repo>/legacy (audit scripts only)
  FLY_PYTHON   interpreter the job queue launches         default: the current interpreter
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MECH = Path(__file__).resolve().parent
REPO = MECH.parent
_LOCAL = MECH / "paths_local.json"
_local = json.loads(_LOCAL.read_text(encoding="utf-8")) if _LOCAL.exists() else {}


def _get(name: str, default) -> Path:
    value = os.environ.get(name) or _local.get(name)
    return Path(value) if value else Path(default)


HARNESS = _get("FLY_HARNESS", REPO / "harness")
DATA = Path(os.environ["MB_DATA"]) if os.environ.get("MB_DATA") else _get("FLY_DATA", Path.home() / "fly-data")
CORPUS = _get("FLY_CORPUS", REPO / "playground" / "gpf" / "data" / "tinyshakespeare.txt")
OUT = _get("FLY_OUT", REPO / "outputs")
LEGACY = _get("FLY_LEGACY", REPO / "legacy")
PYTHON = os.environ.get("FLY_PYTHON") or _local.get("FLY_PYTHON") or sys.executable


def data_dir() -> Path:
    """The connectome folder, downloading flybrain's prebuilt copy (~260 MB) if it is missing."""
    try:
        from flybrain.data import ensure_data
        return Path(ensure_data(DATA))
    except ImportError:
        return DATA


def out(*parts: str) -> Path:
    """A path under FLY_OUT; its parent folder is created."""
    p = OUT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def use_harness() -> None:
    """Put the harness on sys.path (lm/ modules and sitecustomize)."""
    for p in (HARNESS, HARNESS / "lm"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
