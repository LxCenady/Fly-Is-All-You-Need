"""Append-only JSONL progress log shared by experiments and monitor."""

import json
import os
import time
from pathlib import Path

PATH = os.environ.get("FLYLOG", str(Path(__file__).parent / "runs.jsonl"))


def emit(run: str, cfg: str, status: str = "running", **fields):
    rec = {"ts": time.time(), "run": run, "cfg": cfg, "status": status}
    rec.update(fields)
    with open(PATH, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
