"""Overnight job queue with time-dependent concurrency.

Jobs run in list order.  A job {"barrier": true} waits until every earlier job
has finished.  Concurrency is 3 before 09:00 and 1 from 09:00 on (running jobs
are never killed; only new starts are throttled).  Each job's stdout/stderr go
to <logdir>/<name>.log/.err; progress goes to <logdir>/queue_status.json.
Usage: queue_runner.py QUEUE.json LOGDIR
"""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

PY = r"D:\flybrain_lm_cuda\.venv\Scripts\python.exe"
CWD = r"D:\flybrain_lm_cuda"


_START = dt.datetime.now()
# the first 09:00 after the runner starts (tomorrow morning if started after 09:00)
_DEADLINE = _START.replace(hour=9, minute=0, second=0, microsecond=0)
if _DEADLINE <= _START:
    _DEADLINE += dt.timedelta(days=1)


_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 3   # optional max concurrency


def limit():
    return 1 if dt.datetime.now() >= _DEADLINE else _CAP


def main(qfile, logdir):
    jobs = json.load(open(qfile, encoding="utf-8"))
    log = Path(logdir); log.mkdir(parents=True, exist_ok=True)
    running, done, status = {}, [], {}
    i = 0

    def write_status():
        (log / "queue_status.json").write_text(json.dumps(
            {"time": dt.datetime.now().isoformat(timespec="seconds"), "limit": limit(),
             "running": list(running), "done": done, "remaining": [j["name"] for j in jobs[i:]],
             "status": status}, indent=1), encoding="utf-8")

    while i < len(jobs) or running:
        for name, p in list(running.items()):
            if p.poll() is not None:
                status[name] = {"exit": p.returncode, "end": dt.datetime.now().isoformat(timespec="seconds")}
                done.append(name); del running[name]
        if i < len(jobs):
            j = jobs[i]
            if j.get("barrier"):
                if not running:
                    status[j["name"]] = {"barrier_passed": dt.datetime.now().isoformat(timespec="seconds")}
                    done.append(j["name"]); i += 1
            elif len(running) < limit():
                out = open(log / f"{j['name']}.log", "w", encoding="utf-8")
                err = open(log / f"{j['name']}.err", "w", encoding="utf-8")
                running[j["name"]] = subprocess.Popen([PY] + j["args"], stdout=out, stderr=err, cwd=CWD)
                status[j["name"]] = {"start": dt.datetime.now().isoformat(timespec="seconds")}
                i += 1
                write_status()
                time.sleep(5)
                continue
        write_status()
        time.sleep(20)
    write_status()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
