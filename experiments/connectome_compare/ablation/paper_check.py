"""Re-check of the LM paper's memory numbers by feature set. One job: bench caches -> ablation/paper_check.json.

Worm (E:/gpf_bench_worm3*, codes 3-5): the paper's "all features" = counts + voltage + trace of
the same 217 neurons. Fly (E:/gpf_bench_malecns2): records which feature blocks are all zero.
The real connectome is the oldest cache in each bench folder (as repro/lm_stats.py assumes).

    python ablation/paper_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "uctf"))
from uctf.plugins.probes import ridge_span  # noqa: E402
from uctf.plugins.tasks import load_split  # noqa: E402

WORM = ["gpf_bench_worm3", "gpf_bench_worm3_code4", "gpf_bench_worm3_code5"]
SETS = {"all": ["spikes", "voltage", "trace"], "no trace": ["spikes", "voltage"],
        "voltage": ["voltage"], "counts": ["spikes"], "counts+trace": ["spikes", "trace"]}


def real_cache(folder):
    return sorted(Path("E:/", folder).glob("features_*.npy"), key=lambda p: p.stat().st_mtime)[0]


def blocks(npy):
    dims = json.loads(npy.with_suffix(".json").read_text())
    o, sl = 0, {}
    for k, n in dims.items():
        sl[k] = slice(o, o + n)
        o += n
    return sl


def main():
    data = REPO / "playground/gpf/data"
    ids = load_split(data / "tinyshakespeare.txt", 20001, 5000, 0, data / "vocab.json").ids
    out = {"worm": {}, "fly_zero_blocks": []}
    for folder in WORM:
        f = real_cache(folder)
        X, sl = np.load(f), blocks(f)
        out["worm"][folder] = {}
        for name, ks in SETS.items():
            span, _ = ridge_span(np.hstack([X[:, sl[k]] for k in ks]).astype(np.float32), ids, 20001, 8)
            out["worm"][folder][name] = [round(float(v), 4) for v in span]
            print(folder, name, out["worm"][folder][name][:4], file=sys.stderr, flush=True)
    f = real_cache("gpf_bench_malecns2")
    X, sl = np.load(f, mmap_mode="r"), blocks(f)
    out["fly_cache"] = f.name
    out["fly_zero_blocks"] = [k for k, s in sl.items() if float(np.abs(X[:, s]).max()) == 0.0]
    (HERE / "paper_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
