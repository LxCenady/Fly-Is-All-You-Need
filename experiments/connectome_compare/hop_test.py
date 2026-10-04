"""Does memory follow path latency? One job: spec + window-feature npz -> hop_test json (+ table).

Hop distance of every readout neuron from the input population on the chemical graph (after the
spec's transforms; an edge = any nonzero weight, pre -> post; multi-source BFS). Readout neurons
are grouped by distance; from each group the same number of neurons (--m, 3 draws) decode the
character k = 0..3 back (shuffled text). Prediction: farther neurons carry relatively more of
the previous characters (later responses), from the full-token counts and from step 0.

    python hop_test.py specs/fly_win.json features_q/fly_win_c*.npz [--m 150]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paths import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "uctf"))
from uctf.core.network import load_connectome  # noqa: E402
from uctf.plugins.probes import ridge_span  # noqa: E402
from simulate import text_ids  # noqa: E402

N_TR = 20001


def hops(W, sources, max_hops=10):
    """Shortest hop count from any source along pre -> post edges (W rows = post)."""
    A = (W != 0).astype(np.float32).tocsr()          # A[post, pre]
    dist = np.full(W.shape[0], -1, np.int64)
    dist[sources] = 0
    front = np.zeros(W.shape[0], np.float32)
    front[sources] = 1
    for h in range(1, max_hops + 1):
        reach = (A @ front) > 0
        new = reach & (dist < 0)
        if not new.any():
            break
        dist[new] = h
        front = new.astype(np.float32)
    return dist


def mean_latency(W, sources, max_hops=6):
    """Weighted latency: input u at the sources, drive at hop h = (|W|^h u); per neuron the
    drive-weighted mean of h. A structural, simulation-free latency (absolute weights)."""
    A = abs(W).astype(np.float32).tocsr()
    u = np.zeros(W.shape[0], np.float32)
    u[sources] = 1
    num, den = np.zeros(W.shape[0]), np.zeros(W.shape[0])
    for h in range(1, max_hops + 1):
        u = A @ u
        u /= max(float(u.max()), 1e-12)                  # keep the scale; ratios within a hop matter
        num += h * u
        den += u
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--m", type=int, default=150)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--latency", action="store_true", help="group by weighted latency tertiles")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    cx = load_connectome(spec)
    W = cx.layer("chemical").matrix
    inp = np.asarray(spec["populations"]["INPUT"]["ids"])
    ro = np.sort(np.asarray(spec["populations"]["READOUT"]["ids"]))
    d = hops(W, inp)[ro]
    if a.latency:
        lat = mean_latency(W, inp)[ro]
        cuts = np.nanquantile(lat, [1 / 3, 2 / 3])
        d = np.digitize(np.nan_to_num(lat, nan=np.nanmax(lat)), cuts)   # 0 = earliest third
        print("latency tertile cuts", np.round(cuts, 3), file=sys.stderr)
    groups = {}
    for h in sorted(set(d.tolist())):
        idx = np.flatnonzero(d == h)
        print(f"hop {h}: {len(idx)} readout neurons", file=sys.stderr)
        if len(idx) >= a.m:
            groups[h] = idx
    res = {h: {"spikes": [], "w0": []} for h in groups}
    for f in a.files:
        z = np.load(f)
        X, info = z["X"], json.loads(str(z["info"]))
        assert info["dims"]["spikes"] == len(ro), "readout order/size mismatch"
        ids, _ = text_ids(shuffle=info.get("shuffle"))
        start, off = {}, 0
        for k, n in info["dims"].items():
            start[k] = off
            off += n
        for h, idx in groups.items():
            for dr in range(a.draws):
                pick = np.sort(np.random.default_rng(2000 + dr).choice(idx, a.m, replace=False))
                for b in ("spikes", "w0"):
                    span, _ = ridge_span(X[:, start[b] + pick].astype(np.float32), ids, N_TR, 3)
                    res[h][b].append(span)
    out = {"counts": {int(h): int((d == h).sum()) for h in set(d.tolist())}, "m": a.m}
    print(f"\n{Path(a.spec).stem}: M = {a.m} per group; columns k = 0..3")
    for h in groups:
        out[int(h)] = {}
        for b in ("spikes", "w0"):
            m = np.mean(res[h][b], 0)
            out[int(h)][b] = [round(float(v), 4) for v in m]
            print(f"hop {h} {b:6s} " + " ".join(f"{v:6.3f}" for v in m))
    Path(Path(__file__).parent / f"hop_test_{Path(a.spec).stem}{'_lat' if a.latency else ''}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
