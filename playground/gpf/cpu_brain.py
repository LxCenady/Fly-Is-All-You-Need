"""GPF-1 on the CPU with numpy only: the same simulation as FlyRuntime (gpf/brain.py), no GPU.

The frozen MaleCNS connectome ships as one numpy bundle (gpf1_cpu.npz, made by `prepare`
from flybrain's data; sensory inputs removed as in GPF-1).  Synaptic input is event-driven:
only the outgoing weights (CSC columns) of neurons that fired are summed, about 1 % of
neurons per step, which on a laptop CPU is faster than the GPU version.  Against FlyRuntime
on 3,000 characters: identical spikes, features within float32 rounding, identical BPC.

Connectome: MaleCNS v1.0 (Berg et al. 2026, Cell; FlyEM / HHMI Janelia, CC BY 4.0), as
prebuilt by flybrain (github.com/alextitonis/fly.ai, MIT).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .brain import PROTOCOL

DT, TAU = 0.020, 0.100                      # flybrain FlyBrain.dt, FlyBrain.tau
BUNDLE = "gpf1_cpu.npz"
BUNDLE_URL = ("https://github.com/LxCenady/Fly-Is-All-You-Need/releases/download/"
              "gpf1-cpu-data-v1/gpf1_cpu.npz")
BUNDLE_SHA256 = "096d6a3d06a1e7902682478dc23c47601d35d09a3bd0f978c81c5480960f60c4"


def bundle_path() -> Path | None:
    """The CPU bundle: $GPF_BRAIN_BUNDLE, next to the package data, or in ~/.gpf."""
    cands = [os.environ.get("GPF_BRAIN_BUNDLE"),
             Path(__file__).parent / "data" / BUNDLE,
             Path.home() / ".gpf" / BUNDLE]
    return next((Path(c) for c in cands if c and Path(c).exists()), None)


def fetch(dest: Path | None = None, log=print) -> Path:
    """Download the CPU bundle (~138 MB) from the project's data release into ~/.gpf and
    check its SHA-256.  numpy-only, so the lite packages can use it too."""
    import hashlib
    import urllib.request
    dest = Path(dest) if dest else Path.home() / ".gpf" / BUNDLE
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log(f"downloading the GPF-1 connectome (~138 MB) to {dest} ...")
    h = hashlib.sha256()
    with urllib.request.urlopen(BUNDLE_URL) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != BUNDLE_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("download corrupted (SHA-256 mismatch); please try again")
    tmp.replace(dest)
    log("done: GPF-1 now runs on the CPU, no GPU needed")
    return dest


def prepare(data_dir, out_path):
    """flybrain data folder (brain.npz + weights.npz) -> the CPU bundle.  Needs scipy once."""
    from scipy import sparse
    meta = np.load(Path(data_dir) / "brain.npz")
    W = sparse.load_npz(Path(data_dir) / "weights.npz")
    sc = meta["superclass"].astype(str)
    sensory = np.char.find(sc, "sensory") >= 0
    W = (sparse.diags((~sensory).astype(np.float32)) @ W.tocsr()).tocsr()
    W.sum_duplicates()
    W.sort_indices()
    ct = meta["cell_type"].astype(str)
    has = lambda s: np.char.find(ct, s) >= 0
    mbon = has("MBON")
    dan = has("PAM") | has("PPL") | has("PPM")
    post = ~mbon & ~dan & ~has("KC") & ~sensory
    # the central readout ranking of FlyRuntime._rank_targets (rows = postsynaptic)
    rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    keep = mbon[W.indices] & post[rows]
    score = np.bincount(rows[keep], weights=np.abs(W.data[keep]).astype(np.float64),
                        minlength=W.shape[0]).astype(np.float32)
    degree = np.bincount(rows[keep], minlength=W.shape[0])
    ids = np.flatnonzero(degree > 0)
    ids = ids[np.argsort(-score[ids], kind="stable")]
    C = W.tocsc()
    np.savez_compressed(out_path, indptr=C.indptr.astype(np.int64), indices=C.indices.astype(np.int32),
                        weights=C.data.astype(np.float32), cell_type=ct, superclass=sc,
                        positions=meta["positions"].astype(np.float32), central_rank=ids.astype(np.int64))


class CPURuntime:
    """Drop-in for FlyRuntime: reset(), step_token(t), map_payload(), record_activity / last_activity."""

    GROUPS = ["other", "PN", "KC", "MBON", "DAN"]

    def __init__(self, vocab_size: int, protocol: dict | None = None, bundle=None):
        path = Path(bundle) if bundle else bundle_path()
        if path is None:
            raise RuntimeError(f"GPF-1 CPU bundle {BUNDLE} not found (set GPF_BRAIN_BUNDLE)")
        z = np.load(path)
        p = dict(PROTOCOL, **(protocol or {}))
        self.p = p
        self.indptr, self.indices, self.weights = z["indptr"], z["indices"], z["weights"]
        self.n = len(self.indptr) - 1
        ct = z["cell_type"].astype(str)
        self.superclass = z["superclass"].astype(str)
        self.positions = z["positions"]
        has = lambda s: np.char.find(ct, s) >= 0
        pn = np.flatnonzero(has("PN"))
        self.kc, self.mbon = np.flatnonzero(has("KC")), np.flatnonzero(has("MBON"))
        dan = has("PAM") | has("PPL") | has("PPM")
        self.central = z["central_rank"][:p["central_top"]]
        rng = np.random.default_rng(p["encoder_seed"])          # flylm.Encoder codes, seed 3
        self.codes = [np.sort(pn[rng.choice(len(pn), size=p["active"], replace=False)])
                      for _ in range(vocab_size)]
        self.record = np.concatenate([self.mbon, self.central]).astype(np.int64)
        self.vdecay = np.float32(np.exp(-DT / TAU))
        self.decay = np.float32(np.exp(-DT / p["trace_tau"]))
        self.gain, self.tonic = np.float32(p["gain"]), np.float32(p["tonic"])
        self.slot = {}
        for name, ids in (("kc", self.kc), ("mbon", self.mbon), ("central", self.central)):
            s = np.full(self.n, -1, np.int32)
            s[ids] = np.arange(len(ids), dtype=np.int32)
            self.slot[name] = s
        self.group = np.zeros(self.n, np.uint8)                   # 0 other, 1 PN, 2 KC, 3 MBON, 4 DAN
        self.group[pn], self.group[self.kc] = 1, 2
        self.group[self.mbon], self.group[np.flatnonzero(dan)] = 3, 4
        self.record_activity = False
        self.last_activity = None
        self.reset()

    def reset(self):
        self.v = np.zeros(self.n, np.float32)
        self.fired = np.empty(0, np.int64)
        self.trace_central = np.zeros(len(self.central), np.float32)

    def _synaptic_input(self, fired):
        if not len(fired):
            return np.zeros(self.n, np.float32)
        lo, hi = self.indptr[fired], self.indptr[fired + 1]
        lens = hi - lo
        sel = np.repeat(lo - np.cumsum(np.r_[0, lens[:-1]]), lens) + np.arange(lens.sum())
        return np.bincount(self.indices[sel], weights=self.weights[sel], minlength=self.n).astype(np.float32)

    def _step(self):
        current = self._synaptic_input(self.fired) * self.gain
        self.v *= self.vdecay
        self.v += current + self.tonic
        v_pre = self.v[self.record].copy()
        fired = np.flatnonzero(self.v >= 1.0)
        self.v[fired] = 0.0
        self.fired = fired
        return fired, v_pre

    def _count(self, fired, name, size):
        s = self.slot[name][fired]
        return np.bincount(s[s >= 0], minlength=size).astype(np.float32)

    def step_token(self, token: int) -> np.ndarray:
        p = self.p
        nk, nm, nc = len(self.kc), len(self.mbon), len(self.central)
        kc_counts, mbon_counts = np.zeros(nk, np.float32), np.zeros(nm, np.float32)
        vm, vc, fired_steps = [], [], []
        for step in range(p["k"]):
            self.v[self.codes[int(token)]] += np.float32(p["drive"] * (1.0 if step == 0 else p["sustain"]))
            fired, v_pre = self._step()
            if self.record_activity:
                fired_steps.append(fired)
            vm.append(v_pre[:nm]); vc.append(v_pre[nm:])
            self.trace_central *= self.decay
            self.trace_central += self._count(fired, "central", nc)
            mbon_counts += self._count(fired, "mbon", nm)
            kc_counts += self._count(fired, "kc", nk)
        if self.record_activity:
            spikes = int(sum(len(f) for f in fired_steps))
            ids = np.unique(np.concatenate(fired_steps)) if spikes else np.zeros(0, np.int64)
            active = np.bincount(self.group[ids], minlength=len(self.GROUPS))
            self.last_activity = {"ids": ids.astype(np.uint32), "spikes": spikes,
                                  "active": {g: int(active[i]) for i, g in enumerate(self.GROUPS)}}
        return np.concatenate([kc_counts, np.asarray(vm, np.float32).mean(0), mbon_counts,
                               self.trace_central, np.asarray(vc, np.float32).mean(0)]).astype(np.float32)

    def map_payload(self, max_px: int = 4095) -> dict:
        """Same brain view as FlyRuntime.map_payload."""
        import base64
        pos = np.asarray(self.positions, np.float64)
        x, z = -pos[:, 0], pos[:, 2]
        for a in (x, z):
            a[~np.isfinite(a)] = np.nanmedian(a)
        x0, x1 = np.percentile(x, [0.2, 99.8]); z0, z1 = np.percentile(z, [0.2, 99.8])
        span = max(x1 - x0, z1 - z0)
        u = np.clip((x - x0) / span, 0, 1); v = np.clip((z - z0) / span, 0, 1)
        q = np.stack([u, v], 1) * max_px
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

        def centroid(mask):
            return [round(float(u[mask].mean()), 4), round(float(v[mask].mean()), 4)] if mask.any() else None
        side = u < np.median(u)
        labels = []
        for name, mask, sides in (("mushroom body", self.group == 2, (~side,)),
                                  ("optic lobe", self.superclass == "ol_intrinsic", (side, ~side)),
                                  ("antennal lobe", self.group == 1, (side,))):
            for s in sides:
                c = centroid(mask & s)
                if c:
                    labels.append({"text": name, "u": c[0], "v": c[1]})
        c = centroid(self.superclass == "vnc_intrinsic")
        if c:
            labels.append({"text": "ventral nerve cord", "u": c[0], "v": c[1]})
        return {"n": int(len(u)), "scale": max_px, "w": round(float((x1 - x0) / span), 4),
                "h": round(float((z1 - z0) / span), 4), "xy": b64(q.astype(np.uint16)),
                "group": b64(self.group), "groups": self.GROUPS, "labels": labels,
                "totals": {g: int((self.group == i).sum()) for i, g in enumerate(self.GROUPS)}}
