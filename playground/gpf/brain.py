"""Standalone runtime for GPF-1: the frozen fly connectome, one character at a time.

Runs on the GPU with flybrain (CuPy + CUDA 12); without a GPU, gpf/cpu_brain.py runs the same
simulation on the CPU with numpy only (identical spikes).  It reproduces the feature
path used in the research code (mechanism/lm_mech.py, frozen mode) without that code's
experiment harness or hard-coded paths.  Data: flybrain's own data location, or the folder
given by GPF_FLY_DATA / FLY_DATA.

Per character (6 steps of 20 ms): inject the character's PN code (full drive on step 1, half
after), record MBON and central voltages just before threshold, count KC/MBON/central spikes.
Features: KC counts (4064) | mean MBON pre-reset voltage (97) | MBON spike counts (97) |
central traces (256) | mean central pre-reset voltage (256).
"""
from __future__ import annotations

import os

import numpy as np

# Frozen protocol (mechanism/m1_core.protocol_args with the GPF-1 operating point).
PROTOCOL = dict(active=160, drive=1.5, gain=1.5, tonic=0.05, k=6, sustain=0.5,
                trace_tau=0.1, noise_hz=0.0, seed=20260920, encoder_seed=3, central_top=256)


def _patch_cupy_includes() -> None:
    from uctf.core.backend import patch_cupy_includes
    patch_cupy_includes()


def data_dir():
    d = os.environ.get("GPF_FLY_DATA") or os.environ.get("FLY_DATA")
    return d if d else None


def gpu_available() -> bool:
    """flybrain with CuPy importable (the GPU runtime, FlyRuntime)."""
    import importlib.util
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("flybrain", "cupy"))
    except (ImportError, ValueError):
        return False


def available() -> bool:
    """GPF-1 can run: on the GPU, or on the CPU from the bundled connectome (gpf/cpu_brain.py)."""
    from .cpu_brain import bundle_path
    return gpu_available() or bundle_path() is not None


def use_gpu(device: str = "auto") -> bool:
    """GPU unless device is "cpu", GPF_DEVICE=cpu, or flybrain/CuPy are missing."""
    if device == "cpu" or os.environ.get("GPF_DEVICE", "").lower() == "cpu":
        return False
    return gpu_available()


class FlyRuntime:
    """Frozen whole-brain simulation driven by characters; returns per-character features."""

    def __init__(self, vocab_size: int, protocol: dict | None = None):
        _patch_cupy_includes()
        from flybrain import FlyBrain
        p = dict(PROTOCOL, **(protocol or {}))
        self.p = p
        brain = FlyBrain(data=data_dir(), device="cuda", batch=1, seed=p["seed"], sensory_input=False)
        brain.noise_hz = np.float32(p["noise_hz"]); brain.gain = np.float32(p["gain"])
        brain.tonic = np.float32(p["tonic"])
        xp = brain.xp
        ct = np.asarray(brain.cell_type).astype(str); ss = np.asarray(brain.superclass).astype(str)
        has = lambda s: np.char.find(ct, s) >= 0
        kc, mbon = np.flatnonzero(has("KC")), np.flatnonzero(has("MBON"))
        dan_mask = has("PAM") | has("PPL") | has("PPM")
        # central readout: the 256 non-KC/MBON/DAN, non-sensory neurons with most MBON input
        post = ~has("MBON") & ~dan_mask & ~has("KC") & (np.char.find(ss, "sensory") < 0)
        central = self._rank_targets(brain, has("MBON"), post, p["central_top"])
        # PN codes: one fixed random subset of PNs per character (flylm.Encoder, seed 3)
        pn = np.flatnonzero(has("PN"))
        rng = np.random.default_rng(p["encoder_seed"])
        codes = [np.sort(pn[rng.choice(len(pn), size=p["active"], replace=False)]) for _ in range(vocab_size)]
        self.codes = [xp.asarray(c) for c in codes]
        self.brain, self.xp = brain, xp
        self.kc, self.mbon, self.central = kc, mbon, central
        self.kc_slot, self.mbon_slot, self.central_slot = (self._slots(brain, ids) for ids in (kc, mbon, central))
        self.record = xp.asarray(np.concatenate([mbon, central]).astype(np.int64))
        self.decay = np.float32(np.exp(-brain.dt / p["trace_tau"]))
        # activity view (web UI): neuron groups of the memory circuit, recorded only when asked
        self.group = np.zeros(brain.n, np.uint8)              # 0 other, 1 PN, 2 KC, 3 MBON, 4 DAN
        self.group[pn], self.group[kc], self.group[mbon], self.group[np.flatnonzero(dan_mask)] = 1, 2, 3, 4
        self.superclass = ss
        self.record_activity = False
        self.last_activity = None
        self.reset()

    GROUPS = ["other", "PN", "KC", "MBON", "DAN"]

    def map_payload(self, max_px: int = 4095) -> dict:
        """Soma positions of every neuron, projected like the fly.ai dashboard (x flipped, x-z
        plane, as seen from above/behind), quantised to uint16, plus group codes and region labels."""
        import base64
        pos = np.asarray(self.brain.positions, np.float64)
        x, z = -pos[:, 0], pos[:, 2]
        for a in (x, z):
            bad = ~np.isfinite(a)
            a[bad] = np.nanmedian(a)
        x0, x1 = np.percentile(x, [0.2, 99.8]); z0, z1 = np.percentile(z, [0.2, 99.8])
        span = max(x1 - x0, z1 - z0)
        u = np.clip((x - x0) / span, 0, 1); v = np.clip((z - z0) / span, 0, 1)
        q = np.stack([u, v], 1) * max_px
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

        def centroid(mask):
            return [round(float(u[mask].mean()), 4), round(float(v[mask].mean()), 4)] if mask.any() else None
        side = u < np.median(u)
        labels = []                          # the page skips any label that would overlap an earlier one
        for name, mask, sides in (("mushroom body", self.group == 2, (~side,)),       # priority order
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

    @staticmethod
    def _slots(brain, ids):
        slot = brain.xp.full(brain.n, -1, dtype=brain.xp.int32)
        slot[brain.xp.asarray(ids)] = brain.xp.arange(len(ids), dtype=brain.xp.int32)
        return slot

    @staticmethod
    def _rank_targets(brain, source_mask, post_mask, top):
        idx, data, ind = brain._W.indices.get(), brain._W.data.get(), brain._W.indptr.get()
        score = np.zeros(brain.n, np.float32); degree = np.zeros(brain.n, np.int32)
        for post in np.flatnonzero(post_mask):
            lo, hi = int(ind[post]), int(ind[post + 1])
            keep = source_mask[idx[lo:hi]]
            if keep.any():
                score[post] = np.abs(data[lo:hi][keep]).sum(); degree[post] = int(keep.sum())
        ids = np.flatnonzero(degree > 0)
        ids = ids[np.argsort(-score[ids], kind="stable")]
        return ids[:min(int(top), len(ids))].astype(np.int64)

    def reset(self):
        xp = self.xp
        self.brain.reset(seed=self.p["seed"])
        self.trace_central = xp.zeros(len(self.central), xp.float32)
        self.trace_mbon = xp.zeros(len(self.mbon), xp.float32)

    def _step(self):
        b, xp = self.brain, self.xp
        current = b.synaptic_input(b.fired) * b.gain
        b.v *= b.decay
        b.v += current + b.tonic
        b.v += (b.rng.random((b.n, b.batch)) < b.noise_hz * b.dt) * xp.float32(b.noise_amp)
        if b.refractory_steps:
            b.v[(b.steps - b.last_spike) <= b.refractory_steps] = 0.0
        v_pre = b.v[self.record].copy()
        fired = xp.flatnonzero(b.v >= 1.0)
        b.v.ravel()[fired] = 0.0
        if b.refractory_steps:
            b.last_spike.ravel()[fired] = b.steps
        b.fired = fired; b.steps += 1
        return fired, v_pre

    @staticmethod
    def _add(fired, slots, *targets):
        import cupyx
        if not fired.size:
            return
        s = slots[fired]; s = s[s >= 0]
        if not s.size:
            return
        for t in targets:
            cupyx.scatter_add(t, s, np.float32(1.0))

    def step_token(self, token: int) -> np.ndarray:
        b, xp, p = self.brain, self.xp, self.p
        kc_counts = xp.zeros(len(self.kc), xp.float32)
        mbon_counts = xp.zeros(len(self.mbon), xp.float32)
        central_counts = xp.zeros(len(self.central), xp.float32)
        nm = len(self.mbon); vm, vc = [], []
        fired_steps = []
        for step in range(p["k"]):
            b.v[self.codes[int(token)], 0] += np.float32(p["drive"] * (1.0 if step == 0 else p["sustain"]))
            fired, v_pre = self._step()
            if self.record_activity:
                fired_steps.append(fired)
            v_pre = xp.asnumpy(v_pre[:, 0]).astype(np.float32)
            vm.append(v_pre[:nm]); vc.append(v_pre[nm:])
            self.trace_central *= self.decay
            self.trace_mbon *= self.decay
            self._add(fired, self.central_slot, self.trace_central, central_counts)
            self._add(fired, self.mbon_slot, self.trace_mbon, mbon_counts)
            self._add(fired, self.kc_slot, kc_counts)
        if self.record_activity:
            spikes = int(sum(int(f.size) for f in fired_steps))
            ids = xp.asnumpy(xp.unique(xp.concatenate(fired_steps))) if spikes else np.zeros(0, np.int64)
            active = np.bincount(self.group[ids], minlength=len(self.GROUPS)) if len(ids) else np.zeros(5, int)
            self.last_activity = {"ids": ids.astype(np.uint32), "spikes": spikes,
                                  "active": {g: int(active[i]) for i, g in enumerate(self.GROUPS)}}
        return np.concatenate([xp.asnumpy(kc_counts), np.asarray(vm, np.float32).mean(0),
                               xp.asnumpy(mbon_counts), xp.asnumpy(self.trace_central),
                               np.asarray(vc, np.float32).mean(0)]).astype(np.float32)


# ------------------------------------------------------------------ substrates
# Substrates are run through UCTF (uctf/run.py).  A brain spec with {"spec": ...} is any connectome
# described by a UCTF spec; {"substrate": "flybrain-malecns-v1", ...} is FlyRuntime above, which the
# bundled GPF-1 was trained with (the same features as UCTF's malecns-v1 spec to float32 rounding).
def _legacy(vocab_size, spec, device="auto"):
    proto = {"active": spec["pn_active"], "drive": float(spec["drive_scale"]),
             "encoder_seed": spec.get("encoder_seed", PROTOCOL["encoder_seed"])}
    if use_gpu(device):
        return FlyRuntime(vocab_size, proto)
    from .cpu_brain import CPURuntime        # same spikes as FlyRuntime, numpy only
    return CPURuntime(vocab_size, proto)


LEGACY = "flybrain-malecns-v1"


def make_substrate(vocab_size: int, spec: dict, device: str = "auto", log=print):
    from uctf import run
    if "spec" not in spec and "substrate" not in spec:
        # saved before brain specs named their substrate: the bundled GPF-1 readout
        spec = {**spec, "substrate": LEGACY}
    return run.make_substrate(vocab_size, spec, device=device, log=log)


def _register():
    from uctf import run
    run.register(LEGACY, _legacy)


_register()
