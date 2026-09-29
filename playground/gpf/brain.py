"""Standalone runtime for GPF-1: the frozen fly connectome, one character at a time.

Depends only on flybrain (GPU build: CuPy + CUDA 12) and numpy.  It reproduces the feature
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
from pathlib import Path

import numpy as np

# Frozen protocol (mechanism/m1_core.protocol_args with the GPF-1 operating point).
PROTOCOL = dict(active=160, drive=1.5, gain=1.5, tonic=0.05, k=6, sustain=0.5,
                trace_tau=0.1, noise_hz=0.0, seed=20260920, encoder_seed=3, central_top=256)


def _patch_cupy_includes() -> None:
    """Some CuPy installs cannot find their own headers when NVRTC compiles a new kernel.
    Add CuPy's include folders to every compilation (harmless where not needed)."""
    try:
        import cupy
        import cupy.cuda.compiler as compiler
    except Exception:
        return
    if getattr(compiler, "_gpf_include_patch", False):
        return
    inc = Path(cupy.__path__[0]) / "_core" / "include"
    if not (inc / "cupy" / "complex.cuh").exists():
        return
    flags = [f"-I{inc}"] + [f"-I{p}" for p in (inc / "cupy" / "_cccl", inc / "cupy" / "_cccl" / "thrust",
                                               inc / "cupy" / "_cccl" / "libcudacxx", inc / "cupy" / "_cccl" / "cub")
                            if p.exists()]
    original = compiler._compile_with_cache_cuda

    def patched(source, options=(), *a, **kw):
        opts = tuple(options)
        for f in flags:
            if f not in opts:
                opts += (f,)
        return original(source, opts, *a, **kw)

    compiler._compile_with_cache_cuda = patched
    compiler._gpf_include_patch = True


def data_dir():
    d = os.environ.get("GPF_FLY_DATA") or os.environ.get("FLY_DATA")
    return d if d else None


def available() -> bool:
    import importlib.util
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("flybrain", "cupy"))
    except (ImportError, ValueError):
        return False


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
        self.reset()

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
        for step in range(p["k"]):
            b.v[self.codes[int(token)], 0] += np.float32(p["drive"] * (1.0 if step == 0 else p["sustain"]))
            fired, v_pre = self._step()
            v_pre = xp.asnumpy(v_pre[:, 0]).astype(np.float32)
            vm.append(v_pre[:nm]); vc.append(v_pre[nm:])
            self.trace_central *= self.decay
            self.trace_mbon *= self.decay
            self._add(fired, self.central_slot, self.trace_central, central_counts)
            self._add(fired, self.mbon_slot, self.trace_mbon, mbon_counts)
            self._add(fired, self.kc_slot, kc_counts)
        return np.concatenate([xp.asnumpy(kc_counts), np.asarray(vm, np.float32).mean(0),
                               xp.asnumpy(mbon_counts), xp.asnumpy(self.trace_central),
                               np.asarray(vc, np.float32).mean(0)]).astype(np.float32)
