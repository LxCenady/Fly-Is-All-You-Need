"""A connectome as a language-model substrate: characters in, neuron features out.

A spec (JSON; built-ins in gpf/connectome/specs/) says which connectome to load, how neurons
behave, which neurons receive the input and what is read out:

{
  "name": "malecns-v1",
  "connectome": {"loader": "flybrain", "cut_inputs_to": {"col": "superclass", "contains": "sensory"}},
  "populations": {"KC": {"col": "cell_type", "contains": "KC"}, ...},
  "neuron": {"dt": 0.02, "tau": 0.1, "threshold": 1.0, "reset": 0.0, "gain": 1.5, "tonic": 0.05,
             "noise_hz": 0.0, "noise_amp": 0.22, "seed": 20260920},
  "input": {"population": "PN", "active": 160, "drive": 1.5, "steps": 6, "sustain": 0.5, "code_seed": 3},
  "readout": [{"name": "kc", "population": "KC", "feature": "counts"},
              {"name": "mbon_v", "population": "MBON", "feature": "voltage"},
              {"name": "central_trace", "population": "CENTRAL", "feature": "trace", "tau": 0.1}],
  "classes": {"populations": ["KC", "PN"], "else_col": "superclass"},      (for class-preserving rewiring)
  "view": {"project": ["-x", "z"], "groups": [{"name": "KC", "population": "KC", "color": "#facc15"}],
           "labels": [{"text": "mushroom body", "population": "KC", "half": "right"}]}
}

Dynamics (leaky integrate-and-fire, as in flybrain): each step
    v <- exp(-dt/tau) v + gain * W @ spikes + tonic (+ noise);  v >= threshold -> spike, v = reset
Each character injects its input code (a fixed random subset of the input population, `active`
neurons) for `steps` steps: `drive` on the first, `sustain * drive` after.  Readout features per
character, in spec order:
    counts   spikes per neuron during the character
    voltage  mean pre-reset membrane voltage over the character's steps
    trace    spike trace (decay exp(-dt/tau) per step) at the end of the character
The network state carries over from character to character; reset() silences it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import controls
from .data import Connectome, load
from .select import Selector

SPECS = Path(__file__).parent / "specs"


def builtin_specs() -> list[str]:
    return sorted(p.stem for p in SPECS.glob("*.json"))


def load_spec(ref) -> dict:
    """A spec dict, a built-in name (see builtin_specs()) or a path to a JSON file."""
    if isinstance(ref, dict):
        return ref
    p = Path(ref)
    if not p.suffix and (SPECS / f"{ref}.json").exists():
        p = SPECS / f"{ref}.json"
    if not p.exists():
        raise FileNotFoundError(f"no connectome spec {ref!r}; built-in: {', '.join(builtin_specs())}")
    spec = json.loads(p.read_text(encoding="utf-8"))
    src = spec.get("connectome", {})
    if src.get("loader", "folder") == "folder" and "path" in src and not Path(src["path"]).is_absolute():
        src["path"] = str((p.parent / src["path"]).resolve())      # relative to the spec file
    spec.setdefault("name", p.stem)
    return spec


def _xp(device: str):
    if device in ("auto", "cuda"):
        try:
            import cupy
            from ..brain import _patch_cupy_includes
            _patch_cupy_includes()
            if cupy.cuda.runtime.getDeviceCount() > 0:
                return cupy
        except Exception:                                   # noqa: BLE001
            if device == "cuda":
                raise
    return np


class LIFSubstrate:
    """Frozen spiking network driven by characters.  control: None, {"rewire": "full" | "class",
    "seed": s} (populations are chosen on the real wiring first, so the readout neurons are the
    same cells in the control)."""

    GROUP_OTHER = "other"

    def __init__(self, vocab_size: int, spec, control: dict | None = None, device: str = "auto",
                 connectome: Connectome | None = None, log=print):
        spec = load_spec(spec)
        self.spec, self.control = spec, control
        raw = connectome if connectome is not None else load(spec["connectome"])
        cut = spec["connectome"].get("cut_inputs_to")
        sel = Selector(raw, spec.get("populations"))
        cut_mask = sel.mask(cut) if cut else None
        cx = raw.cut_inputs(cut_mask) if cut else raw
        sel = Selector(cx, spec.get("populations"))           # populations: always on the real wiring
        inp = spec["input"]
        self.pops = {name: sel.ids(name) for name in spec.get("populations", {})}
        in_ids = sel.ids(inp["population"])
        readouts = [(r["name"], sel.ids(r["population"]), r["feature"], float(r.get("tau", 0.1)))
                    for r in spec["readout"]]
        self.control_info = None
        if control and control.get("rewire"):
            # rewire the full connectome, then cut the same inputs (the order of paper_lm's controls)
            labels = (controls.class_labels(sel, spec.get("classes", {})) if control["rewire"] == "class"
                      else None)
            cx, self.control_info = controls.rewire(raw, int(control.get("seed", 0)), labels)
            if cut:
                cx = cx.cut_inputs(cut_mask)
            log(f"control: rewired connectome ({self.control_info['mode']}, seed {self.control_info['seed']}, "
                f"{self.control_info['merged_fraction']:.1%} of edges merged)")
        self.cx, self.sel = cx, sel
        nr = spec.get("neuron", {})
        self.dt, self.tau = float(nr.get("dt", 0.02)), float(nr.get("tau", 0.1))
        self.threshold, self.v_reset = np.float32(nr.get("threshold", 1.0)), np.float32(nr.get("reset", 0.0))
        self.gain, self.tonic = np.float32(nr.get("gain", 1.5)), np.float32(nr.get("tonic", 0.05))
        self.noise_hz, self.noise_amp = float(nr.get("noise_hz", 0.0)), np.float32(nr.get("noise_amp", 0.22))
        self.seed = int(nr.get("seed", 0))
        self.decay = np.float32(np.exp(-self.dt / self.tau))
        self.k, self.drive, self.sustain = int(inp.get("steps", 6)), float(inp.get("drive", 1.5)), float(inp.get("sustain", 0.5))
        active = int(inp["active"])
        if active > len(in_ids):
            raise ValueError(f"input.active = {active} but the input population has {len(in_ids)} neurons")
        rng = np.random.default_rng(int(inp.get("code_seed", 3)))
        codes = [np.sort(in_ids[rng.choice(len(in_ids), size=active, replace=False)]) for _ in range(vocab_size)]
        xp = self.xp = _xp(device)
        # electrical synapses: I_i = gap_gain * sum_j Gn_ij (v_j - v_i), Gn = G with each row divided
        # by max(row sum, 1), so the coupling a neuron feels is bounded
        self.gap_gain = np.float32(nr.get("gap_gain", 0.0))
        Gn = None
        if cx.G is not None and self.gap_gain:
            from scipy import sparse as sp
            deg = np.asarray(cx.G.sum(1)).ravel()
            Gn = (sp.diags((1.0 / np.maximum(deg, 1.0)).astype(np.float32)) @ cx.G).tocsr().astype(np.float32)
        if xp is np:
            self.W, self.G = cx.W, Gn
        else:
            from cupyx.scipy import sparse as cusparse
            self.W = cusparse.csr_matrix(cx.W)
            self.G = cusparse.csr_matrix(Gn) if Gn is not None else None
        self.Gdeg = (xp.asarray(np.asarray(Gn.sum(1)).ravel().astype(np.float32)) if Gn is not None else None)
        self.codes = [xp.asarray(c) for c in codes]
        self.readouts = [(name, xp.asarray(ids), feat, np.float32(np.exp(-self.dt / tau)))
                         for name, ids, feat, tau in readouts]
        self.readout_sizes = {name: len(ids) for name, ids, _, _ in readouts}
        self.n_features = sum(self.readout_sizes.values())
        vids = [ids for _, ids, feat, _ in readouts if feat == "voltage"]
        self.record = xp.asarray(np.concatenate(vids).astype(np.int64)) if vids else None
        self._view = spec.get("view", {})
        self.group = self._groups()
        self.record_activity = False
        self.last_activity = None
        self.reset()

    # ------------------------------------------------------------------ dynamics
    def reset(self):
        xp = self.xp
        self.rng = xp.random.default_rng(self.seed)
        self.v = xp.zeros(self.cx.n, xp.float32)
        self.spikes = xp.zeros(self.cx.n, xp.float32)
        self.traces = {name: xp.zeros(len(ids), xp.float32) for name, ids, feat, _ in self.readouts if feat == "trace"}

    def _step(self):
        xp = self.xp
        current = self.W @ self.spikes * self.gain
        gap = (self.G @ self.v - self.Gdeg * self.v) * self.gap_gain if self.G is not None else None
        self.v *= self.decay
        self.v += current + self.tonic
        if gap is not None:
            self.v += gap
        if self.noise_hz:
            self.v += (self.rng.random(self.cx.n) < self.noise_hz * self.dt) * self.noise_amp
        v_pre = self.v[self.record] if self.record is not None else None
        fired = xp.flatnonzero(self.v >= self.threshold)
        self.v[fired] = self.v_reset
        self.spikes = xp.zeros(self.cx.n, xp.float32)
        self.spikes[fired] = 1.0
        return fired, v_pre

    def step_token(self, token: int) -> np.ndarray:
        xp = self.xp
        counts = {name: xp.zeros(len(ids), xp.float32) for name, ids, feat, _ in self.readouts if feat == "counts"}
        volts, fired_steps = [], []
        for step in range(self.k):
            self.v[self.codes[int(token)]] += np.float32(self.drive * (1.0 if step == 0 else self.sustain))
            fired, v_pre = self._step()
            if self.record_activity:
                fired_steps.append(fired)
            if v_pre is not None:
                volts.append(xp.asnumpy(v_pre) if xp is not np else v_pre.copy())
            for name, ids, feat, dec in self.readouts:
                if feat == "counts":
                    counts[name] += self.spikes[ids]
                elif feat == "trace":
                    self.traces[name] *= dec
                    self.traces[name] += self.spikes[ids]
        vmean = np.asarray(volts, np.float32).mean(0) if volts else None
        out, vo = [], 0
        for name, ids, feat, _ in self.readouts:
            if feat == "counts":
                out.append(self._host(counts[name]))
            elif feat == "trace":
                out.append(self._host(self.traces[name]))
            elif feat == "voltage":
                out.append(vmean[vo:vo + len(ids)]); vo += len(ids)
            else:
                raise ValueError(f"unknown readout feature {feat!r} (counts, voltage, trace)")
        if self.record_activity:
            self._activity(fired_steps)
        return np.concatenate(out).astype(np.float32)

    def _host(self, a):
        return a if self.xp is np else self.xp.asnumpy(a)

    # ------------------------------------------------------------------ brain view
    def _groups(self) -> np.ndarray:
        g = np.zeros(self.cx.n, np.uint8)
        for i, grp in enumerate(self._view.get("groups", []), 1):
            g[self.sel.ids(grp["population"])] = i
        return g

    def group_names(self) -> list[str]:
        return [self.GROUP_OTHER] + [grp["name"] for grp in self._view.get("groups", [])]

    def _activity(self, fired_steps):
        xp = self.xp
        spikes = int(sum(int(f.size) for f in fired_steps))
        ids = (self._host(xp.unique(xp.concatenate(fired_steps))) if spikes else np.zeros(0, np.int64))
        names = self.group_names()
        active = np.bincount(self.group[ids], minlength=len(names)) if len(ids) else np.zeros(len(names), int)
        self.last_activity = {"ids": ids.astype(np.uint32), "spikes": spikes,
                              "active": {g: int(active[i]) for i, g in enumerate(names)}}

    def map_payload(self, max_px: int = 4095) -> dict:
        """Neuron positions projected to 2-D (spec view.project, e.g. ["-x", "z"]), quantised to uint16,
        with group codes, colours and region labels, for the web UI's brain view."""
        import base64
        if self.cx.positions is None:           # no anatomy: a schematic, one column per view group
            rng = np.random.default_rng(0)
            pos = np.zeros((self.cx.n, 3))
            pos[:, 0] = -(self.group.astype(float) + rng.uniform(-0.3, 0.3, self.cx.n))
            pos[:, 2] = rng.uniform(0, max(1, len(self.group_names()) - 1), self.cx.n)
        else:
            pos = np.asarray(self.cx.positions, np.float64)
        axes = []
        for a in self._view.get("project", ["-x", "z"]):
            col = pos[:, "xyz".index(a[-1])].copy()
            axes.append(-col if a.startswith("-") else col)
        x, z = axes
        for a in (x, z):
            a[~np.isfinite(a)] = np.nanmedian(a)
        x0, x1 = np.percentile(x, [0.2, 99.8]); z0, z1 = np.percentile(z, [0.2, 99.8])
        span = max(x1 - x0, z1 - z0)
        u = np.clip((x - x0) / span, 0, 1); v = np.clip((z - z0) / span, 0, 1)
        q = np.stack([u, v], 1) * max_px
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
        right = u >= np.median(u)
        labels = []
        for lab in self._view.get("labels", []):
            m = self.sel.mask(lab["population"])
            halves = {"left": [~right], "right": [right], "both": [~right, right]}.get(lab.get("half", "all"), [None])
            for h in halves:
                mm = m if h is None else m & h
                if mm.any():
                    labels.append({"text": lab["text"], "u": round(float(u[mm].mean()), 4),
                                   "v": round(float(v[mm].mean()), 4)})
        names = self.group_names()
        return {"n": int(len(u)), "scale": max_px, "w": round(float((x1 - x0) / span), 4),
                "h": round(float((z1 - z0) / span), 4), "xy": b64(q.astype(np.uint16)),
                "group": b64(self.group), "groups": names, "labels": labels,
                "colors": [self._view.get("other_color", "#f59e0b")] + [g.get("color", "#cccccc")
                                                                        for g in self._view.get("groups", [])],
                "descriptions": [self._view.get("other_text", "everything else")] +
                                [g.get("text", g["name"]) for g in self._view.get("groups", [])],
                "totals": {g: int((self.group == i).sum()) for i, g in enumerate(names)},
                **{k: self._view[k] for k in ("title", "caption", "credit") if k in self._view}}
