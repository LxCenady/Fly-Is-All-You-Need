"""The brain view: neuron positions and activity groups for a drawing.

Configured by the spec's "view" section (all optional):

    "view": {"project": ["-x", "z"],
             "groups": [{"name": "KC", "population": "KC",
                         "color": "#facc15", "text": "Kenyon cells"}],
             "labels": [{"text": "MB", "population": "KC", "half": "both"}],
             "other_color": "#f59e0b", "other_text": "everything else",
             "title": "...", "caption": "...", "credit": "..."}

Group 0 is every neuron in no group. Without positions the view is a
schematic: one column per group.
"""
from __future__ import annotations

import base64

import numpy as np

OTHER = "other"


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


class View:
    def __init__(self, net):
        self.net = net
        self.cfg = net.spec.get("view", {})
        groups = self.cfg.get("groups", [])
        self.names = [OTHER] + [g["name"] for g in groups]
        self.group = np.zeros(net.cx.n, np.uint8)
        for i, g in enumerate(groups, 1):
            self.group[net.sel.ids(g["population"])] = i

    def activity(self, fired: dict | None) -> dict | None:
        """Spikes and active neurons per group in the last token."""
        if fired is None:
            return None
        ids = fired["ids"]
        active = np.bincount(self.group[ids], minlength=len(self.names))
        per_group = {g: int(active[i]) for i, g in enumerate(self.names)}
        return {**fired, "active": per_group}

    def payload(self, max_px: int = 4095) -> dict:
        """Positions projected to 2-D and quantised to uint16, with group
        codes, colours and region labels."""
        u, v, w, h = self._project()
        cfg = self.cfg
        groups = cfg.get("groups", [])
        colors = [cfg.get("other_color", "#f59e0b")]
        colors += [g.get("color", "#cccccc") for g in groups]
        texts = [cfg.get("other_text", "everything else")]
        texts += [g.get("text", g["name"]) for g in groups]
        totals = {g: int((self.group == i).sum())
                  for i, g in enumerate(self.names)}
        xy = (np.stack([u, v], 1) * max_px).astype(np.uint16)
        extra = {k: cfg[k] for k in ("title", "caption", "credit")
                 if k in cfg}
        return {"n": int(len(u)), "scale": max_px,
                "w": round(w, 4), "h": round(h, 4),
                "xy": _b64(xy), "group": _b64(self.group),
                "groups": self.names, "labels": self._labels(u, v),
                "colors": colors, "descriptions": texts,
                "totals": totals, **extra}

    def _positions(self) -> np.ndarray:
        cx = self.net.cx
        if cx.positions is not None:
            return np.asarray(cx.positions, np.float64)
        rng = np.random.default_rng(0)            # schematic
        pos = np.zeros((cx.n, 3))
        jitter = rng.uniform(-0.3, 0.3, cx.n)
        pos[:, 0] = -(self.group.astype(float) + jitter)
        pos[:, 2] = rng.uniform(0, max(1, len(self.names) - 1), cx.n)
        return pos

    def _project(self):
        pos = self._positions()
        axes = []
        for a in self.cfg.get("project", ["-x", "z"]):
            col = pos[:, "xyz".index(a[-1])].copy()
            axes.append(-col if a.startswith("-") else col)
        x, z = axes
        for a in (x, z):
            a[~np.isfinite(a)] = np.nanmedian(a)
        x0, x1 = np.percentile(x, [0.2, 99.8])
        z0, z1 = np.percentile(z, [0.2, 99.8])
        span = max(x1 - x0, z1 - z0)
        u = np.clip((x - x0) / span, 0, 1)
        v = np.clip((z - z0) / span, 0, 1)
        return u, v, float((x1 - x0) / span), float((z1 - z0) / span)

    def _labels(self, u, v) -> list:
        right = u >= np.median(u)
        halves = {"left": [~right], "right": [right],
                  "both": [~right, right]}
        out = []
        for lab in self.cfg.get("labels", []):
            m = self.net.sel.mask(lab["population"])
            for h in halves.get(lab.get("half", "all"), [None]):
                mm = m if h is None else m & h
                if mm.any():
                    out.append({"text": lab["text"],
                                "u": round(float(u[mm].mean()), 4),
                                "v": round(float(v[mm].mean()), 4)})
        return out
