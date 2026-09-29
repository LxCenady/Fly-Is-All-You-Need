"""Raster icons from the final logo geometry: app icon PNGs (green tile, white mark) and a
multi-size Windows .ico.  Output: gpf/assets/gpf-icon-{16..512}.png, gpf/assets/gpf.ico."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from PIL import Image

import make_logo_final as L

OUT = Path(__file__).parent / "gpf" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
GREEN = "#10a37f"


def icon_png(px, path):
    fig = plt.figure(figsize=(1, 1), dpi=px)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(100, 0); ax.axis("off")
    fig.patch.set_alpha(0)
    ax.add_patch(FancyBboxPatch((0, 0), 100, 100, boxstyle="round,pad=0,rounding_size=22", fc=GREEN, ec="none"))
    x0, y0, x1, y1 = L.bbox()
    pad = L.STROKE / 2 + 2
    s = 100 / (max(x1 - x0, y1 - y0) + 2 * pad) * 0.72            # same framing as gpf-app-icon.svg
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lw = L.STROKE * s * px / 100 * 72 / px                          # stroke units -> points at this dpi
    for h in L.hexagons():
        pts = [(50 + s * (x - cx), 50 + s * (y - cy)) for x, y in h]
        ax.add_patch(Polygon(pts, closed=True, fill=False, ec="white", lw=lw, joinstyle="round"))
    fig.savefig(path, dpi=px, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    sizes = [16, 24, 32, 48, 64, 128, 256, 512]
    for px in sizes:
        icon_png(px, OUT / f"gpf-icon-{px}.png")
    Image.open(OUT / "gpf-icon-256.png").save(OUT / "gpf.ico", sizes=[(s, s) for s in sizes if s <= 256])
    print("ok", sorted(p.name for p in OUT.iterdir()))
