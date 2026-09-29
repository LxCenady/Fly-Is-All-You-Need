"""GPF logo, final: V8c geometry with fully continuous outlines.

Six pointy-top hexagons around the centre (ring distance 21, circumradius 15.6), the left and
right trios pulled together so the inner vertical edges of the top pair and of the bottom pair
fuse on the midline (left/right hemispheres).  Neighbouring links overlap into double bands
(the interlock) and enclose a central hexagon (a facet of the compound eye).  Every outline is a
closed path; the mark uses currentColor.
"""
import math
from pathlib import Path

OUT = Path(__file__).parent / "logo"
C, D, R, STROKE = 50.0, 21.0, 15.6, 3.6


def hexagons():
    apothem = R * math.cos(math.radians(30))
    shift = apothem - D * math.cos(math.radians(60))
    out = []
    for a in (300, 0, 60, 120, 180, 240):
        cx = C + D * math.cos(math.radians(a)); cy = C + D * math.sin(math.radians(a))
        cx += shift if cx > C else -shift
        out.append([(cx + R * math.cos(math.radians(30 + 60 * k)), cy + R * math.sin(math.radians(30 + 60 * k)))
                    for k in range(6)])
    return out


def bbox():
    pts = [p for h in hexagons() for p in h]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def body(color="currentColor", stroke=STROKE):
    x0, y0, x1, y1 = bbox()
    pad = stroke / 2 + 2
    s = 100 / (max(x1 - x0, y1 - y0) + 2 * pad)                  # fit the mark into 0-100
    tx, ty = 50 - s * (x0 + x1) / 2, 50 - s * (y0 + y1) / 2
    paths = "".join('<path d="M' + " L".join(f"{x:.3f} {y:.3f}" for x, y in h) + ' Z"/>' for h in hexagons())
    return (f'<g transform="translate({tx:.3f} {ty:.3f}) scale({s:.4f})" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linejoin="round">{paths}</g>')


def svg(color="currentColor", size=None, bg=None, radius=0):
    wh = f' width="{size}" height="{size}"' if size else ""
    back = f'<rect width="100" height="100" rx="{radius}" fill="{bg}"/>' if bg else ""
    inner = body(color) if not bg else f'<g transform="translate(50 50) scale(0.72) translate(-50 -50)">{body(color)}</g>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"{wh}>{back}{inner}</svg>'


if __name__ == "__main__":
    (OUT / "gpf-logo.svg").write_text(svg(), encoding="utf-8")                       # currentColor
    (OUT / "gpf-logo-black.svg").write_text(svg("#111111", 512), encoding="utf-8")
    (OUT / "gpf-app-icon.svg").write_text(svg("#ffffff", 512, bg="#10a37f", radius=22), encoding="utf-8")
    sheet = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 300" width="820" height="300">'
             '<rect width="820" height="300" fill="#f2f2f0"/>'
             '<rect x="20" y="20" width="260" height="260" rx="20" fill="#fff"/>'
             f'<g transform="translate(40 40) scale(2.2)">{body("#111")}</g>'
             '<rect x="300" y="20" width="260" height="260" rx="20" fill="#171717"/>'
             f'<g transform="translate(320 40) scale(2.2)">{body("#ececec")}</g>'
             f'<g transform="translate(590 30) scale(0.64)">{body("#111")}</g>'
             f'<g transform="translate(680 40) scale(0.32)">{body("#111")}</g>'
             f'<g transform="translate(730 45) scale(0.16)">{body("#111")}</g>'
             '<rect x="590" y="130" width="100" height="100" rx="22" fill="#10a37f"/>'
             f'<g transform="translate(604 144) scale(0.72)">{body("#fff")}</g>'
             '<rect x="710" y="150" width="60" height="60" rx="13" fill="#10a37f"/>'
             f'<g transform="translate(718 158) scale(0.44)">{body("#fff")}</g>'
             '</svg>')
    (OUT / "final_preview.svg").write_text(sheet, encoding="utf-8")
    print("ok")
