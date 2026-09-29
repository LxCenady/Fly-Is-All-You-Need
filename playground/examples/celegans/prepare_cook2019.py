"""C. elegans hermaphrodite connectome (Cook et al. 2019, Nature 571:63-71) for GPF.

Downloads two small files from OpenWorm's ConnectomeToolbox (MIT; ~320 KB) unless present,
and writes neurons.csv and edges.csv in GPF's import format:

    python prepare_cook2019.py
    python -m gpf import --neurons neurons.csv --edges edges.csv --out cook2019 \\
        --sign-col nt --inhibitory gaba --type-col type
    python -m gpf bench --substrate celegans-cook2019.json --data <text> ...

Cells: every cell in the edge list (neurons, muscles and a few other cells), with a coarse class
(sensory, interneuron, motor, pharynx, muscle, other) from the cell table's "Type".  Signs: the
26 GABAergic neurons of McIntire et al. 1993 (DD1-6, VD1-13, RMED/V/L/R, AVL, DVB, RIS) are
inhibitory, every other cell excitatory.  This is a simplification: glutamate and acetylcholine
also act through inhibitory receptors in the worm, and a full neurotransmitter atlas exists
(Wang et al. 2024, eLife 13:RP95402).  No positions: the brain view shows a schematic.
"""
import csv
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw"
BASE = "https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data/"
FILES = ["herm_full_edgelist.csv", "all_cell_info.csv"]
GABA = ({f"DD{i}" for i in range(1, 7)} | {f"VD{i}" for i in range(1, 14)}
        | {"RMED", "RMEV", "RMEL", "RMER", "AVL", "DVB", "RIS"})
SENSORY = ("amphid", "cephalic", "mechanosensory", "touch", "phasmid", "sensory", "o2", "nocicept", "labial",
           "deirid")


def name(s: str) -> str:
    """DD01 -> DD1, AS09 -> AS9 (the edge list pads numbers; the cell table does not)."""
    s = s.strip()
    return re.sub(r"^([A-Z]+)0+(\d+)$", r"\1\2", s)


def cell_class(typ: str, cell: str) -> str:
    t = typ.lower()
    if "muscle" in t or "BWM" in cell or cell.startswith(("pm", "vm", "um")):
        return "muscle"
    if "pharyngeal" in t:
        return "pharynx"
    if "motor" in t:
        return "motor"
    if "interneuron" in t:
        return "interneuron"
    if any(k in t for k in SENSORY):
        return "sensory"
    return "other"


def main():
    RAW.mkdir(exist_ok=True)
    for f in FILES:
        if not (RAW / f).exists():
            print("downloading", f)
            urllib.request.urlretrieve(BASE + f, RAW / f)
    info = {name(r["Cell name"]): r for r in csv.DictReader(open(RAW / FILES[1], encoding="utf-8"))}
    edges = [(name(r["Source"]), name(r["Target"]), r["Weight"].strip(), r["Type"].strip())
             for r in csv.DictReader(open(RAW / FILES[0], encoding="utf-8"))]
    cells = sorted({e[0] for e in edges} | {e[1] for e in edges})
    rows = []
    for c in cells:
        typ = info.get(c, {}).get("Type", "")
        rows.append({"id": c, "cell_type": c, "type_desc": typ or "unknown", "class": cell_class(typ, c),
                     "nt": "GABA" if c in GABA else "other"})
    with open(HERE / "neurons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(HERE / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["pre", "post", "weight", "type"]); w.writerows(edges)
    from collections import Counter
    print(len(rows), "cells:", dict(Counter(r["class"] for r in rows)))
    print("GABAergic found:", sum(r["nt"] == "GABA" for r in rows), "of", len(GABA))
    print(len(edges), "edges:", dict(Counter(e[3] for e in edges)))


if __name__ == "__main__":
    main()
