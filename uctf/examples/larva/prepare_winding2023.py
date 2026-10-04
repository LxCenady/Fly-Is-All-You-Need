"""Drosophila larva brain connectome (Winding et al. 2023, Science 379:eadd9330) for UCTF.

Downloads the paper's Supplementary Data S1 (~1.1 MB zip; mirrored by the Betzel lab at
github.com/brain-networks/larval-drosophila-connectome) unless present, checks its SHA-256,
and writes neurons.csv and edges.csv in UCTF's import format:

    python prepare_winding2023.py
    python -m uctf import --neurons neurons.csv --edges edges.csv --out winding2023 \\
        --sign-col nt_assumed --inhibitory inhibitory
    python select_gain.py                  # operating point, by activity only
    python -m uctf bench --substrate larva-winding2023.json --data <text> ...

Neurons: the 2,952 neurons of the all-to-all connectivity matrix; 2,606 carry a cell type
from annotations.csv, the other 346 are "unannotated". Edges: the all-to-all matrix (rows =
presynaptic, columns = postsynaptic; checked: KC->MBON 17,752 synapses, MBON->KC 195), synapse
counts summed over axon/dendrite compartments. Signs: the supplementary data has no
neurotransmitters. Neurons of cell type LN (local neurons, including the GABAergic broad LNs
and the APL) are taken as inhibitory, every other neuron as excitatory. This is a strong
simplification. No positions: the brain view shows a schematic.
"""
import csv
import hashlib
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw"
URL = ("https://raw.githubusercontent.com/brain-networks/larval-drosophila-connectome/"
       "main/Supplementary-Data-S1.zip")
SHA256 = "8c1f43809ed5d527ba61b154e377cc21da26383a75eda8aab85ce05607a72a4c"
INHIBITORY = {"LN"}
DIR = RAW / "Supplementary-Data-S1"


def fetch():
    RAW.mkdir(exist_ok=True)
    z = RAW / "Supplementary-Data-S1.zip"
    if not z.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, z)
    if hashlib.sha256(z.read_bytes()).hexdigest() != SHA256:
        raise SystemExit(f"{z} does not match the expected SHA-256; delete it and retry")
    if not DIR.exists():
        zipfile.ZipFile(z).extractall(RAW)


def annotations():
    """skeleton id -> (cell type, side, pair partner, extra annotation)."""
    out = {}
    with open(DIR / "annotations.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            left, right = r["left_id"], r["right_id"]
            for sid, side, partner in ((left, "left", right), (right, "right", left)):
                if sid != "no pair":
                    pair = partner if partner != "no pair" else ""
                    out[sid] = (r["celltype"], side, pair, r["additional_annotations"])
    return out


def main():
    fetch()
    ann = annotations()
    with open(DIR / "all-all_connectivity_matrix.csv", encoding="utf-8") as f:
        ids = f.readline().strip().split(",")[1:]
        edges = []
        for line in f:
            row = line.strip().split(",")
            pre = row[0]
            for post, w in zip(ids, row[1:]):
                if w not in ("0", "0.0", ""):
                    edges.append((pre, post, w))
    rows = []
    for sid in ids:
        ctype, side, pair, extra = ann.get(sid, ("unannotated", "", "", ""))
        nt = "inhibitory (assumed)" if ctype in INHIBITORY else "excitatory (assumed)"
        rows.append({"id": sid, "cell_type": ctype, "class": ctype, "side": side,
                     "pair": pair, "annotation": extra, "nt_assumed": nt})
    with open(HERE / "neurons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(HERE / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pre", "post", "weight"])
        w.writerows(edges)
    from collections import Counter
    print(len(rows), "neurons:", dict(Counter(r["class"] for r in rows).most_common()))
    print(len(edges), "connections,", int(sum(float(e[2]) for e in edges)), "synapses")


if __name__ == "__main__":
    main()
