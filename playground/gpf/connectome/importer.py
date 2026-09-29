"""Turn a neuron table and an edge list into a connectome folder that GPF can simulate.

    python -m gpf import --neurons neurons.csv --edges edges.csv --out my-connectome \\
        [--id-col id --pre-col pre --post-col post --weight-col weight] \\
        [--sign-col nt --inhibitory gaba,glutamate,histamine] [--no-normalise]

neurons.csv: one row per neuron with an id column and any annotation columns (cell type, class,
side, ...); x, y, z columns (optional) give positions for the brain view.  edges.csv: one row per
connection (pre id, post id, synapse count).  Weights are synapse counts, negative when the
presynaptic neuron's sign column names an inhibitory transmitter, and each neuron's inputs are
divided by its total absolute input (at least 1), flybrain's recipe for MaleCNS.  Then write a
spec (see gpf/connectome/specs/ and examples/) that points at the folder.
"""
from __future__ import annotations

import argparse

from .data import import_edges


def main(argv):
    ap = argparse.ArgumentParser(prog="gpf import", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--pre-col", default="pre")
    ap.add_argument("--post-col", default="post")
    ap.add_argument("--weight-col", default="weight")
    ap.add_argument("--sign-col", help="column whose value marks inhibitory neurons (e.g. neurotransmitter)")
    ap.add_argument("--inhibitory", default="gaba,glutamate,histamine",
                    help="comma-separated substrings of --sign-col meaning inhibitory")
    ap.add_argument("--no-normalise", action="store_true", help="keep raw signed synapse counts")
    ap.add_argument("--type-col", help="edge column marking electrical synapses (gap junctions)")
    ap.add_argument("--electrical", default="electrical", help="value of --type-col meaning electrical")
    a = ap.parse_args(argv)
    cx = import_edges(a.neurons, a.edges, a.out, a.id_col, a.pre_col, a.post_col, a.weight_col, a.sign_col,
                      tuple(s.strip().lower() for s in a.inhibitory.split(",") if s.strip()),
                      not a.no_normalise, type_col=a.type_col, electrical=a.electrical.lower())
    inh = int((cx.W.data < 0).sum())
    gap = f", {cx.G.nnz // 2:,} gap junctions" if cx.G is not None else ""
    print(f"{cx.n:,} neurons, {cx.W.nnz:,} connections ({inh:,} inhibitory){gap}; annotations: "
          f"{', '.join(cx.annotations)}; positions: {'yes' if cx.positions is not None else 'no'}\n"
          f"written to {a.out}")
