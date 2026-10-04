# Cross-connectome memory experiments (work in progress)

Exploratory scripts behind the cross-connectome comparison (worm, larva, adult fly) and the
follow-up studies of memory decay (q) and encoding strength (x0). One script, one job; outputs
go to folders next to the scripts (`features*/`, `logs*/`, `target/`, `results_*.json`), which
are git-ignored.

- `paths.py`: repository location (`FLY_REPO`, default: the author's checkout). Shell scripts
  take the interpreter from `PY`.
- `simulate.py`: one condition (spec, gain, input code; `--shuffle SEED` for shuffled text;
  `--set PATH=VALUE` to override the spec), then cache the readout features.
- `analyze.py`: memory span from cached features (`--features both|spikes|trace`).
- `grid.sh SPEC PREFIX DEVICE LIST`: run a list of conditions on shuffled text.
- `q_fit.py`, `x0_table.py`: encoding term x0 and decay from the memory curves.
- `window_test.py`, `hop_test.py`: where in the token window the memory sits, and whether it
  follows path latency.
- `colab_fly.ipynb`: run the fly grid (`x0_fly.list`) on a Colab GPU.

The worm and larva specs point at local connectome folders. Build them first with
`uctf/examples/celegans` and `uctf/examples/larva/prepare_winding2023.py`, then rerun
`protocol.py`. The fly spec downloads its data through flybrain.
