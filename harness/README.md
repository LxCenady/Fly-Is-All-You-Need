# harness

The experiment harness from the earlier stage of the project (originally `flybrain_lm/lm/`):
token encoding, the frozen/plastic mushroom-body probe, the KC->MBON plasticity rule and the
reference protocol. The research scripts in `../mechanism` build on it through
`mechanism/m1_core.py`. Only the 15 modules those scripts actually import are included; they
are unchanged except that `sitecustomize.py` no longer hard-codes an install path.

Needs [flybrain](https://github.com/alextitonis/fly.ai) with GPU support
(`pip install "flybrain[gpu]==0.1.0"`). Paths are configured in `../mechanism/paths.py`.
