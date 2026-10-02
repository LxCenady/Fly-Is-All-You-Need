<p align="center"><img src="playground/gpf/assets/gpf-app-icon.svg" width="112" alt="GPF logo"></p>

# Fly Is All You Need

## GPF: Generative Pretrained Fly

**A text generator whose brain is a simulated fruit fly.** Type the start of a line and GPF
writes on, one character at a time. Its main model, GPF-1, is a complete fruit-fly nervous
system (166,700 simulated neurons, wired as in the real animal). You can watch its neurons
fire while it writes, and compare it side by side with an n-gram model and a small GRU.

Download it from [Releases](https://github.com/LxCenady/Fly-Is-All-You-Need/releases). No
GPU, Python or internet connection needed:

- **Windows:** `gpf-full-…-windows-x64.exe` (with the fly brain, ~160 MB), double-click it.
- **Ubuntu / Debian:** `gpf-full_…_amd64.deb`, then `sudo apt install ./gpf-full_…_amd64.deb`.
- **Lite** (`gpf-lite-…`, ~20 MB): everything except the fly brain; `gpf get-brain` downloads it
  later.

GPF-1 runs on an ordinary CPU (about 100 characters a second on a laptop) with exactly the
same spikes as on a GPU. The simulation comes from [flybrain](https://github.com/alextitonis/fly.ai)
(MIT); see the [GPF README](playground/).

## UCTF: bring your own connectome

GPF's connectome models come from [UCTF](uctf/), the Universal Connectome Training Framework:
import any wiring diagram, describe it in a JSON spec, and train and benchmark it against
n-gram baselines and rewired copies of itself. It has been run on two species so far: the fly
(MaleCNS v1.0) and the whole C. elegans worm (Cook et al. 2019). Both show the same pattern:
a few characters of memory, far behind a 5-gram, and the real wiring is not special.

UCTF is a small kernel plus plugins. Connectome sources, wiring transforms, neuron and synapse
models (including per-transmitter synapses and gap junctions), input encoders, readout
features, tasks, probes and baselines are all named plugins that a spec composes. Other
packages can add plugins through the `uctf.plugins` entry-point group. The protocol is in
[`uctf/PLUGINS.md`](uctf/PLUGINS.md).

## The research behind it

This is independent research (a high-school project) on a spiking model of the adult male
*Drosophila* connectome (MaleCNS v1.0). All findings are about the model, not claims about
the fly. There are two papers:

- **[Three Addresses of a Connectome-Constrained Mushroom-Body Memory](paper/mechanism_paper.pdf)**:
  where a mushroom-body memory is stored, what retrieves it and how it reaches downstream
  neurons, tested with causal interventions.
- **[Teaching a Fly Connectome to Predict Text](paper_lm/lm_report.pdf)**: the language-model
  project that GPF comes from. The result is negative: the frozen connectome acts as a
  roughly four-character memory, and a 5-gram model beats it.

Each number in both papers is traced to its script and output file in
[`docs/CLAIMS_EVIDENCE.md`](docs/CLAIMS_EVIDENCE.md). A simulator bug that affected
earlier plasticity results, and the corrections it forced, are documented in
[`audit_20260928/`](audit_20260928/) and
[`mechanism/PAPER_OVERTURN_20260928.md`](mechanism/PAPER_OVERTURN_20260928.md).

## Repository

```
playground/       GPF (web UI, terminal UI, packaging)
uctf/             UCTF, the connectome training framework (package, examples, tests)
paper/            mechanism paper (PDF, LaTeX source, figures)
paper_lm/         language-model report (PDF, LaTeX source, figures)
mechanism/        experiment scripts; RESULTS_20260928.md = result register
harness/          simulation harness used by mechanism/
results/          JSON outputs behind the figures
audit_20260928/   evidence for the simulator bug
docs/             claims-to-evidence map, literature index
```

## Reproducing

Everything needed is in [`repro/`](repro/). [`repro/ENVIRONMENT.md`](repro/ENVIRONMENT.md) has
the details; in short (the research code needs an NVIDIA GPU; the ~260 MB connectome downloads on first use):

```
python -m pip install --require-hashes -r repro/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
python repro/make.py env               # the environment matches the lock
python repro/make.py test              # simulator regression test + framework tests
python repro/make.py verify-figures    # every figure of both papers, rebuilt from results/
python repro/make.py headline OUT      # rerun the headline experiments (about 1 h), then:
python repro/make.py compare OUT
```

- **Use the patched simulator.** flybrain 0.1.0 from PyPI moves cached synapse offsets at the
  first GPU sparse product, so plasticity writes land on the wrong synapses. The lock installs
  the fixed fork, flybrain 0.1.0.post1 ([release](https://github.com/LxCenady/Fly-Is-All-You-Need/releases/tag/flybrain-0.1.0.post1),
  [`third_party/flybrain`](third_party/flybrain/), proposed upstream as
  [alextitonis/fly.ai#10](https://github.com/alextitonis/fly.ai/pull/10)).
- **Where each result comes from.** `results/MANIFEST.json` records the hash and provenance of
  every result file. [`repro/RERUN_REPORT.md`](repro/RERUN_REPORT.md) shows the headline
  experiments rerun in the pinned environment.
- **How strong each claim is.** [`docs/CLAIMS_EVIDENCE.md`](docs/CLAIMS_EVIDENCE.md) classifies
  every claim as in-model empirical, by construction or exploratory. Confidence intervals and
  the pre-registered confirmatory run are in `repro/`.
- **Nothing is hard-coded.** `mechanism/paths.py` reads data and output locations from
  environment variables (`FLY_DATA`, `FLY_OUT`, …) or a git-ignored
  `mechanism/paths_local.json`.

## Credits and license

- Connectome: MaleCNS v1.0 by FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and
  Google Research, CC BY 4.0. Berg, S. et al. (2026), *Cell* 189(18), 5504–5526.e15.
- Simulator: [flybrain](https://github.com/alextitonis/fly.ai) by Alex Titonis (MIT).
- Code in this repository: MIT (see `LICENSE`).
- AI tools (Anthropic Claude, OpenAI Codex, DeepSeek via OpenCode) were used extensively
  for code, experiments, audits and drafting. The author directed the research and is
  responsible for the content; see the statements in both papers.
