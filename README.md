<p align="center"><img src="playground/gpf/assets/gpf-app-icon.svg" width="112" alt="GPF logo"></p>

# Fly Is All You Need

## GPF: Generative Pretrained Fly

**A text generator whose brain is a simulated fruit fly.** Type the start of a line and GPF
writes on, one character at a time. Its main model, GPF-1, is a complete fruit-fly nervous
system (166,700 simulated neurons, wired as in the real animal). You can watch its neurons
fire while it writes, and compare it side by side with an n-gram model and a small GRU.

- **Windows:** download `gpf-lite-…-windows-x64.exe` from
  [Releases](https://github.com/LxCenady/Fly-Is-All-You-Need/releases) and double-click it.
- **Ubuntu / Debian:** download `gpf-lite_…_amd64.deb`, then `sudo apt install ./gpf-lite_…_amd64.deb`.
- **The fly model (GPF-1)** needs an NVIDIA GPU and [flybrain](https://github.com/alextitonis/fly.ai)
  (required dependency). See the [GPF README](playground/) for setup.

## UCTF: bring your own connectome

GPF's connectome models come from [UCTF](uctf/), the Universal Connectome Training Framework:
import any wiring diagram, describe it in a JSON spec, and train and benchmark it against
n-gram baselines and rewired copies of itself. It has been run on two species so far: the fly
(MaleCNS v1.0) and the whole C. elegans worm (Cook et al. 2019). Both show the same pattern:
a few characters of memory, far behind a 5-gram, and the real wiring is not special.

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

To reproduce the experiments (NVIDIA GPU needed; flybrain downloads the ~260 MB connectome
to `~/fly-data` on first use):

```
pip install "flybrain[gpu]==0.1.0" scipy
python mechanism/lm_mech.py --out outputs/test.json --train 900 --val 100 --active 160 --scale 1.5
```

Nothing is hard-coded: `mechanism/paths.py` reads data and output locations from
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
