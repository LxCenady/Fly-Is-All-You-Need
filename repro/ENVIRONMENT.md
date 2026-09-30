# Reproducing the results: environment, data, commands

## 1. Environment

The recorded environment is:

| | |
|---|---|
| OS | Windows 11 (x64); the Linux CI builds GPF on Ubuntu |
| Python | 3.12.14 |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop (8 GB), driver 591.59, CUDA runtime 12.9 (via `cuda-toolkit` wheels) |
| Simulator | **flybrain 0.1.0.post1**, the patched fork in `third_party/flybrain` (see below) |
| Everything else | exact versions and sha256 hashes in [`requirements-lock.txt`](requirements-lock.txt) |

Install into a fresh virtual environment:

```
python -m venv .venv && .venv/Scripts/activate        # or: source .venv/bin/activate
python -m pip install --require-hashes -r repro/requirements-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
python repro/make.py env                                # check versions against the lock
```

`torch` is used only by the GRU baselines; drop its line from the lock if you do not need them.

### Why a fork of flybrain

flybrain 0.1.0 as released on PyPI has a bug: the first GPU sparse product can re-sort the weight
matrix in place, which moves cached edge offsets. Every plasticity experiment then writes to
the wrong synapses, silently.

The fix is 8 lines and is published as `flybrain 0.1.0.post1`:
- release asset:
  <https://github.com/LxCenady/Fly-Is-All-You-Need/releases/tag/flybrain-0.1.0.post1>,
  sha256 `6c46867accf76b82e181bbe682e3f581e77d32e494d5c96fc135512e5e6b5d5c`;
- rebuild bit-for-bit with `third_party/flybrain/build_wheel.sh`;
- proposed upstream as <https://github.com/alextitonis/fly.ai/pull/10>.

The exact diff is in `third_party/flybrain/PATCH.md`. **Do not use flybrain 0.1.0 from PyPI.**
`mechanism/m1_core.py` refuses to run on it: it checks edge identity after the first sparse
product.

## 2. Data

| File | Source | sha256 |
|---|---|---|
| `brain.npz` | flybrain's prebuilt MaleCNS v1.0 (downloaded on first use; flybrain checks the hash) | `cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca` |
| `weights.npz` | same | `c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c` |
| `playground/gpf/data/tinyshakespeare.txt` | Karpathy's char-rnn (in the repository) | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |

Locations come from environment variables or a git-ignored `mechanism/paths_local.json`: see
`mechanism/paths.py`. The only one you usually need is `FLY_DATA` (the folder holding the two
brain files).

## 3. Commands

```
python repro/make.py test              # flybrain CSR-identity regression test (GPU) + UCTF invariants
python repro/make.py figures           # every figure of both papers, from results/ (CPU, about 1 min)
python repro/make.py verify-figures    # the same, compared pixel by pixel with the committed PNGs
python repro/make.py headline OUTDIR   # rerun the headline experiments (GPU, about 1 h)
python repro/make.py compare OUTDIR    # compare that rerun with results/ -> repro/RERUN_REPORT.md
python repro/make.py manifest          # results/MANIFEST.json: sha256 and provenance of every result
```

GPU jobs run one at a time at below-normal priority with at most 4 CPU threads (`FLY_THREADS`).
Every rerun output gets a `.manifest.json` beside it with:
- the command and the git commit;
- the environment (including the sha256 of flybrain's `brain.py` and of the lock);
- the data hashes, the timing, and the output hash.

## 4. What is bit-reproducible and what is not

- **Bit-for-bit:**
  - the flybrain wheel;
  - every figure (PDFs have a fixed timestamp);
  - CPU-only analyses;
  - spike counts and anything computed only from spikes.
- **To float32 rounding:**
  - membrane voltages, and quantities computed from them.

  GPU sparse products are not bit-deterministic: voltages differ at the last float32 bit
  (about 1e-7) from run to run on the same machine. `repro/compare.py` therefore compares
  numbers with a tolerance (default 1e-5 + 1e-3 relative) and lists every difference beyond it.
- Quantities that sit at the numerical floor (e.g. a difference of about 1e-7 between two
  replays) are only meaningful as "at the floor"; their exact values change between runs.
