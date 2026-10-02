GPF (Generative Pretrained Fly)
Copyright (c) 2026 Quanxi Li. Part of https://github.com/LxCenady/Fly-Is-All-You-Need

Bundled third-party components and data
- Python (PSF License), numpy (BSD-3-Clause). Packaged with PyInstaller, whose bootloader is
  distributed under the GPL with an exception that permits bundling with any program.
- TinyShakespeare corpus, from Andrej Karpathy's char-rnn (MIT License); the text is by
  William Shakespeare and in the public domain.
- Trained model weights (GRU, connectome readout): produced by this project.

Full edition only (gpf-full packages; not in the lite packages)
- The GPF-1 connectome bundle (gpf/data/gpf1_cpu.npz): the MaleCNS v1.0 connectome as prebuilt
  by flybrain (MIT License, https://github.com/alextitonis/fly.ai), converted to a compact
  numpy format with synapses onto sensory neurons removed. MaleCNS v1.0 connectome data by
  FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and Google Research, CC BY 4.0.
  Cite: S. Berg et al., "Sexual dimorphism in the complete Drosophila male central nervous
  system connectome", Cell 189(18):5504-5526.e15 (2026).
  The lite packages download the same bundle on request (`gpf get-brain`).

Running from source with a GPU instead uses flybrain (MIT), CuPy (MIT) and NVIDIA CUDA
libraries (NVIDIA CUDA EULA); these are not bundled in any package.
