"""GPF, the Generative Pretrained Fly: continue a prompt with a fly connectome (pretrained by
evolution, fine-tuned on 20k characters of Shakespeare), or with an n-gram or a GRU for comparison.

Connectome models, training and benchmarks come from UCTF, the Universal Connectome Training
Framework (../uctf).  Installed (pip install -e uctf) or not, a checkout of the repository works:
the repository's uctf/ folder is used when the package is not installed."""
__version__ = "0.1.0"

import importlib.util as _u
import sys as _sys
from pathlib import Path as _Path

if _u.find_spec("uctf") is None:
    _repo = _Path(__file__).resolve().parents[2] / "uctf"
    if (_repo / "uctf" / "__init__.py").exists():
        _sys.path.insert(0, str(_repo))
