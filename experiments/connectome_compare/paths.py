"""Where the repository is. FLY_REPO overrides the default (this machine's checkout)."""
import os
from pathlib import Path

REPO = Path(os.environ.get("FLY_REPO", r"D:/苍蝇。/Fly-Is-All-You-Need"))
