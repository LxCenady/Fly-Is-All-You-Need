"""Start the GPF web UI from any working directory: python run_web.py [port]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gpf.web import main  # noqa: E402

main(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
