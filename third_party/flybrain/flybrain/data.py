"""Where the brain files live, and fetching them.

FlyBrain needs two files built from the MaleCNS v1.0 connectome: weights.npz (the signed,
normalized connection matrix) and brain.npz (cell types, sides, positions, readout groups,
eye layout). They are too big for the package, so the first FlyBrain() downloads a prebuilt
copy (~260 MB) into $FLY_DATA (default ~/fly-data). `flybrain build` makes the same files
from the original MaleCNS release instead.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(os.environ.get("FLY_DATA", Path.home() / "fly-data"))

RELEASE_URL = os.environ.get("FLYBRAIN_DATA_URL",
                             "https://github.com/alextitonis/fly.ai/releases/download/brain-v1")

# sha256 of the prebuilt files (166,700 neurons, 25,582,938 connections)
FILES = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}


def has_data(data: Path | str = DATA) -> bool:
    """True if both brain files are in `data`."""
    return all((Path(data) / name).exists() for name in FILES)


def download(data: Path | str = DATA, url: str = RELEASE_URL, force: bool = False) -> Path:
    """Fetch the prebuilt brain files into `data`, checking their sha256. Files already
    there are kept unless force=True."""
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = data / name
        if target.exists() and not force:
            continue
        partial = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        print(f"downloading {name}", file=sys.stderr)
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/{name}") as response, open(partial, "wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", file=sys.stderr)
            print(file=sys.stderr)
        except (urllib.error.URLError, OSError) as e:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"could not download {name} from {url}: {e}") from e
        if digest.hexdigest() != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"{name} from {url} has the wrong checksum; not using it")
        partial.replace(target)
    return data


def ensure_data(data: Path | str | None = None) -> Path:
    """The data folder, downloading the prebuilt brain first if it isn't there yet."""
    data = DATA if data is None else Path(data)
    if has_data(data):
        return data
    print(f"flybrain: no brain files in {data}; downloading the prebuilt brain (~260 MB, once)", file=sys.stderr)
    try:
        download(data)
    except RuntimeError as e:
        raise FileNotFoundError(
            f"no brain files in {data} and the download failed ({e}). Build them from the MaleCNS "
            f"release instead: pip install \"flybrain[build]\" && flybrain build --data \"{data}\"") from e
    return data
