"""Command line: `flybrain download`, `flybrain build`, `flybrain info` (or `python -m flybrain ...`)."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .data import DATA, FILES, RELEASE_URL, download, has_data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="flybrain", description="The MaleCNS fruit fly connectome as a spiking network.")
    parser.add_argument("--version", action="version", version=f"flybrain {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("download", help="fetch the prebuilt brain files (~260 MB)")
    fetch.add_argument("--data", type=Path, default=DATA, help=f"where to put them (default {DATA}, or $FLY_DATA)")
    fetch.add_argument("--url", default=RELEASE_URL, help="base URL of the files (or $FLYBRAIN_DATA_URL)")
    fetch.add_argument("--force", action="store_true", help="download again even if the files are there")

    build = sub.add_parser("build", help="download MaleCNS v1.0 (~1.1 GB) and build the brain files from it")
    build.add_argument("--data", type=Path, default=DATA, help=f"data folder (default {DATA}, or $FLY_DATA)")

    info = sub.add_parser("info", help="show the data folder and whether a GPU is usable")
    info.add_argument("--data", type=Path, default=DATA)

    args = parser.parse_args(argv)
    if args.command == "download":
        download(args.data, args.url, force=args.force)
        print(f"brain files in {args.data}")
    elif args.command == "build":
        try:
            from .build import build as build_brain
        except ImportError as e:
            raise SystemExit(f"building needs pandas and pyarrow ({e}): pip install \"flybrain[build]\"")
        build_brain(args.data)
    else:
        from .brain import cuda_available
        print(f"flybrain {__version__}")
        print(f"data folder: {args.data}")
        for name in FILES:
            path = args.data / name
            print(f"  {name}: {f'{path.stat().st_size / 1e6:,.0f} MB' if path.exists() else 'missing'}")
        if not has_data(args.data):
            print("  run `flybrain download` (or just create a FlyBrain) to fetch them")
        gpu = "available" if cuda_available() else 'not available (pip install "flybrain[gpu]")'
        print(f"cuda: {gpu}")


if __name__ == "__main__":
    main()
