#!/bin/sh
# Reproducible build of the released wheel (LF line endings are enforced by .gitattributes).
# Expected: flybrain-0.1.0.post1-py3-none-any.whl
#   sha256 6c46867accf76b82e181bbe682e3f581e77d32e494d5c96fc135512e5e6b5d5c
set -e
cd "$(dirname "$0")"
rm -rf dist build *.egg-info
SOURCE_DATE_EPOCH=1790640000 python -m pip wheel . --no-deps --no-build-isolation -w dist
rm -rf build *.egg-info
sha256sum dist/*.whl 2>/dev/null || shasum -a 256 dist/*.whl
