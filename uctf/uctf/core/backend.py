"""Where arrays live: numpy on the CPU, or CuPy on an NVIDIA GPU."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def array_module(device: str = "auto"):
    """CuPy when device is "auto" or "cuda" and a GPU is present, else numpy."""
    if device == "cpu":
        return np
    try:
        import cupy
        patch_cupy_includes()
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:                                   # noqa: BLE001
        if device == "cuda":
            raise
    return np


def to_host(a):
    """A numpy array, whatever module `a` lives on."""
    return a.get() if hasattr(a, "get") else a


def sparse_on(xp, matrix):
    """A scipy CSR matrix on the array module `xp`."""
    if xp is np:
        return matrix
    from cupyx.scipy import sparse as cusparse
    return cusparse.csr_matrix(matrix)


def patch_cupy_includes() -> None:
    """Some CuPy installs cannot find their own headers when NVRTC compiles a
    kernel; add CuPy's include folders to every compilation."""
    try:
        import cupy
        import cupy.cuda.compiler as compiler
    except Exception:                                   # noqa: BLE001
        return
    if getattr(compiler, "_gpf_include_patch", False):
        return
    inc = Path(cupy.__path__[0]) / "_core" / "include"
    if not (inc / "cupy" / "complex.cuh").exists():
        return
    cccl = inc / "cupy" / "_cccl"
    extra = (cccl, cccl / "thrust", cccl / "libcudacxx", cccl / "cub")
    flags = [f"-I{inc}"] + [f"-I{p}" for p in extra if p.exists()]
    original = compiler._compile_with_cache_cuda

    def patched(source, options=(), *a, **kw):
        opts = tuple(options) + tuple(f for f in flags if f not in options)
        return original(source, opts, *a, **kw)

    compiler._compile_with_cache_cuda = patched
    compiler._gpf_include_patch = True
