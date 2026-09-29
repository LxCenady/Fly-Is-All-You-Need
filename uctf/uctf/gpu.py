"""GPU helpers: pick numpy or CuPy, and work around CuPy installs that cannot find their headers."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def patch_cupy_includes() -> None:
    """Some CuPy installs cannot find their own headers when NVRTC compiles a new kernel.
    Add CuPy's include folders to every compilation (harmless where not needed)."""
    try:
        import cupy
        import cupy.cuda.compiler as compiler
    except Exception:                                       # noqa: BLE001
        return
    if getattr(compiler, "_gpf_include_patch", False):
        return
    inc = Path(cupy.__path__[0]) / "_core" / "include"
    if not (inc / "cupy" / "complex.cuh").exists():
        return
    flags = [f"-I{inc}"] + [f"-I{p}" for p in (inc / "cupy" / "_cccl", inc / "cupy" / "_cccl" / "thrust",
                                               inc / "cupy" / "_cccl" / "libcudacxx", inc / "cupy" / "_cccl" / "cub")
                            if p.exists()]
    original = compiler._compile_with_cache_cuda

    def patched(source, options=(), *a, **kw):
        opts = tuple(options)
        for f in flags:
            if f not in opts:
                opts += (f,)
        return original(source, opts, *a, **kw)

    compiler._compile_with_cache_cuda = patched
    compiler._gpf_include_patch = True


def array_module(device: str = "auto"):
    """CuPy on an NVIDIA GPU when device is auto/cuda and one is available, else numpy."""
    if device in ("auto", "cuda"):
        try:
            import cupy
            patch_cupy_includes()
            if cupy.cuda.runtime.getDeviceCount() > 0:
                return cupy
        except Exception:                                   # noqa: BLE001
            if device == "cuda":
                raise
    return np
