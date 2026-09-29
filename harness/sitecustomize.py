"""CuPy compatibility hook, imported by the harness modules and mechanism/m1_core.py.

Some CuPy installs cannot find their own headers when NVRTC compiles a fresh elementwise
kernel (the pre-existing kernel cache can hide this).  Add CuPy's include folders, including
its vendored CCCL/Thrust tree, to every compilation.  Harmless where it is not needed.
"""
from pathlib import Path


def _install_cupy_include_path() -> None:
    try:
        import cupy
        import cupy.cuda.compiler as compiler
    except Exception:
        return
    if getattr(compiler, "_flybrain_include_patch", False):
        return
    include = Path(cupy.__path__[0]) / "_core" / "include"
    if not (include / "cupy" / "complex.cuh").exists():
        return
    cccl = include / "cupy" / "_cccl"
    flags = [f"-I{include}"] + [f"-I{p}" for p in (cccl, cccl / "thrust", cccl / "libcudacxx", cccl / "cub")
                                if p.exists()]
    original = compiler._compile_with_cache_cuda

    def patched(source, options=(), *args, **kwargs):
        opts = tuple(options)
        for flag in flags:
            if flag not in opts:
                opts += (flag,)
        return original(source, opts, *args, **kwargs)

    compiler._compile_with_cache_cuda = patched
    compiler._flybrain_include_patch = True


_install_cupy_include_path()
