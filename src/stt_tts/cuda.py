from __future__ import annotations

import ctypes
import glob
import logging
import os

logger = logging.getLogger(__name__)

# Modules shipped by the nvidia-*-cu12 pip wheels that carry the CUDA 12
# userspace libraries CTranslate2 (faster-whisper's backend) loads at runtime.
_NVIDIA_LIB_MODULES = ("nvidia.cublas.lib", "nvidia.cudnn.lib")

_preloaded = False


def nvidia_lib_dirs() -> list[str]:
    """Directories of pip-installed nvidia-*-cu12 shared libraries, if present."""
    import importlib.util

    dirs: list[str] = []
    for module in _NVIDIA_LIB_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            dirs.extend(spec.submodule_search_locations)
    return dirs


def preload_cuda_libraries() -> bool:
    """Best-effort: make pip-installed CUDA 12 cuBLAS/cuDNN libs loadable.

    CTranslate2 dynamically loads ``libcublas.so.12`` and ``libcudnn_ops.so.9``
    at runtime but does not bundle them. When they come from the
    ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` wheels they sit in
    site-packages, off the default loader path. Preloading them with
    ``RTLD_GLOBAL`` lets CTranslate2 resolve them without the caller exporting
    ``LD_LIBRARY_PATH`` (which would otherwise be required, and is easy to miss
    under systemd). No-op and never raises if the wheels are absent. Returns
    True if at least one library was preloaded.
    """
    global _preloaded
    if _preloaded:
        return True
    dirs = nvidia_lib_dirs()
    if not dirs:
        return False

    # Also help lazy loads / child processes. This does not change how the
    # current process resolves already-loaded NEEDED entries, but is harmless.
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [*dirs, current] if current else list(dirs)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)

    preloaded = False
    for directory in dirs:
        for path in sorted(glob.glob(os.path.join(directory, "*.so*"))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                preloaded = True
            except OSError as exc:  # pragma: no cover - host/library specific
                logger.debug("Could not preload %s: %s", path, exc)
    _preloaded = preloaded
    return preloaded
