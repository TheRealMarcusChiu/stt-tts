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
_status_cache: dict | None = None


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


def cuda_status() -> dict:
    """Probe whether CTranslate2 (faster-whisper's backend) can see a CUDA GPU.

    Returns a dict with ``available``, ``device_count``, ``libs_found`` and a
    human-readable ``detail``. ``device_count`` reflects the GPU visible to the
    backend's CUDA runtime; full inference additionally needs cuBLAS/cuDNN, which
    ``libs_found`` reports when they come from the nvidia-*-cu12 pip wheels. The
    result is cached per process and the probe never raises.
    """
    global _status_cache
    if _status_cache is not None:
        return _status_cache

    libs_found = bool(nvidia_lib_dirs())
    try:
        import ctranslate2
    except ImportError:
        _status_cache = {
            "available": False,
            "device_count": None,
            "libs_found": libs_found,
            "detail": "ctranslate2 is not installed (CPU-only install).",
        }
        return _status_cache

    try:
        if libs_found:
            preload_cuda_libraries()
        count = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # noqa: BLE001 - host-specific; never fail /health
        _status_cache = {
            "available": False,
            "device_count": None,
            "libs_found": libs_found,
            "detail": f"CUDA probe failed: {exc}",
        }
        return _status_cache

    _status_cache = {
        "available": count > 0,
        "device_count": count,
        "libs_found": libs_found,
        "detail": (
            f"CTranslate2 sees {count} CUDA device(s)."
            if count > 0
            else "No CUDA device visible to CTranslate2."
        ),
    }
    return _status_cache
