from __future__ import annotations

from stt_tts.cuda import nvidia_lib_dirs, preload_cuda_libraries


def test_nvidia_lib_dirs_absent_without_wheels():
    # The nvidia-*-cu12 wheels are not installed in the test environment.
    assert nvidia_lib_dirs() == []


def test_preload_is_noop_without_wheels():
    # Returns False and must never raise when the CUDA wheels are absent.
    assert preload_cuda_libraries() is False
