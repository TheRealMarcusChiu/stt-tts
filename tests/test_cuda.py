from __future__ import annotations

from stt_tts.cuda import cuda_status, nvidia_lib_dirs, preload_cuda_libraries


def test_nvidia_lib_dirs_absent_without_wheels():
    # The nvidia-*-cu12 wheels are not installed in the test environment.
    assert nvidia_lib_dirs() == []


def test_preload_is_noop_without_wheels():
    # Returns False and must never raise when the CUDA wheels are absent.
    assert preload_cuda_libraries() is False


def test_cuda_status_structure():
    status = cuda_status()
    assert set(status) >= {"available", "device_count", "libs_found", "detail"}
    assert isinstance(status["available"], bool)
    assert isinstance(status["libs_found"], bool)
    assert status["detail"]
