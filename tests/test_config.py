from __future__ import annotations

from stt_tts.config import Settings


def test_defaults():
    settings = Settings()
    assert settings.default_stt_model == "large-v3-turbo"
    assert settings.default_tts_model == "kokoro"
    assert settings.device == "auto"
    assert "large-v3-turbo" in settings.stt_model_list
    assert "kokoro" in settings.tts_model_list


def test_csv_lists(monkeypatch):
    monkeypatch.setenv("STT_MODELS", "small, base ,tiny")
    monkeypatch.setenv("TTS_MODELS", "kokoro")
    settings = Settings()
    assert settings.stt_model_list == ["small", "base", "tiny"]
    assert settings.tts_model_list == ["kokoro"]


def test_device_override(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda")
    assert Settings().device == "cuda"
