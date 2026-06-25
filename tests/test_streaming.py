from __future__ import annotations

import io
import wave

from stt_tts.engines.base import (
    AudioChunk,
    EngineNotInstalled,
    STTEngine,
    TTSEngine,
)


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


def _upload():
    return {"file": ("audio.wav", _wav_bytes(), "audio/wav")}


def test_stt_streaming_sse(client):
    response = client.post(
        "/v1/audio/transcriptions", files=_upload(), data={"model": "fake-stt", "stream": "true"}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    assert '"info"' in body
    assert '"segment"' in body
    assert '"done"' in body
    assert "[DONE]" in body


def test_tts_streaming_wav(client):
    response = client.post(
        "/v1/audio/speech",
        json={"model": "fake-tts", "input": "hi", "stream": True, "response_format": "wav"},
    )
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"


def test_tts_streaming_pcm(client):
    response = client.post(
        "/v1/audio/speech",
        json={"model": "fake-tts", "input": "hi", "stream": True, "response_format": "pcm"},
    )
    assert response.status_code == 200
    assert response.headers["x-sample-rate"] == "24000"
    assert len(response.content) == 400


class _NotInstalledSTT(STTEngine):
    model_id = "x"

    def ensure_ready(self) -> None:
        raise EngineNotInstalled("install the stt extra")

    def transcribe(self, audio, *, language=None, **options):
        raise EngineNotInstalled("install the stt extra")


class _NotInstalledTTS(TTSEngine):
    name = "x"

    def ensure_ready(self, voice=None) -> None:
        raise EngineNotInstalled("install the tts extra")

    def sample_rate(self, voice=None) -> int:
        return 24000

    def voices(self) -> list[str]:
        return []

    def synthesize(self, text, *, voice=None, speed=1.0, **options):
        raise EngineNotInstalled("install the tts extra")
        yield AudioChunk(b"", 24000)  # pragma: no cover


def test_stt_missing_dependency_returns_503(make_client):
    client = make_client(stt=_NotInstalledSTT())
    response = client.post("/v1/audio/transcriptions", files=_upload(), data={"model": "fake-stt"})
    assert response.status_code == 503


def test_tts_missing_dependency_returns_503(make_client):
    client = make_client(tts=_NotInstalledTTS())
    response = client.post("/v1/audio/speech", json={"model": "fake-tts", "input": "hi"})
    assert response.status_code == 503


class _LoadFailureSTT(STTEngine):
    model_id = "x"

    def ensure_ready(self) -> None:
        raise RuntimeError("Unable to load libcudnn_ops.so.9")

    def transcribe(self, audio, *, language=None, **options):  # pragma: no cover
        raise RuntimeError("Unable to load libcudnn_ops.so.9")


class _LoadFailureTTS(TTSEngine):
    name = "x"

    def ensure_ready(self, voice=None) -> None:
        raise RuntimeError("could not load voice weights")

    def sample_rate(self, voice=None) -> int:  # pragma: no cover
        return 24000

    def voices(self) -> list[str]:  # pragma: no cover
        return []

    def synthesize(self, text, *, voice=None, speed=1.0, **options):  # pragma: no cover
        raise RuntimeError("could not load voice weights")
        yield AudioChunk(b"", 24000)


def test_stt_model_load_failure_returns_503(make_client):
    client = make_client(stt=_LoadFailureSTT())
    response = client.post("/v1/audio/transcriptions", files=_upload(), data={"model": "fake-stt"})
    assert response.status_code == 503
    assert "libcudnn" in response.json()["detail"]


def test_tts_model_load_failure_returns_503(make_client):
    client = make_client(tts=_LoadFailureTTS())
    response = client.post("/v1/audio/speech", json={"model": "fake-tts", "input": "hi"})
    assert response.status_code == 503
