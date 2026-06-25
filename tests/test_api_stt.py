from __future__ import annotations

import io
import os
import wave

from stt_tts.engines.base import STTEngine, TranscriptionInfo, TranscriptionSegment


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


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "fake-stt" in body["stt_models"]
    assert "fake-tts" in body["tts_models"]
    # CUDA readiness block is always present and well-typed.
    cuda = body["cuda"]
    assert isinstance(cuda["available"], bool)
    assert isinstance(cuda["libs_found"], bool)
    assert "device_count" in cuda
    assert cuda["detail"]


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    entries = {entry["id"]: entry for entry in response.json()["data"]}
    assert entries["fake-stt"]["type"] == "stt"
    assert entries["fake-tts"]["type"] == "tts"
    assert entries["fake-tts"]["voices"]


def test_transcribe_json(client):
    response = client.post("/v1/audio/transcriptions", files=_upload(), data={"model": "fake-stt"})
    assert response.status_code == 200
    assert response.json()["text"] == "Hello world."


def test_transcribe_text(client):
    response = client.post(
        "/v1/audio/transcriptions", files=_upload(), data={"response_format": "text"}
    )
    assert response.status_code == 200
    assert response.text.strip() == "Hello world."


def test_transcribe_verbose_json(client):
    response = client.post(
        "/v1/audio/transcriptions", files=_upload(), data={"response_format": "verbose_json"}
    )
    body = response.json()
    assert body["language"] == "en"
    assert len(body["segments"]) == 2
    assert body["segments"][0]["text"] == "Hello"


def test_transcribe_srt(client):
    response = client.post(
        "/v1/audio/transcriptions", files=_upload(), data={"response_format": "srt"}
    )
    assert response.status_code == 200
    assert "-->" in response.text
    assert "00:00:00,000" in response.text


def test_unknown_model_returns_404(client):
    response = client.post("/v1/audio/transcriptions", files=_upload(), data={"model": "nope"})
    assert response.status_code == 404


def test_empty_file_returns_422(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"", "audio/wav")},
        data={"model": "fake-stt"},
    )
    assert response.status_code == 422


def test_bad_response_format_returns_422(client):
    response = client.post(
        "/v1/audio/transcriptions", files=_upload(), data={"response_format": "weird"}
    )
    assert response.status_code == 422


class _PathCapturingSTT(STTEngine):
    model_id = "fake-stt"

    def __init__(self) -> None:
        self.seen_audio: object = None
        self.existed = False
        self.contents: bytes | None = None
        self.options: dict = {}

    def transcribe(self, audio, *, language=None, **options):
        # The route must hand the engine a real filesystem path, not raw bytes.
        self.seen_audio = audio
        self.existed = isinstance(audio, str) and os.path.exists(audio)
        if self.existed:
            with open(audio, "rb") as handle:
                self.contents = handle.read()
        self.options = options
        info = TranscriptionInfo(language="en", duration=1.0)
        return info, iter([TranscriptionSegment(id=0, start=0.0, end=1.0, text="ok")])


def test_upload_is_passed_as_temp_file_path(make_client):
    # An .mp4 (or any container) upload is written to a temp file and the path,
    # not the bytes, is handed to faster-whisper.
    engine = _PathCapturingSTT()
    client = make_client(stt=engine)
    payload = b"\x00\x01\x02fake-mp4-bytes\x03\x04"
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.mp4", payload, "video/mp4")},
        data={"model": "fake-stt"},
    )
    assert response.status_code == 200
    assert isinstance(engine.seen_audio, str)
    assert engine.existed
    assert engine.contents == payload


def test_vad_filter_defaults_to_none(make_client):
    engine = _PathCapturingSTT()
    client = make_client(stt=engine)
    response = client.post("/v1/audio/transcriptions", files=_upload(), data={"model": "fake-stt"})
    assert response.status_code == 200
    # None => the engine applies its configured default.
    assert engine.options.get("vad_filter") is None


def test_vad_filter_can_be_disabled_per_request(make_client):
    engine = _PathCapturingSTT()
    client = make_client(stt=engine)
    response = client.post(
        "/v1/audio/transcriptions",
        files=_upload(),
        data={"model": "fake-stt", "vad_filter": "false"},
    )
    assert response.status_code == 200
    assert engine.options.get("vad_filter") is False
