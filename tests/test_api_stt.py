from __future__ import annotations

import io
import wave


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
