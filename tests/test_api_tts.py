from __future__ import annotations


def test_speech_wav(client):
    response = client.post("/v1/audio/speech", json={"model": "fake-tts", "input": "hi"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content[:4] == b"RIFF"


def test_speech_pcm(client):
    response = client.post(
        "/v1/audio/speech", json={"model": "fake-tts", "input": "hi", "response_format": "pcm"}
    )
    assert response.status_code == 200
    assert response.headers["x-sample-rate"] == "24000"
    # Two fake chunks of 200 bytes each.
    assert len(response.content) == 400


def test_speech_default_model(client):
    response = client.post("/v1/audio/speech", json={"input": "hi"})
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"


def test_speech_unknown_model_returns_404(client):
    response = client.post("/v1/audio/speech", json={"model": "nope", "input": "hi"})
    assert response.status_code == 404


def test_speech_empty_input_returns_422(client):
    response = client.post("/v1/audio/speech", json={"model": "fake-tts", "input": ""})
    assert response.status_code == 422


def test_speech_speed_out_of_range_returns_422(client):
    response = client.post(
        "/v1/audio/speech", json={"model": "fake-tts", "input": "hi", "speed": 99}
    )
    assert response.status_code == 422
