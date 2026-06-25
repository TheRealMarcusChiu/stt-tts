from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from stt_tts.config import Settings
from stt_tts.engines.base import (
    AudioChunk,
    ModelNotFound,
    STTEngine,
    TranscriptionInfo,
    TranscriptionSegment,
    TTSEngine,
)
from stt_tts.main import create_app


class FakeSTTEngine(STTEngine):
    def __init__(self, model_id: str = "fake-stt") -> None:
        self.model_id = model_id
        self.ready = False

    def ensure_ready(self) -> None:
        self.ready = True

    def transcribe(
        self, audio: bytes, *, language: str | None = None, **options: object
    ) -> tuple[TranscriptionInfo, Iterator[TranscriptionSegment]]:
        info = TranscriptionInfo(language=language or "en", language_probability=0.99, duration=1.0)
        segments = [
            TranscriptionSegment(id=0, start=0.0, end=0.5, text="Hello"),
            TranscriptionSegment(id=1, start=0.5, end=1.0, text=" world."),
        ]
        return info, iter(segments)


class FakeTTSEngine(TTSEngine):
    name = "fake-tts"

    def __init__(self, sample_rate: int = 24000, voices: list[str] | None = None) -> None:
        self._sample_rate = sample_rate
        self._voices = voices or ["af_heart", "am_michael"]
        self.ready = False

    def ensure_ready(self, voice: str | None = None) -> None:
        self.ready = True

    def sample_rate(self, voice: str | None = None) -> int:
        return self._sample_rate

    def voices(self) -> list[str]:
        return list(self._voices)

    def synthesize(
        self, text: str, *, voice: str | None = None, speed: float = 1.0, **options: object
    ) -> Iterator[AudioChunk]:
        frame = b"\x00\x00" * 100  # 200 bytes of silence per chunk
        yield AudioChunk(data=frame, sample_rate=self._sample_rate)
        yield AudioChunk(data=frame, sample_rate=self._sample_rate)


class FakeManager:
    def __init__(
        self,
        settings: Settings,
        stt: STTEngine | None = None,
        tts: TTSEngine | None = None,
    ) -> None:
        self.settings = settings
        self._stt = stt or FakeSTTEngine()
        self._tts = tts or FakeTTSEngine()

    def stt_models(self) -> list[str]:
        return ["fake-stt"]

    def tts_models(self) -> list[str]:
        return ["fake-tts"]

    def get_stt(self, model: str | None) -> STTEngine:
        if model not in (None, "fake-stt", "whisper-1"):
            raise ModelNotFound(f"Unknown STT model '{model}'.")
        return self._stt

    def get_tts(self, model: str | None) -> TTSEngine:
        if model not in (None, "fake-tts", "tts-1"):
            raise ModelNotFound(f"Unknown TTS model '{model}'.")
        return self._tts

    def voices(self, model: str | None) -> list[str]:
        return self.get_tts(model).voices()

    def default_voice_for(self, model: str | None) -> str:
        return "af_heart"


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def make_client(settings: Settings):
    def _make(stt: STTEngine | None = None, tts: TTSEngine | None = None) -> TestClient:
        manager = FakeManager(settings, stt=stt, tts=tts)
        return TestClient(create_app(settings=settings, manager=manager))

    return _make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
