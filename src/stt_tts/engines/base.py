from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


class EngineError(RuntimeError):
    """Base class for engine-related failures."""


class ModelNotFound(EngineError):
    """Raised when a requested model or voice is not available."""


class EngineNotInstalled(EngineError):
    """Raised when the optional dependency backing an engine is not installed."""


@dataclass
class TranscriptionInfo:
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None


@dataclass
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass
class AudioChunk:
    data: bytes  # raw little-endian 16-bit PCM
    sample_rate: int
    channels: int = 1
    sample_width: int = 2  # bytes per sample


class STTEngine(ABC):
    """Speech-to-text engine interface."""

    model_id: str

    def ensure_ready(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Eagerly load the model so dependency/load errors surface up front."""

    @abstractmethod
    def transcribe(
        self, audio: str | bytes, *, language: str | None = None, **options: object
    ) -> tuple[TranscriptionInfo, Iterator[TranscriptionSegment]]:
        """Return transcription info plus a (possibly lazy) segment iterator.

        The segment iterator may perform decoding work as it is consumed, which
        is what enables streaming partial transcripts.
        """
        raise NotImplementedError


class TTSEngine(ABC):
    """Text-to-speech engine interface."""

    name: str

    def ensure_ready(self, voice: str | None = None) -> None:  # noqa: B027 - optional hook
        """Eagerly load the voice/model so errors surface before streaming."""

    @abstractmethod
    def sample_rate(self, voice: str | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def voices(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def synthesize(
        self, text: str, *, voice: str | None = None, speed: float = 1.0, **options: object
    ) -> Iterator[AudioChunk]:
        """Yield audio chunks as they are synthesized (streaming-friendly)."""
        raise NotImplementedError
