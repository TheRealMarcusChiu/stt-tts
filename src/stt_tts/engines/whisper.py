from __future__ import annotations

import io
from collections.abc import Iterator

from .base import (
    EngineNotInstalled,
    STTEngine,
    TranscriptionInfo,
    TranscriptionSegment,
)


class FasterWhisperEngine(STTEngine):
    """STT engine backed by faster-whisper (CTranslate2)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        compute_type: str = "default",
        download_root: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 5,
    ) -> None:
        self.model_id = model_id
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root
        self._vad_filter = vad_filter
        self._beam_size = beam_size
        self._model = None

    def ensure_ready(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise EngineNotInstalled(
                "faster-whisper is not installed. Install with: pip install 'stt-tts[stt]'"
            ) from exc
        self._model = WhisperModel(
            self.model_id,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._download_root,
        )

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        vad_filter: bool | None = None,
        beam_size: int | None = None,
        **_: object,
    ) -> tuple[TranscriptionInfo, Iterator[TranscriptionSegment]]:
        self.ensure_ready()
        source = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio
        segments, info = self._model.transcribe(
            source,
            language=language,
            beam_size=beam_size or self._beam_size,
            vad_filter=self._vad_filter if vad_filter is None else vad_filter,
            word_timestamps=word_timestamps,
        )
        transcription_info = TranscriptionInfo(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

        def _iter_segments() -> Iterator[TranscriptionSegment]:
            for segment in segments:
                yield TranscriptionSegment(
                    id=segment.id,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text,
                )

        return transcription_info, _iter_segments()
