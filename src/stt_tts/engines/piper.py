from __future__ import annotations

import os
from collections.abc import Iterator

from .base import AudioChunk, EngineNotInstalled, ModelNotFound, TTSEngine


class PiperEngine(TTSEngine):
    """TTS engine backed by piper-tts (piper1-gpl). Voice = a local .onnx file."""

    name = "piper"

    def __init__(
        self,
        *,
        voices_dir: str,
        default_voice: str = "en_US-lessac-medium",
        use_cuda: bool = False,
    ) -> None:
        self._voices_dir = voices_dir
        self._default_voice = default_voice
        self._use_cuda = use_cuda
        self._loaded: dict[str, object] = {}

    def _load(self, voice: str | None):
        voice = voice or self._default_voice
        cached = self._loaded.get(voice)
        if cached is not None:
            return cached
        try:
            from piper import PiperVoice
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise EngineNotInstalled(
                "piper-tts is not installed. Install with: pip install 'stt-tts[piper]'"
            ) from exc
        model_path = os.path.join(self._voices_dir, f"{voice}.onnx")
        if not os.path.exists(model_path):
            raise ModelNotFound(
                f"Piper voice '{voice}' not found at {model_path}. "
                f"Download it with: python -m piper.download_voices {voice}"
            )
        loaded = PiperVoice.load(model_path, use_cuda=self._use_cuda)
        self._loaded[voice] = loaded
        return loaded

    def ensure_ready(self, voice: str | None = None) -> None:
        self._load(voice)

    def sample_rate(self, voice: str | None = None) -> int:
        return self._load(voice).config.sample_rate

    def voices(self) -> list[str]:
        if not os.path.isdir(self._voices_dir):
            return [self._default_voice]
        found = sorted(
            name[:-5] for name in os.listdir(self._voices_dir) if name.endswith(".onnx")
        )
        return found or [self._default_voice]

    def synthesize(
        self, text: str, *, voice: str | None = None, speed: float = 1.0, **_: object
    ) -> Iterator[AudioChunk]:
        loaded = self._load(voice)
        syn_config = None
        if speed and speed != 1.0:
            from piper import SynthesisConfig

            # length_scale > 1 slows speech down, so invert the requested speed.
            syn_config = SynthesisConfig(length_scale=1.0 / speed)
        for chunk in loaded.synthesize(text, syn_config=syn_config):
            yield AudioChunk(
                data=chunk.audio_int16_bytes,
                sample_rate=chunk.sample_rate,
                channels=getattr(chunk, "sample_channels", 1),
                sample_width=getattr(chunk, "sample_width", 2),
            )
