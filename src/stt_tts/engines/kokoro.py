from __future__ import annotations

from collections.abc import Iterator

from .base import AudioChunk, EngineNotInstalled, TTSEngine

# A representative subset of Kokoro's built-in voices. The first letter encodes
# the language (a/b = American/British English, e=es, f=fr, h=hi, i=it, j=ja,
# p=pt-BR, z=zh) and the second the gender (f/m).
KOKORO_VOICES = (
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_michael",
    "am_adam",
    "am_eric",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
)

_SAMPLE_RATE = 24000
_LANG_PREFIXES = "abefhijpz"


class KokoroEngine(TTSEngine):
    """TTS engine backed by the kokoro package (Kokoro-82M, Apache-2.0, 24 kHz)."""

    name = "kokoro"

    def __init__(
        self,
        *,
        lang_code: str = "a",
        default_voice: str = "af_heart",
        device: str | None = None,
    ) -> None:
        self._default_lang = lang_code
        self._default_voice = default_voice
        self._device = device
        self._pipelines: dict[str, object] = {}

    def _lang_for_voice(self, voice: str | None) -> str:
        if voice and voice[0] in _LANG_PREFIXES:
            return voice[0]
        return self._default_lang

    def _pipeline(self, lang_code: str):
        cached = self._pipelines.get(lang_code)
        if cached is not None:
            return cached
        try:
            from kokoro import KPipeline
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise EngineNotInstalled(
                "kokoro is not installed. Install with: pip install 'stt-tts[kokoro]' "
                "and the espeak-ng system package."
            ) from exc
        pipeline = (
            KPipeline(lang_code=lang_code)
            if self._device is None
            else KPipeline(lang_code=lang_code, device=self._device)
        )
        self._pipelines[lang_code] = pipeline
        return pipeline

    def ensure_ready(self, voice: str | None = None) -> None:
        self._pipeline(self._lang_for_voice(voice or self._default_voice))

    def sample_rate(self, voice: str | None = None) -> int:
        return _SAMPLE_RATE

    def voices(self) -> list[str]:
        return list(KOKORO_VOICES)

    def synthesize(
        self, text: str, *, voice: str | None = None, speed: float = 1.0, **_: object
    ) -> Iterator[AudioChunk]:
        import numpy as np

        voice = voice or self._default_voice
        pipeline = self._pipeline(self._lang_for_voice(voice))
        for _graphemes, _phonemes, audio in pipeline(
            text, voice=voice, speed=speed, split_pattern=r"\n+"
        ):
            array = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio, dtype="float32")
            pcm = (np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            yield AudioChunk(data=pcm, sample_rate=_SAMPLE_RATE)
