from __future__ import annotations

from ..config import Settings
from .base import ModelNotFound, STTEngine, TTSEngine
from .kokoro import KokoroEngine
from .piper import PiperEngine
from .whisper import FasterWhisperEngine

_STT_ALIASES = {"whisper-1"}
_TTS_ALIASES = {"tts-1", "tts-1-hd"}


class EngineManager:
    """Resolves request-supplied model names to lazily-constructed, cached engines."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._stt: dict[str, STTEngine] = {}
        self._tts: dict[str, TTSEngine] = {}

    # --- Speech-to-text ---
    def stt_models(self) -> list[str]:
        return self.settings.stt_model_list

    def resolve_stt(self, model: str | None) -> str:
        if not model or model in _STT_ALIASES:
            return self.settings.default_stt_model
        return model

    def get_stt(self, model: str | None) -> STTEngine:
        model_id = self.resolve_stt(model)
        allowed = self.settings.stt_model_list
        if not self.settings.stt_allow_any_model and model_id not in allowed:
            raise ModelNotFound(
                f"Unknown STT model '{model_id}'. Available: {', '.join(allowed)}."
            )
        if model_id not in self._stt:
            self._stt[model_id] = FasterWhisperEngine(
                model_id,
                device=self.settings.device,
                compute_type=self.settings.stt_compute_type,
                download_root=self.settings.model_cache_dir,
                vad_filter=self.settings.stt_vad_filter,
                beam_size=self.settings.stt_beam_size,
            )
        return self._stt[model_id]

    # --- Text-to-speech ---
    def tts_models(self) -> list[str]:
        return self.settings.tts_model_list

    def resolve_tts(self, model: str | None) -> str:
        if not model or model in _TTS_ALIASES:
            return self.settings.default_tts_model
        return model

    def get_tts(self, model: str | None) -> TTSEngine:
        name = self.resolve_tts(model)
        if name not in self.settings.tts_model_list:
            raise ModelNotFound(
                f"Unknown TTS model '{name}'. Available: {', '.join(self.settings.tts_model_list)}."
            )
        if name not in self._tts:
            self._tts[name] = self._build_tts(name)
        return self._tts[name]

    def _build_tts(self, name: str) -> TTSEngine:
        device = None if self.settings.device == "auto" else self.settings.device
        if name == "kokoro":
            return KokoroEngine(
                lang_code=self.settings.kokoro_lang_code,
                default_voice=self.settings.default_voice,
                device=device,
            )
        if name == "piper":
            return PiperEngine(
                voices_dir=self.settings.piper_voices_dir,
                default_voice=self.settings.piper_default_voice,
                use_cuda=self.settings.device == "cuda",
            )
        raise ModelNotFound(f"No TTS engine implementation for '{name}'.")

    def voices(self, model: str | None) -> list[str]:
        return self.get_tts(model).voices()

    def default_voice_for(self, model: str | None) -> str:
        if self.resolve_tts(model) == "piper":
            return self.settings.piper_default_voice
        return self.settings.default_voice
