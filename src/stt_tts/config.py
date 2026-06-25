from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables / .env."""

    app_name: str = "STT/TTS API"
    host: str = "0.0.0.0"
    port: int = 8000

    # Compute device for all engines: "auto" | "cuda" | "cpu".
    device: str = "auto"

    # --- Speech-to-text (faster-whisper) ---
    default_stt_model: str = "large-v3-turbo"
    # Comma-separated allowlist of selectable STT models (kept as a string to
    # avoid pydantic-settings JSON-env parsing pitfalls; see *_model_list below).
    stt_models: str = "large-v3-turbo,large-v3,distil-large-v3,medium,small,base,tiny"
    stt_allow_any_model: bool = False
    # CTranslate2 compute type: default | float16 | int8 | int8_float16 | float32.
    stt_compute_type: str = "default"
    stt_vad_filter: bool = True
    stt_beam_size: int = 5

    # --- Text-to-speech ---
    default_tts_model: str = "kokoro"
    tts_models: str = "kokoro,piper"
    default_voice: str = "af_heart"
    kokoro_lang_code: str = "a"
    piper_voices_dir: str = "/models/piper"
    piper_default_voice: str = "en_US-lessac-medium"

    # Where downloaded model weights are cached (faster-whisper download_root).
    model_cache_dir: str | None = Field(default=None, alias="MODEL_CACHE_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @property
    def stt_model_list(self) -> list[str]:
        return [item.strip() for item in self.stt_models.split(",") if item.strip()]

    @property
    def tts_model_list(self) -> list[str]:
        return [item.strip() for item in self.tts_models.split(",") if item.strip()]
