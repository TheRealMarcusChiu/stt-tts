from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

STT_RESPONSE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")
TTS_RESPONSE_FORMATS = ("wav", "pcm")


class TranscriptionSegmentModel(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegmentModel] | None = None


class SpeechRequest(BaseModel):
    """Body for POST /v1/audio/speech."""

    model: str | None = None
    input: str = Field(min_length=1, max_length=8192)
    voice: str | None = None
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream: bool = False


class ModelInfo(BaseModel):
    id: str
    type: Literal["stt", "tts"]
    voices: list[str] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    stt_models: list[str]
    tts_models: list[str]
