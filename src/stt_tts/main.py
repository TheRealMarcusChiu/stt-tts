from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from stt_tts.audio import pcm16_to_wav, streaming_wav_header
from stt_tts.config import Settings
from stt_tts.cuda import cuda_status
from stt_tts.engines.base import (
    EngineNotInstalled,
    ModelNotFound,
    STTEngine,
    TranscriptionInfo,
    TranscriptionSegment,
    TTSEngine,
)
from stt_tts.engines.manager import EngineManager
from stt_tts.models import (
    STT_RESPONSE_FORMATS,
    CudaInfo,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    SpeechRequest,
)


def _timestamp(seconds: float, millis_sep: str) -> str:
    if seconds < 0:
        seconds = 0.0
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        millis = 999
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"


def _to_srt(segments: list[TranscriptionSegment]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(f"{_timestamp(segment.start, ',')} --> {_timestamp(segment.end, ',')}")
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines)


def _to_vtt(segments: list[TranscriptionSegment]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for segment in segments:
        lines.append(f"{_timestamp(segment.start, '.')} --> {_timestamp(segment.end, '.')}")
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _collect_transcription(
    engine: STTEngine, audio: bytes, language: str | None, options: dict
) -> dict:
    info, segments = engine.transcribe(audio, language=language, **options)
    materialized = list(segments)
    text = "".join(segment.text for segment in materialized).strip()
    return {"info": info, "segments": materialized, "text": text}


def _transcribe_sse(
    engine: STTEngine, audio: bytes, language: str | None, options: dict
) -> Iterator[str]:
    info, segments = engine.transcribe(audio, language=language, **options)
    yield _sse(
        {
            "type": "info",
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }
    )
    parts: list[str] = []
    for segment in segments:
        parts.append(segment.text)
        yield _sse(
            {
                "type": "segment",
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
        )
    yield _sse({"type": "done", "text": "".join(parts).strip()})
    yield "data: [DONE]\n\n"


def _synthesize_full(
    engine: TTSEngine, text: str, voice: str | None, speed: float
) -> tuple[bytes, int, int, int]:
    pcm = bytearray()
    sample_rate: int | None = None
    channels = 1
    sample_width = 2
    for chunk in engine.synthesize(text, voice=voice, speed=speed):
        sample_rate = chunk.sample_rate
        channels = chunk.channels
        sample_width = chunk.sample_width
        pcm += chunk.data
    if sample_rate is None:
        sample_rate = engine.sample_rate(voice)
    return bytes(pcm), sample_rate, channels, sample_width


def _synthesize_stream(
    engine: TTSEngine, text: str, voice: str | None, speed: float, response_format: str
) -> Iterator[bytes]:
    chunks = engine.synthesize(text, voice=voice, speed=speed)
    first = next(chunks, None)
    if first is None:
        return
    if response_format == "wav":
        yield streaming_wav_header(first.sample_rate, first.channels, first.sample_width)
    yield first.data
    for chunk in chunks:
        yield chunk.data


def create_app(settings: Settings | None = None, manager: EngineManager | None = None) -> FastAPI:
    settings = settings or Settings()
    manager = manager or EngineManager(settings)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Local speech-to-text and text-to-speech HTTP API with per-request "
            "model/voice selection and streaming."
        ),
    )

    def get_manager() -> EngineManager:
        return manager

    manager_dependency = Depends(get_manager)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        # cuda_status() may import the CTranslate2 C extension on first call;
        # run it off the event loop. The result is cached per process.
        cuda = await run_in_threadpool(cuda_status)
        return HealthResponse(
            status="ok",
            device=settings.device,
            cuda=CudaInfo(**cuda),
            stt_models=manager.stt_models(),
            tts_models=manager.tts_models(),
        )

    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models(mgr: EngineManager = manager_dependency) -> ModelsResponse:
        data = [ModelInfo(id=model_id, type="stt") for model_id in mgr.stt_models()]
        for model_id in mgr.tts_models():
            try:
                voices = mgr.voices(model_id)
            except Exception:
                voices = []
            data.append(ModelInfo(id=model_id, type="tts", voices=voices))
        return ModelsResponse(data=data)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: Annotated[UploadFile, File(description="Audio file to transcribe")],
        mgr: EngineManager = manager_dependency,
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
        stream: Annotated[bool, Form()] = False,
        word_timestamps: Annotated[bool, Form()] = False,
    ):
        if response_format not in STT_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Unsupported response_format '{response_format}'. "
                    f"Choose one of: {', '.join(STT_RESPONSE_FORMATS)}."
                ),
            )
        try:
            engine = mgr.get_stt(model)
        except ModelNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        audio = await file.read()
        if not audio:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty audio file."
            )

        # Warm the model now so dependency/load errors become proper HTTP errors
        # instead of failing mid-stream after the response has started.
        try:
            await run_in_threadpool(engine.ensure_ready)
        except EngineNotInstalled as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - model load (GPU/cuDNN/download) failure
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"STT model failed to load: {exc}. Check the GPU/cuDNN libraries "
                    "(faster-whisper needs cuDNN 9 / cuBLAS for CUDA 12) or set DEVICE=cpu."
                ),
            ) from exc

        options = {"word_timestamps": word_timestamps}

        if stream:
            generator = _transcribe_sse(engine, audio, language, options)
            return StreamingResponse(
                iterate_in_threadpool(generator), media_type="text/event-stream"
            )

        try:
            result = await run_in_threadpool(
                _collect_transcription, engine, audio, language, options
            )
        except EngineNotInstalled as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface backend failures as 500
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Transcription failed: {exc}"
            ) from exc

        info: TranscriptionInfo = result["info"]
        segments: list[TranscriptionSegment] = result["segments"]
        text: str = result["text"]

        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "srt":
            return PlainTextResponse(_to_srt(segments))
        if response_format == "vtt":
            return PlainTextResponse(_to_vtt(segments))
        if response_format == "verbose_json":
            return JSONResponse(
                {
                    "task": "transcribe",
                    "language": info.language,
                    "duration": info.duration,
                    "text": text,
                    "segments": [
                        {"id": s.id, "start": s.start, "end": s.end, "text": s.text}
                        for s in segments
                    ],
                }
            )
        return JSONResponse({"text": text})

    @app.post("/v1/audio/speech")
    async def speech(payload: SpeechRequest, mgr: EngineManager = manager_dependency):
        try:
            engine = mgr.get_tts(payload.model)
        except ModelNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        voice = payload.voice or mgr.default_voice_for(payload.model)

        try:
            await run_in_threadpool(engine.ensure_ready, voice)
        except EngineNotInstalled as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except ModelNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - model/voice load (GPU/weights) failure
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"TTS model '{payload.model or 'default'}' failed to load: {exc}",
            ) from exc

        if payload.stream:
            generator = _synthesize_stream(
                engine, payload.input, voice, payload.speed, payload.response_format
            )
            if payload.response_format == "wav":
                return StreamingResponse(
                    iterate_in_threadpool(generator), media_type="audio/wav"
                )
            sample_rate = engine.sample_rate(voice)
            headers = {
                "X-Sample-Rate": str(sample_rate),
                "X-Audio-Channels": "1",
                "X-Audio-Bits": "16",
            }
            return StreamingResponse(
                iterate_in_threadpool(generator),
                media_type="application/octet-stream",
                headers=headers,
            )

        try:
            pcm, sample_rate, channels, sample_width = await run_in_threadpool(
                _synthesize_full, engine, payload.input, voice, payload.speed
            )
        except EngineNotInstalled as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except ModelNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface backend failures as 500
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Speech synthesis failed: {exc}"
            ) from exc

        if payload.response_format == "pcm":
            return Response(
                content=pcm,
                media_type="application/octet-stream",
                headers={
                    "X-Sample-Rate": str(sample_rate),
                    "X-Audio-Channels": str(channels),
                    "X-Audio-Bits": str(sample_width * 8),
                },
            )
        return Response(
            content=pcm16_to_wav(pcm, sample_rate, channels, sample_width),
            media_type="audio/wav",
        )

    return app


app = create_app()


def run() -> None:
    settings = Settings()
    uvicorn.run("stt_tts.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
