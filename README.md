# STT/TTS API

A small, local-first FastAPI server that exposes two HTTP APIs:

- **`POST /v1/audio/transcriptions`** — speech-to-text (audio file → transcript)
- **`POST /v1/audio/speech`** — text-to-speech (text → audio)

The model (and, for TTS, the voice) is chosen **per HTTP request**, and both
endpoints support **streaming**. Everything runs locally — no external API
calls. The request/response shapes mirror the OpenAI audio API so existing
clients and the companion [`image-tagging-api`](https://github.com/TheRealMarcusChiu/image-tagging-api)
audio provider can talk to it directly.

## Engines

| Direction | Engine | Models / voices | License |
|---|---|---|---|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `large-v3-turbo` (default), `large-v3`, `distil-large-v3`, `medium`, `small`, `base`, `tiny`, … | MIT |
| TTS | [Kokoro](https://github.com/hexgrad/kokoro) (default) | 50+ named voices (`af_heart`, `am_michael`, `bf_emma`, …), multilingual via `lang_code` | Apache-2.0 |
| TTS | [Piper](https://github.com/OHF-Voice/piper1-gpl) | any downloaded `<voice>.onnx` | GPL-3.0 |

The model selected on the STT endpoint maps to a faster-whisper model size; the
model selected on the TTS endpoint chooses the **engine** (`kokoro` / `piper`)
and `voice` chooses the voice within it.

## Features

- Per-request `model` and (for TTS) `voice` selection
- Streaming for both endpoints (`stream=true`)
  - STT streams transcript segments as Server-Sent Events
  - TTS streams audio (WAV with a streaming header, or raw PCM) as it is generated
- STT response formats: `json` (default), `text`, `verbose_json`, `srt`, `vtt`
- TTS response formats: `wav` (default), `pcm`
- `GET /v1/models` lists available STT/TTS models and TTS voices
- `GET /health` reports device and configured models
- Blocking model inference runs in a threadpool so it never stalls the event loop
- Heavy ML dependencies are optional extras and lazily imported, so the package
  installs and the test suite runs without a GPU

## Quick start

The base install (no ML engines) is enough to import the app and run the tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
ruff check .
```

To actually run the server you need the engine extras plus the `espeak-ng`
system package (G2P backend for Kokoro and Piper):

```bash
sudo apt-get install -y espeak-ng libsndfile1     # Debian/Ubuntu
pip install -e '.[all]'                            # faster-whisper + kokoro + piper
cp .env.example .env
uvicorn stt_tts.main:app --host 0.0.0.0 --port 8000
```

> **GPU note:** Kokoro pulls in PyTorch and faster-whisper uses CTranslate2.
> For RTX 4070 / CUDA acceleration set `DEVICE=cuda` and install a CUDA build of
> PyTorch (and `onnxruntime-gpu` if you use Piper on GPU). The CPU default
> works everywhere but is slower.

Open interactive docs at http://localhost:8000/docs.

### Piper voices

Piper loads a voice from a local `.onnx` file in `PIPER_VOICES_DIR`:

```bash
python -m piper.download_voices en_US-lessac-medium   # downloads .onnx + .onnx.json
```

## Speech-to-text

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=large-v3-turbo \
  -F language=en \
  -F response_format=json \
  -F file=@/path/to/audio.mp3
```

```json
{ "text": "the transcript of the audio" }
```

`response_format=verbose_json` adds `language`, `duration`, and per-segment
timings; `text`, `srt`, and `vtt` return the corresponding plain-text formats.

### Streaming transcription (SSE)

```bash
curl -N -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=large-v3-turbo \
  -F stream=true \
  -F file=@/path/to/audio.mp3
```

```
data: {"type": "info", "language": "en", "language_probability": 0.99, "duration": 12.3}
data: {"type": "segment", "id": 0, "start": 0.0, "end": 3.2, "text": " Hello there."}
data: {"type": "segment", "id": 1, "start": 3.2, "end": 6.0, "text": " This streams."}
data: {"type": "done", "text": "Hello there. This streams."}
data: [DONE]
```

## Text-to-speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"af_heart","input":"Hello from a local server.","response_format":"wav"}' \
  --output speech.wav
```

Body fields: `model` (engine), `input` (text), `voice`, `response_format`
(`wav`|`pcm`), `speed` (0.25–4.0), `stream`.

### Streaming speech

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"af_heart","input":"Streaming audio as it is generated.","stream":true,"response_format":"pcm"}' \
  --output speech.pcm
```

For `response_format=pcm` the sample rate is returned in the `X-Sample-Rate`
response header (16-bit mono). `wav` streaming emits a WAV header up front
followed by PCM frames.

## Configuration

All settings come from environment variables / `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` for all engines |
| `DEFAULT_STT_MODEL` | `large-v3-turbo` | model used when a request omits `model` |
| `STT_MODELS` | `large-v3-turbo,…,tiny` | comma-separated allowlist of selectable STT models |
| `STT_COMPUTE_TYPE` | `default` | CTranslate2 compute type (`float16`, `int8`, …) |
| `STT_VAD_FILTER` | `true` | drop non-speech with VAD |
| `STT_ALLOW_ANY_MODEL` | `false` | allow any faster-whisper id, not just the allowlist |
| `DEFAULT_TTS_MODEL` | `kokoro` | TTS engine used when a request omits `model` |
| `TTS_MODELS` | `kokoro,piper` | enabled TTS engines |
| `DEFAULT_VOICE` | `af_heart` | default Kokoro voice |
| `KOKORO_LANG_CODE` | `a` | Kokoro language code |
| `PIPER_VOICES_DIR` | `/models/piper` | directory of Piper `.onnx` voices |
| `PIPER_DEFAULT_VOICE` | `en_US-lessac-medium` | default Piper voice |
| `MODEL_CACHE_DIR` | _unset_ | faster-whisper weight cache dir |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | server bind |

## Docker

```bash
docker compose up --build      # GPU via NVIDIA Container Toolkit (see compose file)
```

or plain Docker (CPU):

```bash
docker build -t stt-tts .
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/models:/models" stt-tts
```

## systemd

`stt-tts.service` runs the API directly on a Linux host / Proxmox LXC. Adjust
`WorkingDirectory`, the `.venv` path, and uncomment `EnvironmentFile` as needed,
then:

```bash
sudo cp stt-tts.service /etc/systemd/system/stt-tts.service
sudo systemctl daemon-reload
sudo systemctl enable --now stt-tts
```

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
```

The test suite injects fake engines through `create_app(manager=...)`, so it
exercises the full HTTP surface (routing, streaming, formats, error mapping)
without downloading models or needing a GPU.

## Project status

The HTTP layer, request/response handling, streaming, formats, and error
mapping are covered by tests. The real model backends (faster-whisper, Kokoro,
Piper) are integrated against their documented APIs but must be exercised on a
machine with the engines and a GPU installed — they are not run in CI.
