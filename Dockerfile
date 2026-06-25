FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# espeak-ng: G2P backend required by Kokoro and Piper.
# libsndfile1: required by soundfile (Kokoro audio I/O).
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Installs the STT + TTS engines. NOTE: this pulls the default (CPU) build of
# torch via Kokoro. For RTX 4070 / CUDA acceleration, build on an NVIDIA CUDA
# base image or pre-install a CUDA torch wheel before `pip install '.[all]'`.
RUN pip install --no-cache-dir '.[all]'

EXPOSE 8000

CMD ["uvicorn", "stt_tts.main:app", "--host", "0.0.0.0", "--port", "8000"]
