from __future__ import annotations

import io
import struct
import wave

from stt_tts.audio import pcm16_to_wav, streaming_wav_header


def test_pcm16_to_wav_roundtrip():
    pcm = b"\x01\x00" * 1000
    data = pcm16_to_wav(pcm, 24000, channels=1, sample_width=2)
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data)) as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(1000) == pcm


def test_streaming_wav_header():
    header = streaming_wav_header(16000, channels=1, sample_width=2)
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert b"fmt " in header
    assert b"data" in header
    sample_rate = struct.unpack("<I", header[24:28])[0]
    assert sample_rate == 16000
