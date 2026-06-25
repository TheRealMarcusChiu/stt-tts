from __future__ import annotations

import io
import struct
import wave


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw little-endian PCM samples in a complete WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def streaming_wav_header(sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Build a WAV header for a stream of unknown length.

    The RIFF and data chunk sizes are set to the 32-bit max as a placeholder so
    players can begin playback before the full PCM payload is known.
    """
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    bits_per_sample = sample_width * 8
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH",
            16,  # PCM fmt chunk size
            1,  # audio format = PCM
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )
