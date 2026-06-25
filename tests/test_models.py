from __future__ import annotations

import pytest
from pydantic import ValidationError

from stt_tts.models import SpeechRequest


def test_speech_request_defaults():
    request = SpeechRequest(input="hello")
    assert request.response_format == "wav"
    assert request.speed == 1.0
    assert request.stream is False
    assert request.model is None


def test_speech_request_empty_input():
    with pytest.raises(ValidationError):
        SpeechRequest(input="")


def test_speech_request_speed_bounds():
    with pytest.raises(ValidationError):
        SpeechRequest(input="hi", speed=10)


def test_speech_request_bad_format():
    with pytest.raises(ValidationError):
        SpeechRequest(input="hi", response_format="mp3")
