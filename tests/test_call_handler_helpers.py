"""Unit tests for CallHandler's pure helper functions: filler-word filtering
and Twilio Gather speech-recognition hints. Both are regression coverage for
real issues found in production call transcripts (see call_handler.py's
comments on _FILLER_WORDS and _build_speech_hints for the specifics).

Run with: venv/Scripts/python -m pytest -q tests/test_call_handler_helpers.py
"""
from call_handler import _build_speech_hints, _format_time_hint, _is_filler_only


def test_pure_filler_transcripts_are_detected():
    for transcript in ["Mm-hmm", "Uh,", "Um,", "hmm", "Uh, um", "Huh?"]:
        assert _is_filler_only(transcript) is True, transcript


def test_real_content_is_not_treated_as_filler():
    for transcript in [
        "I have a skin problem and I want to see a doctor",
        "yes",  # carries real meaning (confirming a slot) - must NOT be filtered
        "no",
        "okay, 22nd",
        "Um, I need an appointment",  # filler prefix but real content follows
        "930",
    ]:
        assert _is_filler_only(transcript) is False, transcript


def test_empty_and_whitespace_only_are_not_filler():
    # Handled separately by the `not transcript` check in call_handler - not
    # this function's job, so it must not also claim these as filler.
    assert _is_filler_only("") is False
    assert _is_filler_only("   ") is False


def test_format_time_hint_covers_midnight_noon_and_half_hours():
    assert _format_time_hint("09:00") == "9 AM"
    assert _format_time_hint("09:30") == "9:30 AM"
    assert _format_time_hint("00:00") == "12 AM"
    assert _format_time_hint("12:00") == "12 PM"
    assert _format_time_hint("16:00") == "4 PM"
    assert _format_time_hint("23:45") == "11:45 PM"


def test_build_speech_hints_dedupes_and_sorts():
    slots = [
        {"id": 1, "date": "2026-07-21", "time": "16:00"},
        {"id": 2, "date": "2026-07-22", "time": "09:30"},
        {"id": 3, "date": "2026-07-22", "time": "16:00"},  # same time as slot 1 - must not duplicate
    ]
    assert _build_speech_hints(slots) == "4 PM, 9:30 AM"


def test_build_speech_hints_handles_no_slots():
    assert _build_speech_hints([]) == ""
