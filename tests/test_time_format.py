"""Unit tests for the shared 24-hour -> 12-hour spoken time conversion.

Regression coverage for a real production bug: raw 24-hour "HH:MM" slot
times fed directly to gpt-oss-20b were garbled when the model tried to
verbalize them itself (a 13:00 slot came out as "3 PM, at 13" in a live
call) - see time_format.py and ai/conversation.py's _build_system_prompt.

Run with: venv/Scripts/python -m pytest -q tests/test_time_format.py
"""
from time_format import format_time_12h


def test_format_time_12h_covers_midnight_noon_and_half_hours():
    assert format_time_12h("09:00") == "9 AM"
    assert format_time_12h("09:30") == "9:30 AM"
    assert format_time_12h("00:00") == "12 AM"
    assert format_time_12h("12:00") == "12 PM"
    assert format_time_12h("16:00") == "4 PM"
    assert format_time_12h("23:45") == "11:45 PM"


def test_format_time_12h_handles_the_confusing_13_00_slot():
    """The exact slot that garbled into "3 PM, at 13" in a live call - must
    now already be unambiguous 12-hour form before it ever reaches the LLM."""
    assert format_time_12h("13:00") == "1 PM"
