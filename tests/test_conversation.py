"""Phase 2 verification: drives the Groq-backed conversation logic directly
with scripted caller turns - no Twilio, no real phone call required.

Needs a real GROQ_API_KEY in .env. If it's missing, or the call fails (e.g.
an invalid key), tests are skipped rather than failed, since that's a
configuration issue, not a code defect.

Run with: venv/Scripts/python -m pytest -s tests/test_conversation.py
"""
from datetime import date, timedelta

import pytest

from ai.conversation import FALLBACK_MESSAGE, ConversationManager
from ai.intent_interpreter import BookingIntent, TurnUnderstanding
from config import load_config

DEMO_BUSINESS = {
    "id": 1,
    "name": "Radiant Skin Clinic",
    "vertical": "clinic",
    "language_pref": "hindi",
}


def _demo_slots():
    # Dates relative to "today" (not hardcoded) so "tomorrow"/"kal" always
    # resolves to a real offered slot regardless of when the suite runs.
    tomorrow = date.today() + timedelta(days=1)
    day_after = date.today() + timedelta(days=2)
    return [
        {"id": 1, "date": tomorrow.isoformat(), "time": "10:00"},
        {"id": 2, "date": tomorrow.isoformat(), "time": "15:30"},
        {"id": 3, "date": day_after.isoformat(), "time": "11:00"},
    ]


@pytest.fixture
def manager():
    config = load_config()
    if not config.groq_api_key:
        pytest.skip("GROQ_API_KEY not set - see .env.example")
    return ConversationManager(config.groq_api_key, config.groq_model)


def _say(manager, call_id, transcript, label):
    result = manager.get_reply(call_id, transcript)
    print(f"\nCaller ({label}): {transcript}")
    print(f"Agent: {result['reply_text']}")
    if result["reply_text"] == FALLBACK_MESSAGE:
        pytest.skip("Groq call failed (invalid/missing GROQ_API_KEY?) - set a real key in .env to run this test")
    if result["booking"]:
        print(f"  -> booking: {result['booking']}")
    return result


def test_happy_path_booking_in_hindi(manager):
    call_id = "TEST-HAPPY-1"
    slots = _demo_slots()
    greeting = manager.start_session(call_id, DEMO_BUSINESS, slots)
    print(f"\nGreeting: {greeting}")

    r1 = _say(manager, call_id, "Mujhe skin check up ke liye appointment chahiye", "Hindi")
    assert not r1["hangup"]

    r2 = _say(manager, call_id, "Kal subah 10 baje ho to accha rahega", "Hindi")
    assert not r2["hangup"]

    r3 = _say(manager, call_id, "Mera naam Priya Sharma hai", "Hindi")
    assert not r3["hangup"]

    r4 = _say(manager, call_id, "Haan, theek hai", "Hindi")
    assert r4["hangup"] is True
    assert r4["booking"] is not None
    assert r4["booking"]["customer_name"]
    print(f"\nFinal booking: {r4['booking']}")


def test_declines_out_of_scope_question(manager):
    call_id = "TEST-SCOPE-1"
    manager.start_session(call_id, DEMO_BUSINESS, _demo_slots())

    result = _say(manager, call_id, "Doctor visit ka price kitna hai?", "Hindi (pricing question)")

    assert result["booking"] is None
    assert not result["hangup"]
    # Loose check: the model should redirect rather than actually quote a price.
    assert "call you back" in result["reply_text"].lower() or "someone from" in result["reply_text"].lower()


def test_code_switched_english_hindi(manager):
    call_id = "TEST-CODESWITCH-1"
    manager.start_session(call_id, DEMO_BUSINESS, _demo_slots())

    result = _say(manager, call_id, "Hi, I need an appointment, ek acne problem hai", "English/Hindi mix")

    assert result["reply_text"]
    assert not result["hangup"]


class _AlwaysUnclearInterpreter:
    """Simulates the LLM classifying every turn as UNCLEAR with zero
    extracted entities - NOT a simulated API failure (interpreted=True, so
    the give-up-after-3-failures safety net doesn't fire). This is the
    worst realistic case for the deterministic slot-matching layer, and
    lets this regression test run with no API key at all."""

    def interpret(self, caller_speech, context):
        return TurnUnderstanding(intent=BookingIntent.UNCLEAR, confidence=0.0, interpreted=True)


def test_fragmented_speech_never_repeats_the_identical_question():
    """Regression test for the real production bug that motivated the
    BookingStage rewrite: a caller answering "preferred date and time" with
    "tomorrow", "Friday", "17", and "17 July 2026" while the old code never
    matched any of them to a real slot and just re-asked the exact same
    question forever. Runs fully offline (no GROQ_API_KEY needed) since it
    targets the deterministic slot-matching layer, using a worst-case fake
    interpreter that never understands anything."""
    manager = ConversationManager.__new__(ConversationManager)
    manager._intent_interpreter = _AlwaysUnclearInterpreter()
    manager._sessions = {}

    call_id = "TEST-FRAGMENTED-1"
    business = {**DEMO_BUSINESS, "language_pref": "english"}
    # One slot per date so "tomorrow" alone is unambiguous - a separate
    # test (in test_slot_matching.py) already covers the multi-slot-per-day
    # "which time?" clarification path.
    slots = [
        {"id": 1, "date": (date.today() + timedelta(days=1)).isoformat(), "time": "10:00"},
        {"id": 2, "date": (date.today() + timedelta(days=2)).isoformat(), "time": "15:30"},
        {"id": 3, "date": (date.today() + timedelta(days=3)).isoformat(), "time": "11:00"},
    ]
    manager.start_session(call_id, business, slots)

    turns = [
        "I",
        "have a skin disease and I want to look for a doctor.",
        "have a circular ring",
        "worms.",
        "So",
        "I have to look for a doctor to clear.",
        "Uh, yes, uh, can we book an appointment for tomorrow?",
    ]

    replies = []
    for turn in turns:
        result = manager.get_reply(call_id, turn)
        replies.append(result["reply_text"])
        if result["hangup"]:
            break

    consecutive_repeats = sum(1 for a, b in zip(replies, replies[1:]) if a == b)
    # The old bug repeated the identical question every single turn from the
    # second exchange onward; the new deterministic layer should resolve
    # "tomorrow" to a real slot well before that, so repeats stay rare.
    assert consecutive_repeats <= 1, f"Repeated identical replies too often: {replies}"

    # "tomorrow" is unambiguous here (only one slot falls on that date), so
    # the flow should have moved on to asking for the caller's name.
    assert "name" in replies[-1].lower()
