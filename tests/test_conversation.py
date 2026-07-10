"""Phase 2 verification: drives the Groq-backed conversation logic directly
with scripted caller turns - no Twilio, no real phone call required.

Needs a real GROQ_API_KEY in .env. If it's missing, or the call fails (e.g.
an invalid key), tests are skipped rather than failed, since that's a
configuration issue, not a code defect.

Run with: venv/Scripts/python -m pytest -s tests/test_conversation.py
"""
import pytest

from ai.conversation import FALLBACK_MESSAGE, ConversationManager
from config import load_config

DEMO_BUSINESS = {
    "id": 1,
    "name": "Radiant Skin Clinic",
    "vertical": "clinic",
    "language_pref": "hindi",
}
DEMO_SLOTS = [
    {"id": 1, "date": "2026-07-12", "time": "10:00"},
    {"id": 2, "date": "2026-07-12", "time": "15:30"},
    {"id": 3, "date": "2026-07-13", "time": "11:00"},
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
    greeting = manager.start_session(call_id, DEMO_BUSINESS, DEMO_SLOTS)
    print(f"\nGreeting: {greeting}")

    r1 = _say(manager, call_id, "Mujhe skin check up ke liye appointment chahiye", "Hindi")
    assert not r1["hangup"]

    r2 = _say(manager, call_id, "Kal subah mein koi time ho to accha rahega", "Hindi")
    assert not r2["hangup"]

    r3 = _say(manager, call_id, "Haan 10 baje wala theek hai, mera naam Priya Sharma hai", "Hindi")
    assert r3["hangup"] is True
    assert r3["booking"] is not None
    assert r3["booking"]["customer_name"]
    print(f"\nFinal booking: {r3['booking']}")


def test_declines_out_of_scope_question(manager):
    call_id = "TEST-SCOPE-1"
    manager.start_session(call_id, DEMO_BUSINESS, DEMO_SLOTS)

    result = _say(manager, call_id, "Doctor visit ka price kitna hai?", "Hindi (pricing question)")

    assert result["booking"] is None
    assert not result["hangup"]
    # Loose check: the model should redirect rather than actually quote a price.
    assert "call you back" in result["reply_text"].lower() or "someone from" in result["reply_text"].lower()


def test_code_switched_english_hindi(manager):
    call_id = "TEST-CODESWITCH-1"
    manager.start_session(call_id, DEMO_BUSINESS, DEMO_SLOTS)

    result = _say(manager, call_id, "Hi, I need an appointment, ek acne problem hai", "English/Hindi mix")

    assert result["reply_text"]
    assert not result["hangup"]
