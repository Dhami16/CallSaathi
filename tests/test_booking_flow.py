"""Phase 3 verification: simulates a full call -> booking without touching
Groq or Twilio, and asserts the SQLite records and mock notifications are
correct.

Run with: venv/Scripts/python -m pytest -s tests/test_booking_flow.py
"""
import pytest

import seed_data
from booking.db import get_connection
from booking.repository import BookingRepository, SlotUnavailableError
from call_handler import CallHandler
from notifications.mock_service import MockNotificationService
from telephony.base import TelephonyProvider


class FakeTelephonyProvider(TelephonyProvider):
    """Bypasses Twilio entirely - passes normalized dicts straight through,
    since these tests exercise CallHandler's own logic, not the adapter."""

    def parse_incoming_call(self, raw_request_data):
        return raw_request_data

    def parse_speech_result(self, raw_request_data):
        return raw_request_data

    def build_greeting_response(self, greeting_text, gather_action_url, language="english"):
        return {"type": "greeting", "text": greeting_text}

    def build_reply_response(self, reply_text, hangup=False):
        return {"type": "reply", "text": reply_text, "hangup": hangup}


class FakeConversationManager:
    """Scripted stand-in for ai.conversation.ConversationManager: greets,
    then confirms a booking on the very next turn - no LLM call involved."""

    def __init__(self):
        self.sessions = {}

    def start_session(self, call_id, business, slots):
        self.sessions[call_id] = {"slots": {s["id"]: s for s in slots}}
        return f"Hello! You've reached {business['name']}."

    def get_reply(self, call_id, transcript):
        slot = next(iter(self.sessions[call_id]["slots"].values()))
        return {
            "reply_text": f"Booked for {slot['date']} at {slot['time']}!",
            "hangup": True,
            "booking": {
                "slot_id": slot["id"],
                "slot_date": slot["date"],
                "slot_time": slot["time"],
                "customer_name": "Priya Sharma",
                "reason": "skin consultation",
            },
        }

    def get_transcript(self, call_id):
        return "Caller: Mujhe ek appointment chahiye\nAgent: Booked for the slot!"

    def get_duration_seconds(self, call_id):
        return 42

    def end_session(self, call_id):
        self.sessions.pop(call_id, None)


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test_callsaathi.db")
    seed_data.seed(path)
    return path


def test_full_call_creates_booking_and_notifications(db_path, caplog):
    repository = BookingRepository(db_path)
    handler = CallHandler(
        telephony_provider=FakeTelephonyProvider(),
        conversation_manager=FakeConversationManager(),
        booking_repository=repository,
        notification_service=MockNotificationService(),
    )

    conn = get_connection(db_path)
    business = conn.execute("SELECT * FROM businesses LIMIT 1").fetchone()
    available_before = conn.execute(
        "SELECT COUNT(*) c FROM slots WHERE business_id = ? AND is_booked = 0", (business["id"],)
    ).fetchone()["c"]
    conn.close()

    call_data = {
        "caller_number": "+919876500000",
        "call_id": "CALL-TEST-1",
        "called_number": business["phone_number"],
    }

    with caplog.at_level("INFO"):
        greeting = handler.handle_incoming_call(call_data, gather_action_url="http://test/voice/handle-input")
        print("Greeting:", greeting)

        speech_data = {**call_data, "transcript": "Mujhe kal appointment chahiye", "confidence": 0.9}
        reply = handler.handle_speech_input(speech_data)
        print("Reply:", reply)

    assert reply["hangup"] is True

    conn = get_connection(db_path)
    booking_row = conn.execute("SELECT * FROM bookings WHERE customer_name = 'Priya Sharma'").fetchone()
    assert booking_row is not None
    assert booking_row["reason"] == "skin consultation"
    assert booking_row["customer_phone"] == "+919876500000"

    slot_row = conn.execute("SELECT * FROM slots WHERE id = ?", (booking_row["slot_id"],)).fetchone()
    assert slot_row["is_booked"] == 1

    call_log_row = conn.execute("SELECT * FROM call_logs WHERE caller_phone = '+919876500000'").fetchone()
    assert call_log_row is not None
    assert call_log_row["outcome"] == "booked"
    assert call_log_row["duration_seconds"] == 42
    assert "Caller: Mujhe ek appointment chahiye" in call_log_row["transcript"]

    available_after = conn.execute(
        "SELECT COUNT(*) c FROM slots WHERE business_id = ? AND is_booked = 0", (business["id"],)
    ).fetchone()["c"]
    conn.close()
    assert available_after == available_before - 1

    owner_logs = [r.getMessage() for r in caplog.records if "OWNER NOTIFICATION" in r.getMessage()]
    customer_logs = [r.getMessage() for r in caplog.records if "CUSTOMER NOTIFICATION" in r.getMessage()]

    assert len(owner_logs) == 1
    assert "Priya Sharma" in owner_logs[0]
    assert "skin consultation" in owner_logs[0]
    assert "Caller: Mujhe ek appointment chahiye" in owner_logs[0]  # full transcript, never a summary

    assert len(customer_logs) == 1
    assert "confirmed" in customer_logs[0]


def test_double_booking_same_slot_is_rejected(db_path):
    repository = BookingRepository(db_path)
    conn = get_connection(db_path)
    business = conn.execute("SELECT * FROM businesses LIMIT 1").fetchone()
    slot = conn.execute(
        "SELECT * FROM slots WHERE business_id = ? AND is_booked = 0 LIMIT 1", (business["id"],)
    ).fetchone()
    conn.close()

    repository.book_slot(business["id"], slot["id"], "First Caller", "+911111111111", "reason A")

    with pytest.raises(SlotUnavailableError):
        repository.book_slot(business["id"], slot["id"], "Second Caller", "+922222222222", "reason B")
