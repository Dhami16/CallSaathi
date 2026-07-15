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

    def build_continue_response(self, sentence_text, continue_url):
        return {"type": "continue", "text": sentence_text, "continue_url": continue_url}


class FakeConversationManager:
    """Scripted stand-in for ai.conversation.ConversationManager: greets,
    then confirms a booking on the very next turn in a single (non-streamed)
    sentence - no LLM call involved."""

    def __init__(self):
        self.sessions = {}

    def start_session(self, call_id, business, slots):
        self.sessions[call_id] = {"slots": {s["id"]: s for s in slots}}
        return f"Hello! You've reached {business['name']}."

    def start_streaming_reply(self, call_id, transcript):
        slot = next(iter(self.sessions[call_id]["slots"].values()))
        return {
            "sentence": f"Booked for {slot['date']} at {slot['time']}!",
            "more_coming": False,
            "hangup": True,
            "booking": {
                "slot_id": slot["id"],
                "slot_date": slot["date"],
                "slot_time": slot["time"],
                "customer_name": "Priya Sharma",
                "reason": "skin consultation",
            },
        }

    def get_next_streamed_sentence(self, call_id, sentence_index):
        raise AssertionError("not expected to be called - this fake never sets more_coming=True")

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


def test_full_call_creates_booking_and_notifications(db_path, capsys):
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

    # Don't hardcode a date: seed_data.py generates slots relative to
    # "today," so the first offered slot's date shifts as real time passes.
    # FakeConversationManager below always picks whichever slot it's handed
    # first, i.e. the first one get_available_slots() would return.
    first_slot = repository.get_available_slots(business["id"])[0]

    call_data = {
        "caller_number": "+919876500000",
        "call_id": "CALL-TEST-1",
        "called_number": business["phone_number"],
    }

    greeting = handler.handle_incoming_call(call_data, gather_action_url="http://test/voice/handle-input")
    print("Greeting:", greeting)

    speech_data = {**call_data, "transcript": "Mujhe kal appointment chahiye", "confidence": 0.9}
    reply = handler.handle_speech_input(speech_data, continue_url_base="http://test/voice/continue")
    print("Reply:", reply)

    # structlog's PrintLoggerFactory writes straight to stdout rather than
    # through stdlib logging, so pytest's caplog (which hooks the logging
    # module) can't see it - capsys reads the actual stdout stream instead.
    captured_out = capsys.readouterr().out

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

    turn_rows = conn.execute(
        "SELECT * FROM call_turns WHERE call_id = 'CALL-TEST-1' ORDER BY turn_number"
    ).fetchall()
    conn.close()
    assert available_after == available_before - 1

    # One row for the greeting (turn 0), one for the booking turn (turn 1).
    assert [r["turn_number"] for r in turn_rows] == [0, 1]
    assert turn_rows[0]["response_out"] == "Hello! You've reached Radiant Skin Clinic."
    assert turn_rows[0]["transcript_in"] is None
    assert turn_rows[0]["total_latency_ms"] >= 0
    assert turn_rows[1]["transcript_in"] == "Mujhe kal appointment chahiye"
    assert turn_rows[1]["response_out"] == f"Booked for {first_slot['date']} at {first_slot['time']}!"
    assert turn_rows[1]["llm_latency_ms"] >= 0
    # Not measurable given Twilio's <Gather>-based telephony - see booking/db.py.
    assert turn_rows[1]["stt_latency_ms"] is None
    assert turn_rows[1]["tts_latency_ms"] is None

    # Filter on "message=" specifically: both mock_service's actual content
    # log and call_handler's stage-tracking log mention channel=owner/
    # customer, but only mock_service's carries the notification body.
    owner_logs = [line for line in captured_out.splitlines() if "channel=owner" in line and "message=" in line]
    customer_logs = [line for line in captured_out.splitlines() if "channel=customer" in line and "message=" in line]

    assert len(owner_logs) == 1
    assert "Priya Sharma" in owner_logs[0]
    assert "skin consultation" in owner_logs[0]
    assert "Caller: Mujhe ek appointment chahiye" in owner_logs[0]  # full transcript, never a summary

    assert len(customer_logs) == 1
    assert "confirmed" in customer_logs[0]


def test_double_booking_same_slot_by_different_calls_is_rejected(db_path):
    """Two different callers (different call_id) racing for the same slot -
    the second one must be rejected."""
    repository = BookingRepository(db_path)
    conn = get_connection(db_path)
    business = conn.execute("SELECT * FROM businesses LIMIT 1").fetchone()
    slot = conn.execute(
        "SELECT * FROM slots WHERE business_id = ? AND is_booked = 0 LIMIT 1", (business["id"],)
    ).fetchone()
    conn.close()

    repository.book_slot(business["id"], slot["id"], "First Caller", "+911111111111", "reason A", call_id="CALL-A")

    with pytest.raises(SlotUnavailableError):
        repository.book_slot(business["id"], slot["id"], "Second Caller", "+922222222222", "reason B", call_id="CALL-B")


def test_retried_webhook_for_same_call_is_idempotent_not_duplicated(db_path):
    """Task 5: the SAME call_id booking the SAME slot twice (simulating a
    Twilio webhook retry re-processing an already-confirmed booking) must
    not create a second row or re-touch the slot - it should just return
    the existing booking."""
    repository = BookingRepository(db_path)
    conn = get_connection(db_path)
    business = conn.execute("SELECT * FROM businesses LIMIT 1").fetchone()
    slot = conn.execute(
        "SELECT * FROM slots WHERE business_id = ? AND is_booked = 0 LIMIT 1", (business["id"],)
    ).fetchone()
    conn.close()

    first_id, first_created = repository.book_slot(
        business["id"], slot["id"], "Amy", "+911111111111", "checkup", call_id="CALL-RETRY-1"
    )
    second_id, second_created = repository.book_slot(
        business["id"], slot["id"], "Amy", "+911111111111", "checkup", call_id="CALL-RETRY-1"
    )

    assert first_created is True
    assert second_created is False
    assert first_id == second_id

    conn = get_connection(db_path)
    count = conn.execute("SELECT COUNT(*) c FROM bookings WHERE call_id = 'CALL-RETRY-1'").fetchone()["c"]
    conn.close()
    assert count == 1
