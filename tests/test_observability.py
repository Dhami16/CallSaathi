"""Tests for the observability layer added on top of the working MVP:
structured logging, call_turns latency rows, the /internal/stats endpoint,
and Sentry init. None of these need Groq or Twilio.

Run with: venv/Scripts/python -m pytest -s tests/test_observability.py
"""
import os

import pytest
import structlog

import seed_data
from booking.db import get_connection
from booking.repository import BookingRepository
from observability import capture_fallback, init_sentry


# --- Task 1: structured logging -------------------------------------------


def test_structured_logging_does_not_crash_with_missing_fields(capsys):
    """A logger call with no extra kwargs, and one with unusual/missing
    fields, should both just work - structlog must never raise even if a
    caller forgets to pass e.g. call_id."""
    logger = structlog.get_logger("test")

    logger.info("stage", stage="telephony_webhook_received")  # no outcome, no call_id
    logger.warning("stage", stage="booking_result", outcome=None, slot_id=None)  # None values
    logger.error("stage")  # no fields at all beyond the event name

    out = capsys.readouterr().out
    assert "telephony_webhook_received" in out
    assert "booking_result" in out


def test_bound_call_id_appears_on_every_line_within_the_context(capsys):
    logger = structlog.get_logger("test")
    with structlog.contextvars.bound_contextvars(call_id="CALL-XYZ"):
        logger.info("stage", stage="telephony_webhook_received", outcome="success")
        logger.info("stage", stage="speech_captured", outcome="success")

    logger.info("stage", stage="call_ended", outcome="success")  # outside the block

    lines = capsys.readouterr().out.splitlines()
    assert "call_id=CALL-XYZ" in lines[0]
    assert "call_id=CALL-XYZ" in lines[1]
    assert "call_id=CALL-XYZ" not in lines[2]  # contextvars binding doesn't leak past the `with` block


# --- Task 3: Sentry ----------------------------------------------------------


def test_sentry_init_does_not_crash_when_dsn_unset(capsys):
    init_sentry("")  # must not raise
    out = capsys.readouterr().out
    assert "sentry_not_configured" in out


def test_capture_fallback_is_a_safe_noop_without_sentry_configured():
    # No sentry_sdk.init() has been called for this DSN in this test process
    # (or if some other test initialized it, this call must still not raise).
    capture_fallback("test_fallback_event", call_id="CALL-1", business_id=1)


# --- Task 2 + 4: call_turns and /internal/stats -----------------------------


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test_stats.db")
    seed_data.seed(path)
    return path


def _seed_call_turn(repository, call_id, turn_number, total_latency_ms, llm_latency_ms=None):
    repository.create_call_turn(
        call_id=call_id,
        turn_number=turn_number,
        total_latency_ms=total_latency_ms,
        llm_latency_ms=llm_latency_ms,
        transcript_in="hello" if turn_number > 0 else None,
        response_out="hi there",
    )


def test_compute_stats_aggregates_calls_bookings_and_latency(db_path):
    from stats import compute_stats

    repository = BookingRepository(db_path)
    conn = get_connection(db_path)
    business = conn.execute("SELECT * FROM businesses LIMIT 1").fetchone()
    slots = conn.execute(
        "SELECT * FROM slots WHERE business_id = ? AND is_booked = 0 ORDER BY id", (business["id"],)
    ).fetchall()
    conn.close()

    # Call 1: booked successfully. Two turns: greeting + booking turn.
    _seed_call_turn(repository, "CALL-1", 0, total_latency_ms=100)
    _seed_call_turn(repository, "CALL-1", 1, total_latency_ms=900, llm_latency_ms=800)
    repository.book_slot(business["id"], slots[0]["id"], "Amy", "+911111111111", "checkup", call_id="CALL-1")
    repository.create_call_log(business["id"], "+911111111111", "transcript one", "booked", duration_seconds=12)

    # Call 2: ended in a fallback, no booking. One turn.
    _seed_call_turn(repository, "CALL-2", 0, total_latency_ms=300)
    repository.create_call_log(business["id"], "+922222222222", "transcript two", "fallback", duration_seconds=5)

    stats = compute_stats(db_path)

    assert stats["today"]["calls"] == 2
    assert stats["today"]["bookings"] == 1
    assert stats["today"]["fallback_or_error"] == 1
    assert stats["today"]["success_rate"] == pytest.approx(0.5)

    assert stats["this_week"]["calls"] == 2
    assert stats["this_week"]["bookings"] == 1

    latency = stats["latency_ms_this_week"]
    assert latency["avg_total_latency_ms"] == pytest.approx((100 + 900 + 300) / 3, abs=0.1)
    assert latency["avg_llm_latency_ms"] == pytest.approx(800)
    # Never measured given the current Twilio <Gather> architecture.
    assert latency["avg_stt_latency_ms"] is None
    assert latency["avg_tts_latency_ms"] is None


def test_stats_endpoint_requires_correct_token(db_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", db_path)
    monkeypatch.setenv("INTERNAL_STATS_TOKEN", "secret-token-123")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-for-import-only")
    monkeypatch.setenv("VALIDATE_TWILIO_SIGNATURE", "false")

    # app.py reads config at import time, so import it fresh under these env vars.
    import importlib

    import app as app_module

    importlib.reload(app_module)
    client = app_module.app.test_client()

    r_no_token = client.get("/internal/stats")
    assert r_no_token.status_code == 404

    r_wrong_token = client.get("/internal/stats?token=wrong")
    assert r_wrong_token.status_code == 404

    r_ok = client.get("/internal/stats?token=secret-token-123")
    assert r_ok.status_code == 200
    body = r_ok.get_json()
    assert "today" in body and "this_week" in body and "latency_ms_this_week" in body

    r_header = client.get("/internal/stats", headers={"X-Internal-Token": "secret-token-123"})
    assert r_header.status_code == 200
