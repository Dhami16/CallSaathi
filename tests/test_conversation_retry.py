"""Task 4 verification: retry-with-backoff for transient Groq failures
(timeouts, connection errors, 5xx), using a mocked Groq client - no
network, no real API key needed, deterministic.

Run with: venv/Scripts/python -m pytest -q tests/test_conversation_retry.py
"""
from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import APIConnectionError, APITimeoutError, InternalServerError

from ai.conversation import FALLBACK_MESSAGE, TRANSIENT_RETRY_ATTEMPTS, ConversationManager
from booking.db import init_db
from booking.session_store import SQLiteSessionStore

DEMO_BUSINESS = {"id": 1, "name": "Test Clinic", "vertical": "clinic", "language_pref": "english"}
DEMO_SLOTS = [{"id": 1, "date": "2026-07-14", "time": "10:00"}]

_FAKE_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _timeout_error():
    return APITimeoutError(request=_FAKE_REQUEST)


def _connection_error():
    return APIConnectionError(request=_FAKE_REQUEST)


def _internal_server_error():
    response = httpx.Response(500, request=_FAKE_REQUEST, json={"error": {"message": "boom"}})
    return InternalServerError("boom", response=response, body=None)


def _make_completion(content: str):
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=content))]
    return completion


@pytest.fixture
def manager(tmp_path):
    db_path = str(tmp_path / "test_retry_sessions.db")
    init_db(db_path)
    return ConversationManager("fake-key-not-used", session_store=SQLiteSessionStore(db_path))


@pytest.mark.parametrize("first_error_factory", [_timeout_error, _connection_error, _internal_server_error])
def test_transient_error_retries_then_succeeds(manager, first_error_factory):
    call_id = f"CALL-RETRY-SUCCESS-{first_error_factory.__name__}"
    manager.start_session(call_id, DEMO_BUSINESS, DEMO_SLOTS)

    mock_create = MagicMock(
        side_effect=[first_error_factory(), _make_completion("Sure, what's the reason for your visit?")]
    )
    with patch.object(manager._client.chat.completions, "create", mock_create):
        result = manager.get_reply(call_id, "Hi, I need an appointment")

    assert mock_create.call_count == 2  # failed once, succeeded on retry - well within the bound
    assert result["reply_text"] == "Sure, what's the reason for your visit?"
    assert not result["hangup"]


def test_transient_error_exhausts_bounded_retries_then_falls_back(manager):
    """Confirms retries are bounded (not unbounded - a live call can't hang
    forever), and that once exhausted, the existing graceful fallback still
    triggers and still reports a non-fatal Sentry event."""
    call_id = "CALL-RETRY-EXHAUSTED"
    manager.start_session(call_id, DEMO_BUSINESS, DEMO_SLOTS)

    mock_create = MagicMock(side_effect=_timeout_error())  # raises the same error every call

    with patch.object(manager._client.chat.completions, "create", mock_create):
        with patch("ai.conversation.capture_fallback") as mock_capture_fallback:
            result = manager.get_reply(call_id, "Hi, I need an appointment")

    assert mock_create.call_count == TRANSIENT_RETRY_ATTEMPTS  # bounded: 1 original + 2 retries, then gives up
    assert result["reply_text"] == FALLBACK_MESSAGE
    assert result["hangup"] is True
    assert result["booking"] is None

    mock_capture_fallback.assert_called_once()
    assert mock_capture_fallback.call_args.args[0] == "groq_api_call_failed"


def test_harmony_glitch_retry_is_unaffected_by_transient_retry_logic(manager):
    """The pre-existing harmony tool-call-glitch retry (BadRequestError,
    immediate retry, no backoff) is a different failure mode from transient
    network/5xx errors and must keep working exactly as before."""
    from groq import BadRequestError

    call_id = "CALL-HARMONY-GLITCH"
    manager.start_session(call_id, DEMO_BUSINESS, DEMO_SLOTS)

    glitch_response = httpx.Response(
        400,
        request=_FAKE_REQUEST,
        json={"error": {"message": "Tool choice is none, but model called a tool", "code": "tool_use_failed"}},
    )
    glitch_error = BadRequestError(
        "Tool choice is none, but model called a tool",
        response=glitch_response,
        body={"error": {"code": "tool_use_failed"}},
    )

    mock_create = MagicMock(side_effect=[glitch_error, _make_completion("Sure, what's the reason for your visit?")])
    with patch.object(manager._client.chat.completions, "create", mock_create):
        result = manager.get_reply(call_id, "Hi, I need an appointment")

    assert mock_create.call_count == 2
    assert result["reply_text"] == "Sure, what's the reason for your visit?"
    assert not result["hangup"]
