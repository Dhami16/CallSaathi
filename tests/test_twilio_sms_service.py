"""Notification session verification: TwilioSMSNotificationService sends
real SMS via a Twilio client, with a bounded retry on transient failures,
never raises out to the caller (a failed notification must never fail the
booking that already happened), and respects an optional allowlist guardrail
for safe live testing.

Uses a fake Twilio client (no real network/API calls) - see FakeMessages/
FakeClient below.

Run with: venv/Scripts/python -m pytest -q tests/test_twilio_sms_service.py
"""
from unittest.mock import patch

import pytest
from twilio.base.exceptions import TwilioRestException

from notifications.mock_service import MockNotificationService
from notifications.twilio_sms_service import TwilioSMSNotificationService

DEMO_BUSINESS = {"id": 1, "name": "Radiant Skin Clinic", "vertical": "clinic"}
DEMO_BOOKING = {
    "slot_id": 1,
    "customer_name": "Rajdeep",
    "customer_phone": "+911111111111",
    "reason": "skin problem",
    "slot_date": "2026-07-23",
    "slot_time": "15:00",
}
DEMO_TRANSCRIPT = "caller: hi\nassistant: hello"


class FakeMessages:
    def __init__(self, error_sequence=None):
        # error_sequence: list of exceptions to raise on successive calls,
        # None entries mean "succeed this call". Exhausting the list means
        # every subsequent call succeeds.
        self._error_sequence = list(error_sequence or [])
        self.calls = []

    def create(self, to, from_, body):
        self.calls.append({"to": to, "from_": from_, "body": body})
        if self._error_sequence:
            error = self._error_sequence.pop(0)
            if error is not None:
                raise error


class FakeClient:
    def __init__(self, error_sequence=None):
        self.messages = FakeMessages(error_sequence)


def _transient_error():
    return TwilioRestException(status=500, uri="/Messages", msg="server error")


def _permanent_error():
    return TwilioRestException(status=400, uri="/Messages", msg="invalid number", code=21211)


def _service(client, owner_phone="+919999999999", allowlist=frozenset()):
    return TwilioSMSNotificationService(
        account_sid="AC-fake",
        auth_token="fake-token",
        from_number="+15005550006",
        owner_phone=owner_phone,
        allowlist=allowlist,
        client=client,
    )


# --- successful sends -------------------------------------------------------


def test_notify_owner_sends_full_transcript_and_booking_details():
    client = FakeClient()
    service = _service(client)

    service.notify_owner(DEMO_BUSINESS, DEMO_BOOKING, DEMO_TRANSCRIPT)

    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    assert call["to"] == "+919999999999"
    assert call["from_"] == "+15005550006"
    assert "Rajdeep" in call["body"]
    assert "skin problem" in call["body"]
    assert DEMO_TRANSCRIPT in call["body"]


def test_notify_customer_sends_short_confirmation():
    client = FakeClient()
    service = _service(client)

    service.notify_customer(DEMO_BUSINESS, DEMO_BOOKING)

    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    assert call["to"] == "+911111111111"
    assert "Radiant Skin Clinic" in call["body"]
    assert "2026-07-23" in call["body"]


def test_notify_owner_skipped_with_no_owner_phone_configured():
    client = FakeClient()
    service = _service(client, owner_phone="")

    service.notify_owner(DEMO_BUSINESS, DEMO_BOOKING, DEMO_TRANSCRIPT)

    assert client.messages.calls == []


# --- retry + failure handling ------------------------------------------------


def test_transient_failure_retries_then_succeeds():
    client = FakeClient(error_sequence=[_transient_error()])
    service = _service(client)

    service.notify_customer(DEMO_BUSINESS, DEMO_BOOKING)  # must not raise

    assert len(client.messages.calls) == 2  # 1 failed attempt + 1 successful retry


@patch("notifications.twilio_sms_service.capture_fallback")
def test_transient_failure_exhausts_retries_then_logs_and_reports_without_raising(mock_capture_fallback):
    client = FakeClient(error_sequence=[_transient_error(), _transient_error()])
    service = _service(client)

    service.notify_customer(DEMO_BUSINESS, DEMO_BOOKING)  # must not raise - booking already succeeded

    assert len(client.messages.calls) == 2  # SMS_RETRY_ATTEMPTS = 2, both exhausted
    mock_capture_fallback.assert_called_once()
    assert mock_capture_fallback.call_args.args[0] == "notification_send_failed"


@patch("notifications.twilio_sms_service.capture_fallback")
def test_permanent_failure_is_not_retried(mock_capture_fallback):
    client = FakeClient(error_sequence=[_permanent_error()])
    service = _service(client)

    service.notify_owner(DEMO_BUSINESS, DEMO_BOOKING, DEMO_TRANSCRIPT)  # must not raise

    assert len(client.messages.calls) == 1  # no retry attempted for a 4xx
    mock_capture_fallback.assert_called_once()


# --- allowlist guardrail ------------------------------------------------------


def test_allowlist_skips_recipients_not_on_it():
    client = FakeClient()
    service = _service(client, allowlist=frozenset({"+911111111111"}))  # only the customer number

    service.notify_owner(DEMO_BUSINESS, DEMO_BOOKING, DEMO_TRANSCRIPT)  # owner phone not on allowlist

    assert client.messages.calls == []


def test_allowlist_allows_recipients_on_it():
    client = FakeClient()
    service = _service(client, allowlist=frozenset({"+911111111111"}))

    service.notify_customer(DEMO_BUSINESS, DEMO_BOOKING)  # customer phone IS on allowlist

    assert len(client.messages.calls) == 1


def test_empty_allowlist_sends_to_anyone():
    client = FakeClient()
    service = _service(client, allowlist=frozenset())

    service.notify_owner(DEMO_BUSINESS, DEMO_BOOKING, DEMO_TRANSCRIPT)
    service.notify_customer(DEMO_BUSINESS, DEMO_BOOKING)

    assert len(client.messages.calls) == 2


# --- mock service is untouched and still the safe default -------------------


def test_mock_notification_service_still_logs_instead_of_sending(capsys):
    service = MockNotificationService()

    service.notify_owner(DEMO_BUSINESS, DEMO_BOOKING, DEMO_TRANSCRIPT)
    service.notify_customer(DEMO_BUSINESS, DEMO_BOOKING)

    out = capsys.readouterr().out
    assert "notification_sent" in out
    assert "Rajdeep" in out
