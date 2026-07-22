"""Real, billed SMS delivery via Twilio's Programmable Messaging API.

WhatsApp is NOT implemented here on purpose: Twilio's WhatsApp Sandbox
requires each recipient to manually opt in with a join code first (unworkable
for arbitrary real customers), and real production WhatsApp needs Meta
Business Manager verification that hasn't happened yet. SMS has no such
opt-in requirement, so it's the right fit for pilot stage. When WhatsApp
verification is done, it can be added as another NotificationService
implementation behind this same interface - booking logic won't need to
change.
"""
import requests
import structlog
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from notifications.base import NotificationService
from observability import capture_fallback

logger = structlog.get_logger(__name__)

SMS_RETRY_ATTEMPTS = 2  # 1 original attempt + 1 retry - brief and bounded, mirrors ai/conversation.py's pattern


def _is_transient_sms_error(exc: BaseException) -> bool:
    """Only retry failures a second identical attempt could plausibly fix:
    a 5xx from Twilio's API, or a network-level connection/timeout error
    from the underlying HTTP client. A 4xx (invalid number, unverified
    trial-account recipient, etc.) is not retried - sending the exact same
    request again can't make a bad number valid."""
    if isinstance(exc, TwilioRestException):
        return exc.status is None or exc.status >= 500
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


class TwilioSMSNotificationService(NotificationService):
    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        owner_phone: str = "",
        allowlist: frozenset[str] = frozenset(),
        client=None,
    ):
        self._client = client if client is not None else Client(account_sid, auth_token)
        self._from_number = from_number
        self._owner_phone = owner_phone
        # Empty allowlist means "send to anyone" - see .env.example. Only a
        # guardrail for initial live testing, not a pilot-stage restriction.
        self._allowlist = allowlist

    def notify_owner(self, business: dict, booking: dict, transcript: str) -> None:
        if not self._owner_phone:
            logger.warning("notification_skipped", channel="owner", reason="no_owner_phone_configured")
            return
        message = (
            f"New booking: {booking['customer_name']} ({booking['customer_phone']})\n"
            f"When: {booking['slot_date']} at {booking['slot_time']}\n"
            f"Reason: {booking['reason']}\n"
            f"--- Full call transcript ---\n{transcript}\n"
            f"--- end transcript ---"
        )
        self._send(self._owner_phone, message, channel="owner")

    def notify_customer(self, business: dict, booking: dict) -> None:
        message = (
            f"Your appointment with {business['name']} is confirmed for "
            f"{booking['slot_date']} at {booking['slot_time']}. See you then!"
        )
        self._send(booking["customer_phone"], message, channel="customer")

    def _send(self, to_number: str, body: str, channel: str) -> None:
        if self._allowlist and to_number not in self._allowlist:
            logger.warning(
                "notification_skipped", channel=channel, to=to_number, reason="not_in_allowlist"
            )
            return

        try:
            self._send_with_retry(to_number, body)
        except Exception as exc:
            # A failed SMS must never fail the call/booking itself - the
            # booking already happened and is real regardless of whether the
            # notification about it was delivered.
            logger.error("notification_send_failed", channel=channel, to=to_number, error=str(exc))
            capture_fallback("notification_send_failed", channel=channel)
            return

        logger.info("notification_sent", channel=channel, to=to_number)

    @retry(
        retry=retry_if_exception(_is_transient_sms_error),
        stop=stop_after_attempt(SMS_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=0.5, max=2),
        reraise=True,
    )
    def _send_with_retry(self, to_number: str, body: str) -> None:
        self._client.messages.create(to=to_number, from_=self._from_number, body=body)
