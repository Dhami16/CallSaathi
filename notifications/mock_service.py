"""Logs what would be sent to the owner and customer instead of actually
sending it. Swap for a real WhatsApp/SMS implementation of
NotificationService later without touching booking logic."""
import structlog

from notifications.base import NotificationService

logger = structlog.get_logger(__name__)


class MockNotificationService(NotificationService):
    def notify_owner(self, business: dict, booking: dict, transcript: str) -> None:
        message = (
            f"[OWNER NOTIFICATION -> {business['name']}]\n"
            f"New booking: {booking['customer_name']} ({booking['customer_phone']})\n"
            f"When: {booking['slot_date']} at {booking['slot_time']}\n"
            f"Reason: {booking['reason']}\n"
            f"--- Full call transcript ---\n{transcript}\n"
            f"--- end transcript ---"
        )
        logger.info("notification_sent", channel="owner", message=message)

    def notify_customer(self, business: dict, booking: dict) -> None:
        message = (
            f"[CUSTOMER NOTIFICATION -> {booking['customer_phone']}]\n"
            f"Your appointment with {business['name']} is confirmed for "
            f"{booking['slot_date']} at {booking['slot_time']}. See you then!"
        )
        logger.info("notification_sent", channel="customer", message=message)
