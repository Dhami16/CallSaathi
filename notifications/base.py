"""Notification interface. A real WhatsApp/SMS-sending implementation can
be swapped in later by implementing this same interface - booking logic
never depends on how a notification is actually delivered."""
from abc import ABC, abstractmethod


class NotificationService(ABC):
    @abstractmethod
    def notify_owner(self, business: dict, booking: dict, transcript: str) -> None:
        """Send the business owner the full call transcript plus booking
        details. Never a summary - this is the core trust mechanic."""

    @abstractmethod
    def notify_customer(self, business: dict, booking: dict) -> None:
        """Send the customer a short booking confirmation."""
