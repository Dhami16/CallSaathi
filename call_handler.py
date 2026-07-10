"""Orchestrates a call end to end: telephony parsing -> AI conversation ->
booking -> notifications. This is where the three subsystems meet; app.py
stays a thin Flask/webhook layer and never sees these details.
"""
import logging

from ai.conversation import ConversationManager
from booking.db import init_db
from booking.repository import BookingRepository, SlotUnavailableError
from notifications.base import NotificationService
from notifications.mock_service import MockNotificationService
from telephony.base import TelephonyProvider

logger = logging.getLogger(__name__)

_NO_BUSINESS_MESSAGE = "Sorry, this number isn't set up yet. Please try again later."
_REPEAT_MESSAGE = "Sorry, could you please repeat that?"
_SLOT_TAKEN_MESSAGE = "Sorry, that slot was just taken. Let me have someone call you back shortly."


class CallHandler:
    def __init__(
        self,
        telephony_provider: TelephonyProvider,
        conversation_manager: ConversationManager,
        booking_repository: BookingRepository,
        notification_service: NotificationService,
    ):
        self.telephony = telephony_provider
        self.conversation = conversation_manager
        self.repository = booking_repository
        self.notifications = notification_service
        # Per-call bookkeeping the conversation manager doesn't need to know
        # about (which business this call belongs to). Keyed by call_id,
        # same lifetime as a ConversationManager session.
        self._calls: dict[str, dict] = {}

    def handle_incoming_call(self, raw_request_data: dict, gather_action_url: str) -> str:
        call = self.telephony.parse_incoming_call(raw_request_data)
        business = self.repository.get_business_by_phone(call["called_number"])

        if business is None:
            logger.error("No active business configured for called_number=%s", call["called_number"])
            return self.telephony.build_reply_response(_NO_BUSINESS_MESSAGE, hangup=True)

        slots = self.repository.get_available_slots(business["id"])
        greeting = self.conversation.start_session(call["call_id"], business, slots)
        self._calls[call["call_id"]] = {"business": business}
        return self.telephony.build_greeting_response(
            greeting, gather_action_url, business.get("language_pref", "english")
        )

    def handle_speech_input(self, raw_request_data: dict) -> str:
        speech = self.telephony.parse_speech_result(raw_request_data)
        call_id = speech["call_id"]
        transcript = speech["transcript"]

        if not transcript:
            return self.telephony.build_reply_response(_REPEAT_MESSAGE, hangup=False)

        result = self.conversation.get_reply(call_id, transcript)
        reply_text = result["reply_text"]

        if result["booking"]:
            booked = self._finalize_booking(call_id, speech["caller_number"], result["booking"])
            if not booked:
                reply_text = _SLOT_TAKEN_MESSAGE

        if result["hangup"]:
            self.conversation.end_session(call_id)
            self._calls.pop(call_id, None)

        return self.telephony.build_reply_response(reply_text, hangup=result["hangup"])

    def _finalize_booking(self, call_id: str, caller_number: str, booking: dict) -> bool:
        call_meta = self._calls.get(call_id)
        if call_meta is None:
            logger.error("Booking confirmed for unknown call_id=%s", call_id)
            return False
        business = call_meta["business"]

        try:
            self.repository.book_slot(
                business_id=business["id"],
                slot_id=booking["slot_id"],
                customer_name=booking["customer_name"],
                customer_phone=caller_number,
                reason=booking["reason"],
            )
        except SlotUnavailableError:
            logger.warning(
                "call_id=%s: slot_id=%s for business_id=%s was already booked by another caller",
                call_id,
                booking["slot_id"],
                business["id"],
            )
            return False

        transcript = self.conversation.get_transcript(call_id)
        duration = self.conversation.get_duration_seconds(call_id)
        self.repository.create_call_log(
            business_id=business["id"],
            caller_phone=caller_number,
            transcript=transcript,
            outcome="booked",
            duration_seconds=duration,
        )

        booking_record = {**booking, "customer_phone": caller_number}
        self.notifications.notify_owner(business, booking_record, transcript)
        self.notifications.notify_customer(business, booking_record)
        return True


def build_default_call_handler(config, telephony_provider: TelephonyProvider) -> CallHandler:
    """Wires up the concrete production dependencies (Groq, SQLite, mock
    notifications) into a CallHandler. Tests build a CallHandler directly
    with their own fakes/temp DB instead of using this factory."""
    if not config.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to run the app (see .env.example)")

    init_db(config.database_path)

    return CallHandler(
        telephony_provider=telephony_provider,
        conversation_manager=ConversationManager(config.groq_api_key, config.groq_model),
        booking_repository=BookingRepository(config.database_path),
        notification_service=MockNotificationService(),
    )
