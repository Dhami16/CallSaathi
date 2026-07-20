"""Orchestrates a call end to end: telephony parsing -> AI conversation ->
booking -> notifications. This is where the three subsystems meet; app.py
stays a thin Flask/webhook layer and never sees these details.
"""
import time

import sentry_sdk
import structlog

from ai.conversation import FALLBACK_MESSAGE, ConversationManager
from booking.db import init_db
from booking.repository import BookingRepository, SlotUnavailableError
from booking.session_store import SQLiteSessionStore
from notifications.base import NotificationService
from notifications.mock_service import MockNotificationService
from observability import capture_fallback
from telephony.base import TelephonyProvider

logger = structlog.get_logger(__name__)

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
        # about (which business this call belongs to, and a turn counter for
        # call_turns rows). Keyed by call_id, same lifetime as a
        # ConversationManager session.
        self._calls: dict[str, dict] = {}

    def handle_incoming_call(self, raw_request_data: dict, gather_action_url: str) -> str:
        turn_start = time.monotonic()
        call = self.telephony.parse_incoming_call(raw_request_data)
        call_id = call["call_id"]

        with structlog.contextvars.bound_contextvars(call_id=call_id):
            sentry_sdk.set_tag("call_id", call_id)
            logger.info(
                "stage", stage="telephony_webhook_received", outcome="success", direction="incoming_call"
            )

            business = self.repository.get_business_by_phone(call["called_number"])
            if business is None:
                logger.error(
                    "stage",
                    stage="call_ended",
                    outcome="error",
                    reason="no_business_configured",
                    called_number=call["called_number"],
                )
                capture_fallback("no_business_configured_for_called_number", call_id=call_id)
                self._write_turn(call_id, turn_number=0, turn_start=turn_start, response_out=_NO_BUSINESS_MESSAGE)
                return self.telephony.build_reply_response(_NO_BUSINESS_MESSAGE, hangup=True)

            sentry_sdk.set_tag("business_id", business["id"])
            slots = self.repository.get_available_slots(business["id"])
            greeting = self.conversation.start_session(call_id, business, slots)
            self._calls[call_id] = {"business": business, "turn_number": 0}

            self._write_turn(call_id, turn_number=0, turn_start=turn_start, response_out=greeting, llm_latency_ms=0)

            return self.telephony.build_greeting_response(
                greeting, gather_action_url, business.get("language_pref", "english")
            )

    def handle_speech_input(self, raw_request_data: dict, continue_url_base: str) -> str:
        """First webhook hit of a turn. Kicks off progressive (sentence-by-
        sentence) delivery and returns as soon as the first thing to say is
        ready - see ai/conversation.py's start_streaming_reply. Subsequent
        sentences of THIS SAME turn are fetched via handle_continue, driven
        by Twilio's <Redirect>."""
        turn_start = time.monotonic()
        speech = self.telephony.parse_speech_result(raw_request_data)
        call_id = speech["call_id"]
        transcript = speech["transcript"]

        with structlog.contextvars.bound_contextvars(call_id=call_id):
            sentry_sdk.set_tag("call_id", call_id)
            call_meta = self._calls.get(call_id)
            turn_number = None
            if call_meta is not None:
                sentry_sdk.set_tag("business_id", call_meta["business"]["id"])
                call_meta["turn_number"] += 1
                turn_number = call_meta["turn_number"]

            logger.info(
                "stage",
                stage="telephony_webhook_received",
                outcome="success",
                direction="speech_result",
                confidence=speech["confidence"],
            )

            if not transcript:
                logger.warning("stage", stage="speech_captured", outcome="error", reason="empty_transcript")
                self._write_turn(
                    call_id, turn_number=turn_number, turn_start=turn_start, transcript_in="", response_out=_REPEAT_MESSAGE
                )
                language = call_meta["business"].get("language_pref", "english") if call_meta is not None else "english"
                return self.telephony.build_reply_response(_REPEAT_MESSAGE, hangup=False, language=language)

            logger.info(
                "stage", stage="speech_captured", outcome="success", transcript_length=len(transcript)
            )

            result = self.conversation.start_streaming_reply(call_id, transcript)

            return self._process_streaming_result(
                call_id,
                speech["caller_number"],
                turn_number,
                turn_start,
                transcript,
                result,
                next_sentence_index=1,
                continue_url_base=continue_url_base,
            )

    def handle_continue(self, raw_request_data: dict, continue_url_base: str, sentence_index: int) -> str:
        """Handles a <Redirect> hit fetching the next sentence of a turn
        already started by handle_speech_input. `sentence_index` is carried
        in the redirect URL's query string since each hit is a fresh,
        possibly different-process request with no memory of its own."""
        turn_start = time.monotonic()
        call = self.telephony.parse_incoming_call(raw_request_data)
        call_id = call["call_id"]

        with structlog.contextvars.bound_contextvars(call_id=call_id):
            sentry_sdk.set_tag("call_id", call_id)
            call_meta = self._calls.get(call_id)
            turn_number = call_meta["turn_number"] if call_meta is not None else None
            if call_meta is not None:
                sentry_sdk.set_tag("business_id", call_meta["business"]["id"])

            result = self.conversation.get_next_streamed_sentence(call_id, sentence_index)

            return self._process_streaming_result(
                call_id,
                call["caller_number"],
                turn_number,
                turn_start,
                transcript_in=None,
                result=result,
                next_sentence_index=sentence_index + 1,
                continue_url_base=continue_url_base,
            )

    def _process_streaming_result(
        self,
        call_id: str,
        caller_number: str,
        turn_number: int | None,
        turn_start: float,
        transcript_in: str | None,
        result: dict,
        next_sentence_index: int,
        continue_url_base: str,
    ) -> str:
        sentence = result["sentence"]
        is_fallback = sentence == FALLBACK_MESSAGE and result["hangup"] and result["booking"] is None

        logger.info(
            "stage",
            stage="ai_response_generated",
            outcome="fallback_triggered" if is_fallback else "success",
            hangup=result["hangup"],
            has_booking=bool(result["booking"]),
            more_coming=result["more_coming"],
        )

        if result["booking"]:
            logger.info("stage", stage="booking_attempted", slot_id=result["booking"]["slot_id"])
            booked = self._finalize_booking(call_id, caller_number, result["booking"])
            if not booked:
                sentence = _SLOT_TAKEN_MESSAGE
        elif is_fallback:
            call_meta = self._calls.get(call_id)
            if call_meta is not None:
                # The AI layer gave up (Groq down/glitched/timed out
                # mid-stream) and the call is ending without a booking.
                # Persist this so /internal/stats can count it, not just Sentry.
                self._log_non_booking_outcome(call_id, call_meta["business"], caller_number, "fallback")

        if result["hangup"]:
            logger.info("stage", stage="call_ended", outcome="fallback_triggered" if is_fallback else "success")
            self.conversation.end_session(call_id)
            self._calls.pop(call_id, None)

        self._write_turn(
            call_id,
            turn_number=turn_number,
            turn_start=turn_start,
            llm_latency_ms=int((time.monotonic() - turn_start) * 1000),
            transcript_in=transcript_in,
            response_out=sentence,
        )

        if result["more_coming"]:
            continue_url = f"{continue_url_base}?idx={next_sentence_index}"
            return self.telephony.build_continue_response(sentence, continue_url)

        call_meta = self._calls.get(call_id)
        language = call_meta["business"].get("language_pref", "english") if call_meta is not None else "english"
        return self.telephony.build_reply_response(sentence or "", hangup=result["hangup"], language=language)

    def _finalize_booking(self, call_id: str, caller_number: str, booking: dict) -> bool:
        call_meta = self._calls.get(call_id)
        if call_meta is None:
            logger.error("stage", stage="booking_result", outcome="error", reason="unknown_call_id")
            return False
        business = call_meta["business"]

        try:
            booking_id, created = self.repository.book_slot(
                business_id=business["id"],
                slot_id=booking["slot_id"],
                customer_name=booking["customer_name"],
                customer_phone=caller_number,
                reason=booking["reason"],
                call_id=call_id,
            )
        except SlotUnavailableError:
            logger.warning(
                "stage", stage="booking_result", outcome="error", reason="slot_unavailable", slot_id=booking["slot_id"]
            )
            capture_fallback("slot_already_booked", call_id=call_id, business_id=business["id"])
            self._log_non_booking_outcome(call_id, business, caller_number, "slot_unavailable")
            return False

        if not created:
            # Idempotent replay: this call_id already has a booking (e.g. a
            # Twilio webhook retry after a network blip). The first
            # successful pass already wrote the call_log and sent
            # notifications - doing it again would double-notify the owner
            # for one real booking, so just report success and stop here.
            logger.info(
                "stage", stage="booking_result", outcome="success", reason="idempotent_replay", booking_id=booking_id
            )
            return True

        logger.info("stage", stage="booking_result", outcome="success", slot_id=booking["slot_id"])

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
        logger.info("stage", stage="notification_sent", outcome="success", channel="owner")
        self.notifications.notify_customer(business, booking_record)
        logger.info("stage", stage="notification_sent", outcome="success", channel="customer")
        return True

    def _log_non_booking_outcome(self, call_id: str, business: dict, caller_number: str, outcome: str) -> None:
        """Persists a call_logs row for a call that ended WITHOUT a booking
        (fallback or a lost slot-availability race), so /internal/stats can
        compute error/fallback counts from our own DB rather than only from
        Sentry. call_logs originally only ever recorded successful
        bookings; this is a deliberate, minimal broadening of that, not a
        change to the booking transaction itself."""
        transcript = self.conversation.get_transcript(call_id)
        duration = self.conversation.get_duration_seconds(call_id)
        self.repository.create_call_log(
            business_id=business["id"],
            caller_phone=caller_number,
            transcript=transcript,
            outcome=outcome,
            duration_seconds=duration,
        )

    def _write_turn(
        self,
        call_id: str,
        turn_start: float,
        turn_number: int | None = None,
        llm_latency_ms: int | None = None,
        transcript_in: str | None = None,
        response_out: str | None = None,
    ) -> None:
        total_latency_ms = int((time.monotonic() - turn_start) * 1000)
        self.repository.create_call_turn(
            call_id=call_id,
            turn_number=turn_number if turn_number is not None else -1,
            total_latency_ms=total_latency_ms,
            llm_latency_ms=llm_latency_ms,
            # stt_latency_ms / tts_latency_ms intentionally omitted (None):
            # see booking/db.py's call_turns comment for why these can't be
            # measured given Twilio's <Gather>-based architecture.
            transcript_in=transcript_in,
            response_out=response_out,
        )


def build_default_call_handler(config, telephony_provider: TelephonyProvider) -> CallHandler:
    """Wires up the concrete production dependencies (Groq, SQLite, mock
    notifications) into a CallHandler. Tests build a CallHandler directly
    with their own fakes/temp DB instead of using this factory."""
    if not config.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to run the app (see .env.example)")

    init_db(config.database_path)

    return CallHandler(
        telephony_provider=telephony_provider,
        conversation_manager=ConversationManager(
            config.groq_api_key,
            config.groq_model,
            session_store=SQLiteSessionStore(config.database_path),
            session_ttl_seconds=config.session_ttl_seconds,
        ),
        booking_repository=BookingRepository(config.database_path),
        notification_service=MockNotificationService(),
    )
