"""Conversational booking agent backed by Groq's openai/gpt-oss-20b model.

Conversation history lives in an externalized SessionStore (see
booking/session_store.py), not an in-memory dict - a plain dict silently
breaks the moment there's more than one gunicorn worker process, since
Twilio's sequential webhook hits for one call can land on different
workers.

Booking is driven by an explicit `BookingStage` state machine, not by
scanning an LLM's own reply text for a marker. Each turn: a deterministic,
non-LLM parser (ai/slot_matching.py) tries to resolve the caller's speech
to one of the offered slots first; a separate structured-understanding
Groq call (ai/intent_interpreter.py) classifies intent and extracts
whatever the deterministic parser couldn't (name, reason, out-of-scope
replies); then a stage-keyed handler here decides the next state and the
spoken reply from code/templates. A booking is only ever created once the
CONFIRM stage's affirmative branch fires - there is no text marker to
parse and nothing for the model to accidentally "just decide" to emit.

This replaces an earlier design where one free-form Groq call per turn both
decided everything and spoke the reply, with no code-level tracking of what
was already known about the caller's request. That design could (and did,
in production) loop the exact same clarifying question forever whenever it
failed to resolve a caller's spoken date into an actual slot - see
ai/slot_matching.py's module docstring for the bug this fixes.
"""
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime

from groq import Groq

from ai.intent_interpreter import BookingIntent, BookingIntentInterpreter
from ai.slot_matching import match_requested_slot, parse_datetime_request, slots_on_date

from booking.session_store import DEFAULT_TTL_SECONDS, SessionStore
from observability import capture_fallback

logger = structlog.get_logger(__name__)

MAX_TURNS = 12  # safety valve so a stuck/confused caller can't loop forever
REQUEST_TIMEOUT_SECONDS = 8.0

FALLBACK_MESSAGE = "I'm having trouble right now, let me have someone call you back shortly."

_PLACEHOLDER_TOKENS = {"unknown", "n/a", "na", "customer", "caller", "name", ""}

# Cheap, deterministic guards that corroborate (or short-circuit) the LLM's
# intent classification for the highest-stakes decisions - ending the call
# and confirming a booking - so a borderline-confidence classification can't
# strand an otherwise-clear caller answer.
_END_CALL_RE = re.compile(
    r"\b(bye|goodbye|that'?s all|nothing else|no thanks|no,? thank you|hang up|alvida|"
    r"dhonnobad|bas itna hi)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_RE = re.compile(
    r"\b(yes|yeah|yep|sure|ok|okay|correct|confirm|right|haan|ha|theek\s*hai|thik\s*(?:ache|achhe))\b",
    re.IGNORECASE,
)


def _looks_like_end_call(text: str) -> bool:
    return bool(_END_CALL_RE.search(text))


def _looks_like_affirmative(text: str) -> bool:
    return bool(_AFFIRMATIVE_RE.search(text)) and len(text.strip().split()) <= 6


def _looks_substantive(text: str) -> bool:
    return len(text.strip().split()) >= 2


_NAME_REJECT_RE = re.compile(r"[?0-9]")


def _looks_like_name(text: str) -> bool:
    """Heuristic guard for the raw-transcript fallback in the NAME stage: a
    real name is short and isn't a question or a date/number-laden
    sentence. Without this, any substantive but unrelated turn (e.g. the
    caller re-describing their symptoms) would get booked in as the
    customer's name whenever the LLM didn't extract one."""
    text = text.strip()
    words = text.split()
    return 1 <= len(words) <= 4 and not _NAME_REJECT_RE.search(text)


def _speak_date(iso_date: str) -> str:
    return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d %B")


def _speak_time(hh_mm: str) -> str:
    spoken = datetime.strptime(hh_mm, "%H:%M").strftime("%I:%M %p")
    return spoken.lstrip("0") or spoken


def _speak_slot(slot: dict) -> str:
    return f"{_speak_date(slot['date'])} at {_speak_time(slot['time'])}"


def _join_or(items: list[str], language: str) -> str:
    joiner = {"hindi": "ya", "bengali": "ba"}.get(language, "or")
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])}, {joiner} {items[-1]}"


def _offer_slots_by_number(offered_slots: list[dict], language: str, nudge: bool = False) -> str:
    numbered = ", ".join(f"{i}) {_speak_slot(s)}" for i, s in enumerate(offered_slots, 1))
    templates = {
        "hindi": f"Chaliye aasan karte hain - {numbered}. Kaunsa number theek rahega?",
        "bengali": f"Cholun shohoj kori - {numbered}. Kon number ta bhalo hobe?",
        "english": f"Let's make this easy - {numbered}. Which number works for you?",
    }
    reply = templates.get(language, templates["english"])
    if nudge:
        # A second consecutive miss on this exact prompt - vary the wording
        # rather than repeat the identical line again.
        suffixes = {"hindi": " Bas 1, 2 ya 3 boliye.", "bengali": " Shudhu 1, 2 ba 3 bolun.", "english": " Just say 1, 2, or 3."}
        reply += suffixes.get(language, suffixes["english"])
    return reply


# Not anchored to line start/end; romanized (Hinglish/Benglish) rather than
# native script for the same reason as the greeting - the Twilio voice used
# (Polly.Aditi) is a Hindi/English voice and doesn't render Bengali script
# reliably. A future iteration could pick a per-language voice.
_GREETINGS = {
    "hindi": "Namaste! Aapne {business} se call kiya hai. Main aapki appointment book karne mein madad karunga. Bataiye, kis wajah se call kar rahe hain?",
    "bengali": "Nomoshkar! Apni {business} e phone korechen. Ami apnar appointment book korte shahajyo korbo. Bolun, ki karone phone korchen?",
    "english": "Hello! You've reached {business}. I can help you book an appointment. Could you tell me what this is regarding?",
}

_REASON_NUDGE = {
    "hindi": "Ji, bataiye - main sun raha hoon.",
    "bengali": "Ji, bolun - ami shunchi.",
    "english": "Go ahead, I'm listening.",
}
_ASK_REASON_RETRY = {
    "hindi": "Thodi si detail mein bataiye - kis wajah se appointment chahiye?",
    "bengali": "Ektu bistarito bolun - ki karone appointment ta lagbe?",
    "english": "In a few words - what's the appointment for?",
}

_ASK_TIMING = {
    "hindi": "Accha, samajh gaya. Aapko kis din aur kis time appointment chahiye?",
    "bengali": "Bhalo, bujhte perechi. Apni kon diner o kon shomoyer jonyo appointment chan?",
    "english": "Got it. What day and time would work for you?",
}
_ASK_TIMING_RETRY = {
    "hindi": "Maaf kijiye, date samajh nahi paaya. 'Kal' ya 'Friday' jaisa kuch bata sakte hain?",
    "bengali": "Dukkhito, tarikh ta bujhte parini. 'Agamikal' ba 'Friday'-r moto kichu bolte paren?",
    "english": "Sorry, I didn't catch a date there - could you say something like 'tomorrow' or 'Friday'?",
}
_NO_SLOTS_AVAILABLE = {
    "hindi": "Maaf kijiye, abhi koi slot available nahi hai. Koi aapko jald hi call karega.",
    "bengali": "Dukkhito, ekhon kono slot nei. Keu apnake shiggiri phone korbe.",
    "english": "Sorry, there are no slots available right now - someone will call you back shortly.",
}

_ASK_NAME = {
    "hindi": "Theek hai, aur aapka naam kya hai?",
    "bengali": "Thik ache, ebong apnar naam ki?",
    "english": "Great. And what name should I book this under?",
}
_ASK_NAME_RETRY = {
    "hindi": "Maaf kijiye, sirf aapka naam bata dijiye.",
    "bengali": "Dukkhito, shudhu apnar naam ta bolun.",
    "english": "Sorry, could you just say your name?",
}

_CONFIRM_PROMPT = {
    "hindi": "Toh {name} ke naam se, {reason} ke liye, {date} ko {time} baje appointment book kar doon?",
    "bengali": "Tahole {name}-r naame, {reason}-r jonyo, {date} tarikhe {time} shomoy e appointment book kore dei?",
    "english": "So should I book this for {name}, for {reason}, on {date} at {time}?",
}
_CONFIRM_RETRY = {
    "hindi": "Ek baar aur - {name} ke naam se {date} ko {time} baje book karoon, theek hai?",
    "bengali": "Arek baar - {name}-r naame {date} tarikhe {time} shomoy e book kori, thik ache?",
    "english": "Just to confirm - book this for {name} on {date} at {time}?",
}
_NO_MORE_SLOTS = {
    "hindi": "Maaf kijiye, aur koi slot nahi bacha hai. Koi aapko jald hi call karega.",
    "bengali": "Dukkhito, ar kono slot nei. Keu apnake shiggiri phone korbe.",
    "english": "Sorry, there are no other slots left - someone will call you back shortly.",
}

_CONFIRMATIONS = {
    "hindi": "Bahut badhiya, aapki appointment {date} ko {time} baje book ho gayi hai. Dhanyavaad!",
    "bengali": "Chomotkar, apnar appointment {date} tarikhe {time} shomoy e book hoye geche. Dhonnobad!",
    "english": "Great, your appointment is booked for {date} at {time}. Thank you for calling!",
}

_OUT_OF_SCOPE_FALLBACK = {
    "hindi": "Main appointment book karne mein madad kar sakta hoon, aur baaki sawal ka jawab call back karke koi denga.",
    "bengali": "Ami appointment book korte shahajyo korte pari, r baki proshner uttor keu callback kore debe.",
    "english": "I can help you book an appointment, and someone from the business can answer that when they call you back.",
}

_GOODBYE = {
    "hindi": "Theek hai, call karne ke liye dhanyavaad!",
    "bengali": "Thik ache, phone korar jonyo dhonnobad!",
    "english": "Okay, thank you for calling!",
}


class BookingStage:
    REASON = "reason"
    TIMING = "timing"
    NAME = "name"
    CONFIRM = "confirm"
    CLOSING = "closing"


@dataclass
class ConversationState:
    business: dict
    slots: dict
    stage: str = BookingStage.REASON
    offered_slots: list = field(default_factory=list)
    reason: str | None = None
    # ISO date string; set only while narrowing a date that has more than
    # one offered slot on it, so a bare follow-up like "10am" resolves
    # against the date already established rather than being ambiguous.
    target_date: str | None = None
    matched_slot: dict | None = None
    customer_name: str | None = None
    messages: list = field(default_factory=list)
    turns: int = 0
    consecutive_unclear: int = 0
    reason_attempts: int = 0
    unresolved_timing_attempts: int = 0
    name_attempts: int = 0
    confirm_attempts: int = 0
    started_at: float = field(default_factory=time.monotonic)


class ConversationManager:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        self._groq = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        self._intent_interpreter = BookingIntentInterpreter(self._groq, model)
        self._sessions: dict[str, ConversationState] = {}

    def start_session(self, call_id: str, business: dict, slots: list[dict]) -> str:
        """Seeds conversation state and returns a template greeting (no LLM
        round-trip needed to pick up the call - keeps time-to-first-word low)."""
        language = business.get("language_pref", "english")
        greeting = _GREETINGS.get(language, _GREETINGS["english"]).format(business=business["name"])
        session = ConversationState(business=business, slots={s["id"]: s for s in slots})
        session.messages.append({"speaker": "agent", "text": greeting})
        self._sessions[call_id] = session
        return greeting

    def get_reply(self, call_id: str, transcript: str) -> dict:
        """Returns {"reply_text": str, "hangup": bool, "booking": dict | None}"""
        session = self._store.get(call_id)
        if session is None:
            logger.error("stage", stage="ai_response_generated", outcome="error", reason="unknown_call_id")
            return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}

        session.turns += 1
        session.messages.append({"speaker": "caller", "text": transcript})

        if session.turns > MAX_TURNS:
            logger.warning("call_id=%s exceeded MAX_TURNS without a booking", call_id)
            return self._finish(session, FALLBACK_MESSAGE, hangup=True)

        language = session.business.get("language_pref", "english")
        today = date.today()

        understanding = self._intent_interpreter.interpret(transcript, self._context(session))

        if understanding.interpreted:
            session.consecutive_unclear = 0
        else:
            session.consecutive_unclear += 1
            if session.consecutive_unclear >= 3:
                logger.warning(
                    "call_id=%s: repeated interpretation failures, ending call gracefully", call_id
                )
                return self._finish(session, FALLBACK_MESSAGE, hangup=True)

        if _looks_like_end_call(transcript) or understanding.intent == BookingIntent.END_CALL:
            return self._finish(session, _GOODBYE.get(language, _GOODBYE["english"]), hangup=True)

        if understanding.intent == BookingIntent.OUT_OF_SCOPE:
            reply = understanding.assistant_reply or _OUT_OF_SCOPE_FALLBACK.get(
                language, _OUT_OF_SCOPE_FALLBACK["english"]
            )
            return self._finish(session, reply, hangup=False)

        if understanding.intent == BookingIntent.ASK_REPEAT_SLOTS and session.offered_slots:
            return self._finish(session, _offer_slots_by_number(session.offered_slots, language), hangup=False)

        if session.stage == BookingStage.REASON:
            reply_text, hangup, booking = self._handle_reason_stage(session, transcript, understanding, today, language)
        elif session.stage == BookingStage.TIMING:
            reply_text, hangup, booking = self._handle_timing_stage(session, transcript, understanding, today, language)
        elif session.stage == BookingStage.NAME:
            reply_text, hangup, booking = self._handle_name_stage(session, transcript, understanding, language)
        elif session.stage == BookingStage.CONFIRM:
            reply_text, hangup, booking = self._handle_confirm_stage(session, transcript, understanding, language)
        else:
            reply_text, hangup, booking = FALLBACK_MESSAGE, True, None

        return self._finish(session, reply_text, hangup=hangup, booking=booking)

    def _context(self, session: ConversationState) -> dict:
        return {
            "stage": session.stage,
            "reason": session.reason,
            "matched_slot": (
                {"date": session.matched_slot["date"], "time": session.matched_slot["time"]}
                if session.matched_slot
                else None
            ),
            "customer_name": session.customer_name,
            "offered_slots": [
                {"option": i, "date": s["date"], "time": s["time"]}
                for i, s in enumerate(session.offered_slots, 1)
            ],
            "business_vertical": session.business.get("vertical"),
            "recent_turns": session.messages[-6:],
        }

    def _finish(self, session: ConversationState, reply_text: str, hangup: bool, booking: dict | None = None) -> dict:
        session.messages.append({"speaker": "agent", "text": reply_text})
        if hangup:
            session.stage = BookingStage.CLOSING
        return {"reply_text": reply_text, "hangup": hangup, "booking": booking}

    def _handle_reason_stage(self, session, transcript, understanding, today, language):
        reason = understanding.reason
        if not reason and _looks_substantive(transcript):
            reason = transcript.strip()
        if not reason:
            session.reason_attempts += 1
            template = _ASK_REASON_RETRY if session.reason_attempts >= 2 else _REASON_NUDGE
            return template.get(language, template["english"]), False, None

        session.reason = reason[:200]
        session.stage = BookingStage.TIMING
        session.offered_slots = list(session.slots.values())[:3]
        if not session.offered_slots:
            return _NO_SLOTS_AVAILABLE.get(language, _NO_SLOTS_AVAILABLE["english"]), True, None

        # The caller may have already mentioned a date/time in the same
        # breath as their reason ("I need a skin appointment tomorrow") -
        # check before asking a question they've effectively already answered.
        slot = match_requested_slot(transcript, session.offered_slots, today)
        if slot is not None:
            session.matched_slot = slot
            return self._advance_past_timing(session, language)

        return _ASK_TIMING.get(language, _ASK_TIMING["english"]), False, None

    def _handle_timing_stage(self, session, transcript, understanding, today, language):
        slot = match_requested_slot(transcript, session.offered_slots, today, assume_scheduling_context=True)

        if slot is None and session.target_date:
            # Already narrowed to one date last turn; this turn is likely
            # just a bare time ("10am") answering that narrowed question.
            narrowed = slots_on_date(session.offered_slots, date.fromisoformat(session.target_date))
            parsed = parse_datetime_request(transcript, today, assume_scheduling_context=True)
            candidates = narrowed
            if parsed.time:
                candidates = [s for s in candidates if s["time"] == parsed.time.strftime("%H:%M")]
            elif understanding.selected_option:
                index = understanding.selected_option - 1
                candidates = [narrowed[index]] if 0 <= index < len(narrowed) else []
            if len(candidates) == 1:
                slot = candidates[0]

        if slot is None and understanding.selected_option:
            index = understanding.selected_option - 1
            if 0 <= index < len(session.offered_slots):
                slot = session.offered_slots[index]

        if slot is None and understanding.target_date:
            try:
                llm_date = date.fromisoformat(understanding.target_date)
            except ValueError:
                llm_date = None
            if llm_date:
                candidates = slots_on_date(session.offered_slots, llm_date)
                if understanding.target_time:
                    candidates = [s for s in candidates if s["time"] == understanding.target_time]
                if len(candidates) == 1:
                    slot = candidates[0]

        if slot is not None:
            session.matched_slot = slot
            session.target_date = None
            session.unresolved_timing_attempts = 0
            return self._advance_past_timing(session, language)

        parsed = parse_datetime_request(transcript, today, assume_scheduling_context=True)
        requested_date = parsed.date
        if requested_date is None and understanding.target_date:
            try:
                requested_date = date.fromisoformat(understanding.target_date)
            except ValueError:
                requested_date = None

        if requested_date:
            session.unresolved_timing_attempts = 0
            on_date = slots_on_date(session.offered_slots, requested_date)
            if on_date:
                session.target_date = requested_date.isoformat()
                times = _join_or([_speak_time(s["time"]) for s in on_date], language)
                templates = {
                    "hindi": f"{_speak_date(requested_date.isoformat())} ko mere paas {times} available hai - kaunsa theek rahega?",
                    "bengali": f"{_speak_date(requested_date.isoformat())} tarikhe amar kache {times} ache - kontai bhalo hobe?",
                    "english": f"For {_speak_date(requested_date.isoformat())}, I have {times} - which works for you?",
                }
                return templates.get(language, templates["english"]), False, None

            session.target_date = None
            listed = _join_or([_speak_slot(s) for s in session.offered_slots], language)
            templates = {
                "hindi": f"Maaf kijiye, {_speak_date(requested_date.isoformat())} ko koi slot nahi hai. Mere paas {listed} available hai - koi theek rahega?",
                "bengali": f"Dukkhito, {_speak_date(requested_date.isoformat())} tarikhe kono slot nei. Amar kache {listed} ache - kono ekta cholbe?",
                "english": f"Sorry, nothing's open on {_speak_date(requested_date.isoformat())}. I have {listed} - would any of those work?",
            }
            return templates.get(language, templates["english"]), False, None

        session.unresolved_timing_attempts += 1
        if session.unresolved_timing_attempts >= 2:
            nudge = session.unresolved_timing_attempts > 2
            return _offer_slots_by_number(session.offered_slots, language, nudge=nudge), False, None
        return _ASK_TIMING_RETRY.get(language, _ASK_TIMING_RETRY["english"]), False, None

    def _advance_past_timing(self, session, language):
        session.stage = BookingStage.NAME
        return _ASK_NAME.get(language, _ASK_NAME["english"]), False, None

    def _handle_name_stage(self, session, transcript, understanding, language):
        name = understanding.customer_name
        if not name and _looks_like_name(transcript):
            name = transcript.strip()

        normalized = (name or "").strip("[]").lower()
        if not name or normalized in _PLACEHOLDER_TOKENS:
            session.name_attempts += 1
            reply = _ASK_NAME_RETRY.get(language, _ASK_NAME_RETRY["english"])
            if session.name_attempts > 1:
                suffixes = {"hindi": " Bas apna naam boliye.", "bengali": " Shudhu apnar naam ta bolun.", "english": " Just your name is fine."}
                reply += suffixes.get(language, suffixes["english"])
            return reply, False, None

        session.customer_name = name[:100]
        session.stage = BookingStage.CONFIRM
        prompt = _CONFIRM_PROMPT.get(language, _CONFIRM_PROMPT["english"])
        return (
            prompt.format(
                name=session.customer_name,
                reason=session.reason,
                date=_speak_date(session.matched_slot["date"]),
                time=_speak_time(session.matched_slot["time"]),
            ),
            False,
            None,
        )

    def _handle_confirm_stage(self, session, transcript, understanding, language):
        wants_confirm = (
            understanding.intent == BookingIntent.CONFIRM_BOOKING
            and understanding.confidence >= BookingIntentInterpreter.MIN_ACTION_CONFIDENCE
        ) or (
            _looks_like_affirmative(transcript)
            and understanding.intent
            not in (BookingIntent.WANTS_DIFFERENT_SLOT, BookingIntent.OUT_OF_SCOPE, BookingIntent.END_CALL)
        )
        if wants_confirm:
            booking = {
                "slot_id": session.matched_slot["id"],
                "slot_date": session.matched_slot["date"],
                "slot_time": session.matched_slot["time"],
                "customer_name": session.customer_name,
                "reason": session.reason,
            }
            confirmation = _CONFIRMATIONS.get(language, _CONFIRMATIONS["english"]).format(
                date=_speak_date(session.matched_slot["date"]), time=_speak_time(session.matched_slot["time"])
            )
            return confirmation, True, booking

        if understanding.intent == BookingIntent.WANTS_DIFFERENT_SLOT:
            session.offered_slots = [s for s in session.offered_slots if s["id"] != session.matched_slot["id"]]
            session.matched_slot = None
            session.target_date = None
            session.unresolved_timing_attempts = 0
            session.stage = BookingStage.TIMING
            if not session.offered_slots:
                return _NO_MORE_SLOTS.get(language, _NO_MORE_SLOTS["english"]), True, None
            return _ASK_TIMING.get(language, _ASK_TIMING["english"]), False, None

        session.confirm_attempts += 1
        reply = _CONFIRM_RETRY.get(language, _CONFIRM_RETRY["english"]).format(
            name=session.customer_name,
            date=_speak_date(session.matched_slot["date"]),
            time=_speak_time(session.matched_slot["time"]),
        )
        if session.confirm_attempts > 1:
            # A second consecutive miss on this exact prompt - vary the
            # wording rather than repeat the identical line again.
            suffixes = {"hindi": " Sirf haan ya na boliye.", "bengali": " Shudhu hain ba na bolun.", "english": " Just say yes or no."}
            reply += suffixes.get(language, suffixes["english"])
        return reply, False, None

    def get_transcript(self, call_id: str) -> str:
        session = self._store.get(call_id)
        if not session:
            return ""
        lines = []
        for message in session.messages:
            speaker = "Caller" if message["speaker"] == "caller" else "Agent"
            lines.append(f"{speaker}: {message['text']}")
        return "\n".join(lines)

    def get_duration_seconds(self, call_id: str) -> int:
        session = self._store.get(call_id)
        if not session:
            return 0
        return int(time.monotonic() - session.started_at)

    def end_session(self, call_id: str) -> None:
        self._store.delete(call_id)
        self._store.delete(self._stream_key(call_id))
