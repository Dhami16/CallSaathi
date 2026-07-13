"""Conversational booking agent backed by Groq's openai/gpt-oss-20b model.

Conversation history lives in an externalized SessionStore (see
booking/session_store.py), not an in-memory dict - a plain dict silently
breaks the moment there's more than one gunicorn worker process, since
Twilio's sequential webhook hits for one call can land on different
workers.

Booking confirmation uses a plain-text marker line (BOOKING_CONFIRMED: {...})
rather than the Groq API's native tool-calling. In testing, openai/gpt-oss-20b
on Groq reliably called a declared `confirm_booking` tool on the very first
turn with empty or placeholder arguments (e.g. slot_id 0, customer_name set
to the raw transcript) regardless of prompt wording forbidding it - a real,
reproducible reliability problem with this model/provider combination, not
occasional flakiness. Asking for a text marker instead keeps the model in
plain chat mode (which it follows correctly) while still giving us a
structured, parseable signal - and total control over validating it before
ever touching the database.
"""
import json
import re
import time

import structlog
from groq import APIConnectionError, APITimeoutError, BadRequestError, Groq, InternalServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from booking.session_store import DEFAULT_TTL_SECONDS, SessionStore
from observability import capture_fallback

logger = structlog.get_logger(__name__)

MAX_TURNS = 8  # safety valve so a stuck/confused caller can't loop forever
REQUEST_TIMEOUT_SECONDS = 8.0

# Groq's serving stack for openai/gpt-oss-20b intermittently misparses the
# model's own harmony-format "final channel" output as an attempted tool
# call (error code "tool_use_failed"), even when this app sends zero tools
# in the request. Confirmed in testing: ~1 in 2 calls, no discernible
# trigger. It's transient - retrying the identical request succeeds. This
# is a distinct failure mode from genuine network/server transience below,
# so it gets its own immediate (no-backoff) retry loop.
MAX_GROQ_ATTEMPTS = 4

# Genuine transient failures (timeout, connection drop, 5xx) get a small,
# bounded number of retries with short backoff - not unbounded, so a live
# phone call can't hang indefinitely waiting on Groq. Deliberately excludes
# RateLimitError (429): that wasn't asked for and arguably deserves
# different handling (backing off much further), so left alone for now.
_RETRYABLE_TRANSIENT_ERRORS = (APITimeoutError, APIConnectionError, InternalServerError)
TRANSIENT_RETRY_ATTEMPTS = 3  # 1 original attempt + 2 retries

FALLBACK_MESSAGE = "I'm having trouble right now, let me have someone call you back shortly."
_UNSURE_MESSAGE = "Sorry, could you say that again?"

_PLACEHOLDER_TOKENS = {"unknown", "n/a", "na", "customer", "caller", "name", ""}

# Not anchored to line start/end: the model sometimes appends this marker
# inline after other spoken text on the same line rather than on its own
# line, despite the prompt asking for "a final line".
_BOOKING_MARKER_RE = re.compile(r"BOOKING_CONFIRMED:\s*(\{[^{}]*\})")


def _is_harmony_tool_glitch(error: BadRequestError) -> bool:
    body = error.body if isinstance(error.body, dict) else {}
    return (body.get("error") or {}).get("code") == "tool_use_failed"


# Greetings/confirmations are romanized (Hinglish/Benglish) rather than
# native script: the Twilio voice used (Polly.Aditi) is a Hindi/English
# voice and doesn't render Bengali script reliably. A future iteration
# could pick a per-language voice; out of scope for this MVP.
_GREETINGS = {
    "hindi": "Namaste! Aapne {business} se call kiya hai. Main aapki appointment book karne mein madad karunga. Bataiye, kis wajah se call kar rahe hain?",
    "bengali": "Nomoshkar! Apni {business} e phone korechen. Ami apnar appointment book korte shahajyo korbo. Bolun, ki karone phone korchen?",
    "english": "Hello! You've reached {business}. I can help you book an appointment. Could you tell me what this is regarding?",
}

_CONFIRMATIONS = {
    "hindi": "Bahut badhiya, aapki appointment {date} ko {time} baje book ho gayi hai. Dhanyavaad!",
    "bengali": "Chomotkar, apnar appointment {date} tarikhe {time} shomoy e book hoye geche. Dhonnobad!",
    "english": "Great, your appointment is booked for {date} at {time}. Thank you for calling!",
}


def _build_system_prompt(business: dict, slots: list[dict]) -> str:
    if slots:
        slot_lines = "\n".join(f"- id={s['id']}: {s['date']} at {s['time']}" for s in slots[:3])
    else:
        slot_lines = "(no slots currently available - apologize and say someone will call back)"

    return f"""You are a phone booking assistant for {business['name']}, a {business['vertical']}.

Your ONLY job on this call:
1. The greeting is already done. Ask the caller's reason for calling.
2. Ask their preferred date/time.
3. Offer these available slots (never invent others):
{slot_lines}
4. Only once the caller has clearly agreed to one specific slot AND you know
   their real name AND the reason for the visit, add a final line to your
   reply in EXACTLY this format (the caller never hears this line, so write
   it in plain English regardless of the conversation's language):
   BOOKING_CONFIRMED: {{"slot_id": <id>, "customer_name": "<name>", "reason": "<reason>"}}
   Do not add this line until all three pieces of information are genuinely
   known from what the caller told you - never guess or use a placeholder.
5. Do not add the BOOKING_CONFIRMED line in the same reply as your very
   first response to the caller - you need at least one full exchange
   (reason, then timing) before a booking can be legitimate.

Rules:
- Speak in whatever mix of Hindi, Bengali and English the caller uses. Match their language naturally.
- Never answer medical advice, pricing, or service-specific questions. If asked, say something like:
  "I can help you book an appointment, and someone from the business can answer that when they call you back."
- Keep every spoken reply to 1-2 short sentences. This is a live phone call, not a chat.
- Never make up slots, prices, or business details not given to you here.
- The caller's phone number is already known from caller ID - never ask for it.
"""


class ConversationManager:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        session_store: SessionStore = None,
        session_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self._client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        self._model = model
        if session_store is None:
            raise ValueError("session_store is required (see booking/session_store.py)")
        self._store = session_store
        self._session_ttl_seconds = session_ttl_seconds

    def start_session(self, call_id: str, business: dict, slots: list[dict]) -> str:
        """Seeds conversation state and returns a template greeting (no LLM
        round-trip needed to pick up the call - keeps time-to-first-word low)."""
        language = business.get("language_pref", "english")
        greeting = _GREETINGS.get(language, _GREETINGS["english"]).format(business=business["name"])
        session = {
            "messages": [
                {"role": "system", "content": _build_system_prompt(business, slots)},
                {"role": "assistant", "content": greeting},
            ],
            "business": business,
            # Stored as a list, not a dict keyed by id: session state is
            # JSON-serialized for the external store, and JSON object keys
            # are always strings, which would silently turn int slot ids
            # into strings on every round trip. Rebuilt into a dict locally
            # wherever id lookups are needed.
            "slots": slots,
            # Wall-clock time, not time.monotonic(): monotonic clocks are
            # only comparable within a single process's uptime - meaningless
            # once this session is written by one worker process and read
            # back by a different one.
            "started_at": time.time(),
            "turns": 0,
        }
        self._store.set(call_id, session, self._session_ttl_seconds)
        return greeting

    def get_reply(self, call_id: str, transcript: str) -> dict:
        """Returns {"reply_text": str, "hangup": bool, "booking": dict | None}"""
        session = self._store.get(call_id)
        if session is None:
            logger.error("stage", stage="ai_response_generated", outcome="error", reason="unknown_call_id")
            return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}

        session["turns"] += 1
        if session["turns"] > MAX_TURNS:
            logger.warning(
                "stage", stage="ai_response_generated", outcome="fallback_triggered", reason="max_turns_exceeded"
            )
            capture_fallback("max_turns_exceeded", call_id=call_id)
            return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}

        session["messages"].append({"role": "user", "content": transcript})

        content = None
        for attempt in range(1, MAX_GROQ_ATTEMPTS + 1):
            try:
                completion = self._call_groq(session["messages"])
                content = (completion.choices[0].message.content or "").strip()
                break
            except BadRequestError as e:
                if _is_harmony_tool_glitch(e) and attempt < MAX_GROQ_ATTEMPTS:
                    logger.warning(
                        "stage",
                        stage="ai_response_generated",
                        outcome="retry",
                        reason="groq_harmony_tool_glitch",
                        attempt=attempt,
                        max_attempts=MAX_GROQ_ATTEMPTS,
                    )
                    continue
                logger.exception("stage", stage="ai_response_generated", outcome="error", reason="groq_api_call_failed")
                capture_fallback("groq_api_call_failed", call_id=call_id)
                return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}
            except Exception:
                # Reached when a genuinely transient error (timeout,
                # connection drop, 5xx) exhausts _call_groq's own tenacity
                # retries and re-raises, or for any other unexpected error.
                logger.exception("stage", stage="ai_response_generated", outcome="error", reason="groq_api_call_failed")
                capture_fallback("groq_api_call_failed", call_id=call_id)
                return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}

        if not content:
            logger.error("stage", stage="ai_response_generated", outcome="error", reason="empty_content")
            capture_fallback("groq_returned_empty_content", call_id=call_id)
            return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}

        slots_by_id = {s["id"]: s for s in session["slots"]}

        marker_match = _BOOKING_MARKER_RE.search(content)
        if marker_match:
            result = self._handle_booking_marker(call_id, session, slots_by_id, content, marker_match)
            self._store.set(call_id, session, self._session_ttl_seconds)
            return result

        session["messages"].append({"role": "assistant", "content": content})
        self._store.set(call_id, session, self._session_ttl_seconds)
        return {"reply_text": content, "hangup": False, "booking": None}

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_TRANSIENT_ERRORS),
        stop=stop_after_attempt(TRANSIENT_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=0.5, max=2),
        reraise=True,
    )
    def _call_groq(self, messages: list[dict]):
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            # gpt-oss's hidden reasoning tokens count against this budget;
            # too low a limit truncates the real answer.
            max_tokens=600,
            # Hides gpt-oss's internal "harmony" reasoning channel tokens.
            # Without this, Groq's serving stack sometimes misparses that
            # raw output as an attempted tool call (error code
            # tool_use_failed) even with zero tools declared - confirmed as
            # the root cause in testing.
            reasoning_format="hidden",
        )

    def _handle_booking_marker(
        self, call_id: str, session: dict, slots_by_id: dict, content: str, match: re.Match
    ) -> dict:
        spoken_text = _BOOKING_MARKER_RE.sub("", content).strip()

        try:
            data = json.loads(match.group(1))
            slot_id = int(data["slot_id"])
            customer_name = str(data["customer_name"]).strip()
            reason = str(data["reason"]).strip()
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning(
                "stage", stage="ai_response_generated", outcome="error", reason="malformed_booking_marker"
            )
            session["messages"].append({"role": "assistant", "content": content})
            return {"reply_text": spoken_text or _UNSURE_MESSAGE, "hangup": False, "booking": None}

        slot = slots_by_id.get(slot_id)
        invalid = (
            slot is None
            or not customer_name
            or customer_name.strip("[]").lower() in _PLACEHOLDER_TOKENS
            or not reason
        )
        if invalid:
            # The model jumped the gun before it actually had everything -
            # don't book, just keep talking. Far better UX than dropping the
            # call, and this is our own text parsing, so there's no API
            # error risk in simply continuing the conversation.
            logger.warning(
                "stage",
                stage="ai_response_generated",
                outcome="error",
                reason="premature_booking_marker",
                slot_id=slot_id,
            )
            session["messages"].append({"role": "assistant", "content": content})
            return {"reply_text": spoken_text or _UNSURE_MESSAGE, "hangup": False, "booking": None}

        language = session["business"].get("language_pref", "english")
        reply_text = _CONFIRMATIONS.get(language, _CONFIRMATIONS["english"]).format(
            date=slot["date"], time=slot["time"]
        )
        booking = {
            "slot_id": slot_id,
            "slot_date": slot["date"],
            "slot_time": slot["time"],
            "customer_name": customer_name,
            "reason": reason,
        }
        return {"reply_text": reply_text, "hangup": True, "booking": booking}

    def get_transcript(self, call_id: str) -> str:
        session = self._store.get(call_id)
        if not session:
            return ""
        lines = []
        for m in session["messages"]:
            if m["role"] == "system":
                continue
            speaker = "Caller" if m["role"] == "user" else "Agent"
            content = _BOOKING_MARKER_RE.sub("", m["content"]).strip()
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def get_duration_seconds(self, call_id: str) -> int:
        session = self._store.get(call_id)
        if not session:
            return 0
        return int(time.time() - session["started_at"])

    def end_session(self, call_id: str) -> None:
        self._store.delete(call_id)
