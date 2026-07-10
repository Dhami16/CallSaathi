"""Conversational booking agent backed by Groq's openai/gpt-oss-20b model.

Conversation history is kept in an in-memory dict keyed by call_id, since
webhook hits are otherwise stateless. This is fine for a single-process MVP
per spec and does not survive a process restart.

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
import logging
import re
import time

from groq import BadRequestError, Groq

logger = logging.getLogger(__name__)

MAX_TURNS = 8  # safety valve so a stuck/confused caller can't loop forever
REQUEST_TIMEOUT_SECONDS = 8.0

# Groq's serving stack for openai/gpt-oss-20b intermittently misparses the
# model's own harmony-format "final channel" output as an attempted tool
# call (error code "tool_use_failed"), even when this app sends zero tools
# in the request. Confirmed in testing: ~1 in 2 calls, no discernible
# trigger. It's transient - retrying the identical request succeeds.
MAX_GROQ_ATTEMPTS = 4

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
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        self._client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        self._model = model
        self._sessions: dict[str, dict] = {}

    def start_session(self, call_id: str, business: dict, slots: list[dict]) -> str:
        """Seeds conversation state and returns a template greeting (no LLM
        round-trip needed to pick up the call - keeps time-to-first-word low)."""
        language = business.get("language_pref", "english")
        greeting = _GREETINGS.get(language, _GREETINGS["english"]).format(business=business["name"])
        self._sessions[call_id] = {
            "messages": [
                {"role": "system", "content": _build_system_prompt(business, slots)},
                {"role": "assistant", "content": greeting},
            ],
            "business": business,
            "slots": {s["id"]: s for s in slots},
            "started_at": time.monotonic(),
            "turns": 0,
        }
        return greeting

    def get_reply(self, call_id: str, transcript: str) -> dict:
        """Returns {"reply_text": str, "hangup": bool, "booking": dict | None}"""
        session = self._sessions.get(call_id)
        if session is None:
            logger.error("get_reply called for unknown call_id=%s", call_id)
            return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}

        session["turns"] += 1
        if session["turns"] > MAX_TURNS:
            logger.warning("call_id=%s exceeded MAX_TURNS without a booking", call_id)
            return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}

        session["messages"].append({"role": "user", "content": transcript})

        content = None
        for attempt in range(1, MAX_GROQ_ATTEMPTS + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=session["messages"],
                    temperature=0.3,
                    # gpt-oss's hidden reasoning tokens count against this
                    # budget; too low a limit truncates the real answer.
                    max_tokens=600,
                    # Hides gpt-oss's internal "harmony" reasoning channel
                    # tokens. Without this, Groq's serving stack sometimes
                    # misparses that raw output as an attempted tool call
                    # (error code tool_use_failed) even with zero tools
                    # declared - confirmed as the root cause in testing.
                    reasoning_format="hidden",
                )
                content = (completion.choices[0].message.content or "").strip()
                break
            except BadRequestError as e:
                if _is_harmony_tool_glitch(e) and attempt < MAX_GROQ_ATTEMPTS:
                    logger.warning(
                        "call_id=%s: Groq harmony tool-call glitch on attempt %d/%d, retrying",
                        call_id,
                        attempt,
                        MAX_GROQ_ATTEMPTS,
                    )
                    continue
                logger.exception("Groq API call failed for call_id=%s", call_id)
                return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}
            except Exception:
                logger.exception("Groq API call failed for call_id=%s", call_id)
                return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}

        if not content:
            logger.error("Groq returned empty content for call_id=%s", call_id)
            return {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}

        marker_match = _BOOKING_MARKER_RE.search(content)
        if marker_match:
            return self._handle_booking_marker(call_id, session, content, marker_match)

        session["messages"].append({"role": "assistant", "content": content})
        return {"reply_text": content, "hangup": False, "booking": None}

    def _handle_booking_marker(self, call_id: str, session: dict, content: str, match: re.Match) -> dict:
        spoken_text = _BOOKING_MARKER_RE.sub("", content).strip()

        try:
            data = json.loads(match.group(1))
            slot_id = int(data["slot_id"])
            customer_name = str(data["customer_name"]).strip()
            reason = str(data["reason"]).strip()
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning("call_id=%s: malformed BOOKING_CONFIRMED marker, ignoring: %r", call_id, content)
            session["messages"].append({"role": "assistant", "content": content})
            return {"reply_text": spoken_text or _UNSURE_MESSAGE, "hangup": False, "booking": None}

        slot = session["slots"].get(slot_id)
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
                "call_id=%s: premature/invalid booking marker (slot_id=%s name=%r reason=%r)",
                call_id,
                slot_id,
                customer_name,
                reason,
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
        session = self._sessions.get(call_id)
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
        session = self._sessions.get(call_id)
        if not session:
            return 0
        return int(time.monotonic() - session["started_at"])

    def end_session(self, call_id: str) -> None:
        self._sessions.pop(call_id, None)
