"""
ai/intent_interpreter.py

Structured per-turn understanding of the caller's speech, decoupled from
spoken-reply generation. Modeled on the ContextualIntentInterpreter pattern
used by CallSaathi's sibling project (EduGuardian-AI-Voice): classify intent
and extract entities via a single forced Groq tool call, JSON-mode fallback
if tool-calling fails, then validate every field server-side before it's
ever trusted - the model's raw string output never drives a booking
directly. This replaces the old approach of asking one free-form LLM call
to both decide *and* speak, which had no code-level tracking of what was
already known and could loop the same question forever.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 6.0


class BookingIntent(str, Enum):
    STATE_REASON = "state_reason"
    STATE_TIMING = "state_timing"
    SELECT_SLOT = "select_slot"
    PROVIDE_NAME = "provide_name"
    CONFIRM_BOOKING = "confirm_booking"
    WANTS_DIFFERENT_SLOT = "wants_different_slot"
    ASK_REPEAT_SLOTS = "ask_repeat_slots"
    OUT_OF_SCOPE = "out_of_scope"
    END_CALL = "end_call"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class TurnUnderstanding:
    intent: BookingIntent
    confidence: float
    reason: str | None = None
    target_date: str | None = None
    target_time: str | None = None
    customer_name: str | None = None
    selected_option: int | None = None
    requires_clarification: bool = False
    reasoning: str = ""
    assistant_reply: str | None = None
    interpreted: bool = True

    @classmethod
    def unclear(cls, reasoning: str = "") -> "TurnUnderstanding":
        return cls(
            intent=BookingIntent.UNCLEAR,
            confidence=0.0,
            requires_clarification=True,
            reasoning=reasoning,
            interpreted=False,
        )


_VALID_INTENTS = ", ".join(intent.value for intent in BookingIntent)

_TOOL_SYSTEM_PROMPT = (
    "You are a routing component for a phone appointment-booking assistant serving "
    "small Indian businesses (clinics, salons, coaching centers). Callers speak a mix "
    "of Hindi, Bengali and English. Interpret ONLY the latest caller turn, using the "
    "supplied conversation state (current stage, what's already known, slots on offer, "
    "last agent message) for context. Resolve short answers like 'haan'/'yes'/'ok' "
    "using the active stage. Extract target_date as YYYY-MM-DD and target_time as "
    "HH:MM only when the caller's words clearly resolve to one calendar date/time - "
    "otherwise leave both null; a separate deterministic parser handles ambiguous "
    "phrasing, so guessing here is worse than leaving it null. Classify select_slot "
    "only when the caller clearly picks one of the offered slots (by number, ordinal, "
    "or a date/time matching one of them). Classify confirm_booking only when the "
    "caller affirmatively agrees to book the slot currently proposed to them; classify "
    "wants_different_slot when they reject the proposed slot in favor of another time. "
    "Classify end_call when the caller wants to hang up or says nothing further is "
    "needed. Never claim a booking succeeded - this tool only classifies and extracts, "
    "it never books anything. Call classify_caller_turn exactly once."
)

_JSON_FALLBACK_SYSTEM_PROMPT = (
    "Return one JSON object only, with exactly these keys: intent, confidence, reason, "
    "target_date, target_time, customer_name, selected_option, requires_clarification, "
    "reasoning, assistant_reply. Use null for anything unknown this turn. "
    f"Valid intents: {_VALID_INTENTS}. "
    "Interpret only the latest caller speech, using the conversation state for context. "
    "For out_of_scope turns (pricing, medical advice, anything unrelated to booking), "
    "assistant_reply must be a natural one-sentence redirect back to booking; for every "
    "other intent, assistant_reply must be null since the app replies deterministically. "
    "Never claim a booking succeeded."
)

CLASSIFY_TURN_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_caller_turn",
        "description": (
            "Classify the caller's latest turn on a phone appointment-booking call "
            "using the supplied conversation state. Classifies and extracts only - "
            "never books anything."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": [intent.value for intent in BookingIntent]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "The caller's stated reason for the visit, briefly paraphrased; null if not stated this turn.",
                },
                "target_date": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Resolved appointment date as YYYY-MM-DD; null if not clearly resolved.",
                },
                "target_time": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Resolved appointment time as HH:MM (24-hour); null if not clearly resolved.",
                },
                "customer_name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "The caller's real name if stated this turn; null otherwise.",
                },
                "selected_option": {
                    "anyOf": [{"type": "integer", "minimum": 1, "maximum": 3}, {"type": "null"}],
                    "description": "1, 2 or 3 if the caller picked one of the numbered offered slots; null otherwise.",
                },
                "requires_clarification": {"type": "boolean"},
                "reasoning": {"type": "string", "description": "One short explanation grounded in the conversation state."},
                "assistant_reply": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Natural spoken reply, ONLY for out_of_scope turns; null otherwise since the app replies deterministically.",
                },
            },
            "required": [
                "intent", "confidence", "reason", "target_date", "target_time",
                "customer_name", "selected_option", "requires_clarification",
                "reasoning", "assistant_reply",
            ],
        },
    },
}


class BookingIntentInterpreter:
    MIN_ACTION_CONFIDENCE = 0.6

    def __init__(self, groq_client, model: str):
        self._groq = groq_client
        self._model = model

    def interpret(self, caller_speech: str, context: dict[str, Any]) -> TurnUnderstanding:
        user_content = (
            "LATEST CALLER SPEECH:\n"
            f"{caller_speech}\n\n"
            "CONVERSATION STATE:\n"
            f"{json.dumps(context, ensure_ascii=True)}\n\n"
            "Call classify_caller_turn with flat arguments matching its schema. "
            "Use null for anything unknown this turn."
        )
        try:
            response = self._groq.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _TOOL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                tools=[CLASSIFY_TURN_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_caller_turn"}},
                temperature=0,
                max_tokens=220,
            )
            calls = response.choices[0].message.tool_calls or []
            if not calls:
                return self._interpret_json_fallback(caller_speech, context)
            arguments = calls[0].function.arguments
            data = json.loads(arguments) if isinstance(arguments, str) else arguments
            return _validate_understanding(data)
        except Exception as exc:
            # Groq's serving of openai/gpt-oss-20b has been observed to
            # intermittently misparse the model's own hidden reasoning
            # output as a malformed tool call (BadRequestError
            # "tool_use_failed"). Falling back to plain JSON mode - a
            # different code path on Groq's side - recovers from exactly
            # this without needing a bespoke retry loop.
            logger.warning("Tool-call intent interpretation failed, falling back to JSON mode: %s", exc)
            return self._interpret_json_fallback(caller_speech, context)

    def _interpret_json_fallback(self, caller_speech: str, context: dict[str, Any]) -> TurnUnderstanding:
        try:
            response = self._groq.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _JSON_FALLBACK_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"latest_caller_speech": caller_speech, "conversation_state": context},
                            ensure_ascii=True,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=220,
            )
            return _validate_understanding(json.loads(response.choices[0].message.content))
        except Exception as exc:
            logger.warning("JSON-mode intent fallback also failed: %s", exc)
            return TurnUnderstanding.unclear(str(exc))


def _validate_understanding(data: Any) -> TurnUnderstanding:
    if not isinstance(data, dict):
        return TurnUnderstanding.unclear("Tool arguments were not an object.")
    try:
        intent = BookingIntent(data.get("intent", "unclear"))
    except ValueError:
        intent = BookingIntent.UNCLEAR
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0

    reason = _valid_text(data.get("reason"))
    target_date = _valid_date(data.get("target_date"))
    target_time = _valid_time(data.get("target_time"))
    customer_name = _valid_text(data.get("customer_name"))

    option = data.get("selected_option")
    option = option if isinstance(option, int) and 1 <= option <= 3 else None

    assistant_reply = data.get("assistant_reply")
    if not isinstance(assistant_reply, str) or not assistant_reply.strip():
        assistant_reply = None
    else:
        assistant_reply = assistant_reply.strip()[:400]

    return TurnUnderstanding(
        intent=intent,
        confidence=confidence,
        reason=reason,
        target_date=target_date,
        target_time=target_time,
        customer_name=customer_name,
        selected_option=option,
        requires_clarification=bool(data.get("requires_clarification", False)),
        reasoning=str(data.get("reasoning", ""))[:240],
        assistant_reply=assistant_reply,
        interpreted=True,
    )


def _valid_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped[:200] if stripped else None


def _valid_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _valid_time(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", value)
    return value if match else None
