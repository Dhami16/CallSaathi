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
import threading
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
_BOOKING_MARKER_TRIGGER = "BOOKING_CONFIRMED:"

# Progressive delivery (see SentenceStreamer / start_streaming_reply below):
# how long the FIRST webhook hit for a turn will wait for the first sentence
# (generous - covers the same retries get_reply() would do), and how long a
# later /voice/continue hit will wait for the NEXT sentence (short - by then
# the stream is already flowing, so a real gap here means something's wrong,
# not just normal generation time).
STREAM_FIRST_SENTENCE_TIMEOUT_SECONDS = 8.0
STREAM_NEXT_SENTENCE_TIMEOUT_SECONDS = 4.0
STREAM_POLL_INTERVAL_SECONDS = 0.15


def _is_harmony_tool_glitch(error: BadRequestError) -> bool:
    body = error.body if isinstance(error.body, dict) else {}
    return (body.get("error") or {}).get("code") == "tool_use_failed"


class SentenceStreamer:
    """Detects sentence boundaries in a token stream, releasing each
    sentence as soon as it's complete (i.e. as soon as sentence-ending
    punctuation is followed by whitespace in the buffer).

    Known, accepted limitation: a BOOKING_CONFIRMED marker is only detected
    once its own text starts appearing in the buffer. Any sentence that
    completed and was already released in an *earlier* feed() call cannot
    be un-released - so if the model ever precedes the marker with more
    than a token or two of spoken lead-in, that lead-in may already have
    been spoken by the time the marker is recognized, in addition to the
    deterministic confirmation _handle_booking_marker produces. In testing
    with the current prompt, marker turns had zero leading spoken text, so
    this hasn't been observed in practice - but it's a real trade-off, not
    an oversight: an earlier design held back one sentence at a time to
    close this gap, but that meant the last two sentences of EVERY reply
    (not just booking-confirming ones) were never released progressively,
    which defeated the point for this app's typical 1-2 sentence replies.
    Delivering promptly was chosen over perfect marker-adjacency safety.
    """

    _SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self):
        self._buffer = ""
        self.marker_seen = False

    def feed(self, delta: str) -> list[str]:
        """Feed a chunk of newly streamed text. Returns any sentences now
        safe to release for delivery (may be empty)."""
        if self.marker_seen or not delta:
            return []

        self._buffer += delta

        if _BOOKING_MARKER_TRIGGER in self._buffer:
            # Never release anything still buffered once the marker starts
            # appearing - see the class docstring for what this does and
            # doesn't protect against.
            self.marker_seen = True
            self._buffer = ""
            return []

        released = []
        parts = self._SENTENCE_END_RE.split(self._buffer)
        if len(parts) > 1:
            for complete in parts[:-1]:
                complete = complete.strip()
                if complete:
                    released.append(complete)
            self._buffer = parts[-1]
        return released

    def finish(self) -> list[str]:
        """Call once the stream has ended. Returns the final sentence, if
        any is still buffered (it never had trailing whitespace to signal
        its own completion) - empty if a marker was seen."""
        if self.marker_seen:
            return []
        tail = self._buffer.strip()
        self._buffer = ""
        return [tail] if tail else []


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


_NO_SLOTS_TEXT = "(no slots currently available - apologize and say someone will call back)"

# --- Per-conversation-stage prompt templates -------------------------------
#
# Split out (rather than one long f-string) so each stage of the flow can be
# read, edited, or reused independently - e.g. a future per-vertical variant
# could swap just _REASON_GATHERING_STAGE without touching booking
# confirmation. _build_system_prompt below assembles these into ONE final
# prompt whose text is unchanged from before this split.
#
# IMPORTANT: keep the assembled result's *structure* (numbered steps,
# explicit rules) intact when editing these - see README's latency section.
# Collapsing this same content into flowing prose was measured to roughly
# double gpt-oss's hidden reasoning tokens and made replies slower, not
# shorter, despite fewer prompt tokens. That finding is about the shape of
# the final text the model sees, not how the Python building it is
# organized - so this refactor only changes the source layout, not the
# prompt itself.

_INTRO_STAGE = "You are a phone booking assistant for {name}, a {vertical}."

_REASON_GATHERING_STAGE = "1. The greeting is already done. Ask the caller's reason for calling."

_SLOT_OFFERING_STAGE = """2. Ask their preferred date/time.
3. Offer these available slots (never invent others):
{slot_lines}"""

_CONFIRMATION_STAGE = """4. Only once the caller has clearly agreed to one specific slot AND you know
   their real name AND reason, add a final line in EXACTLY this format
   (never spoken aloud, always in English):
   BOOKING_CONFIRMED: {"slot_id": <id>, "customer_name": "<name>", "reason": "<reason>"}
   Never guess or use a placeholder for name/reason.
5. Never add that line in your very first reply - ask reason and timing first."""

_LANGUAGE_MATCHING_RULE = """- Reply in the SAME language the caller's most recent message used. If they
  just spoke English, reply in English - do not default to Hindi. If they
  spoke Hindi, Bengali, or a mix, mirror that same mix. Switch again
  whenever they do."""

_DECLINE_OUT_OF_SCOPE_RULE = (
    '- Decline medical/pricing/service questions: "I can help you book an\n'
    "  appointment, and someone from the business can answer that when they call\n"
    '  you back."'
)

_GENERAL_RULES = """- 1-2 short sentences per reply - this is a live phone call.
- Never invent slots, prices, or details not given here.
- Caller's phone number is already known from caller ID - never ask for it."""


def _build_system_prompt(business: dict, slots: list[dict]) -> str:
    if slots:
        slot_lines = "\n".join(f"- id={s['id']}: {s['date']} at {s['time']}" for s in slots[:3])
    else:
        slot_lines = _NO_SLOTS_TEXT

    intro = _INTRO_STAGE.format(name=business["name"], vertical=business["vertical"])
    flow = "\n".join(
        [
            _REASON_GATHERING_STAGE,
            _SLOT_OFFERING_STAGE.format(slot_lines=slot_lines),
            _CONFIRMATION_STAGE,
        ]
    )
    rules = "\n".join([_LANGUAGE_MATCHING_RULE, _DECLINE_OUT_OF_SCOPE_RULE, _GENERAL_RULES])

    return f"""{intro}

Your ONLY job on this call:
{flow}

Rules:
{rules}
"""


class ConversationManager:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        session_store: SessionStore = None,
        session_ttl_seconds: int = DEFAULT_TTL_SECONDS,
        stream_first_sentence_timeout_seconds: float = STREAM_FIRST_SENTENCE_TIMEOUT_SECONDS,
        stream_next_sentence_timeout_seconds: float = STREAM_NEXT_SENTENCE_TIMEOUT_SECONDS,
        stream_poll_interval_seconds: float = STREAM_POLL_INTERVAL_SECONDS,
    ):
        self._client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        self._model = model
        if session_store is None:
            raise ValueError("session_store is required (see booking/session_store.py)")
        self._store = session_store
        self._session_ttl_seconds = session_ttl_seconds
        # Overridable (rather than always using the module constants
        # directly) so tests can use short timeouts instead of waiting the
        # real several-second production values.
        self._stream_first_timeout = stream_first_sentence_timeout_seconds
        self._stream_next_timeout = stream_next_sentence_timeout_seconds
        self._stream_poll_interval = stream_poll_interval_seconds

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

    # --- Progressive (sentence-level streaming) delivery ------------------
    #
    # Twilio's TwiML model only lets us return one complete document per
    # webhook hit, so "streaming" here means: run the Groq call with
    # stream=True in a background thread, extract sentences as they
    # complete, and let each subsequent /voice/continue hit (triggered by
    # a <Redirect> in the previous response) pick up the next one from the
    # shared session store. This reuses the exact same store as
    # conversation session state (see booking/session_store.py) rather than
    # a separate mechanism, and reuses _handle_booking_marker (unchanged)
    # for the actual booking-confirmation decision once the full text is
    # assembled - streaming only changes how the text is produced and
    # delivered, never how it's validated or turned into conversation
    # history/bookings.

    def _stream_key(self, call_id: str) -> str:
        return f"stream:{call_id}"

    def start_streaming_reply(self, call_id: str, transcript: str) -> dict:
        """Starts a streaming turn in a background thread and returns as
        soon as the first thing to say is ready. Returns
        {"sentence": str | None, "more_coming": bool, "hangup": bool,
        "booking": dict | None}. If more_coming is True, the caller should
        speak `sentence` and redirect to a /voice/continue-style endpoint
        for sentence_index=1, then 2, etc., via get_next_streamed_sentence."""
        session = self._store.get(call_id)
        if session is None:
            logger.error("stage", stage="ai_response_generated", outcome="error", reason="unknown_call_id")
            return {"sentence": FALLBACK_MESSAGE, "more_coming": False, "hangup": True, "booking": None}

        session["turns"] += 1
        if session["turns"] > MAX_TURNS:
            logger.warning(
                "stage", stage="ai_response_generated", outcome="fallback_triggered", reason="max_turns_exceeded"
            )
            capture_fallback("max_turns_exceeded", call_id=call_id)
            return {"sentence": FALLBACK_MESSAGE, "more_coming": False, "hangup": True, "booking": None}

        session["messages"].append({"role": "user", "content": transcript})
        turn_number = session["turns"]
        self._store.set(call_id, session, self._session_ttl_seconds)

        stream_key = self._stream_key(call_id)
        self._store.set(
            stream_key,
            {"turn_number": turn_number, "sentences": [], "done": False, "finalized": None},
            self._session_ttl_seconds,
        )

        thread = threading.Thread(target=self._run_stream_worker, args=(call_id, turn_number), daemon=True)
        thread.start()

        return self._consume_sentence(call_id, turn_number, sentence_index=0, timeout=self._stream_first_timeout)

    def get_next_streamed_sentence(self, call_id: str, sentence_index: int) -> dict:
        """Called for each /voice/continue hit. Returns the same shape as
        start_streaming_reply, for sentence number `sentence_index` of the
        call's currently in-progress turn."""
        state = self._store.get(self._stream_key(call_id))
        if state is None:
            logger.error("stage", stage="ai_response_generated", outcome="error", reason="unknown_streaming_turn")
            capture_fallback("unknown_streaming_turn", call_id=call_id)
            return {"sentence": FALLBACK_MESSAGE, "more_coming": False, "hangup": True, "booking": None}
        return self._consume_sentence(
            call_id, state["turn_number"], sentence_index, timeout=self._stream_next_timeout
        )

    def _consume_sentence(self, call_id: str, turn_number: int, sentence_index: int, timeout: float) -> dict:
        stream_key = self._stream_key(call_id)
        deadline = time.time() + timeout

        while True:
            state = self._store.get(stream_key)
            if state is None or state["turn_number"] != turn_number:
                logger.warning(
                    "stage", stage="ai_response_generated", outcome="error", reason="streaming_state_lost"
                )
                capture_fallback("streaming_state_lost", call_id=call_id)
                return {"sentence": FALLBACK_MESSAGE, "more_coming": False, "hangup": True, "booking": None}

            sentences = state["sentences"]
            if len(sentences) > sentence_index:
                sentence = sentences[sentence_index]
                finalized = state.get("finalized")
                has_extra = finalized is not None and finalized.get("reply_text")
                is_last_ever = state["done"] and len(sentences) == sentence_index + 1
                if is_last_ever and not has_extra:
                    hangup = finalized["hangup"] if finalized else False
                    booking = finalized.get("booking") if finalized else None
                    return {"sentence": sentence, "more_coming": False, "hangup": hangup, "booking": booking}
                return {"sentence": sentence, "more_coming": True, "hangup": False, "booking": None}

            if state["done"]:
                finalized = state.get("finalized") or {"reply_text": None, "hangup": False, "booking": None}
                if finalized.get("reply_text"):
                    return {
                        "sentence": finalized["reply_text"],
                        "more_coming": False,
                        "hangup": finalized["hangup"],
                        "booking": finalized.get("booking"),
                    }
                # Fully delivered via progressive sentences already, nothing extra to add.
                return {"sentence": None, "more_coming": False, "hangup": False, "booking": None}

            if time.time() >= deadline:
                # LLM hasn't produced the next sentence yet and we've waited
                # a sane amount of time - never error out, fall back
                # gracefully exactly like a Groq API failure would.
                logger.warning(
                    "stage", stage="ai_response_generated", outcome="fallback_triggered", reason="streaming_timeout"
                )
                capture_fallback("streaming_reply_timeout", call_id=call_id)
                return {"sentence": FALLBACK_MESSAGE, "more_coming": False, "hangup": True, "booking": None}

            time.sleep(self._stream_poll_interval)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_TRANSIENT_ERRORS),
        stop=stop_after_attempt(TRANSIENT_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=0.5, max=2),
        reraise=True,
    )
    def _call_groq_stream(self, messages: list[dict]):
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
            reasoning_format="hidden",
            stream=True,
        )

    def _run_stream_worker(self, call_id: str, turn_number: int) -> None:
        """Runs in a background thread, started by start_streaming_reply.
        Streams the Groq completion, writing sentences to the shared
        session store as they complete so /voice/continue hits (which may
        land on a different worker process) can pick them up. The harmony-
        glitch retry loop mirrors get_reply()'s exactly - only genuinely
        transient errors (via _call_groq_stream's own tenacity retry) and
        the harmony glitch are retried, and only before any content has
        been produced; a failure mid-stream (after some sentences may
        already be queued) is treated as final for this turn rather than
        restarted, since restarting would regenerate content the caller may
        have already partially heard.
        """
        stream_key = self._stream_key(call_id)

        stream = None
        for attempt in range(1, MAX_GROQ_ATTEMPTS + 1):
            session = self._store.get(call_id)
            if session is None or session["turns"] != turn_number:
                return  # turn no longer current (call ended/moved on) - nothing to do
            try:
                stream = self._call_groq_stream(session["messages"])
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
                self._finish_stream_with_fallback(stream_key, turn_number)
                return
            except Exception:
                logger.exception("stage", stage="ai_response_generated", outcome="error", reason="groq_api_call_failed")
                capture_fallback("groq_api_call_failed", call_id=call_id)
                self._finish_stream_with_fallback(stream_key, turn_number)
                return

        if stream is None:
            self._finish_stream_with_fallback(stream_key, turn_number)
            return

        streamer = SentenceStreamer()
        full_text = ""
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                full_text += delta
                newly_released = streamer.feed(delta)
                if newly_released:
                    self._append_sentences(stream_key, turn_number, newly_released)
        except Exception:
            logger.exception("stage", stage="ai_response_generated", outcome="error", reason="groq_stream_interrupted")
            capture_fallback("groq_stream_interrupted", call_id=call_id)
            self._finish_stream_with_fallback(stream_key, turn_number)
            return

        final_sentences = streamer.finish()

        if not full_text.strip():
            logger.error("stage", stage="ai_response_generated", outcome="error", reason="empty_content")
            capture_fallback("groq_returned_empty_content", call_id=call_id)
            self._finish_stream_with_fallback(stream_key, turn_number)
            return

        session = self._store.get(call_id)
        if session is None or session["turns"] != turn_number:
            return
        slots_by_id = {s["id"]: s for s in session["slots"]}
        marker_match = _BOOKING_MARKER_RE.search(full_text)
        if marker_match:
            finalized = self._handle_booking_marker(call_id, session, slots_by_id, full_text, marker_match)
        else:
            session["messages"].append({"role": "assistant", "content": full_text})
            finalized = {"reply_text": None, "hangup": False, "booking": None}
        self._store.set(call_id, session, self._session_ttl_seconds)

        # The final sentence(s) and done/finalized are written together in
        # ONE store update - writing them separately would leave a window
        # where a poller could see the last sentence appended but done
        # still False, wrongly concluding more_coming=True for what's
        # actually the final sentence (a real race hit during testing).
        state = self._store.get(stream_key)
        if state is None or state["turn_number"] != turn_number:
            return
        if final_sentences:
            state["sentences"].extend(final_sentences)
        state["done"] = True
        state["finalized"] = finalized
        self._store.set(stream_key, state, self._session_ttl_seconds)

    def _append_sentences(self, stream_key: str, turn_number: int, sentences: list[str]) -> None:
        state = self._store.get(stream_key)
        if state is None or state["turn_number"] != turn_number:
            return
        state["sentences"].extend(sentences)
        self._store.set(stream_key, state, self._session_ttl_seconds)

    def _finish_stream_with_fallback(self, stream_key: str, turn_number: int) -> None:
        state = self._store.get(stream_key)
        if state is None or state["turn_number"] != turn_number:
            return
        state["done"] = True
        state["finalized"] = {"reply_text": FALLBACK_MESSAGE, "hangup": True, "booking": None}
        self._store.set(stream_key, state, self._session_ttl_seconds)

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
        self._store.delete(self._stream_key(call_id))
