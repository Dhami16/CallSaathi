"""Task 3 verification: sentence-level progressive delivery.

Uses a mocked Groq streaming client (no network) so sentence timing and
edge cases are deterministic. Covers: a multi-sentence reply delivered
progressively, a single-sentence reply still working correctly, the
"LLM produces nothing more in time" timeout/fallback path, and the
booking-marker-suppression behavior in SentenceStreamer directly.

Run with: venv/Scripts/python -m pytest -q tests/test_streaming.py
"""
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import APIError

from ai.conversation import FALLBACK_MESSAGE, ConversationManager, SentenceStreamer
from booking.db import init_db
from booking.session_store import SQLiteSessionStore

DEMO_BUSINESS = {"id": 1, "name": "Test Clinic", "vertical": "clinic", "language_pref": "english"}
DEMO_SLOTS = [{"id": 1, "date": "2026-07-15", "time": "10:00"}]


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _fake_stream(token_chunks, delay_before_index=None, delay_seconds=0.0):
    """token_chunks: list of text fragments to yield as separate chunks.
    delay_before_index: if set, sleeps delay_seconds right before yielding
    that chunk index - used to simulate the LLM stalling mid-generation."""

    def generator():
        for i, text in enumerate(token_chunks):
            if delay_before_index is not None and i == delay_before_index:
                time.sleep(delay_seconds)
            yield _FakeChunk(text)

    return generator()


@pytest.fixture
def manager(tmp_path):
    db_path = str(tmp_path / "test_streaming_sessions.db")
    init_db(db_path)
    return ConversationManager(
        "fake-key-not-used",
        session_store=SQLiteSessionStore(db_path),
        stream_first_sentence_timeout_seconds=2.0,
        stream_next_sentence_timeout_seconds=2.0,
        stream_poll_interval_seconds=0.05,
    )


def _start_session(manager, call_id):
    manager.start_session(call_id, DEMO_BUSINESS, DEMO_SLOTS)


# --- SentenceStreamer unit tests --------------------------------------------


def test_sentence_streamer_releases_sentences_as_they_complete():
    streamer = SentenceStreamer()
    released = []
    released += streamer.feed("Sure, ")
    released += streamer.feed("what's the reason ")
    released += streamer.feed("for your visit? ")
    released += streamer.feed("Also, what time works for you?")
    released += streamer.finish()
    assert released == ["Sure, what's the reason for your visit?", "Also, what time works for you?"]


def test_sentence_streamer_suppresses_marker_and_any_immediately_preceding_text():
    # Marker arriving in the SAME feed() call as its (only) preceding text,
    # with no trailing whitespace released beforehand - the achievable
    # guarantee (see SentenceStreamer's docstring for what's NOT covered:
    # sentences already released in an earlier feed() call before the
    # marker appears can't be un-released).
    streamer = SentenceStreamer()
    released = []
    released += streamer.feed(
        'Great, Priya! BOOKING_CONFIRMED: {"slot_id": 1, "customer_name": "Priya", "reason": "checkup"}'
    )
    released += streamer.finish()
    assert released == []  # nothing spoken - see _handle_booking_marker's deterministic template instead
    assert streamer.marker_seen is True


def test_sentence_streamer_single_sentence_with_no_terminal_punctuation():
    streamer = SentenceStreamer()
    released = list(streamer.feed("Sure, one moment"))
    released += streamer.finish()
    assert released == ["Sure, one moment"]


# --- End-to-end streaming via ConversationManager ---------------------------
#
# Only replies that turn out to have MIN_SENTENCES_TO_STREAM_PROGRESSIVELY
# (3) or more sentences are actually streamed sentence-by-sentence. Shorter
# replies - the common case for this app's prompt-mandated 1-2 sentence
# style - are delivered as a single classic block instead, so Twilio's
# <Gather> is present and the caller can interrupt (barge-in), which
# progressive per-sentence delivery cannot support (no <Gather> on
# intermediate <Say>+<Redirect> steps). See MIN_SENTENCES_TO_STREAM_PROGRESSIVELY.


def test_two_sentence_reply_delivered_as_single_block_not_streamed(manager):
    """The key regression this restructuring fixes: a 1-2 sentence reply
    (the common case) must come back as ONE combined block with
    more_coming=False, so call_handler uses a real <Gather>+<Say> (caller
    can interrupt) instead of the no-Gather <Say>+<Redirect> chain."""
    call_id = "CALL-STREAM-TWO-SENTENCES"
    _start_session(manager, call_id)

    stream = _fake_stream(["Sure, what's the reason ", "for your visit? ", "Also, what time works for you?"])
    # NOTE: 3 chunks of TEXT above split into exactly 2 SENTENCES ("Sure...
    # visit?" and "Also...you?") - deliberately under the streaming threshold.
    with patch.object(manager._client.chat.completions, "create", MagicMock(return_value=stream)):
        result = manager.start_streaming_reply(call_id, "Hi, I need an appointment")

    assert result["sentence"] == "Sure, what's the reason for your visit? Also, what time works for you?"
    assert result["more_coming"] is False
    assert result["hangup"] is False
    assert result["booking"] is None


def test_three_sentence_reply_streams_progressively(manager):
    call_id = "CALL-STREAM-THREE-SENTENCES"
    _start_session(manager, call_id)

    stream = _fake_stream(
        [
            "Sure, what's the reason for your visit? ",
            "We have slots tomorrow. ",
            "Which time works best for you?",
        ]
    )
    with patch.object(manager._client.chat.completions, "create", MagicMock(return_value=stream)):
        first = manager.start_streaming_reply(call_id, "Hi, I need an appointment")

    assert first["sentence"] == "Sure, what's the reason for your visit?"
    assert first["more_coming"] is True
    assert first["hangup"] is False

    second = manager.get_next_streamed_sentence(call_id, 1)
    assert second["sentence"] == "We have slots tomorrow."
    assert second["more_coming"] is True

    third = manager.get_next_streamed_sentence(call_id, 2)
    assert third["sentence"] == "Which time works best for you?"
    assert third["more_coming"] is False
    assert third["hangup"] is False
    assert third["booking"] is None


def test_single_sentence_response_still_works(manager):
    call_id = "CALL-STREAM-SINGLE"
    _start_session(manager, call_id)

    stream = _fake_stream(["Sure, what's the reason for your visit?"])
    with patch.object(manager._client.chat.completions, "create", MagicMock(return_value=stream)):
        result = manager.start_streaming_reply(call_id, "Hi, I need an appointment")

    assert result["sentence"] == "Sure, what's the reason for your visit?"
    assert result["more_coming"] is False
    assert result["hangup"] is False
    assert result["booking"] is None


def test_booking_confirming_turn_uses_deterministic_reply_not_streamed_text(manager):
    """The marker suppresses all progressive sentences (see SentenceStreamer
    tests above); the turn's actual reply comes from the same deterministic
    _CONFIRMATIONS template the non-streaming path uses, exactly as if this
    had gone through get_reply(). No "language" field in the marker here -
    falls back to the business's configured language_pref ("english")."""
    call_id = "CALL-STREAM-BOOKING"
    _start_session(manager, call_id)

    stream = _fake_stream(['BOOKING_CONFIRMED: {"slot_id": 1, "customer_name": "Priya", "reason": "checkup"}'])
    with patch.object(manager._client.chat.completions, "create", MagicMock(return_value=stream)):
        result = manager.start_streaming_reply(call_id, "Yes, that works, I'm Priya")

    assert result["more_coming"] is False
    assert result["hangup"] is True
    assert result["booking"] == {
        "slot_id": 1,
        "slot_date": "2026-07-15",
        "slot_time": "10:00",
        "customer_name": "Priya",
        "reason": "checkup",
    }
    assert "BOOKING_CONFIRMED" not in result["sentence"]
    # Spoken confirmation uses 12-hour form ("10 AM"), not the raw "10:00"
    # stored in the booking dict above (that raw value is what's written to
    # the DB and must stay unconverted).
    assert "2026-07-15" in result["sentence"] and "10 AM" in result["sentence"]


def test_booking_confirmation_uses_callers_actual_language_not_business_default(manager):
    """Regression test for a real production bug: a caller who spoke English
    the entire call still heard a Hindi confirmation, because the
    confirmation used to always pick session["business"]["language_pref"]
    (a fixed per-business setting) regardless of what language the LLM had
    actually been replying in all call. The model now self-reports which
    language it used via the marker's "language" field, and that must win
    over the business default whenever it's present and valid."""
    call_id = "CALL-STREAM-BOOKING-LANG"
    _start_session(manager, call_id)  # DEMO_BUSINESS's language_pref is "english"

    stream = _fake_stream(
        [
            'Great, Priya! BOOKING_CONFIRMED: {"slot_id": 1, "customer_name": "Priya", '
            '"reason": "checkup", "language": "hindi"}'
        ]
    )
    with patch.object(manager._client.chat.completions, "create", MagicMock(return_value=stream)):
        result = manager.start_streaming_reply(call_id, "Haan theek hai, main Priya hoon")

    assert result["hangup"] is True
    assert "Bahut badhiya" in result["sentence"]  # the Hindi _CONFIRMATIONS template
    assert "Great, your appointment" not in result["sentence"]


def test_timeout_when_next_sentence_never_arrives_in_time(manager):
    """A reply confirmed to be 3+ sentences (progressive mode engaged), then
    the LLM stalls indefinitely before the 4th - get_next_streamed_sentence
    must wait briefly, then fall back gracefully rather than hanging or
    raising."""
    call_id = "CALL-STREAM-TIMEOUT"
    _start_session(manager, call_id)

    # Trailing whitespace after each sentence matters: that's what signals a
    # sentence is actually complete (see SentenceStreamer) - without it, the
    # text just sits in the buffer, never released.
    def stalling_generator():
        yield _FakeChunk("Sure, one moment please. ")
        yield _FakeChunk("We have slots tomorrow. ")
        yield _FakeChunk("Which time works for you? ")
        time.sleep(5)  # far longer than the 2s next-sentence timeout used in this test
        yield _FakeChunk("Continuing...")

    with patch.object(manager._client.chat.completions, "create", MagicMock(return_value=stalling_generator())):
        first = manager.start_streaming_reply(call_id, "Hi, I need an appointment")

    assert first["sentence"] == "Sure, one moment please."
    assert first["more_coming"] is True

    second = manager.get_next_streamed_sentence(call_id, 1)
    assert second["sentence"] == "We have slots tomorrow."
    assert second["more_coming"] is True

    third = manager.get_next_streamed_sentence(call_id, 2)
    assert third["sentence"] == "Which time works for you?"
    # Still "more coming" per _consume_sentence's contract at this point:
    # the stream isn't done yet, so this isn't confirmed as the last sentence.
    assert third["more_coming"] is True

    fourth = manager.get_next_streamed_sentence(call_id, 3)
    assert fourth["sentence"] == FALLBACK_MESSAGE
    assert fourth["more_coming"] is False
    assert fourth["hangup"] is True
    assert fourth["booking"] is None


def test_mid_stream_harmony_glitch_retries_when_nothing_spoken_yet(manager):
    """Real bug found in production: Groq's harmony-format glitch can raise
    from WITHIN iterating the stream (a plain groq.APIError from the SSE
    parser, body shaped as {"code": ...} rather than the nested
    {"error": {"code": ...}} a BadRequestError from stream-creation has) -
    not just from the initial call. Confirmed live: this used to fall
    straight to the fallback message on the very first glitch. Since
    nothing has been queued for delivery yet when it happens immediately,
    it must retry instead."""
    call_id = "CALL-STREAM-MIDGLITCH"
    _start_session(manager, call_id)

    fake_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

    def failing_stream():
        raise APIError(
            message="Tool choice is none, but model called a tool",
            request=fake_request,
            body={"code": "tool_use_failed", "message": "boom"},
        )
        yield  # pragma: no cover - unreachable; makes this a generator function

    def successful_stream():
        yield _FakeChunk("Sure, what's the reason for your visit?")

    mock_create = MagicMock(side_effect=[failing_stream(), successful_stream()])
    with patch.object(manager._client.chat.completions, "create", mock_create):
        result = manager.start_streaming_reply(call_id, "Hi, I need an appointment")

    assert mock_create.call_count == 2  # retried once, then succeeded
    assert result["sentence"] == "Sure, what's the reason for your visit?"
    assert result["more_coming"] is False
    assert result["hangup"] is False


def test_mid_stream_harmony_glitch_after_partial_delivery_falls_back_not_retries(manager):
    """Once a sentence has actually been queued for delivery, a later
    mid-stream glitch must NOT retry (that would regenerate content the
    caller may already have heard) - it should fall back gracefully."""
    call_id = "CALL-STREAM-MIDGLITCH-PARTIAL"
    _start_session(manager, call_id)

    fake_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

    def failing_after_one_sentence_stream():
        yield _FakeChunk("Sure, what's the reason for your visit? ")
        raise APIError(
            message="Tool choice is none, but model called a tool",
            request=fake_request,
            body={"code": "tool_use_failed", "message": "boom"},
        )

    mock_create = MagicMock(return_value=failing_after_one_sentence_stream())
    with patch.object(manager._client.chat.completions, "create", mock_create):
        result = manager.start_streaming_reply(call_id, "Hi, I need an appointment")

    assert mock_create.call_count == 1  # not retried
    assert result["sentence"] == FALLBACK_MESSAGE
    assert result["hangup"] is True


def test_streaming_repeated_question_triggers_nudge_and_regenerates(manager):
    """Same production bug covered non-streaming in
    test_conversation_retry.py, exercised through the actual live code path
    (start_streaming_reply / _run_stream_worker, called by call_handler.py
    on every real turn) rather than the non-streaming get_reply."""
    call_id = "CALL-STREAM-REPEAT-NUDGE"
    _start_session(manager, call_id)

    first_stream = _fake_stream(["Could you tell me your name?"])
    with patch.object(manager._client.chat.completions, "create", MagicMock(return_value=first_stream)):
        first = manager.start_streaming_reply(call_id, "Hi, I need an appointment")
    assert first["sentence"] == "Could you tell me your name?"

    # The model is about to repeat itself verbatim - the stream yields the
    # identical text again, then the nudge regeneration (a plain
    # non-streamed call) returns a corrected reply.
    repeat_stream = _fake_stream(["Could you tell me your name?"])
    nudge_completion = MagicMock()
    nudge_completion.choices = [MagicMock(message=MagicMock(content="Great, Rajdeep - what's this for?"))]
    mock_create = MagicMock(side_effect=[repeat_stream, nudge_completion])

    with patch.object(manager._client.chat.completions, "create", mock_create):
        second = manager.start_streaming_reply(call_id, "Rajdeep")

    assert mock_create.call_count == 2
    assert second["sentence"] == "Great, Rajdeep - what's this for?"
    assert second["more_coming"] is False


def test_duplicate_webhook_for_same_turn_does_not_spawn_a_second_worker(manager):
    """Regression test for a real production bug: a Twilio webhook retry (or
    any concurrent duplicate hit) for the exact same turn used to be
    indistinguishable from a new turn, so it spawned its OWN independent
    streaming Groq call, appending into the same shared sentence list as the
    original - the caller heard both completions interleaved sentence by
    sentence in real time. Simulates the race directly: the turn is already
    claimed (as the first, in-flight request would have done) with some
    sentences already streamed in, before the "duplicate" webhook hit calls
    start_streaming_reply. The duplicate must NOT call Groq again, and must
    just observe the already-in-progress state instead."""
    call_id = "CALL-STREAM-DUPLICATE-WEBHOOK"
    _start_session(manager, call_id)

    stream_key = manager._stream_key(call_id)
    # Turn 1 already claimed and partially streamed by the "original" request.
    assert manager._store.claim_turn(call_id, 1) is True
    manager._store.set(
        stream_key,
        {"turn_number": 1, "sentences": ["Sure, what's the reason for your visit?"], "done": False, "finalized": None},
        manager._session_ttl_seconds,
    )

    mock_create = MagicMock(side_effect=AssertionError("Groq must not be called by the duplicate webhook hit"))
    with patch.object(manager._client.chat.completions, "create", mock_create):
        result = manager.start_streaming_reply(call_id, "Hi, I need an appointment")

    assert mock_create.call_count == 0
    # The duplicate observes the ORIGINAL request's in-progress state rather
    # than starting its own - since only one sentence is queued so far (below
    # MIN_SENTENCES_TO_STREAM_PROGRESSIVELY) and the stream isn't done yet,
    # it just waits and then falls back once the timeout elapses, exactly
    # like any other consumer polling an in-progress turn would - it never
    # gets its own separate reply.
    assert result["sentence"] == FALLBACK_MESSAGE
    assert result["hangup"] is True


def test_llm_finishes_before_all_sentences_requested_is_not_an_error(manager):
    """If the caller asks for a sentence index that will never come because
    the (short) reply already fully finished, that's a normal end-of-turn,
    not a timeout/fallback."""
    call_id = "CALL-STREAM-DONE-EARLY"
    _start_session(manager, call_id)

    stream = _fake_stream(["Sure, one moment please."])
    with patch.object(manager._client.chat.completions, "create", MagicMock(return_value=stream)):
        first = manager.start_streaming_reply(call_id, "Hi, I need an appointment")
    assert first["more_coming"] is False  # only one sentence, already known to be the last

    # Give the background worker a brief moment to fully finalize (it does,
    # synchronously, right after the last chunk - this just avoids a race
    # in the test itself checking a second time).
    time.sleep(0.2)
    again = manager.get_next_streamed_sentence(call_id, 1)
    assert again["sentence"] is None
    assert again["more_coming"] is False
    assert again["hangup"] is False
