# CallSaathi

An AI voice agent that answers missed calls for small local businesses in
India (clinics, salons, coaching centers), converses in Hindi/Bengali/English,
books an appointment from a fixed slot list, and notifies the business owner
with the full call transcript.

## What this MVP does

- Answers an inbound call forwarded to a Twilio number (`POST /voice`).
- Greets the caller, asks their reason for calling and preferred time, offers
  up to 3 open slots, and books one once the caller agrees and gives their
  name (`POST /voice/handle-input`, looped via Twilio's `<Gather>`). Replies
  are streamed from Groq and delivered sentence by sentence as they're
  generated (`POST /voice/continue`, driven by TwiML `<Redirect>`) rather
  than waiting for the whole response - see "Progressive delivery" below.
- Persists the booking, the booked slot, and the **full** call transcript to
  SQLite.
- "Notifies" the business owner (full transcript + booking details) and the
  customer (short confirmation) - currently by logging, not by actually
  sending WhatsApp/SMS (see [Non-goals](#non-goals-for-this-mvp)).

## Architecture

```
telephony/            Provider-agnostic interface (TelephonyProvider) +
                       telephony/twilio_adapter.py, the only file allowed to
                       import Twilio-specific types.
ai/conversation.py     Groq-backed dialogue manager: session state lives in
                       booking/session_store.py (not in-memory), booking
                       confirmation via a text marker, retry + error fallback,
                       sentence-by-sentence streamed delivery (SentenceStreamer,
                       start_streaming_reply/get_next_streamed_sentence).
booking/db.py          SQLite schema.
booking/repository.py  Data access: businesses, slots (future-only, IST-aware),
                       bookings (idempotent per call_id), call_logs, call_turns.
booking/session_store.py  Externalized conversation session state (SQLite-
                       backed) - see "Session state" below.
notifications/         NotificationService interface + MockNotificationService.
call_handler.py        Orchestrates telephony -> AI -> booking -> notifications.
                       app.py never touches these subsystems directly.
observability.py       Sentry init + non-fatal fallback event capture.
stats.py               Aggregate queries backing GET /internal/stats.
app.py                 Flask routes only.
seed_data.py           Idempotent demo business + slots.
```

Swapping the telephony vendor means writing a new adapter file implementing
`TelephonyProvider` - nothing else changes. Swapping mock notifications for
real WhatsApp/SMS means writing a new `NotificationService` - booking logic
is untouched.

## Session state: SQLite now, Redis later if needed

Conversation state (message history, offered slots, turn count) used to
live in a plain Python dict keyed by call_id - fine for one process, but it
breaks silently the moment the app runs behind more than one gunicorn
worker: Twilio's sequential webhook hits for a single call can land on
different worker processes, each with separate memory, so the second hit
wouldn't find the first hit's history.

Fixed by externalizing session state behind a `SessionStore` interface
(`booking/session_store.py`), with a SQLite-backed implementation used by
default (`SQLiteSessionStore`, stored in the same DB file as everything
else, table `call_sessions`, JSON-serialized). **Redis was considered and
explicitly not chosen for now**: it's the more scalable long-term option
(native TTL, no single-writer file lock), but it's new infrastructure a
two-person team would have to run (Docker) or pay for/manage (hosted free
tier) even at pilot stage, and SQLite's serialized writes aren't expected
to be a practical bottleneck at pilot call volumes (small, fast
one-row-per-turn writes, not sustained high throughput). If call volume
grows enough that this becomes a real concern, swap in a
`RedisSessionStore` implementing the same interface - nothing in
`ai/conversation.py` or `call_handler.py` would need to change.

Sessions expire after `SESSION_TTL_SECONDS` (default 600s / 10 min) so an
abandoned or crashed call doesn't leak state forever; expired rows are
cleaned up opportunistically on every write (no background job/cron
needed).

## Setup

1. **Install dependencies** (Python 3.11+):
   ```
   python -m venv venv
   venv\Scripts\pip install -r requirements.txt      # Windows
   # source venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
   ```

2. **Copy `.env.example` to `.env`** and fill in:
   - `GROQ_API_KEY` - get one free at https://console.groq.com
   - `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` - from a Twilio trial account
     (https://www.twilio.com/try-twilio); a trial account includes one free
     phone number capable of receiving calls.
   - `TWILIO_PHONE_NUMBER` - the Twilio number you got above. The seed script
     uses this to set the demo business's phone number, overriding whatever
     is in `data/businesses.csv`.

3. **Seed demo data** (imports `data/businesses.csv` and `data/slots.csv`):
   ```
   venv\Scripts\python seed_data.py
   ```
   Edit those CSVs to manage which businesses and slots get seeded - add a
   row to add a business/slot, or edit one in place. Re-running the script
   only inserts rows for businesses that don't already exist yet (matched by
   phone number).

4. **Run the app**:
   ```
   venv\Scripts\python app.py
   ```
   This starts Flask on `http://localhost:5000`.

5. **Expose it publicly with ngrok** (Twilio needs an HTTPS URL it can reach):
   ```
   ngrok http 5000
   ```
   Copy the `https://...ngrok-free.app` URL it prints.

6. **Point your Twilio number's webhook at the tunnel**: in the Twilio
   console, open your phone number's configuration, and under "A call
   comes in" set the webhook to `https://<your-ngrok-subdomain>/voice`
   (HTTP POST).

7. Call your Twilio number from any phone. You should hear the greeting and
   be able to book one of the seeded slots.

To have a real business's missed calls reach this number, they'd set up
call forwarding on no-answer/busy to this Twilio number - no code changes
needed for that part.

## Running the local tests (no phone call needed)

```
venv\Scripts\python -m pytest -q tests/                         # everything except the note below
venv\Scripts\python -m pytest -s tests/test_conversation.py      # needs a real GROQ_API_KEY; skips otherwise
```

| File | What it covers | Needs network/Groq? |
| --- | --- | --- |
| `test_booking_flow.py` | Full call -> booking via `CallHandler` with a scripted fake conversation manager; SQLite rows + mock notification content; idempotent-replay and concurrent-slot-conflict booking scenarios | No |
| `test_slots.py` | Bug fix regression: past slots are never offered, including the exact real-call scenario that surfaced the bug | No |
| `test_session_store.py` | Externalized session state: one store instance's write is readable by another (the multi-worker scenario), TTL expiry, opportunistic cleanup | No |
| `test_conversation_retry.py` | Retry-with-backoff for transient Groq errors (timeout/connection/5xx) via a mocked client; confirms bounded retries, eventual fallback, and the harmony-glitch retry path is unaffected | No |
| `test_observability.py` | Structured logging robustness, `/internal/stats` aggregation, Sentry init safety | No |
| `test_streaming.py` | Progressive delivery: `SentenceStreamer` boundary/marker-suppression logic, multi- and single-sentence replies delivered correctly via a mocked streaming client, the next-sentence timeout/fallback path | No |
| `test_conversation.py` | Live persona sanity check (Hindi, code-switching, out-of-scope refusal) against the real Groq API | **Yes** - skips if `GROQ_API_KEY` is missing/invalid |

## Known limitation: occasional Groq/gpt-oss-20b glitch

During testing, Groq's serving of `openai/gpt-oss-20b` intermittently
(roughly 1 in 10 calls) misparses the model's own internal "harmony format"
output as an attempted tool call and returns a 400
(`tool_use_failed: "Tool choice is none, but model called a tool"`) **even
though this app sends zero tools in the request.** This looks like a
provider-side bug in how gpt-oss's reasoning channels are served, not
something caused by our prompt.

Mitigations already in place (`ai/conversation.py`):
- `reasoning_format="hidden"` and a larger `max_tokens` budget (gpt-oss's
  hidden reasoning tokens count against the limit) cut the glitch rate
  substantially.
- Each turn retries up to `MAX_GROQ_ATTEMPTS` (4) times specifically on this
  error code before giving up.
- If it still fails, the call ends gracefully with the fallback message
  ("I'm having trouble right now...") per spec - never dead air, never a
  silent drop, always logged.

In practice this means the overwhelming majority of calls complete
normally; a small fraction end early with the graceful fallback instead of
a booking. If Groq patches this server-side, `MAX_GROQ_ATTEMPTS` can likely
be lowered back down. For streaming requests specifically, this glitch can
also surface mid-stream with a different error shape - see "Progressive
(sentence-by-sentence) delivery" below for that variant and how retries
handle it.

A related, separate quirk measured during the streaming work (Task 3 of
the performance session): for the specific booking-confirmation turn
(caller has just agreed to a slot and given their name), Groq occasionally
returns **completely empty content** with `finish_reason='stop'` and no
exception at all - measured at roughly 15-20% of attempts in a 20-call
batch, in **both** streaming and non-streaming modes at a similar rate, so
this isn't something progressive delivery introduced. The existing
empty-content fallback (`ai/conversation.py`) already handles it correctly
- graceful message, logged, Sentry-reported - but the rate is worth knowing
if booking-turn fallbacks seem more frequent than other turns.

## Progressive (sentence-by-sentence) delivery

Twilio's TwiML model only allows one complete document per webhook hit, so
"streaming" here means: `ai/conversation.py` runs the Groq call with
`stream=True` in a background thread, extracts sentences as they complete
(`SentenceStreamer`), and writes them to the same session store used for
conversation state (a `stream:<call_id>` key, not a separate mechanism).

**Only replies confirmed to have `MIN_SENTENCES_TO_STREAM_PROGRESSIVELY`
(3) or more sentences are actually streamed sentence-by-sentence** - each
one spoken via `<Say>` then `<Redirect>`ed to `POST /voice/continue?idx=N`
to fetch the next. A reply that finishes at 1-2 sentences (this app's
prompt-mandated common case) is instead delivered as a single classic
`<Gather><Say>...</Say></Gather>` block, exactly like the non-streaming
path. This was a deliberate correction after real testing: Twilio's
`<Gather>` (which is what lets a caller interrupt/barge in) is present on
that single block, but is **absent** from every intermediate
`<Say>+<Redirect>` step of a progressively-streamed reply - so streaming
every reply meant losing the caller's ability to interrupt the agent for
nearly every turn, for a latency win that barely exists on a 1-2 sentence
reply anyway. Genuinely long (3+ sentence) replies still stream, and still
lose mid-reply interruptibility - a trade-off accepted only for the rare
case where it actually matters.

Two more things worth knowing, both found via real production calls, not
just testing:

- **No artificial lag between sentences.** An earlier design held each
  sentence back by one step so a `BOOKING_CONFIRMED` marker appearing right
  after it could suppress it before it was ever spoken. Measured effect: it
  meant the *last two* sentences of every reply never got delivered
  progressively at all. The shipped version releases each sentence
  immediately. The real, accepted residual risk: if the model ever
  precedes a booking-confirming marker with more than a token or two of
  spoken lead-in, that lead-in may already have been spoken in addition to
  the deterministic confirmation template. Not observed in testing, but
  possible - a minor redundancy, not a correctness bug (the booking itself
  is unaffected either way).
- **The harmony-format glitch (see below) can raise from two different
  places for a streaming request, not just one.** Groq's SDK sometimes
  raises it from the initial stream-creation call (a `BadRequestError`,
  same shape as the non-streaming path), but a real production call showed
  it can ALSO raise later, from *within* iterating the stream itself (a
  plain `groq.APIError` from the SSE parser, with a *differently shaped*
  error body - nested `body["error"]["code"]` vs. flat `body["code"]`).
  The retry logic now catches and detects both shapes. A retry only
  happens if nothing has been queued for delivery yet in that attempt;
  once a sentence has actually been queued (meaning the caller may already
  have heard it), a later failure is treated as final for that turn
  instead, since restarting would regenerate content that might not follow
  on from what was already spoken.

Known, out-of-scope limitation carried over from before this session,
inherited rather than introduced: `call_handler.py`'s own `_calls` dict
(business/turn-number bookkeeping, distinct from the externalized
conversation session) is still an in-memory, per-process dict. Under more
than one gunicorn worker, a `/voice/continue` hit landing on a different
worker than the one that started the turn wouldn't find it. This is the
same class of bug the reliability session fixed for conversation history,
but that fix didn't extend to this dict; still out of scope to fix here.

## Speech-recognition locale: always en-IN, regardless of business language

Twilio's `<Gather>` uses one fixed locale per turn - it cannot auto-detect
or switch language mid-call. Two real bugs were found and fixed across
sessions here:

1. `build_reply_response`'s `<Gather>` (used for every turn after the
   greeting) originally never set a `language` locale at all, silently
   falling back to Twilio's own default (`en-US`) regardless of the
   business's configured language.
2. Fixing (1) by using the business's `language_pref` (e.g. `hi-IN` for a
   Hindi-preferred business) for every turn's locale then caused a *worse*
   bug, confirmed against real call data: `hi-IN` **phonetically
   transliterates clear English speech into unreadable Devanagari-script
   garbage** rather than recognizing it as English (e.g. "I have skin
   problem" was transcribed as nonsense Hindi-script text), which directly
   corrupted the LLM's input and caused fallbacks and repeated clarifying
   questions.

Every turn's speech-recognition locale is now always `en-IN` (Indian
English), which tolerates Hindi/English code-switching far better than
`hi-IN` tolerates English, regardless of `language_pref`. That setting
still controls what language the app itself speaks (greeting and
confirmation templates) - just not what locale Twilio listens with.

## Latency notes (from the performance session)

Real `call_turns` data (small sample - 3 calls, 10 LLM turns) showed nearly
all controllable latency is the Groq call itself (non-LLM overhead per turn
was ~0-1ms in almost every turn). Two things worth knowing before touching
`ai/conversation.py`'s prompt or generation params again:

- **`max_tokens=600` is not the bottleneck and should not be lowered.**
  Measured completion_tokens on successful turns typically land around
  60-150 (`finish_reason='stop'`, well under the cap) - the model stops
  itself long before hitting 600. Earlier testing (MVP session) showed real
  truncation/empty-content failures at lower caps (200-400), so the margin
  is deliberate, not slack.
- **A shorter prompt is not automatically a faster one for this model.**
  Collapsing the system prompt's explicit numbered steps into flowing
  prose was tried and measured (interleaved A/B trials against the
  original, controlling for Groq's own time-varying load): it *increased*
  median completion tokens roughly 2x and made latency worse, despite
  having fewer prompt tokens. `openai/gpt-oss-20b`'s hidden reasoning
  tokens (which dominate latency, per `reasoning_format="hidden"`) appear
  to scale with how explicit/structured the prompt is, not its raw length -
  a numbered step-by-step scaffold seems to reduce how much the model
  needs to reason out the flow itself. The shipped prompt keeps that
  numbered structure and only trims genuinely redundant wording (1643 ->
  1181 chars), which measured as a real, modest improvement (~14% lower
  median latency, ~30% fewer completion tokens in testing) rather than a
  regression.

## What "done" looks like for this MVP

- [x] `/health`, `/voice`, `/voice/handle-input` all work behind the Twilio
      abstraction; no Twilio types leak outside `telephony/twilio_adapter.py`.
- [x] A caller can have a multi-turn Hindi/Bengali/English (code-switching)
      conversation, get offered real open slots, and complete a booking.
- [x] Out-of-scope questions (pricing, medical advice) are declined in-scope.
- [x] A confirmed booking marks the slot booked, writes a `bookings` row and
      a `call_logs` row with the **full** transcript, all in SQLite.
- [x] The owner "notification" logs the full transcript + booking details;
      the customer "notification" logs a short confirmation. Both are mocked
      (logged, not sent) by design for this MVP.
- [x] Groq failures/timeouts/malformed responses never leave dead air or an
      un-ended call; transient failures (timeout/connection/5xx) get a
      small bounded retry with backoff before falling back.
- [x] A slot whose date/time has already passed is never offered or booked
      (explicit IST-aware comparison, not a naive/local-time one).
- [x] Conversation session state survives across separate worker processes
      handling the same call (no more in-memory-dict-per-process).
- [x] A retried webhook for an already-confirmed booking is a no-op, not a
      duplicate booking or a duplicate owner/customer notification.
- [ ] Manual-only, not automatable here: getting a Twilio trial number and
      auth token, running ngrok, and pointing the Twilio console's webhook
      at the tunnel URL (steps 2-6 above) - do this once to test a real
      live call end-to-end.
- [x] Replies are delivered sentence-by-sentence as Groq streams them,
      instead of the caller waiting for the entire response.
- Explicitly **not** built (see product spec): owner dashboard, calendar
  sync, payments, multi-location/staff routing, outbound calling, real
  WhatsApp/SMS sending.

## Task 4 research: Twilio Media Streams (not implemented - recommendation only)

Investigated moving from the current TwiML request/response model (Twilio's
built-in `<Gather>` STT + `<Say>` TTS) to Twilio Media Streams: a
WebSocket carrying raw bidirectional audio, paired with a real-time
streaming STT (e.g. Deepgram) and streaming TTS (e.g. ElevenLabs, Deepgram
Aura, Cartesia).

**Scope of the change: closer to a rewrite of the telephony/audio layer
than an addition.** `TelephonyProvider`'s whole interface
(`parse_incoming_call`, `build_reply_response`, etc.) is built around "one
webhook in, one TwiML document out" - Media Streams is a long-lived
WebSocket carrying raw audio frames and JSON control events, a fundamentally
different shape that doesn't fit that interface at all. It would need a new,
parallel interface, and Flask's synchronous WSGI model isn't a natural fit
for a long-lived low-latency bidirectional stream - this would likely need
a different server (ASGI, or a dedicated WebSocket process alongside the
existing Flask app). The AI/booking "brain" (Groq prompt, marker parsing,
database schema, notifications) would mostly carry over; the telephony
transport, STT, and TTS layers would not.

**Cost**: today's per-minute Twilio voice pricing bundles STT (`<Gather>`)
and TTS (`<Say>`) in. Media Streams typically carries its own additional
per-minute fee on top of base voice minutes, plus separate, additional
bills from a real-time STT vendor and a real-time TTS vendor (premium
conversational TTS providers like ElevenLabs are typically priced well
above Twilio's bundled Polly voices). Net effect: more vendor
relationships to manage, and very likely a higher total per-minute cost -
get current quotes from specific vendors before deciding, since that's
where the real number lives, not a per-minute estimate here.

**Realistic latency upside on top of Tasks 2/3**: Task 1's real data showed
the Groq call itself is the dominant cost (hundreds of ms to a few
seconds), not telephony transport - and Tasks 2/3 already address that
directly (tighter prompt, progressive sentence delivery). Media Streams
would mainly shave latency at the edges: skipping the wait for Twilio's
Gather to finalize a transcript before POSTing to us, and removing the
`<Redirect>` HTTP round-trip between sentences (each measured at roughly
50-100ms in this session's live demo). It would also enable real barge-in
(the caller interrupting the agent mid-sentence), which is a genuinely new
capability TwiML can't do at all, not just a latency win. But as long as
the LLM call remains the dominant cost, the latency upside here is
incremental, not transformative.

**Recommendation: defer.** Two-person team, pilot stage, no paying
businesses onboarded yet. This would mean a substantial rewrite of the
telephony layer, two new paid vendor relationships, and re-validating
voice/transcription quality for Hindi/Bengali/English code-switching
against providers that haven't been tested for that - all to chase a
latency win that's likely secondary to what's already been captured from
the actual measured bottleneck. Revisit once there's real evidence (caller
complaints, measured drop-off correlating with latency, or a concrete want
for barge-in as a feature) that specifically implicates the TwiML
transport model rather than the LLM call.
