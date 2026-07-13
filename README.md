# CallSaathi

An AI voice agent that answers missed calls for small local businesses in
India (clinics, salons, coaching centers), converses in Hindi/Bengali/English,
books an appointment from a fixed slot list, and notifies the business owner
with the full call transcript.

## What this MVP does

- Answers an inbound call forwarded to a Twilio number (`POST /voice`).
- Greets the caller, asks their reason for calling and preferred time, offers
  up to 3 open slots, and books one once the caller agrees and gives their
  name (`POST /voice/handle-input`, looped via Twilio's `<Gather>`).
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
                       confirmation via a text marker, retry + error fallback.
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
   - `TWILIO_PHONE_NUMBER` - the Twilio number you got above. This must match
     a `businesses.phone_number` row (the seed script uses this env var).

3. **Seed demo data** (one business + a handful of open slots):
   ```
   venv\Scripts\python seed_data.py
   ```

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
be lowered back down.

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
- Explicitly **not** built (see product spec): owner dashboard, calendar
  sync, payments, multi-location/staff routing, outbound calling, real
  WhatsApp/SMS sending.
