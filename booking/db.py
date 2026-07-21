"""SQLite schema and connection helper. One file, no ORM - the schema is
small and fixed, so plain sqlite3 keeps this readable for a small team."""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS businesses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    vertical TEXT NOT NULL,
    phone_number TEXT NOT NULL UNIQUE,
    language_pref TEXT NOT NULL DEFAULT 'english',
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    is_booked INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_slots_business ON slots(business_id, is_booked);

-- call_id links a booking back to the call that created it, so a retried
-- Twilio webhook (e.g. after a network blip) can be recognized as the same
-- booking attempt instead of creating a duplicate - see
-- BookingRepository.book_slot(). Nullable + a partial unique index (rather
-- than NOT NULL UNIQUE) so pre-existing rows from before this column
-- existed remain valid.
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    slot_id INTEGER NOT NULL REFERENCES slots(id),
    customer_name TEXT NOT NULL,
    customer_phone TEXT NOT NULL,
    reason TEXT NOT NULL,
    call_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS call_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id INTEGER NOT NULL REFERENCES businesses(id),
    caller_phone TEXT NOT NULL,
    transcript TEXT NOT NULL,
    outcome TEXT NOT NULL,
    duration_seconds INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Granular per-turn timing, separate from call_logs (whole-call outcome).
-- One row is written as each turn happens, not batched at call end, so a
-- crashed call still leaves partial data behind.
--
-- stt_latency_ms and tts_latency_ms are always NULL for now: Twilio's
-- <Gather> does speech-to-text before it ever calls our webhook, and
-- text-to-speech happens after we return TwiML, in both cases inside
-- Twilio's infrastructure with no timing handed back to us. Only
-- llm_latency_ms (our Groq call) and total_latency_ms (our whole
-- webhook-handling time) are things we can actually measure.
CREATE TABLE IF NOT EXISTS call_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    stt_latency_ms INTEGER,
    llm_latency_ms INTEGER,
    tts_latency_ms INTEGER,
    total_latency_ms INTEGER NOT NULL,
    transcript_in TEXT,
    response_out TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_call_turns_call_id ON call_turns(call_id);

-- Externalized conversation session state (replaces an in-memory dict that
-- silently broke across gunicorn workers - see booking/session_store.py).
-- session_data is a JSON blob; expires_at bounds how long an abandoned/
-- crashed call's state sticks around.
CREATE TABLE IF NOT EXISTS call_sessions (
    call_id TEXT PRIMARY KEY,
    session_data TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_call_sessions_expires ON call_sessions(expires_at);

-- Real bug found in production: a Twilio webhook retry for the same turn
-- (e.g. a slow Groq response tripping Twilio's own retry-on-timeout
-- behavior) landing on a different gunicorn worker used to read the same
-- pre-increment turn count and spawn its OWN independent streaming Groq
-- call for the identical turn, appending into the same shared sentence
-- list as the original - the caller heard both completions' sentences
-- interleaved in real time. The PRIMARY KEY here makes "claim this
-- (call_id, turn_number)" an atomic insert: only the first request to try
-- succeeds, so only one worker ever streams a given turn - see
-- SessionStore.claim_turn.
CREATE TABLE IF NOT EXISTS stream_turn_claims (
    call_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    claimed_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (call_id, turn_number)
);
"""


def _migrate_add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Handles upgrading a pre-existing DB file created before `column`
    existed. CREATE TABLE IF NOT EXISTS alone won't add a column to an
    already-existing table, so this is needed alongside the schema above."""
    existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate_add_column_if_missing(conn, "bookings", "call_id", "TEXT")
        # Created after the migration above (not inside SCHEMA's
        # executescript) because on a pre-existing DB file, bookings.call_id
        # wouldn't exist yet at the point SCHEMA runs - CREATE TABLE IF NOT
        # EXISTS is a no-op for an already-existing table.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_bookings_call_id ON bookings(call_id) WHERE call_id IS NOT NULL"
        )
        conn.commit()
    finally:
        conn.close()
