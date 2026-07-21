"""Externalized conversation session state.

Why this exists: ai/conversation.py used to hold session state (message
history, offered slots, turn count) in a plain Python dict keyed by
call_id, living only in the process's memory. That's fine for a single
process, but breaks silently the moment the app runs with more than one
gunicorn worker - Twilio's sequential webhook hits for one call (the
initial /voice, then each /voice/handle-input turn) can land on different
worker processes, each with its own separate memory. The second request
just wouldn't find the first request's history, and the conversation would
restart from nothing mid-call with no error raised anywhere.

SQLite vs Redis: Redis is the more scalable, more "correct" long-term
choice for this kind of ephemeral keyed store (native TTL, no
single-writer file lock), but it's a new piece of infrastructure a
two-person team would need to run (Docker locally) or pay for/manage
(hosted free tier) even at pilot stage. SQLite reuses the DB file already
in place, with zero new infrastructure - the write pattern here is tiny,
fast, one-row-per-turn upserts, not sustained high-throughput, so SQLite's
serialized writes aren't expected to be a practical bottleneck until well
past pilot scale. Recommendation: SQLite now, revisit Redis if/when call
volume actually demands it - this is why SessionStore is an interface, so
that swap wouldn't touch ai/conversation.py or call_handler.py at all.
"""
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional

from booking.db import get_connection

DEFAULT_TTL_SECONDS = 600  # 10 minutes - generous for one phone call, bounds abandoned-call storage
_CLAIM_MAX_AGE_SECONDS = 3600  # bounds stream_turn_claims growth for calls that never cleanly end_session


class SessionStore(ABC):
    @abstractmethod
    def get(self, call_id: str) -> Optional[dict]:
        """Returns the session dict, or None if missing/expired."""

    @abstractmethod
    def set(self, call_id: str, session: dict, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        """Stores (or replaces) the session, refreshing its expiry."""

    @abstractmethod
    def delete(self, call_id: str) -> None:
        """Removes the session immediately (call ended normally)."""

    @abstractmethod
    def claim_turn(self, call_id: str, turn_number: int) -> bool:
        """Atomically claims (call_id, turn_number). Returns True the first
        time this is called for a given turn - the caller should proceed to
        start a new streaming worker. Returns False on every subsequent call
        for the SAME turn (a concurrent or retried webhook hit for the exact
        same turn) - the caller must NOT start a second worker, since two
        independent LLM completions for the same turn would both append
        their own sentences into the same shared stream state, interleaving
        two overlapping replies into what the caller hears."""

    @abstractmethod
    def clear_turn_claims(self, call_id: str) -> None:
        """Removes all claimed turns for a call (call ended normally)."""


def _now_utc_naive() -> datetime:
    # Naive-but-UTC to match SQLite's own datetime('now'), which is UTC.
    # This store's expiry math is a separate concern from Bug 1's IST slot
    # comparison - session TTLs aren't user-facing business hours, just an
    # internal bookkeeping window, so UTC is the natural choice here.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _format(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class SQLiteSessionStore(SessionStore):
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get(self, call_id: str) -> Optional[dict]:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT session_data, expires_at FROM call_sessions WHERE call_id = ?", (call_id,)
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] < _format(_now_utc_naive()):
                conn.execute("DELETE FROM call_sessions WHERE call_id = ?", (call_id,))
                conn.commit()
                return None
            return json.loads(row["session_data"])
        finally:
            conn.close()

    def set(self, call_id: str, session: dict, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        expires_at = _format(_now_utc_naive() + timedelta(seconds=ttl_seconds))
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """INSERT INTO call_sessions (call_id, session_data, expires_at, updated_at)
                   VALUES (?, ?, ?, datetime('now'))
                   ON CONFLICT(call_id) DO UPDATE SET
                       session_data = excluded.session_data,
                       expires_at = excluded.expires_at,
                       updated_at = excluded.updated_at""",
                (call_id, json.dumps(session), expires_at),
            )
            # Opportunistic cleanup instead of a background job/cron (which
            # would be new infrastructure/architecture, out of scope here) -
            # cheap, indexed delete, piggybacked on writes that already happen.
            conn.execute("DELETE FROM call_sessions WHERE expires_at < ?", (_format(_now_utc_naive()),))
            conn.commit()
        finally:
            conn.close()

    def delete(self, call_id: str) -> None:
        conn = get_connection(self.db_path)
        try:
            conn.execute("DELETE FROM call_sessions WHERE call_id = ?", (call_id,))
            conn.commit()
        finally:
            conn.close()

    def claim_turn(self, call_id: str, turn_number: int) -> bool:
        conn = get_connection(self.db_path)
        try:
            try:
                conn.execute(
                    "INSERT INTO stream_turn_claims (call_id, turn_number) VALUES (?, ?)",
                    (call_id, turn_number),
                )
                claimed = True
            except sqlite3.IntegrityError:
                claimed = False
            # Opportunistic cleanup, same pattern as set()'s expired-session
            # sweep - piggybacked on writes that already happen rather than
            # a separate cron/background job.
            stale_before = _format(_now_utc_naive() - timedelta(seconds=_CLAIM_MAX_AGE_SECONDS))
            conn.execute("DELETE FROM stream_turn_claims WHERE claimed_at < ?", (stale_before,))
            conn.commit()
            return claimed
        finally:
            conn.close()

    def clear_turn_claims(self, call_id: str) -> None:
        conn = get_connection(self.db_path)
        try:
            conn.execute("DELETE FROM stream_turn_claims WHERE call_id = ?", (call_id,))
            conn.commit()
        finally:
            conn.close()
