"""Data access for businesses, slots, bookings and call logs.

Each method opens its own short-lived connection rather than holding one
long-lived connection on the instance - sqlite3 connections aren't safe to
share across Flask's request threads, and per-call connections are cheap
enough at this scale.
"""
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from booking.db import get_connection

# All businesses this app serves are in India (see product scope) - slot
# date/time strings are interpreted as this timezone, and "now" for
# excluding past slots is computed in this timezone too. Comparing a naive
# datetime.now() (whatever timezone the host OS/process happens to be in)
# against IST business hours is exactly the bug this file used to have.
IST = ZoneInfo("Asia/Kolkata")


class SlotUnavailableError(Exception):
    """Raised when a slot was booked by someone else between being offered
    and being confirmed."""


def _slot_datetime(slot_row) -> datetime:
    naive = datetime.strptime(f"{slot_row['date']} {slot_row['time']}", "%Y-%m-%d %H:%M")
    return naive.replace(tzinfo=IST)


class BookingRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_business_by_phone(self, phone_number: str) -> Optional[dict]:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM businesses WHERE phone_number = ? AND active = 1", (phone_number,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_available_slots(self, business_id: int, limit: int = 3, now: Optional[datetime] = None) -> list[dict]:
        """Returns up to `limit` unbooked slots that are strictly in the
        future relative to `now` (defaults to the real current time in IST;
        tests pass a fixed `now` for determinism).

        The future-vs-past filter is applied in Python rather than SQL:
        date/time are stored as separate TEXT columns, and slot counts per
        business are small (a handful), so fetching all unbooked rows and
        filtering here is simpler and less error-prone than building a
        string-comparable SQL datetime expression.
        """
        now = now if now is not None else datetime.now(IST)

        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                """SELECT * FROM slots
                   WHERE business_id = ? AND is_booked = 0
                   ORDER BY date, time""",
                (business_id,),
            ).fetchall()
        finally:
            conn.close()

        future_slots = [dict(row) for row in rows if _slot_datetime(row) > now]
        return future_slots[:limit]

    def book_slot(
        self,
        business_id: int,
        slot_id: int,
        customer_name: str,
        customer_phone: str,
        reason: str,
        call_id: str,
    ) -> tuple[int, bool]:
        """Marks the slot booked and inserts the booking row atomically.
        Raises SlotUnavailableError if the slot was already taken by a
        different call.

        Idempotent per call_id: if this call_id already has a booking (e.g.
        Twilio retried the webhook after a network blip and we're
        processing the same confirmed booking a second time), returns the
        existing booking instead of creating a duplicate or touching the
        slot again. Returns (booking_id, created) - `created` is False on
        such a replay, so callers can skip re-sending notifications.
        """
        conn = get_connection(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")

            existing = conn.execute("SELECT id FROM bookings WHERE call_id = ?", (call_id,)).fetchone()
            if existing is not None:
                conn.rollback()
                return existing["id"], False

            cursor = conn.execute(
                "UPDATE slots SET is_booked = 1 WHERE id = ? AND business_id = ? AND is_booked = 0",
                (slot_id, business_id),
            )
            if cursor.rowcount == 0:
                conn.rollback()
                raise SlotUnavailableError(f"Slot {slot_id} is no longer available")

            cursor = conn.execute(
                """INSERT INTO bookings (business_id, slot_id, customer_name, customer_phone, reason, call_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (business_id, slot_id, customer_name, customer_phone, reason, call_id),
            )
            conn.commit()
            return cursor.lastrowid, True
        finally:
            conn.close()

    def create_call_log(
        self,
        business_id: int,
        caller_phone: str,
        transcript: str,
        outcome: str,
        duration_seconds: int,
    ) -> int:
        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute(
                """INSERT INTO call_logs (business_id, caller_phone, transcript, outcome, duration_seconds)
                   VALUES (?, ?, ?, ?, ?)""",
                (business_id, caller_phone, transcript, outcome, duration_seconds),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def create_call_turn(
        self,
        call_id: str,
        turn_number: int,
        total_latency_ms: int,
        llm_latency_ms: Optional[int] = None,
        stt_latency_ms: Optional[int] = None,
        tts_latency_ms: Optional[int] = None,
        transcript_in: Optional[str] = None,
        response_out: Optional[str] = None,
    ) -> int:
        """Records one conversational turn's latency breakdown. Written
        immediately per turn (see booking/db.py's call_turns comment for why
        stt/tts latency are always None for now)."""
        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute(
                """INSERT INTO call_turns
                   (call_id, turn_number, stt_latency_ms, llm_latency_ms, tts_latency_ms,
                    total_latency_ms, transcript_in, response_out)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    call_id,
                    turn_number,
                    stt_latency_ms,
                    llm_latency_ms,
                    tts_latency_ms,
                    total_latency_ms,
                    transcript_in,
                    response_out,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_slot(self, slot_id: int) -> Optional[dict]:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute("SELECT * FROM slots WHERE id = ?", (slot_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
