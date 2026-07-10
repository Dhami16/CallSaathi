"""Data access for businesses, slots, bookings and call logs.

Each method opens its own short-lived connection rather than holding one
long-lived connection on the instance - sqlite3 connections aren't safe to
share across Flask's request threads, and per-call connections are cheap
enough at this scale.
"""
import logging
from typing import Optional

from booking.db import get_connection

logger = logging.getLogger(__name__)


class SlotUnavailableError(Exception):
    """Raised when a slot was booked by someone else between being offered
    and being confirmed."""


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

    def get_available_slots(self, business_id: int, limit: int = 3) -> list[dict]:
        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                """SELECT * FROM slots
                   WHERE business_id = ? AND is_booked = 0
                   ORDER BY date, time
                   LIMIT ?""",
                (business_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def book_slot(
        self,
        business_id: int,
        slot_id: int,
        customer_name: str,
        customer_phone: str,
        reason: str,
    ) -> int:
        """Marks the slot booked and inserts the booking row atomically.
        Raises SlotUnavailableError if the slot was already taken."""
        conn = get_connection(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE slots SET is_booked = 1 WHERE id = ? AND business_id = ? AND is_booked = 0",
                (slot_id, business_id),
            )
            if cursor.rowcount == 0:
                conn.rollback()
                raise SlotUnavailableError(f"Slot {slot_id} is no longer available")

            cursor = conn.execute(
                """INSERT INTO bookings (business_id, slot_id, customer_name, customer_phone, reason)
                   VALUES (?, ?, ?, ?, ?)""",
                (business_id, slot_id, customer_name, customer_phone, reason),
            )
            conn.commit()
            return cursor.lastrowid
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

    def get_slot(self, slot_id: int) -> Optional[dict]:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute("SELECT * FROM slots WHERE id = ?", (slot_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
