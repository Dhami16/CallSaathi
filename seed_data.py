"""Idempotent demo data: one business + a handful of open slots, so the app
and tests have something to book against without any manual setup.

Run directly: `python seed_data.py`
"""
import datetime
import os

from booking.db import get_connection, init_db

DEMO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+15005550006")


def seed(db_path: str) -> int:
    """Returns the demo business id (creates it if missing)."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id FROM businesses WHERE phone_number = ?", (DEMO_PHONE_NUMBER,)
        ).fetchone()
        if row:
            return row["id"]

        cursor = conn.execute(
            """INSERT INTO businesses (name, vertical, phone_number, language_pref, active)
               VALUES (?, ?, ?, ?, 1)""",
            ("Radiant Skin Clinic", "clinic", DEMO_PHONE_NUMBER, "hindi"),
        )
        business_id = cursor.lastrowid

        today = datetime.date.today()
        for day_offset, time_str in [(1, "10:00"), (1, "15:30"), (2, "11:00"), (2, "16:00"), (3, "09:30")]:
            slot_date = (today + datetime.timedelta(days=day_offset)).isoformat()
            conn.execute(
                "INSERT INTO slots (business_id, date, time, is_booked) VALUES (?, ?, ?, 0)",
                (business_id, slot_date, time_str),
            )

        conn.commit()
        return business_id
    finally:
        conn.close()


if __name__ == "__main__":
    from config import load_config

    config = load_config()
    business_id = seed(config.database_path)
    print(f"Seeded demo business id={business_id} at {config.database_path}")
