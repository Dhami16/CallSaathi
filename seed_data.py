"""Idempotent data import from data/businesses.csv and data/slots.csv, so
the app and tests have something to book against without any manual setup.

Run directly: `python seed_data.py`
"""
import csv
import datetime
import os
from pathlib import Path

import config  # noqa: F401 - importing this loads .env before we read os.getenv below
from booking.db import get_connection, init_db

DATA_DIR = Path(__file__).parent / "data"
BUSINESSES_CSV = DATA_DIR / "businesses.csv"
SLOTS_CSV = DATA_DIR / "slots.csv"

# The one row whose phone number must match whatever Twilio number the
# current environment is actually wired up to - see README's .env setup step.
DEMO_BUSINESS_NAME = "Radiant Skin Clinic"

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _next_occurrence(weekday_name: str, today: datetime.date) -> datetime.date:
    """Next future date (never today) matching the given weekday name."""
    target = _WEEKDAYS.index(weekday_name.strip().lower())
    days_ahead = (target - today.weekday() - 1) % 7 + 1
    return today + datetime.timedelta(days=days_ahead)


def _to_24_hour(time_12h: str) -> str:
    return datetime.datetime.strptime(time_12h.strip(), "%I:%M %p").strftime("%H:%M")


def seed(db_path: str) -> int:
    """Imports businesses.csv and slots.csv. Returns the first business id."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        first_business_id = None
        name_to_id = {}
        newly_created_ids = set()

        with open(BUSINESSES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row["name"]
                phone_number = row["phone_number"]
                if name == DEMO_BUSINESS_NAME:
                    phone_number = os.getenv("TWILIO_PHONE_NUMBER", phone_number)

                existing = conn.execute(
                    "SELECT id FROM businesses WHERE phone_number = ?", (phone_number,)
                ).fetchone()
                if existing:
                    business_id = existing["id"]
                else:
                    cursor = conn.execute(
                        """INSERT INTO businesses (name, vertical, phone_number, language_pref, active)
                           VALUES (?, ?, ?, ?, ?)""",
                        (name, row["vertical"], phone_number, row["language_pref"], int(row["active"])),
                    )
                    business_id = cursor.lastrowid
                    newly_created_ids.add(business_id)

                name_to_id[name] = business_id
                if first_business_id is None:
                    first_business_id = business_id

        # Slots are only seeded for businesses created in this run - an
        # already-existing business was seeded (and possibly booked into)
        # by an earlier run, so re-adding its starter slots would duplicate
        # them.
        today = datetime.date.today()
        with open(SLOTS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                business_id = name_to_id.get(row["business_name"])
                if business_id is None or business_id not in newly_created_ids:
                    continue

                slot_date = _next_occurrence(row["days"], today).isoformat()
                conn.execute(
                    "INSERT INTO slots (business_id, date, time, is_booked) VALUES (?, ?, ?, 0)",
                    (business_id, slot_date, _to_24_hour(row["time"])),
                )

        conn.commit()
        return first_business_id
    finally:
        conn.close()


if __name__ == "__main__":
    from config import load_config

    config = load_config()
    business_id = seed(config.database_path)
    print(f"Seeded businesses from {BUSINESSES_CSV} and slots from {SLOTS_CSV} at {config.database_path}")
