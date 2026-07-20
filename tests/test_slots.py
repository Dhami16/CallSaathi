"""Bug 1 regression tests: a slot whose date/time has already passed must
never be offered to a caller. The root cause was that
BookingRepository.get_available_slots had NO time-based filter at all
(only is_booked=0) - confirmed against the real production database, where
a slot for "2026-07-13 16:00" was offered and booked during a call at
2026-07-13 22:45 IST, ~6.5 hours after the slot's time had passed.

Run with: venv/Scripts/python -m pytest -q tests/test_slots.py
"""
from datetime import datetime

import pytest

from booking.db import get_connection, init_db
from booking.repository import IST, BookingRepository, SlotUnavailableError


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test_slots.db")
    init_db(path)
    conn = get_connection(path)
    conn.execute(
        "INSERT INTO businesses (id, name, vertical, phone_number, language_pref, active) "
        "VALUES (1, 'Test Clinic', 'clinic', '+10000000000', 'english', 1)"
    )
    conn.commit()
    conn.close()
    return path


def _insert_slot(db_path, slot_id, date, time_str, is_booked=0):
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO slots (id, business_id, date, time, is_booked) VALUES (?, 1, ?, ?, ?)",
        (slot_id, date, time_str, is_booked),
    )
    conn.commit()
    conn.close()


def test_past_slots_excluded_future_slots_kept(db_path):
    fixed_now = datetime(2026, 7, 13, 17, 15, tzinfo=IST)

    _insert_slot(db_path, 1, "2026-07-13", "09:00")  # past: today, earlier time
    _insert_slot(db_path, 2, "2026-07-12", "23:59")  # past: yesterday
    _insert_slot(db_path, 3, "2026-07-13", "17:16")  # future: today, one minute later
    _insert_slot(db_path, 4, "2026-07-13", "20:00")  # future: today, later
    _insert_slot(db_path, 5, "2026-07-14", "09:00")  # future: tomorrow

    repository = BookingRepository(db_path)
    slots = repository.get_available_slots(business_id=1, limit=10, now=fixed_now)

    assert [s["id"] for s in slots] == [3, 4, 5]  # ordered by date, time; past ones excluded


def test_regression_same_day_past_slot_never_offered_or_booked(db_path):
    """Replays the exact real-call scenario: a call at 2026-07-13 22:45 IST
    must never be offered a same-day slot at 16:00 (~6.5 hours already
    past), even though it shares "today"'s date with the call itself."""
    call_time = datetime(2026, 7, 13, 22, 45, tzinfo=IST)

    _insert_slot(db_path, 1, "2026-07-13", "10:00")  # past
    _insert_slot(db_path, 2, "2026-07-13", "15:30")  # past
    _insert_slot(db_path, 3, "2026-07-13", "11:00", is_booked=1)  # already booked (unrelated - see Bug 2)
    _insert_slot(db_path, 4, "2026-07-13", "16:00")  # the exact slot from the real production bug
    _insert_slot(db_path, 5, "2026-07-14", "09:30")  # genuinely in the future

    repository = BookingRepository(db_path)
    slots = repository.get_available_slots(business_id=1, limit=3, now=call_time)

    offered_ids = [s["id"] for s in slots]
    assert 4 not in offered_ids, "the exact past slot from the real production bug must never be offered again"
    assert offered_ids == [5]


def test_get_available_slots_respects_limit_after_filtering_past_slots(db_path):
    fixed_now = datetime(2026, 7, 13, 8, 0, tzinfo=IST)
    for i, hour in enumerate(["09", "10", "11", "12", "13"], start=1):
        _insert_slot(db_path, i, "2026-07-13", f"{hour}:00")

    repository = BookingRepository(db_path)
    slots = repository.get_available_slots(business_id=1, limit=3, now=fixed_now)
    assert len(slots) == 3
    assert [s["id"] for s in slots] == [1, 2, 3]


def test_booked_slot_still_excluded_regardless_of_time(db_path):
    fixed_now = datetime(2026, 7, 13, 8, 0, tzinfo=IST)
    _insert_slot(db_path, 1, "2026-07-14", "10:00", is_booked=1)  # future but already booked

    repository = BookingRepository(db_path)
    slots = repository.get_available_slots(business_id=1, limit=10, now=fixed_now)
    assert slots == []

    with pytest.raises(SlotUnavailableError):
        repository.book_slot(1, 1, "Someone", "+911111111111", "reason", call_id="CALL-X")
