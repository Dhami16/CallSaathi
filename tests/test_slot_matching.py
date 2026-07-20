"""Pure unit tests for ai/slot_matching.py - no API key, no network, always run.

Cases below are drawn directly from the real bug transcript that motivated
this module: a caller answering "preferred date and time" with "tomorrow",
"Friday", "17", and "17 July 2026" while the old code never matched any of
them to a real slot and just re-asked the same question forever.

Run with: venv\\Scripts\\python -m pytest -s tests/test_slot_matching.py
"""
from datetime import date

from ai.slot_matching import match_requested_slot, parse_datetime_request, slots_on_date

TODAY = date(2026, 7, 15)

SLOTS = [
    {"id": 1, "date": "2026-07-17", "time": "10:00"},
    {"id": 2, "date": "2026-07-17", "time": "15:30"},
    {"id": 3, "date": "2026-07-18", "time": "11:00"},
]


def test_parses_tomorrow():
    parsed = parse_datetime_request("Uh, yes, uh, can we book an appointment for tomorrow?", TODAY)
    assert parsed.date == date(2026, 7, 16)


def test_parses_hindi_kal_as_tomorrow():
    # Callers code-switch freely - "kal subah" ("tomorrow morning") must
    # resolve the same way the English word "tomorrow" does.
    parsed = parse_datetime_request("Kal subah mein koi time ho to accha rahega", TODAY)
    assert parsed.date == date(2026, 7, 16)


def test_parses_hindi_time_particle_baje():
    parsed = parse_datetime_request("Haan 10 baje wala theek hai", TODAY, assume_scheduling_context=True)
    assert parsed.time is not None
    assert parsed.time.hour == 10 and parsed.time.minute == 0


def test_parses_bengali_time_particle_tay():
    parsed = parse_datetime_request("10 tay hobe", TODAY, assume_scheduling_context=True)
    assert parsed.time is not None
    assert parsed.time.hour == 10 and parsed.time.minute == 0


def test_parses_weekday_name_to_upcoming_occurrence():
    parsed = parse_datetime_request("Can you book appointment at Friday?", TODAY)
    assert parsed.date is not None
    assert parsed.date.strftime("%A").lower() == "friday"
    assert parsed.date >= TODAY


def test_parses_explicit_month_day_year():
    parsed = parse_datetime_request("Uh, 17 July 2026. Abey,", TODAY)
    assert parsed.date == date(2026, 7, 17)


def test_bare_day_number_needs_scheduling_context():
    # "17." in isolation, with no surrounding scheduling words and no
    # assumed context, is genuinely ambiguous (could be an age, a count...).
    assert parse_datetime_request("17.", TODAY).date is None

    # But when the call site already knows this turn answers a date
    # question (assume_scheduling_context=True), it resolves - this is the
    # actual fix for the fragmented-ASR-turn case in the bug transcript.
    parsed = parse_datetime_request("17.", TODAY, assume_scheduling_context=True)
    assert parsed.date == date(2026, 7, 17)


def test_misheard_weekday_does_not_falsely_match():
    # "Witness day" (ASR mishearing "Wednesday") must not silently resolve
    # to some other date - it should stay unparsed so the conversation layer
    # asks again rather than booking the wrong day.
    assert parse_datetime_request("Witness day.", TODAY, assume_scheduling_context=True).date is None


def test_match_requested_slot_resolves_unambiguous_date_to_single_slot():
    slot = match_requested_slot("17 July 2026", SLOTS, TODAY)
    assert slot is None  # two slots exist on 2026-07-17 - date alone is ambiguous


def test_match_requested_slot_resolves_date_and_time_together():
    slot = match_requested_slot("17 July 2026 at 10am", SLOTS, TODAY)
    assert slot == SLOTS[0]


def test_match_requested_slot_by_ordinal_word():
    assert match_requested_slot("the second one please", SLOTS, TODAY) == SLOTS[1]


def test_match_requested_slot_by_numeric_option():
    assert match_requested_slot("option 3", SLOTS, TODAY) == SLOTS[2]


def test_match_requested_slot_returns_none_for_unmatched_date():
    # 2026-07-20 has no offered slots at all.
    assert match_requested_slot("20 July 2026", SLOTS, TODAY) is None


def test_slots_on_date_lists_all_matches_for_clarification():
    assert slots_on_date(SLOTS, date(2026, 7, 17)) == [SLOTS[0], SLOTS[1]]
    assert slots_on_date(SLOTS, date(2026, 7, 20)) == []
