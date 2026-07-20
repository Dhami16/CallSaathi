"""
ai/slot_matching.py

Deterministic (non-LLM) date/time extraction and slot matching.

Root-cause fix for the observed conversation-flow bug: previously the
caller's spoken date/time was only ever fed into the LLM's own memory of
the conversation, with nothing in code checking whether it had actually
been resolved to a real slot - so a garbled or fragmented answer could
loop the same clarifying question forever. These functions run directly
against the raw caller speech, are pure and stateless, and never depend on
an LLM "remembering" anything, so they're testable with zero API calls.

Unlike a general-purpose calendar, CallSaathi only ever offers a fixed list
of pre-existing slots (the system prompt already says "never invent
others"), so unlike a richer scheduling assistant, an unmatched date/time
here is reported back to the caller rather than synthesized into a new slot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as calendar_date, time as clock_time, timedelta

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTH_NAMES = "|".join(_MONTHS)
# Romanized Hindi/Bengali variants alongside English - callers code-switch
# freely (see ai/conversation.py's greeting templates), so a date/time
# parser that only understood English would miss most real utterances.
_WEEKDAY_NAMES = {
    0: ["monday", "somvar", "sombar"],
    1: ["tuesday", "mangalvar", "mongolbar"],
    2: ["wednesday", "budhvar", "budhbar"],
    3: ["thursday", "guruvar", "brihaspativar", "brihospotibar"],
    4: ["friday", "shukravar", "shukrobar"],
    5: ["saturday", "shanivar", "shonibar"],
    6: ["sunday", "ravivar", "robibar"],
}
_WORD_HOURS = {
    "twelve": 12, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11,
}
_SCHEDULING_KEYWORDS_RE = re.compile(r"\b(?:appointment|slot|slots|booking|date|time)\b")


@dataclass
class ParsedRequest:
    date: calendar_date | None = None
    time: clock_time | None = None


def parse_datetime_request(
    speech: str, today: calendar_date, assume_scheduling_context: bool = False
) -> ParsedRequest:
    """Parse a caller's free-text date/time preference against `today`.

    `assume_scheduling_context` should be True when the call site already
    knows this turn is answering a date/time question (e.g. the TIMING
    stage) - it relaxes the bare-day-number guard so a fragmented ASR turn
    like just "17." (no surrounding words) still resolves, since the
    calling stage - not this isolated utterance - is what establishes that
    context.
    """
    text = speech.lower().strip()
    return ParsedRequest(
        date=_parse_date(text, today, assume_scheduling_context),
        time=_parse_time(text),
    )


def _parse_date(
    text: str, today: calendar_date, assume_scheduling_context: bool
) -> calendar_date | None:
    iso = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", text)
    month_first = re.search(
        rf"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b", text
    )
    day_first = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAMES})(?:\s+(20\d{{2}}))?\b", text
    )

    scheduling_context = assume_scheduling_context or bool(_SCHEDULING_KEYWORDS_RE.search(text))
    day_only = None
    if scheduling_context and not any((iso, numeric, month_first, day_first)):
        day_only = (
            re.search(r"\b(?:on|at|for)\s+(3[01]|[12]\d|[1-9])(?:st|nd|rd|th)?\b", text)
            or re.search(r"\bdate\s+(?:is\s+)?(3[01]|[12]\d|[1-9])(?:st|nd|rd|th)?\b", text)
            or re.fullmatch(r"(3[01]|[12]\d|[1-9])(?:st|nd|rd|th)?\.?", text)
        )

    try:
        if iso:
            return calendar_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        if numeric:
            return calendar_date(int(numeric.group(3)), int(numeric.group(2)), int(numeric.group(1)))
        if month_first or day_first:
            if month_first:
                month, day, year = _MONTHS[month_first.group(1)], int(month_first.group(2)), month_first.group(3)
            else:
                month, day, year = _MONTHS[day_first.group(2)], int(day_first.group(1)), day_first.group(3)
            result = calendar_date(int(year or today.year), month, day)
            if not year and result < today:
                result = result.replace(year=today.year + 1)
            return result
        if day_only:
            day = int(day_only.group(1))
            result = calendar_date(today.year, today.month, day)
            if result < today:
                if today.month == 12:
                    result = calendar_date(today.year + 1, 1, day)
                else:
                    result = calendar_date(today.year, today.month + 1, day)
            return result
    except ValueError:
        return None

    # "kal" is ambiguous in Hindi (can mean yesterday or tomorrow depending
    # on tense), but in a scheduling context the caller is virtually always
    # asking about the future.
    if "day after tomorrow" in text or re.search(r"\b(parso|parsu|porshu)\b", text):
        return today + timedelta(days=2)
    if "tomorrow" in text or re.search(r"\b(kal|agamikal)\b", text):
        return today + timedelta(days=1)
    if "today" in text or re.search(r"\b(aaj|aajke)\b", text):
        return today

    matches = [
        (match.start(), weekday, name)
        for weekday, names in _WEEKDAY_NAMES.items()
        for name in names
        for match in re.finditer(rf"\b{name}\b", text)
    ]
    if matches:
        _, weekday, name = matches[-1] if len(matches) > 1 else matches[0]
        distance = (weekday - today.weekday()) % 7
        if f"next {name}" in text and distance == 0:
            distance = 7
        return today + timedelta(days=distance)

    return None


def _parse_time(text: str) -> clock_time | None:
    time_match = (
        re.search(r"\b(?:to|instead(?:\s+at)?|rather\s+at)\s+(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", text)
        or re.search(r"\b(?:at|around|from|by)\s+(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", text)
        or re.search(r"\b(\d{1,2})[:.](\d{2})\s*(a\.?m\.?|p\.?m\.?)?\b", text)
        or re.search(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)\b", text)
        # Hindi "X baje" / Bengali "X-ta"/"X tay" - "at X o'clock", no am/pm
        # marker, so the trailing empty group keeps the 3-group convention
        # above (meridiem = "") rather than needing separate handling.
        or re.search(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(?:baje|bajkar)()\b", text)
        or re.search(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(?:tay|ta)()\b", text)
    )
    if time_match:
        groups = time_match.groups()
        hour = int(groups[0])
        minute = int(groups[1]) if len(groups) > 2 and groups[1] else 0
        meridiem = (groups[-1] or "").replace(".", "")
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return clock_time(hour, minute)
        return None

    word_match = re.search(r"\b(?:at|around|from|by)\s+(" + "|".join(_WORD_HOURS) + r")\b", text)
    if word_match:
        hour = _WORD_HOURS[word_match.group(1)]
        if "pm" in text and hour < 12:
            hour += 12
        return clock_time(hour, 30 if "thirty" in text else 0)
    return None


def _slot_date(slot: dict) -> calendar_date:
    return calendar_date.fromisoformat(slot["date"])


def _slot_time(slot: dict) -> clock_time:
    hour, minute = slot["time"].split(":")
    return clock_time(int(hour), int(minute))


def slots_on_date(offered_slots: list[dict], requested_date: calendar_date) -> list[dict]:
    """Every offered slot that falls on `requested_date`, in offered order."""
    return [slot for slot in offered_slots if _slot_date(slot) == requested_date]


def match_requested_slot(
    speech: str,
    offered_slots: list[dict],
    today: calendar_date,
    assume_scheduling_context: bool = False,
) -> dict | None:
    """Resolve the caller's speech to exactly one of `offered_slots`, or None
    if it names no slot, or names one ambiguously (multiple/zero matches)."""
    text = speech.lower().strip()

    ordinal_words = {"first": 0, "second": 1, "third": 2}
    for word, index in ordinal_words.items():
        if re.search(rf"\b{word}\b", text) and index < len(offered_slots):
            return offered_slots[index]

    numeric_choice = re.search(r"(?:option\s+([123])|slot\s+([123])|\b([123])(?:st|nd|rd)\b)", text)
    if numeric_choice:
        index = int(next(g for g in numeric_choice.groups() if g)) - 1
        if 0 <= index < len(offered_slots):
            return offered_slots[index]

    parsed = parse_datetime_request(speech, today, assume_scheduling_context)
    if not parsed.date and not parsed.time:
        return None

    candidates = offered_slots
    if parsed.date:
        candidates = [slot for slot in candidates if _slot_date(slot) == parsed.date]
    if parsed.time:
        candidates = [slot for slot in candidates if _slot_time(slot) == parsed.time]

    if len(candidates) == 1:
        return candidates[0]
    return None
