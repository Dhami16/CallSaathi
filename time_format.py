"""Shared 24-hour -> 12-hour spoken time formatting.

Both consumers need the exact same conversion for the exact same underlying
reason: raw 24-hour "HH:MM" slot times force whoever/whatever reads them to
do the AM/PM mental arithmetic themselves. Real production bugs from this:
Twilio's speech recognition misheard "9 AM" as "99 AM" with no spoken-form
hint to bias toward (see call_handler.py's Gather `hints`), and gpt-oss-20b
garbled a 13:00 slot into "3 PM, at 13" when asked to verbalize raw 24-hour
times itself (see ai/conversation.py's system prompt slot listing and
booking confirmation). Doing the conversion once, here, removes that burden
from both Twilio's STT biasing and the LLM's own arithmetic.
"""


def format_time_12h(time_24h: str) -> str:
    """Converts a 24-hour "HH:MM" string into 12-hour spoken form, e.g.
    "16:00" -> "4 PM", "09:30" -> "9:30 AM", "00:00" -> "12 AM"."""
    hour, _, minute = time_24h.partition(":")
    hour, minute = int(hour), int(minute)
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return f"{hour_12} {period}" if minute == 0 else f"{hour_12}:{minute:02d} {period}"
