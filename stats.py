"""Aggregate queries backing GET /internal/stats. A single
query-and-render module by design - this is an internal debugging view for
a two-person team, not a dashboard product.

"Total calls" is counted from call_turns (turn_number = 0, i.e. every call
that got at least a greeting), not call_logs: call_logs is only ever
written when a call ends (booked, or a fallback/slot-conflict outcome from
call_handler.py), so it can't answer "how many calls happened" on its own,
only "how many calls ended in each outcome." call_turns, added in this same
change, has one row per call from the very first turn, making it the right
source for the denominator.
"""
from datetime import datetime, timedelta, timezone

from booking.db import get_connection

_NOTES = [
    "avg_stt_latency_ms and avg_tts_latency_ms are always null: Twilio's "
    "<Gather>-based telephony does speech-to-text before our webhook is "
    "called and text-to-speech after we respond, so neither is visible to "
    "this app to measure (see booking/db.py's call_turns comment).",
    "'calls' counts distinct call_id values from call_turns where "
    "turn_number = 0 (i.e. reached at least the greeting), not call_logs - "
    "call_logs only records how a call ended, not that it happened.",
]


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _period_stats(conn, since: str) -> dict:
    calls = conn.execute(
        "SELECT COUNT(DISTINCT call_id) c FROM call_turns WHERE turn_number = 0 AND created_at >= ?",
        (since,),
    ).fetchone()["c"]
    bookings = conn.execute(
        "SELECT COUNT(*) c FROM call_logs WHERE outcome = 'booked' AND created_at >= ?",
        (since,),
    ).fetchone()["c"]
    fallback_or_error = conn.execute(
        "SELECT COUNT(*) c FROM call_logs WHERE outcome != 'booked' AND created_at >= ?",
        (since,),
    ).fetchone()["c"]
    return {
        "calls": calls,
        "bookings": bookings,
        "fallback_or_error": fallback_or_error,
        "success_rate": round(bookings / calls, 3) if calls else None,
    }


def _avg_latency_ms(conn, since: str) -> dict:
    row = conn.execute(
        """SELECT AVG(total_latency_ms) avg_total, AVG(llm_latency_ms) avg_llm,
                  AVG(stt_latency_ms) avg_stt, AVG(tts_latency_ms) avg_tts
           FROM call_turns WHERE created_at >= ?""",
        (since,),
    ).fetchone()

    def _round(value):
        return round(value, 1) if value is not None else None

    return {
        "avg_total_latency_ms": _round(row["avg_total"]),
        "avg_llm_latency_ms": _round(row["avg_llm"]),
        "avg_stt_latency_ms": _round(row["avg_stt"]),
        "avg_tts_latency_ms": _round(row["avg_tts"]),
    }


def compute_stats(db_path: str) -> dict:
    now = _utc_now_naive()
    today_start = now.strftime("%Y-%m-%d 00:00:00")
    week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection(db_path)
    try:
        return {
            "generated_at_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
            "today": _period_stats(conn, today_start),
            "this_week": _period_stats(conn, week_start),
            "latency_ms_this_week": _avg_latency_ms(conn, week_start),
            "notes": _NOTES,
        }
    finally:
        conn.close()
