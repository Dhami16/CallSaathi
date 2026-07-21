"""Task 3 verification: externalized session state.

The multi-worker failure this replaces: an in-memory dict keyed by call_id
only exists within one process's memory. Two SQLiteSessionStore instances
pointed at the same db_path stand in for two separate gunicorn worker
processes here - if data written through one instance is readable through
the other, that's direct proof the fix actually solves the cross-worker
problem (an in-memory dict could never pass this test, by construction).

Run with: venv/Scripts/python -m pytest -q tests/test_session_store.py
"""
import time

from booking.db import init_db
from booking.session_store import SQLiteSessionStore


def test_session_written_by_one_instance_is_readable_by_another(tmp_path):
    """Simulates the multi-worker scenario: 'worker A' writes a session,
    'worker B' (a separate SQLiteSessionStore instance, same db file) must
    be able to read it back intact."""
    db_path = str(tmp_path / "test_sessions.db")
    init_db(db_path)

    worker_a_store = SQLiteSessionStore(db_path)
    worker_b_store = SQLiteSessionStore(db_path)

    session = {
        "messages": [{"role": "system", "content": "..."}],
        "business": {"id": 1, "name": "Test Clinic"},
        "slots": [{"id": 1, "date": "2026-07-14", "time": "10:00"}],
        "started_at": time.time(),
        "turns": 1,
    }
    worker_a_store.set("CALL-MULTIWORKER-1", session)

    read_back = worker_b_store.get("CALL-MULTIWORKER-1")
    assert read_back is not None
    assert read_back["business"]["name"] == "Test Clinic"
    assert read_back["turns"] == 1
    assert read_back["slots"][0]["date"] == "2026-07-14"

    # And a mutation made via worker B must be visible to worker A -
    # confirming this isn't just a one-way/cached read.
    read_back["turns"] += 1
    worker_b_store.set("CALL-MULTIWORKER-1", read_back)
    assert worker_a_store.get("CALL-MULTIWORKER-1")["turns"] == 2


def test_delete_removes_session_for_all_instances(tmp_path):
    db_path = str(tmp_path / "test_sessions.db")
    init_db(db_path)
    store_a = SQLiteSessionStore(db_path)
    store_b = SQLiteSessionStore(db_path)

    store_a.set("CALL-DEL-1", {"turns": 0})
    assert store_b.get("CALL-DEL-1") is not None

    store_b.delete("CALL-DEL-1")
    assert store_a.get("CALL-DEL-1") is None


def test_expired_session_is_treated_as_missing(tmp_path):
    db_path = str(tmp_path / "test_sessions.db")
    init_db(db_path)
    store = SQLiteSessionStore(db_path)

    store.set("CALL-EXPIRE-1", {"turns": 0}, ttl_seconds=0)
    time.sleep(1.1)  # ttl_seconds=0 means "already expired" once a second has passed

    assert store.get("CALL-EXPIRE-1") is None


def test_expired_sessions_are_purged_on_subsequent_writes(tmp_path):
    """set() opportunistically cleans up expired rows (no background job/
    cron - see session_store.py) - confirm an expired row for a DIFFERENT
    call_id actually gets deleted from the table, not just ignored on read."""
    from booking.db import get_connection

    db_path = str(tmp_path / "test_sessions.db")
    init_db(db_path)
    store = SQLiteSessionStore(db_path)

    store.set("CALL-OLD", {"turns": 0}, ttl_seconds=0)
    time.sleep(1.1)

    store.set("CALL-NEW", {"turns": 0}, ttl_seconds=600)  # triggers the opportunistic purge

    conn = get_connection(db_path)
    remaining_ids = {row["call_id"] for row in conn.execute("SELECT call_id FROM call_sessions")}
    conn.close()

    assert "CALL-OLD" not in remaining_ids
    assert "CALL-NEW" in remaining_ids


def test_claim_turn_succeeds_once_and_fails_on_retry(tmp_path):
    """Regression test for the real production bug this exists to prevent: a
    Twilio webhook retry (or any concurrent duplicate hit) for the exact
    same turn used to be indistinguishable from a genuinely new turn, so it
    would spawn its OWN independent streaming worker - two LLM completions
    for one turn, both appending into the same shared sentence list,
    interleaving into what the caller heard. claim_turn makes "start a
    worker for this turn" a one-time action, shared across separate
    SessionStore instances (i.e. separate gunicorn workers)."""
    db_path = str(tmp_path / "test_claims.db")
    init_db(db_path)
    worker_a_store = SQLiteSessionStore(db_path)
    worker_b_store = SQLiteSessionStore(db_path)

    assert worker_a_store.claim_turn("CALL-CLAIM-1", 1) is True
    # A retried/concurrent webhook hit for the SAME turn, even from a
    # different process/instance, must not also get to claim it.
    assert worker_b_store.claim_turn("CALL-CLAIM-1", 1) is False
    assert worker_a_store.claim_turn("CALL-CLAIM-1", 1) is False

    # A genuinely new turn on the same call must claim successfully.
    assert worker_a_store.claim_turn("CALL-CLAIM-1", 2) is True


def test_clear_turn_claims_allows_call_id_reuse(tmp_path):
    db_path = str(tmp_path / "test_claims.db")
    init_db(db_path)
    store = SQLiteSessionStore(db_path)

    assert store.claim_turn("CALL-CLAIM-2", 1) is True
    store.clear_turn_claims("CALL-CLAIM-2")
    assert store.claim_turn("CALL-CLAIM-2", 1) is True
