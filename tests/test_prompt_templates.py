"""Regression coverage for the per-stage prompt template refactor in
ai/conversation.py - guards against accidentally collapsing the numbered
step structure back into prose, which was measured to roughly double
gpt-oss's hidden reasoning tokens and increase latency (see README).

No network needed - this only tests string assembly.

Run with: venv/Scripts/python -m pytest -q tests/test_prompt_templates.py
"""
from ai.conversation import _build_system_prompt

DEMO_BUSINESS = {"name": "Radiant Skin Clinic", "vertical": "clinic"}
DEMO_SLOTS = [
    {"id": 1, "date": "2026-07-16", "time": "10:00"},
    {"id": 2, "date": "2026-07-16", "time": "15:30"},
]


def test_prompt_keeps_numbered_step_structure():
    prompt = _build_system_prompt(DEMO_BUSINESS, DEMO_SLOTS)
    for step in ["1.", "2.", "3.", "4.", "5."]:
        assert step in prompt


def test_prompt_includes_business_name_and_vertical():
    prompt = _build_system_prompt(DEMO_BUSINESS, DEMO_SLOTS)
    assert "Radiant Skin Clinic" in prompt
    assert "clinic" in prompt


def test_prompt_lists_given_slots_and_no_others():
    prompt = _build_system_prompt(DEMO_BUSINESS, DEMO_SLOTS)
    assert "id=1: 2026-07-16 at 10:00" in prompt
    assert "id=2: 2026-07-16 at 15:30" in prompt


def test_prompt_handles_no_available_slots():
    prompt = _build_system_prompt(DEMO_BUSINESS, [])
    assert "no slots currently available" in prompt


def test_prompt_includes_language_matching_and_decline_rules():
    prompt = _build_system_prompt(DEMO_BUSINESS, DEMO_SLOTS)
    assert "Reply in the SAME language" in prompt
    assert "Decline medical/pricing/service questions" in prompt
    assert "BOOKING_CONFIRMED" in prompt
