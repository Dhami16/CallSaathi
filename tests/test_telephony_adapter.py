"""Regression test for a real bug: build_reply_response's <Gather> (used for
every turn after the greeting) never set a speech-recognition `language`
locale at all, silently falling back to Twilio's own default (en-US)
regardless of the business's configured language - found while
investigating a reported Hindi/English language-matching issue.

Run with: venv/Scripts/python -m pytest -q tests/test_telephony_adapter.py
"""
from telephony.twilio_adapter import TwilioProvider


def test_greeting_response_sets_locale_from_language():
    provider = TwilioProvider()
    twiml = provider.build_greeting_response("Namaste!", "http://test/voice/handle-input", language="hindi")
    assert 'language="hi-IN"' in twiml


def test_reply_response_also_sets_locale_from_language():
    """The bug: this used to omit `language` entirely on every turn after
    the first, defaulting to en-US regardless of the business's language."""
    provider = TwilioProvider()
    twiml = provider.build_reply_response("Aapka din kaisa raha?", hangup=False, language="hindi")
    assert 'language="hi-IN"' in twiml


def test_reply_response_defaults_to_indian_english_locale():
    provider = TwilioProvider()
    twiml = provider.build_reply_response("How can I help?", hangup=False, language="english")
    assert 'language="en-IN"' in twiml


def test_reply_response_hangup_true_does_not_need_a_locale():
    provider = TwilioProvider()
    twiml = provider.build_reply_response("Goodbye!", hangup=True, language="hindi")
    assert "<Gather" not in twiml
    assert "<Hangup" in twiml
