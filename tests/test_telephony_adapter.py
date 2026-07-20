"""Regression tests for two real Twilio-locale bugs found across sessions:

1. build_reply_response's <Gather> (used for every turn after the greeting)
   used to never set a speech-recognition `language` locale at all,
   silently falling back to Twilio's own default (en-US) regardless of the
   business's configured language.
2. Fixing (1) by locale-per-business-language then caused a WORSE bug in
   real production data: hi-IN phonetically transliterates clear English
   speech into unreadable Devanagari-script garbage rather than recognizing
   it as English, corrupting the LLM's input. Every turn's speech
   recognition now always uses en-IN (the code-switch-tolerant locale)
   regardless of business language_pref - that setting still controls what
   language WE speak (greeting/confirmation templates), not what Twilio
   listens with.

Run with: venv/Scripts/python -m pytest -q tests/test_telephony_adapter.py
"""
from telephony.twilio_adapter import TwilioProvider


def test_greeting_response_always_uses_code_switch_tolerant_locale():
    provider = TwilioProvider()
    # Even for a business whose spoken language_pref is Hindi, the listening
    # locale must stay en-IN - a hi-IN Gather was measured to mangle clear
    # English speech into transliterated garbage.
    twiml = provider.build_greeting_response("Namaste!", "http://test/voice/handle-input", language="hindi")
    assert 'language="en-IN"' in twiml
    assert "hi-IN" not in twiml


def test_reply_response_also_uses_code_switch_tolerant_locale():
    """The original bug: this used to omit `language` entirely on every turn
    after the first, defaulting to en-US. The fix for that regressed into
    locale-per-business-language, which real call data showed was worse -
    both are now consistently en-IN regardless of the `language` argument."""
    provider = TwilioProvider()
    twiml = provider.build_reply_response("Aapka din kaisa raha?", hangup=False, language="hindi")
    assert 'language="en-IN"' in twiml


def test_reply_response_hangup_true_does_not_need_a_locale():
    provider = TwilioProvider()
    twiml = provider.build_reply_response("Goodbye!", hangup=True, language="hindi")
    assert "<Gather" not in twiml
    assert "<Hangup" in twiml
