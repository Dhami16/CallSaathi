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
    twiml = provider.build_reply_response(
        "Aapka din kaisa raha?", "http://test/voice/handle-input", hangup=False, language="hindi"
    )
    assert 'language="en-IN"' in twiml


def test_reply_response_gather_always_posts_back_to_handle_input():
    """Regression test: this <Gather> used to omit `action` entirely,
    relying on Twilio's default of re-posting to whatever URL is currently
    handling the request. That silently misroutes the caller's next real
    answer to /voice/continue (a dead end that never reads speech input)
    whenever this response happens to be reached from there - i.e. every
    call whose reply streams 3+ sentences. `action` must always be the
    explicit /voice/handle-input URL, regardless of which endpoint built
    this response."""
    provider = TwilioProvider()
    twiml = provider.build_reply_response(
        "Aapka din kaisa raha?", "http://test/voice/handle-input", hangup=False, language="hindi"
    )
    assert twiml.count('action="http://test/voice/handle-input"') == 2  # main Gather + retry Gather


def test_reply_response_hangup_true_does_not_need_a_locale():
    provider = TwilioProvider()
    twiml = provider.build_reply_response(
        "Goodbye!", "http://test/voice/handle-input", hangup=True, language="hindi"
    )
    assert "<Gather" not in twiml
    assert "<Hangup" in twiml


def test_greeting_response_retries_once_before_giving_up_on_no_input():
    """Real bug found in production: a single no-input timeout used to end
    the whole call immediately with no second chance - unforgiving for a
    real phone caller (background noise, a pause to think, etc.)."""
    provider = TwilioProvider()
    twiml = provider.build_greeting_response("Namaste!", "http://test/voice/handle-input", language="hindi")
    assert twiml.count("<Gather") == 2
    assert "didn't catch that - could you say that again" in twiml
    assert "<Hangup" in twiml


def test_greeting_response_includes_hints_when_given():
    """Regression test: "9 AM" was transcribed as "99 AM" in a real call with
    no hints at all telling STT that appointment times were the likely
    subject - hints must actually reach the TwiML, on both the main and
    retry Gather."""
    provider = TwilioProvider()
    twiml = provider.build_greeting_response(
        "Namaste!", "http://test/voice/handle-input", hints="9 AM, 4 PM"
    )
    assert twiml.count('hints="9 AM, 4 PM"') == 2


def test_greeting_response_omits_hints_attribute_when_not_given():
    provider = TwilioProvider()
    twiml = provider.build_greeting_response("Namaste!", "http://test/voice/handle-input")
    assert "hints=" not in twiml


def test_reply_response_includes_hints_when_given():
    provider = TwilioProvider()
    twiml = provider.build_reply_response(
        "Aapka din kaisa raha?", "http://test/voice/handle-input", hangup=False, hints="9:30 AM, 2 PM"
    )
    assert twiml.count('hints="9:30 AM, 2 PM"') == 2


def test_reply_response_retries_once_before_giving_up_on_no_input():
    provider = TwilioProvider()
    twiml = provider.build_reply_response(
        "Aapka din kaisa raha?", "http://test/voice/handle-input", hangup=False, language="hindi"
    )
    assert twiml.count("<Gather") == 2
    assert "didn't catch that - could you say that again" in twiml
    assert "<Hangup" in twiml
