"""Unit tests for ai/intent_interpreter.py's validation contract, using a
fake Groq client so the tool-call -> TurnUnderstanding pipeline is checked
without a real API key or network call - same spirit as
tests/test_booking_flow.py's fakes for CallHandler's dependencies.

Run with: venv\\Scripts\\python -m pytest -s tests/test_intent_interpreter.py
"""
import json

from ai.intent_interpreter import BookingIntent, BookingIntentInterpreter


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, arguments):
        self.function = _FakeFunction("classify_caller_turn", json.dumps(arguments))


class _FakeMessage:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls = tool_calls
        self.content = content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class FakeGroqClient:
    """Scripted stand-in returning one canned tool-call response per call."""

    def __init__(self, tool_call_arguments):
        self._tool_call_arguments = tool_call_arguments
        self.calls = []

        class _Completions:
            def create(_self, **kwargs):
                self.calls.append(kwargs)
                return _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall(self._tool_call_arguments)]))

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _interpret(tool_call_arguments):
    interpreter = BookingIntentInterpreter(FakeGroqClient(tool_call_arguments), model="test-model")
    return interpreter.interpret("some caller speech", context={"stage": "timing"})


def test_valid_tool_call_is_trusted():
    understanding = _interpret({
        "intent": "select_slot",
        "confidence": 0.92,
        "reason": None,
        "target_date": "2026-07-17",
        "target_time": "10:00",
        "customer_name": None,
        "selected_option": 1,
        "requires_clarification": False,
        "reasoning": "Caller confirmed the first offered slot.",
        "assistant_reply": None,
    })
    assert understanding.intent == BookingIntent.SELECT_SLOT
    assert understanding.confidence == 0.92
    assert understanding.target_date == "2026-07-17"
    assert understanding.target_time == "10:00"
    assert understanding.selected_option == 1


def test_malformed_date_is_rejected_not_trusted_verbatim():
    understanding = _interpret({
        "intent": "select_slot",
        "confidence": 0.9,
        "reason": None,
        "target_date": "not-a-date",
        "target_time": "25:99",
        "customer_name": None,
        "selected_option": None,
        "requires_clarification": False,
        "reasoning": "",
        "assistant_reply": None,
    })
    # The model's raw strings must never be trusted without validation -
    # both fail their format checks and must come back null, not passed
    # through to drive a booking against a bogus date/time.
    assert understanding.target_date is None
    assert understanding.target_time is None


def test_unknown_intent_falls_back_to_unclear():
    understanding = _interpret({
        "intent": "totally_made_up_intent",
        "confidence": 0.99,
        "reason": None,
        "target_date": None,
        "target_time": None,
        "customer_name": None,
        "selected_option": None,
        "requires_clarification": False,
        "reasoning": "",
        "assistant_reply": None,
    })
    assert understanding.intent == BookingIntent.UNCLEAR


def test_out_of_range_selected_option_is_dropped():
    understanding = _interpret({
        "intent": "select_slot",
        "confidence": 0.9,
        "reason": None,
        "target_date": None,
        "target_time": None,
        "customer_name": None,
        "selected_option": 7,
        "requires_clarification": False,
        "reasoning": "",
        "assistant_reply": None,
    })
    assert understanding.selected_option is None


def test_confidence_is_clamped_to_unit_interval():
    understanding = _interpret({
        "intent": "confirm_booking",
        "confidence": 4.5,
        "reason": None,
        "target_date": None,
        "target_time": None,
        "customer_name": None,
        "selected_option": None,
        "requires_clarification": False,
        "reasoning": "",
        "assistant_reply": None,
    })
    assert understanding.confidence == 1.0


def test_no_tool_calls_falls_back_to_json_mode():
    class _NoToolCallClient:
        def __init__(self, json_content):
            self._json_content = json_content

            class _Completions:
                def create(_self, **kwargs):
                    if kwargs.get("tools"):
                        return _FakeResponse(_FakeMessage(tool_calls=None))
                    return _FakeResponse(_FakeMessage(content=self._json_content))

            class _Chat:
                completions = _Completions()

            self.chat = _Chat()

    payload = json.dumps({
        "intent": "out_of_scope",
        "confidence": 0.8,
        "reason": None,
        "target_date": None,
        "target_time": None,
        "customer_name": None,
        "selected_option": None,
        "requires_clarification": False,
        "reasoning": "Pricing question.",
        "assistant_reply": "I can help you book an appointment; someone will call back about pricing.",
    })
    interpreter = BookingIntentInterpreter(_NoToolCallClient(payload), model="test-model")
    understanding = interpreter.interpret("what's the price?", context={"stage": "reason"})
    assert understanding.intent == BookingIntent.OUT_OF_SCOPE
    assert understanding.assistant_reply
