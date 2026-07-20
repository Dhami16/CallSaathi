"""Provider-agnostic telephony interface.

app.py and every business-logic module talk to this interface only. No
provider-specific type (e.g. Twilio's VoiceResponse) may cross this
boundary - concrete providers live entirely inside their own adapter file.
"""
from abc import ABC, abstractmethod
from typing import Any, Mapping


class TelephonyProvider(ABC):
    @abstractmethod
    def parse_incoming_call(self, raw_request_data: Mapping[str, str]) -> dict:
        """Normalize an incoming-call webhook payload.

        Returns: {"caller_number": str, "call_id": str, "called_number": str}
        `called_number` (the number the customer dialed) is included so the
        app can look up which business this call belongs to.
        """

    @abstractmethod
    def parse_speech_result(self, raw_request_data: Mapping[str, str]) -> dict:
        """Normalize a speech-capture webhook payload.

        Returns: {"caller_number": str, "call_id": str, "transcript": str, "confidence": float}
        """

    @abstractmethod
    def build_greeting_response(
        self, greeting_text: str, gather_action_url: str, language: str = "english"
    ) -> Any:
        """Build the response for the very first turn of a call: speak
        `greeting_text` and start listening for speech, posting the result
        to `gather_action_url`."""

    @abstractmethod
    def build_reply_response(self, reply_text: str, hangup: bool = False, language: str = "english") -> Any:
        """Build the response for a follow-up turn: speak `reply_text`, then
        either keep listening for more speech (hangup=False) or end the
        call (hangup=True). `language` sets the speech-recognition locale
        for the next gather, same as build_greeting_response - without it,
        every turn after the first silently falls back to the provider's
        default locale regardless of the business's configured language."""

    @abstractmethod
    def build_continue_response(self, sentence_text: str, continue_url: str) -> Any:
        """Build a mid-turn progressive-delivery response: speak just
        `sentence_text`, then immediately fetch more of this same turn's
        reply from `continue_url` - used while a streamed LLM response is
        still being generated, so the caller starts hearing it sentence by
        sentence rather than waiting for the whole thing."""
