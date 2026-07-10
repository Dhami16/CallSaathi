"""Twilio implementation of the TelephonyProvider interface.

This is the ONLY file in the project allowed to import Twilio-specific
objects (VoiceResponse, Gather, RequestValidator, ...). Everything it hands
back across the interface boundary is either a plain dict or a plain str
(TwiML XML), never a Twilio object.
"""
import logging
from typing import Mapping

from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import Gather, VoiceResponse

from telephony.base import TelephonyProvider

logger = logging.getLogger(__name__)

# Twilio speech-recognition locale per business language_pref. Twilio's
# speech model must pick a single locale per call; it does not truly
# code-switch mid-utterance, so this is a best-effort choice based on the
# business's configured primary language, not a per-word decision.
_LANGUAGE_LOCALES = {
    "hindi": "hi-IN",
    "bengali": "bn-IN",
    "english": "en-IN",
}
_DEFAULT_LOCALE = "en-IN"

# A neutral Indian-English voice available on Twilio's Polly set.
_VOICE = "Polly.Aditi"

_NO_INPUT_MESSAGE = "Sorry, I didn't catch that. Please call back and try again."


class TwilioProvider(TelephonyProvider):
    def parse_incoming_call(self, raw_request_data: Mapping[str, str]) -> dict:
        return {
            "caller_number": raw_request_data.get("From", ""),
            "call_id": raw_request_data.get("CallSid", ""),
            "called_number": raw_request_data.get("To", ""),
        }

    def parse_speech_result(self, raw_request_data: Mapping[str, str]) -> dict:
        raw_confidence = raw_request_data.get("Confidence")
        try:
            confidence = float(raw_confidence) if raw_confidence else 0.0
        except ValueError:
            confidence = 0.0
        return {
            "caller_number": raw_request_data.get("From", ""),
            "call_id": raw_request_data.get("CallSid", ""),
            "transcript": (raw_request_data.get("SpeechResult") or "").strip(),
            "confidence": confidence,
        }

    def build_greeting_response(
        self, greeting_text: str, gather_action_url: str, language: str = "english"
    ) -> str:
        locale = _LANGUAGE_LOCALES.get(language, _DEFAULT_LOCALE)
        response = VoiceResponse()
        gather = Gather(
            input="speech",
            action=gather_action_url,
            method="POST",
            language=locale,
            speech_timeout="auto",
        )
        gather.say(greeting_text, voice=_VOICE)
        response.append(gather)
        # Reached only if the caller says nothing at all and Gather times out.
        response.say(_NO_INPUT_MESSAGE, voice=_VOICE)
        response.hangup()
        return str(response)

    def build_reply_response(self, reply_text: str, hangup: bool = False) -> str:
        response = VoiceResponse()
        if hangup:
            response.say(reply_text, voice=_VOICE)
            response.hangup()
            return str(response)

        # No `action` given: Twilio defaults to re-posting to the current
        # request URL, i.e. /voice/handle-input again, continuing the loop.
        gather = Gather(input="speech", method="POST", speech_timeout="auto")
        gather.say(reply_text, voice=_VOICE)
        response.append(gather)
        response.say(_NO_INPUT_MESSAGE, voice=_VOICE)
        response.hangup()
        return str(response)


def validate_signature(url: str, form_params: Mapping[str, str], signature: str, auth_token: str) -> bool:
    """Verify a webhook request actually came from Twilio.

    Kept as a plain function (not part of TelephonyProvider) since signature
    validation is a Twilio-specific security concern, not a normalized
    telephony operation - but it still lives entirely in this adapter file.
    """
    if not auth_token:
        logger.warning("No Twilio auth token configured; skipping signature validation")
        return True
    validator = RequestValidator(auth_token)
    return validator.validate(url, dict(form_params), signature)
