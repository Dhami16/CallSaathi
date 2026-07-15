"""Twilio implementation of the TelephonyProvider interface.

This is the ONLY file in the project allowed to import Twilio-specific
objects (VoiceResponse, Gather, RequestValidator, ...). Everything it hands
back across the interface boundary is either a plain dict or a plain str
(TwiML XML), never a Twilio object.
"""
from typing import Mapping

import structlog
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import Gather, Redirect, VoiceResponse

from telephony.base import TelephonyProvider

logger = structlog.get_logger(__name__)

# Twilio's Gather can only use ONE speech-recognition locale per call turn -
# it does not auto-detect or switch language mid-utterance. Real production
# data showed hi-IN is actively harmful for this app's actual callers: it
# phonetically TRANSLITERATES clear English speech into Devanagari-script
# garbage (e.g. "I have skin problem" became unreadable Hindi-script text)
# rather than recognizing it as English, which then corrupted the LLM's
# input and directly caused fallbacks and repeated clarifying questions.
# en-IN tolerates Hindi/English code-switching far better than hi-IN
# tolerates English, so it's used for every turn's speech recognition
# regardless of the business's language_pref - that setting still controls
# what language WE speak (greeting/confirmation templates), just not what
# locale Twilio listens with.
_STT_LOCALE = "en-IN"

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
        response = VoiceResponse()
        gather = Gather(
            input="speech",
            action=gather_action_url,
            method="POST",
            language=_STT_LOCALE,
            speech_timeout="auto",
        )
        gather.say(greeting_text, voice=_VOICE)
        response.append(gather)
        # Reached only if the caller says nothing at all and Gather times out.
        response.say(_NO_INPUT_MESSAGE, voice=_VOICE)
        response.hangup()
        return str(response)

    def build_reply_response(self, reply_text: str, hangup: bool = False, language: str = "english") -> str:
        response = VoiceResponse()
        if hangup:
            if reply_text:
                response.say(reply_text, voice=_VOICE)
            response.hangup()
            return str(response)

        # No `action` given: Twilio defaults to re-posting to the current
        # request URL, i.e. /voice/handle-input again, continuing the loop.
        # `language` (kept as a parameter for interface symmetry with
        # build_greeting_response) is intentionally NOT used to pick the
        # locale here - see _STT_LOCALE above for why every turn uses the
        # same code-switch-tolerant locale regardless of business language.
        gather = Gather(input="speech", method="POST", language=_STT_LOCALE, speech_timeout="auto")
        # reply_text can be empty when progressive delivery already spoke
        # everything for this turn via prior <Say>+<Redirect> hits and this
        # call is just the one that confirms "nothing more, start listening".
        if reply_text:
            gather.say(reply_text, voice=_VOICE)
        response.append(gather)
        response.say(_NO_INPUT_MESSAGE, voice=_VOICE)
        response.hangup()
        return str(response)

    def build_continue_response(self, sentence_text: str, continue_url: str) -> str:
        response = VoiceResponse()
        response.say(sentence_text, voice=_VOICE)
        response.append(Redirect(continue_url, method="POST"))
        return str(response)


def validate_signature(url: str, form_params: Mapping[str, str], signature: str, auth_token: str) -> bool:
    """Verify a webhook request actually came from Twilio.

    Kept as a plain function (not part of TelephonyProvider) since signature
    validation is a Twilio-specific security concern, not a normalized
    telephony operation - but it still lives entirely in this adapter file.
    """
    if not auth_token:
        logger.warning("stage", stage="telephony_webhook_received", outcome="error", reason="no_auth_token_configured")
        return True
    validator = RequestValidator(auth_token)
    return validator.validate(url, dict(form_params), signature)
