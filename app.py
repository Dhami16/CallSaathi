"""Flask entrypoint. Routes only - all real work is delegated to
call_handler.CallHandler, which is built from plain interfaces
(TelephonyProvider, ConversationManager, BookingRepository,
NotificationService). Nothing here ever touches a Twilio-specific type.
"""
import logging

from flask import Flask, Response, request

from config import configure_logging, load_config
from telephony.twilio_adapter import TwilioProvider, validate_signature

logger = logging.getLogger(__name__)

config = load_config()
configure_logging(config.log_level)

app = Flask(__name__)
telephony_provider = TwilioProvider()

# call_handler is created lazily on first request so importing app.py never
# requires a GROQ_API_KEY / DB file to exist (keeps `flask routes` and
# import-only smoke tests working without full configuration).
_call_handler = None


def get_call_handler():
    global _call_handler
    if _call_handler is None:
        from call_handler import build_default_call_handler

        _call_handler = build_default_call_handler(config, telephony_provider)
    return _call_handler


def _signature_valid() -> bool:
    if not config.validate_twilio_signature:
        return True
    signature = request.headers.get("X-Twilio-Signature", "")
    return validate_signature(request.url, request.form, signature, config.twilio_auth_token)


@app.post("/voice")
def voice():
    if not _signature_valid():
        logger.warning("Rejected /voice request with invalid Twilio signature")
        return Response(status=403)

    gather_action_url = request.url_root.rstrip("/") + "/voice/handle-input"
    twiml = get_call_handler().handle_incoming_call(dict(request.form), gather_action_url)
    return Response(twiml, mimetype="text/xml")


@app.post("/voice/handle-input")
def voice_handle_input():
    if not _signature_valid():
        logger.warning("Rejected /voice/handle-input request with invalid Twilio signature")
        return Response(status=403)

    twiml = get_call_handler().handle_speech_input(dict(request.form))
    return Response(twiml, mimetype="text/xml")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(port=5000, debug=False)
