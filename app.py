"""Flask entrypoint. Routes only - all real work is delegated to
call_handler.CallHandler, which is built from plain interfaces
(TelephonyProvider, ConversationManager, BookingRepository,
NotificationService). Nothing here ever touches a Twilio-specific type.
"""
import structlog
from flask import Flask, Response, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

from booking.db import init_db
from config import configure_logging, load_config
from observability import init_sentry
from stats import compute_stats
from telephony.twilio_adapter import TwilioProvider, validate_signature

logger = structlog.get_logger(__name__)

config = load_config()
configure_logging(config.log_level, config.env)
init_sentry(config.sentry_dsn)
# Eagerly (not just lazily inside build_default_call_handler) so /health and
# /internal/stats work even before the first webhook hit creates the DB.
init_db(config.database_path)

app = Flask(__name__)
# ngrok (and any reverse proxy) terminates HTTPS and forwards to Flask as
# plain HTTP, setting X-Forwarded-Proto/Host to say so. Without this,
# request.url reports "http://..." while Twilio signed the request using
# the real "https://..." URL, so signature validation always fails behind
# a tunnel/proxy.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
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
        logger.warning("stage", stage="telephony_webhook_received", outcome="error", reason="invalid_signature", route="/voice")
        return Response(status=403)

    gather_action_url = request.url_root.rstrip("/") + "/voice/handle-input"
    twiml = get_call_handler().handle_incoming_call(dict(request.form), gather_action_url)
    return Response(twiml, mimetype="text/xml")


@app.post("/voice/handle-input")
def voice_handle_input():
    if not _signature_valid():
        logger.warning(
            "stage", stage="telephony_webhook_received", outcome="error", reason="invalid_signature", route="/voice/handle-input"
        )
        return Response(status=403)

    continue_url_base = request.url_root.rstrip("/") + "/voice/continue"
    gather_action_url = request.url_root.rstrip("/") + "/voice/handle-input"
    twiml = get_call_handler().handle_speech_input(dict(request.form), continue_url_base, gather_action_url)
    return Response(twiml, mimetype="text/xml")


@app.post("/voice/continue")
def voice_continue():
    """Hit by Twilio's <Redirect> mid-turn, while a streamed LLM response is
    still being progressively delivered sentence by sentence - see
    ai/conversation.py's start_streaming_reply/get_next_streamed_sentence."""
    if not _signature_valid():
        logger.warning(
            "stage", stage="telephony_webhook_received", outcome="error", reason="invalid_signature", route="/voice/continue"
        )
        return Response(status=403)

    sentence_index = int(request.args.get("idx", 0))
    continue_url_base = request.url_root.rstrip("/") + "/voice/continue"
    # Always /voice/handle-input, never this same /voice/continue route - if
    # this turn's streamed reply is finishing here and starts listening
    # again, the caller's next real answer must land on the endpoint that
    # actually reads speech input, not this one (see build_reply_response's
    # docstring for the bug this avoids).
    gather_action_url = request.url_root.rstrip("/") + "/voice/handle-input"
    twiml = get_call_handler().handle_continue(dict(request.form), continue_url_base, sentence_index, gather_action_url)
    return Response(twiml, mimetype="text/xml")


@app.get("/health")
def health():
    return {"status": "ok"}


def _internal_stats_authorized() -> bool:
    """Shared-secret check appropriate for a two-person internal tool - not
    full auth. Missing INTERNAL_STATS_TOKEN in config means the endpoint is
    unusable (fails closed) rather than open."""
    if not config.internal_stats_token:
        return False
    provided = request.headers.get("X-Internal-Token") or request.args.get("token", "")
    return provided == config.internal_stats_token


@app.get("/internal/stats")
def internal_stats():
    if not _internal_stats_authorized():
        return Response(status=404)  # 404, not 403: don't confirm this route exists to an unauthenticated caller
    return jsonify(compute_stats(config.database_path))


if __name__ == "__main__":
    app.run(port=5000, debug=False)
