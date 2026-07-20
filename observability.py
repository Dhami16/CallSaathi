"""Sentry setup and small helpers for non-fatal event capture.

Isolated in one module so the rest of the app doesn't need to care whether
Sentry is actually configured: sentry_sdk's capture_* functions are safe,
silent no-ops when sentry_sdk.init() was never called (e.g. SENTRY_DSN
unset), so call sites don't need their own "if configured" guards.
"""
import sentry_sdk
import structlog
from sentry_sdk.integrations.flask import FlaskIntegration

logger = structlog.get_logger(__name__)


def init_sentry(dsn: str) -> None:
    if not dsn:
        logger.warning("sentry_not_configured", stage="startup", outcome="skipped")
        return

    sentry_sdk.init(
        dsn=dsn,
        integrations=[FlaskIntegration()],
        # Error tracking only, no performance/tracing product - keeps this
        # comfortably within Sentry's free tier.
        traces_sample_rate=0.0,
    )
    logger.info("sentry_configured", stage="startup", outcome="success")


def capture_fallback(message: str, **context) -> None:
    """Report a handled/graceful fallback (e.g. a Groq timeout that was
    caught and turned into a spoken apology) as a non-fatal Sentry event.
    Without this, these are invisible - the call itself didn't crash, so
    nothing would otherwise reach Sentry.

    A no-op if Sentry was never initialized (see init_sentry above).
    """
    with sentry_sdk.new_scope() as scope:
        for key, value in context.items():
            scope.set_tag(key, value)
        sentry_sdk.capture_message(message, level="warning")
