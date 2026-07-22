"""Central configuration loaded from environment variables.

Values are read lazily (not validated at import time) so modules that don't
need a given credential (e.g. booking tests with no Groq key) can still run.
Call `require(...)` helpers in the module that actually needs the value.
"""
import logging
import os
from dataclasses import dataclass

import structlog
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    groq_model: str
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    validate_twilio_signature: bool
    database_path: str
    log_level: str
    public_base_url: str
    env: str
    sentry_dsn: str
    internal_stats_token: str
    session_ttl_seconds: int
    notification_mode: str
    owner_notification_phone: str
    notification_allowlist: frozenset


def _parse_allowlist(raw: str) -> frozenset:
    return frozenset(number.strip() for number in raw.split(",") if number.strip())


def load_config() -> Config:
    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        twilio_phone_number=os.getenv("TWILIO_PHONE_NUMBER", ""),
        validate_twilio_signature=os.getenv("VALIDATE_TWILIO_SIGNATURE", "true").lower() == "true",
        database_path=os.getenv("DATABASE_PATH", "callsaathi.db"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        public_base_url=os.getenv("PUBLIC_BASE_URL", ""),
        env=os.getenv("ENV", "development"),
        sentry_dsn=os.getenv("SENTRY_DSN", ""),
        internal_stats_token=os.getenv("INTERNAL_STATS_TOKEN", ""),
        session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "600")),
        notification_mode=os.getenv("NOTIFICATION_MODE", "mock").strip().lower(),
        owner_notification_phone=os.getenv("OWNER_NOTIFICATION_PHONE", ""),
        notification_allowlist=_parse_allowlist(os.getenv("NOTIFICATION_ALLOWLIST", "")),
    )


def configure_logging(level: str, env: str) -> None:
    """Configures structlog for the whole app. `env == "production"` gets
    single-line JSON (log-aggregator friendly); anything else gets
    structlog's pretty console renderer, since this is what a developer
    actually wants to read in a terminal."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # structlog wraps the stdlib logging module rather than replacing it, so
    # third-party libraries (flask, twilio, groq's httpx client) still log
    # normally; only our own structlog.get_logger() calls get the structured
    # processors below.
    logging.basicConfig(format="%(message)s", level=log_level)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if env == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
