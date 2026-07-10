"""Central configuration loaded from environment variables.

Values are read lazily (not validated at import time) so modules that don't
need a given credential (e.g. booking tests with no Groq key) can still run.
Call `require(...)` helpers in the module that actually needs the value.
"""
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    groq_model: str
    twilio_account_sid: str
    twilio_auth_token: str
    validate_twilio_signature: bool
    database_path: str
    log_level: str
    public_base_url: str


def load_config() -> Config:
    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        validate_twilio_signature=os.getenv("VALIDATE_TWILIO_SIGNATURE", "true").lower() == "true",
        database_path=os.getenv("DATABASE_PATH", "callsaathi.db"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        public_base_url=os.getenv("PUBLIC_BASE_URL", ""),
    )


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
