"""
Centralized configuration.

Why this exists: the original app called os.getenv("GROQ_API_KEY") in three
different files (app_simple_top5_FINAL.py, simple_top5_validator.py, and
inside _groq_client()). If the key were ever renamed or a second key added
for a different provider, that meant hunting through multiple files. This
module is the single place env vars are read and validated, and it fails
loudly and early if something required is missing, instead of failing deep
inside an LLM call with a confusing Groq SDK error.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    jd_model: str = "llama-3.3-70b-versatile"
    validation_model: str = "llama-3.1-8b-instant"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3
    max_upload_mb: int = 10

    @classmethod
    def load(cls) -> "Settings":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Create a .env file (see .env.example) "
                "or export it in your shell before starting the app."
            )
        return cls(groq_api_key=api_key)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
