"""Application configuration and filesystem layout."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_TEMPERATURE = 0.2
MAX_LLM_ATTEMPTS = 3
MAX_VALIDATION_ATTEMPTS = 3

LANDING_DIR = PROJECT_ROOT / "data" / "landing"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze_layer"
SILVER_DIR = PROJECT_ROOT / "data" / "silver_layer"
GOLD_DIR = PROJECT_ROOT / "data" / "gold_layer"
REPORTS_DIR = PROJECT_ROOT / "reports"
STTM_HISTORY_DIR = PROJECT_ROOT / "sttm_history"
AUDIT_DB = PROJECT_ROOT / "audit" / "audit.db"


def ensure_directories() -> None:
    for directory in (
        LANDING_DIR,
        BRONZE_DIR,
        SILVER_DIR,
        GOLD_DIR,
        REPORTS_DIR,
        STTM_HISTORY_DIR,
        AUDIT_DB.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_llm_client():
    """Build the Groq OpenAI-compatible client when credentials are available."""
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY in idamp/.env before using AI features.")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=90.0)
