"""Configuration and environment helpers."""

import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

try:
    # Use OS certificates for HTTPS behind TLS-inspecting proxies.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-flash-latest"


def get_config() -> Dict[str, Any]:
    """Load scoring weights, thresholds, regions and aliases from config.json."""
    with open(PROJECT_ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def setup_environment() -> Dict[str, str]:
    """Load .env and return the Gemini settings."""
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
        "model": os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
    }
