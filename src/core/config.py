"""Lab 11 — OpenAI configuration and API-key setup."""
import os

from dotenv import load_dotenv


# The lab's repo-local .env is authoritative. This also avoids a stale
# PowerShell OPENAI_API_KEY (for example, placeholder text with Unicode).
load_dotenv(override=True)


DEFAULT_OPENAI_MODEL = "gpt-5.6"


def get_openai_model(role: str = "main") -> str:
    """Return a role-specific model override or the shared OpenAI model."""
    role_variable = f"OPENAI_{role.upper()}_MODEL"
    return (
        os.environ.get(role_variable, "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
        or DEFAULT_OPENAI_MODEL
    )


def setup_api_key() -> str:
    """Load and validate OPENAI_API_KEY without ever printing the secret."""
    load_dotenv(override=True)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it to the repo .env file before running."
        )
    if not api_key.isascii() or not api_key.startswith("sk-"):
        raise RuntimeError(
            "OPENAI_API_KEY is malformed. Use the real ASCII key, not placeholder text."
        )
    print("OpenAI API key loaded.")
    return api_key


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
