"""Redaction helpers so credentials never reach logs, storage, or exports."""

from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "x-api-key",
    "openai-api-key",
    "typesafe_api_key",
    "openai_api_key",
    "cookie",
    "set-cookie",
    "token",
    "secret",
    "password",
}

# Common credential shapes: bearer tokens, OpenAI keys, generic long secrets after "key=".
_PATTERNS = [
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[A-Za-z0-9._\-]{8,}"),
]

REDACTED = "[REDACTED]"


def redact_text(text: str, secrets: list[str | None] | None = None) -> str:
    for secret in secrets or []:
        if secret and len(secret) >= 6:
            text = text.replace(secret, REDACTED)
    for pat in _PATTERNS:
        text = pat.sub(lambda m: (m.group(1) if m.lastindex else "") + REDACTED, text)
    return text


def redact(obj: Any, secrets: list[str | None] | None = None) -> Any:
    """Recursively redact sensitive keys and any embedded secret strings."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in SENSITIVE_KEYS:
                out[k] = REDACTED
            else:
                out[k] = redact(v, secrets)
        return out
    if isinstance(obj, list):
        return [redact(v, secrets) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj, secrets)
    return obj
