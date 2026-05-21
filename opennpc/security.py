"""Runtime security settings for OpenNPC services."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Configuration loaded from environment for the API runtime."""

    api_key: str | None = None
    debug_enabled: bool = True
    max_batch_size: int = 128
    max_agent_id_length: int = 96
    max_event_length: int = 512
    max_memory_limit: int = 100

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        env = os.getenv("OPENNPC_ENV", "development").strip().lower()
        return cls(
            api_key=os.getenv("OPENNPC_API_KEY") or None,
            debug_enabled=_env_bool("OPENNPC_DEBUG", env != "production"),
            max_batch_size=_env_int("OPENNPC_MAX_BATCH_SIZE", 128),
            max_agent_id_length=_env_int("OPENNPC_MAX_AGENT_ID_LENGTH", 96),
            max_event_length=_env_int("OPENNPC_MAX_EVENT_LENGTH", 512),
            max_memory_limit=_env_int("OPENNPC_MAX_MEMORY_LIMIT", 100),
        )

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key)


def token_matches(expected: str | None, provided: str | None) -> bool:
    """Compare API tokens without leaking timing information."""

    if not expected:
        return True
    if not provided:
        return False
    return secrets.compare_digest(expected, provided)

