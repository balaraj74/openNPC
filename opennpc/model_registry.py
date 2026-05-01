"""Model metadata and checkpoint validation for OpenNPC RL policies.

Embeds version, architecture, and training metadata into saved checkpoints so
that runtime loading can detect incompatible models before they cause silent
failures.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# Shared across the SDK — bump when checkpoint format changes.
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata embedded in every saved checkpoint."""

    sdk_version: str
    model_type: str  # "ppo" | "dqn"
    state_size: int
    action_count: int
    actions: list[str]
    format_version: int = CHECKPOINT_FORMAT_VERSION
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class CheckpointValidationError(Exception):
    """Raised when a checkpoint fails hard validation (e.g. wrong state_size)."""


class CheckpointVersionWarning(UserWarning):
    """Issued when a checkpoint was saved by a different SDK version."""


def _current_sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("opennpc")
    except Exception:
        return "0.0.0-dev"


def build_metadata(
    model_type: str,
    state_size: int,
    actions: list[str],
    *,
    extra: dict[str, Any] | None = None,
) -> ModelMetadata:
    """Create metadata for a checkpoint about to be saved."""
    return ModelMetadata(
        sdk_version=_current_sdk_version(),
        model_type=model_type,
        state_size=state_size,
        action_count=len(actions),
        actions=list(actions),
        extra=extra or {},
    )


def embed_metadata(checkpoint: dict[str, Any], metadata: ModelMetadata) -> dict[str, Any]:
    """Add a ``_metadata`` key to an existing checkpoint dict."""
    checkpoint["_metadata"] = metadata.to_dict()
    return checkpoint


def validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_model_type: str,
    expected_state_size: int,
    expected_actions: list[str] | None = None,
) -> ModelMetadata | None:
    """Validate a loaded checkpoint's metadata.

    Returns the parsed ``ModelMetadata`` if present, or ``None`` for legacy
    checkpoints that have no metadata (with a deprecation warning).

    Raises ``CheckpointValidationError`` for hard mismatches (wrong state_size
    or model_type) that would cause runtime crashes.

    Emits ``CheckpointVersionWarning`` for soft mismatches (SDK version drift,
    action list changes) that are recoverable but worth knowing about.
    """
    raw = checkpoint.get("_metadata")
    if raw is None:
        warnings.warn(
            "Checkpoint has no embedded metadata. "
            "Re-save with the latest SDK to enable validation.",
            CheckpointVersionWarning,
            stacklevel=2,
        )
        return None

    meta = ModelMetadata.from_dict(raw)

    # --- Hard checks (raise) ---

    if meta.model_type != expected_model_type:
        raise CheckpointValidationError(
            f"Checkpoint model_type is '{meta.model_type}' but loader expects "
            f"'{expected_model_type}'."
        )

    if meta.state_size != expected_state_size:
        raise CheckpointValidationError(
            f"Checkpoint was trained with state_size={meta.state_size} but the "
            f"current environment expects state_size={expected_state_size}. "
            f"Retrain or use a compatible checkpoint."
        )

    # --- Soft checks (warn) ---

    current_version = _current_sdk_version()
    if meta.sdk_version != current_version:
        warnings.warn(
            f"Checkpoint was saved with SDK v{meta.sdk_version}, "
            f"but current SDK is v{current_version}.",
            CheckpointVersionWarning,
            stacklevel=2,
        )

    if expected_actions is not None and sorted(meta.actions) != sorted(expected_actions):
        warnings.warn(
            f"Checkpoint action space {meta.actions} differs from expected "
            f"{expected_actions}. Actions may be remapped at runtime.",
            CheckpointVersionWarning,
            stacklevel=2,
        )

    return meta
