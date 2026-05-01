"""Tests for opennpc.model_registry — checkpoint metadata and validation."""

from __future__ import annotations

import warnings

import pytest

from opennpc.model_registry import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointValidationError,
    CheckpointVersionWarning,
    ModelMetadata,
    build_metadata,
    embed_metadata,
    validate_checkpoint,
)


class TestModelMetadata:
    """ModelMetadata dataclass round-trip tests."""

    def test_to_dict_roundtrip(self):
        meta = build_metadata("ppo", 9, ["attack", "defend", "heal"])
        d = meta.to_dict()
        restored = ModelMetadata.from_dict(d)
        assert restored.model_type == "ppo"
        assert restored.state_size == 9
        assert restored.action_count == 3
        assert restored.actions == ["attack", "defend", "heal"]
        assert restored.format_version == CHECKPOINT_FORMAT_VERSION

    def test_from_dict_ignores_unknown_keys(self):
        d = build_metadata("dqn", 9, ["a"]).to_dict()
        d["future_field"] = "should_be_ignored"
        meta = ModelMetadata.from_dict(d)
        assert meta.model_type == "dqn"

    def test_trained_at_populated(self):
        meta = build_metadata("ppo", 9, ["a"])
        assert meta.trained_at  # not empty
        assert "T" in meta.trained_at  # ISO 8601

    def test_extra_dict_stored(self):
        meta = build_metadata("ppo", 9, ["a"], extra={"lr": 0.001})
        assert meta.extra == {"lr": 0.001}


class TestEmbedMetadata:
    """Ensure metadata is properly injected into checkpoint dicts."""

    def test_metadata_key_added(self):
        ckpt: dict = {"state_dict": {}, "actions": ["a"]}
        meta = build_metadata("ppo", 9, ["a"])
        embed_metadata(ckpt, meta)
        assert "_metadata" in ckpt
        assert ckpt["_metadata"]["model_type"] == "ppo"

    def test_original_keys_preserved(self):
        ckpt: dict = {"state_dict": {"w": 1}, "actions": ["a"]}
        meta = build_metadata("ppo", 9, ["a"])
        embed_metadata(ckpt, meta)
        assert ckpt["state_dict"] == {"w": 1}
        assert ckpt["actions"] == ["a"]


class TestValidateCheckpoint:
    """Validation logic — hard errors vs soft warnings."""

    def _make_checkpoint(
        self,
        model_type: str = "ppo",
        state_size: int = 9,
        actions: list[str] | None = None,
        sdk_version: str | None = None,
    ) -> dict:
        actions = actions or ["attack", "defend"]
        meta = build_metadata(model_type, state_size, actions)
        if sdk_version is not None:
            # Override the auto-detected version.
            meta = ModelMetadata(
                sdk_version=sdk_version,
                model_type=meta.model_type,
                state_size=meta.state_size,
                action_count=meta.action_count,
                actions=meta.actions,
                format_version=meta.format_version,
                trained_at=meta.trained_at,
            )
        ckpt: dict = {"state_dict": {}, "actions": actions}
        embed_metadata(ckpt, meta)
        return ckpt

    # --- Legacy checkpoints (no metadata) ---

    def test_legacy_checkpoint_returns_none_with_warning(self):
        ckpt: dict = {"state_dict": {}, "actions": ["a"]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_checkpoint(
                ckpt, expected_model_type="ppo", expected_state_size=9
            )
        assert result is None
        assert len(w) == 1
        assert issubclass(w[0].category, CheckpointVersionWarning)

    # --- Hard errors ---

    def test_wrong_model_type_raises(self):
        ckpt = self._make_checkpoint(model_type="dqn")
        with pytest.raises(CheckpointValidationError, match="model_type"):
            validate_checkpoint(
                ckpt, expected_model_type="ppo", expected_state_size=9
            )

    def test_wrong_state_size_raises(self):
        ckpt = self._make_checkpoint(state_size=15)
        with pytest.raises(CheckpointValidationError, match="state_size"):
            validate_checkpoint(
                ckpt, expected_model_type="ppo", expected_state_size=9
            )

    # --- Soft warnings ---

    def test_sdk_version_mismatch_warns(self):
        ckpt = self._make_checkpoint(sdk_version="0.0.0-ancient")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            meta = validate_checkpoint(
                ckpt, expected_model_type="ppo", expected_state_size=9
            )
        assert meta is not None
        version_warnings = [x for x in w if issubclass(x.category, CheckpointVersionWarning)]
        assert len(version_warnings) >= 1

    def test_action_mismatch_warns(self):
        ckpt = self._make_checkpoint(actions=["attack", "defend"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            meta = validate_checkpoint(
                ckpt,
                expected_model_type="ppo",
                expected_state_size=9,
                expected_actions=["attack", "heal"],
            )
        assert meta is not None
        action_warnings = [
            x for x in w if "action space" in str(x.message)
        ]
        assert len(action_warnings) == 1

    # --- Happy path ---

    def test_valid_checkpoint_returns_metadata(self):
        ckpt = self._make_checkpoint()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            meta = validate_checkpoint(
                ckpt,
                expected_model_type="ppo",
                expected_state_size=9,
                expected_actions=["attack", "defend"],
            )
        # Only version warnings are acceptable (test env version != installed).
        non_version_warnings = [
            x for x in w if "action space" in str(x.message) or "no embedded metadata" in str(x.message).lower()
        ]
        assert len(non_version_warnings) == 0
        assert meta is not None
        assert meta.model_type == "ppo"
        assert meta.state_size == 9
