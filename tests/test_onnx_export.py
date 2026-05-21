"""Tests for ONNXPolicy and export_onnx() methods on PPO and DQN policies."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")


# ---------------------------------------------------------------------------
# PPO ONNX export
# ---------------------------------------------------------------------------

class TestPPOOnnxExport:
    def _make_policy(self):
        from opennpc.training.ppo import PPOPytorchPolicy
        from opennpc.training.ppo import ActorCritic
        from opennpc.training.ppo import STATE_SIZE

        actions = ["attack", "retreat", "patrol"]
        model = ActorCritic(STATE_SIZE, len(actions))
        return PPOPytorchPolicy(model, actions), STATE_SIZE

    def test_export_onnx_creates_files(self):
        policy, state_size = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "ppo.onnx"
            out = policy.export_onnx(onnx_path)
            assert out == onnx_path
            assert onnx_path.exists()
            meta_path = onnx_path.with_suffix(".onnx.meta.json")
            assert meta_path.exists()

    def test_export_onnx_metadata_contents(self):
        policy, _ = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "ppo.onnx"
            policy.export_onnx(onnx_path)
            meta = json.loads(onnx_path.with_suffix(".onnx.meta.json").read_text())
            assert meta["model_type"] == "ppo"
            assert set(meta["action_space"]) == {"attack", "retreat", "patrol"}
            assert "state_size" in meta

    def test_onnx_policy_loads_ppo_export(self):
        from opennpc.training.onnx_policy import ONNXPolicy

        policy, _ = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "ppo.onnx"
            policy.export_onnx(onnx_path)
            onnx_policy = ONNXPolicy.from_file(onnx_path)
            assert onnx_policy.actions == ["attack", "retreat", "patrol"]

    def test_onnx_policy_selects_valid_action(self):
        from opennpc.training.onnx_policy import ONNXPolicy
        from opennpc.types import AgentConfig, GameState

        policy, _ = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "ppo.onnx"
            policy.export_onnx(onnx_path)
            onnx_policy = ONNXPolicy.from_file(onnx_path)

            config = AgentConfig(agent_id="t", agent_type="enemy", allowed_actions=["attack", "patrol"])
            state = GameState(agent_id="t", location="forest", threat_level=0.5)
            decision = onnx_policy.select_action(config, state, ["attack", "patrol"])
            assert decision.action in ["attack", "patrol"]
            assert 0.0 <= decision.confidence <= 1.0


# ---------------------------------------------------------------------------
# DQN ONNX export
# ---------------------------------------------------------------------------

class TestDQNOnnxExport:
    def _make_policy(self):
        from opennpc.training.dqn import DQNPolicy, QNetwork
        from opennpc.training.dqn import STATE_SIZE

        actions = ["fire", "cover", "reload"]
        model = QNetwork(STATE_SIZE, len(actions))
        return DQNPolicy(model, actions), STATE_SIZE

    def test_export_onnx_creates_files(self):
        policy, _ = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "dqn.onnx"
            out = policy.export_onnx(onnx_path)
            assert out.exists()
            meta_path = onnx_path.with_suffix(".onnx.meta.json")
            assert meta_path.exists()

    def test_export_onnx_metadata_contents(self):
        policy, _ = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "dqn.onnx"
            policy.export_onnx(onnx_path)
            meta = json.loads(onnx_path.with_suffix(".onnx.meta.json").read_text())
            assert meta["model_type"] == "dqn"
            assert "fire" in meta["action_space"]

    def test_onnx_policy_loads_dqn_export(self):
        from opennpc.training.onnx_policy import ONNXPolicy

        policy, _ = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "dqn.onnx"
            policy.export_onnx(onnx_path)
            onnx_policy = ONNXPolicy.from_file(onnx_path)
            assert set(onnx_policy.actions) == {"fire", "cover", "reload"}

    def test_onnx_policy_fallback_when_no_overlap(self):
        from opennpc.training.onnx_policy import ONNXPolicy
        from opennpc.types import AgentConfig, GameState

        policy, _ = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "dqn.onnx"
            policy.export_onnx(onnx_path)
            onnx_policy = ONNXPolicy.from_file(onnx_path)

            config = AgentConfig(agent_id="t", agent_type="npc", allowed_actions=["dance"])
            state = GameState(agent_id="t", location="hub", threat_level=0.0)
            decision = onnx_policy.select_action(config, state, ["dance"])
            # Should fall back to the only available action
            assert decision.action == "dance"

    def test_onnx_policy_idle_when_no_actions(self):
        from opennpc.training.onnx_policy import ONNXPolicy
        from opennpc.types import AgentConfig, GameState

        policy, _ = self._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "dqn.onnx"
            policy.export_onnx(onnx_path)
            onnx_policy = ONNXPolicy.from_file(onnx_path)

            config = AgentConfig(agent_id="t", role="npc", goal="exist", allowed_actions=[])
            state = GameState(agent_id="t", location="hub", threat_level=0.0)
            decision = onnx_policy.select_action(config, state, [])
            assert decision.action == "idle"


# ---------------------------------------------------------------------------
# ONNXPolicy direct unit tests
# ---------------------------------------------------------------------------

class TestONNXPolicyDirect:
    def test_missing_meta_json_raises(self):
        from opennpc.training.onnx_policy import ONNXPolicy

        policy, _ = TestPPOOnnxExport()._make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            policy.export_onnx(onnx_path)
            # Delete the meta file
            meta = onnx_path.with_suffix(".onnx.meta.json")
            meta.unlink()
            with pytest.raises(FileNotFoundError):
                ONNXPolicy.from_file(onnx_path)
