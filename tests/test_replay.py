"""Tests for the replay-to-training converter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from opennpc.experience import ExperienceRecord, RuntimeExperienceLogger
from opennpc.training.replay import ReplayConverter, ReplayStats


def _make_record(
    agent_id: str = "enemy_01",
    action: str = "attack",
    reward: float = 1.0,
    done: bool = False,
) -> ExperienceRecord:
    return ExperienceRecord(
        agent_id=agent_id,
        state={
            "agent_id": agent_id,
            "health": 80,
            "stamina": 100,
            "threat_level": 0.5,
            "distance_to_target": 2.0,
            "target_health": 60,
            "cover_available": True,
        },
        action=action,
        reward=reward,
        next_state={
            "agent_id": agent_id,
            "health": 75,
            "stamina": 95,
            "threat_level": 0.6,
            "distance_to_target": 1.5,
            "target_health": 40,
            "cover_available": True,
        },
        done=done,
        policy_name="heuristic",
    )


class TestReplayConverter:
    def test_from_records_summary(self) -> None:
        records = [
            _make_record(action="attack", reward=1.0),
            _make_record(action="defend", reward=-0.5),
            _make_record(action="attack", reward=2.0, done=True),
        ]
        converter = ReplayConverter.from_records(records)
        stats = converter.summary()

        assert stats.total_transitions == 3
        assert stats.unique_agents == 1
        assert stats.unique_actions == 2
        assert stats.terminal_count == 1
        assert stats.action_distribution["attack"] == 2
        assert stats.action_distribution["defend"] == 1
        assert stats.reward_min == -0.5
        assert stats.reward_max == 2.0

    def test_from_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "test_experience.jsonl"
        records = [
            _make_record(action="attack", reward=1.0),
            _make_record(action="flee", reward=-1.0, done=True),
        ]
        with open(log_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_dict()) + "\n")

        converter = ReplayConverter.from_jsonl(log_path)
        stats = converter.summary()
        assert stats.total_transitions == 2
        assert stats.terminal_count == 1

    def test_from_jsonl_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ReplayConverter.from_jsonl(tmp_path / "nonexistent.jsonl")

    def test_filter_by_agent(self) -> None:
        records = [
            _make_record(agent_id="a", reward=1.0),
            _make_record(agent_id="b", reward=2.0),
            _make_record(agent_id="a", reward=3.0),
        ]
        converter = ReplayConverter.from_records(records)
        filtered = converter.filter(agent_id="a")
        assert filtered.summary().total_transitions == 2

    def test_filter_by_reward(self) -> None:
        records = [
            _make_record(reward=-2.0),
            _make_record(reward=0.0),
            _make_record(reward=5.0),
        ]
        converter = ReplayConverter.from_records(records)
        filtered = converter.filter(min_reward=0.0)
        assert filtered.summary().total_transitions == 2

    def test_filter_by_action(self) -> None:
        records = [
            _make_record(action="attack"),
            _make_record(action="defend"),
            _make_record(action="attack"),
        ]
        converter = ReplayConverter.from_records(records)
        filtered = converter.filter(action="attack")
        assert filtered.summary().total_transitions == 2

    def test_empty_summary(self) -> None:
        converter = ReplayConverter.from_records([])
        stats = converter.summary()
        assert stats.total_transitions == 0
        assert stats.reward_mean == 0.0

    def test_stats_to_dict(self) -> None:
        records = [_make_record(reward=1.5)]
        converter = ReplayConverter.from_records(records)
        stats_dict = converter.summary().to_dict()
        assert isinstance(stats_dict, dict)
        assert "total_transitions" in stats_dict
        assert "action_distribution" in stats_dict


class TestReplayConverterTorch:
    """Tests that require PyTorch."""

    @pytest.fixture(autouse=True)
    def _check_torch(self) -> None:
        pytest.importorskip("torch")

    def test_to_dqn_replay_buffer(self) -> None:
        records = [
            _make_record(action="attack", reward=1.0),
            _make_record(action="defend", reward=-0.5),
        ]
        converter = ReplayConverter.from_records(records)
        buffer = converter.to_dqn_replay_buffer()
        assert len(buffer) == 2

    def test_to_ppo_rollout(self) -> None:
        records = [
            _make_record(action="attack", reward=1.0),
            _make_record(action="move", reward=0.0),
        ]
        converter = ReplayConverter.from_records(records)
        rollout = converter.to_ppo_rollout()
        assert len(rollout["states"]) == 2
        assert len(rollout["actions"]) == 2
        assert len(rollout["rewards"]) == 2

    def test_unknown_action_skipped(self) -> None:
        records = [_make_record(action="teleport_home")]
        converter = ReplayConverter.from_records(records)
        rollout = converter.to_ppo_rollout()
        assert len(rollout["states"]) == 0
