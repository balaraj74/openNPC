"""Tests for the training logger."""

import json

from opennpc.training.logger import TrainingLogger


def test_logger_episode_records(tmp_path) -> None:
    log_file = tmp_path / "training.jsonl"
    logger = TrainingLogger(log_path=log_file)

    logger.begin_episode(0)
    logger.log_step(0, "attack", 1.5, health=90, target_health=80)
    logger.log_step(1, "move", 0.2, health=88, target_health=78)
    episode = logger.end_episode(win=True)

    assert episode.total_reward == 1.7
    assert episode.steps == 2
    assert episode.win is True
    assert log_file.exists()

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["episode"] == 0
    assert record["win"] is True


def test_logger_summary_aggregation() -> None:
    logger = TrainingLogger()

    for i in range(5):
        logger.begin_episode(i)
        logger.log_step(0, "attack", float(i), health=100)
        logger.end_episode(win=(i % 2 == 0))

    summary = logger.summary()
    assert summary["episodes"] == 5
    assert summary["win_rate"] == 0.6  # 3 out of 5
    assert "total_reward_mean" in summary


def test_logger_action_distribution() -> None:
    logger = TrainingLogger()
    logger.begin_episode(0)
    logger.log_step(0, "attack", 1.0)
    logger.log_step(1, "attack", 0.5)
    logger.log_step(2, "move", 0.3)
    logger.end_episode()

    dist = logger.action_distribution()
    assert dist["attack"] == 2
    assert dist["move"] == 1


def test_logger_reward_curve() -> None:
    logger = TrainingLogger()
    for i in range(3):
        logger.begin_episode(i)
        logger.log_step(0, "idle", float(i) * 2)
        logger.end_episode()

    curve = logger.reward_curve()
    assert curve == [0.0, 2.0, 4.0]
