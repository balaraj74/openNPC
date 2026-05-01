import json

from opennpc import RuntimeExperienceLogger


def test_runtime_experience_logger_writes_jsonl(tmp_path) -> None:
    log_path = tmp_path / "experience.jsonl"
    logger = RuntimeExperienceLogger(log_path=log_path)

    record = logger.record(
        agent_id="enemy_01",
        state={"health": 90},
        action="attack",
        reward=1.5,
        next_state={"health": 80},
        policy_name="ppo",
    )

    assert record.action == "attack"
    data = json.loads(log_path.read_text().strip())
    assert data["agent_id"] == "enemy_01"
    assert data["policy_name"] == "ppo"
    assert data["reward"] == 1.5


def test_runtime_experience_summary_and_filtering() -> None:
    logger = RuntimeExperienceLogger(max_records=2)

    logger.record("enemy_01", {}, "move", 0.2, {}, policy_name="heuristic")
    logger.record("enemy_02", {}, "attack", -0.4, {}, policy_name="heuristic", done=True)
    logger.record("enemy_01", {}, "defend", 0.6, {}, policy_name="heuristic")

    summary = logger.summary()
    assert summary["transitions"] == 2
    assert summary["terminal_transitions"] == 1
    assert summary["by_agent"] == {"enemy_02": 1, "enemy_01": 1}
    assert [record.action for record in logger.recent("enemy_01")] == ["defend"]
