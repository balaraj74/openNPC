"""Tests for the player pattern tracker and villain planner."""

from opennpc.strategy import PlayerPatternTracker
from opennpc.villain import VillainPlanner
from opennpc.types import AgentConfig, AgentType, GameState, Goal, Personality


def _villain_config(**kwargs) -> AgentConfig:
    defaults = dict(
        agent_id="villain_01",
        agent_type=AgentType.VILLAIN,
        personality=Personality(aggression=0.7, caution=0.6, patience=0.8),
        goals=[
            Goal("weaken_player", 0.9),
            Goal("control_area", 0.75),
            Goal("survive", 0.55),
            Goal("trigger_strategic_traps", 0.7),
        ],
        allowed_actions=["idle", "move", "attack", "defend", "flee", "seek_cover", "flank", "set_trap", "patrol", "hide"],
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _state(**kwargs) -> GameState:
    defaults = dict(
        agent_id="villain_01",
        agent_type=AgentType.VILLAIN,
        health=80,
        target_health=70,
        distance_to_target=4.0,
        threat_level=0.3,
    )
    defaults.update(kwargs)
    return GameState(**defaults)


# --- PlayerPatternTracker tests ---

def test_empty_tracker_defaults() -> None:
    tracker = PlayerPatternTracker()
    assert tracker.dominant_action() == "unknown"
    assert tracker.aggression_estimate() == 0.5
    assert tracker.predictability() == 0.0
    assert tracker.total_observations == 0


def test_record_and_dominant_action() -> None:
    tracker = PlayerPatternTracker()
    for _ in range(5):
        tracker.record("attack")
    tracker.record("move")
    assert tracker.dominant_action() == "attack"
    assert tracker.total_observations == 6


def test_aggression_estimate_high_for_aggressive_player() -> None:
    tracker = PlayerPatternTracker()
    for action in ["attack", "flank", "charge", "attack", "set_trap"]:
        tracker.record(action)
    assert tracker.aggression_estimate() >= 0.6


def test_predictability_high_for_repetitive_player() -> None:
    tracker = PlayerPatternTracker(window_size=10)
    for _ in range(10):
        tracker.record("attack")
    assert tracker.predictability() == 1.0


def test_counter_recommendation_returns_string() -> None:
    tracker = PlayerPatternTracker()
    for _ in range(10):
        tracker.record("attack")
    rec = tracker.counter_recommendation()
    assert isinstance(rec, str) and len(rec) > 0


def test_snapshot_creates_record() -> None:
    tracker = PlayerPatternTracker()
    tracker.record("move")
    snap = tracker.snapshot()
    assert snap.dominant_action == "move"
    assert snap.timestamp > 0


def test_summary_includes_analysis() -> None:
    tracker = PlayerPatternTracker()
    for _ in range(8):
        tracker.record("attack")
    summary = tracker.summary()
    assert "attack" in summary.lower()
    assert "aggression" in summary.lower()


def test_to_dict_complete() -> None:
    tracker = PlayerPatternTracker()
    tracker.record("move")
    tracker.record("attack")
    d = tracker.to_dict()
    assert "dominant_action" in d
    assert "aggression_estimate" in d
    assert "counter_recommendation" in d
    assert "recent_actions" in d


# --- VillainPlanner tests ---

def test_villain_plan_basic() -> None:
    tracker = PlayerPatternTracker()
    planner = VillainPlanner(pattern_tracker=tracker)
    config = _villain_config()
    state = _state()
    plan = planner.plan(config, state)
    assert plan.next_action in config.allowed_actions or plan.next_action in {"idle"}
    assert len(plan.long_term_strategy) > 0
    assert 0.0 < plan.confidence <= 1.0


def test_villain_retreats_at_low_health() -> None:
    tracker = PlayerPatternTracker()
    planner = VillainPlanner(pattern_tracker=tracker)
    config = _villain_config()
    state = _state(health=15)
    plan = planner.plan(config, state)
    assert plan.long_term_strategy == "retreat_and_rebuild"
    assert plan.next_action in {"flee", "hide"}


def test_villain_exploits_predictable_player() -> None:
    tracker = PlayerPatternTracker(window_size=10)
    for _ in range(10):
        tracker.record("attack")
    planner = VillainPlanner(pattern_tracker=tracker)
    config = _villain_config()
    state = _state()
    plan = planner.plan(config, state)
    assert plan.long_term_strategy == "exploit_predictability"


def test_villain_presses_weak_target() -> None:
    tracker = PlayerPatternTracker()
    planner = VillainPlanner(pattern_tracker=tracker)
    config = _villain_config()
    state = _state(target_health=20)
    plan = planner.plan(config, state)
    assert plan.long_term_strategy == "press_advantage"


def test_villain_adjusts_goal_priorities() -> None:
    tracker = PlayerPatternTracker()
    planner = VillainPlanner(pattern_tracker=tracker)
    config = _villain_config()
    state = _state(health=15)
    plan = planner.plan(config, state)
    survive_goals = [g for g in plan.adjusted_goals if g.name == "survive"]
    assert len(survive_goals) == 1
    assert survive_goals[0].priority > 0.55  # boosted from original


def test_planner_counts_plans() -> None:
    planner = VillainPlanner()
    config = _villain_config()
    state = _state()
    planner.plan(config, state)
    planner.plan(config, state)
    assert planner.plans_generated == 2
    assert "2 plans" in planner.summary()
