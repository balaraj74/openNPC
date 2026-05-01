from opennpc.training.evaluate import benchmark_combat, compare_baseline


def test_compare_baseline_returns_core_metrics() -> None:
    results = compare_baseline(episodes=2, seed=3)

    assert set(results) == {"scripted", "opennpc_heuristic"}
    for metrics in results.values():
        assert metrics["episodes"] == 2
        assert "win_rate" in metrics
        assert "action_diversity" in metrics
        assert "adaptability" in metrics


def test_benchmark_combat_accepts_no_rl_checkpoint() -> None:
    results = benchmark_combat(episodes=1, seed=5)

    assert "scripted" in results
    assert "opennpc_heuristic" in results
