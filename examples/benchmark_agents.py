"""Benchmark scripted, heuristic, and optional trained RL agents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opennpc.training.evaluate import benchmark_combat


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenNPC combat benchmarks.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ppo", type=str, default=None, help="Optional PPO checkpoint path.")
    parser.add_argument("--dqn", type=str, default=None, help="Optional DQN checkpoint path.")
    args = parser.parse_args()

    results = benchmark_combat(
        episodes=args.episodes,
        seed=args.seed,
        ppo_model_path=args.ppo,
        dqn_model_path=args.dqn,
    )
    print("| Agent | Win Rate | Avg Reward | Avg Steps | Diversity | Adaptability |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, metrics in results.items():
        print(
            f"| {name} | {metrics['win_rate']:.2f} | {metrics['average_reward']:.2f} | "
            f"{metrics['avg_steps']:.2f} | {metrics['action_diversity']:.2f} | {metrics['adaptability']:.2f} |"
        )


if __name__ == "__main__":
    main()
