from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opennpc.training.ppo import train_and_save


def main() -> None:
    output = Path("artifacts/ppo_enemy.pt")
    rewards = train_and_save(output, episodes=80)
    print(f"saved={output}")
    print(f"first_reward={rewards[0]:.2f} last_reward={rewards[-1]:.2f}")
    print('runtime_policy={"rl_policy": "ppo", "metadata": {"model_path": "artifacts/ppo_enemy.pt"}}')


if __name__ == "__main__":
    main()
