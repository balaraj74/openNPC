# OpenNPC Benchmarks

Benchmarks are run against the reference `GridCombatEnv` combat simulation.

Command:

```bash
PYTHONPATH=. python3 examples/benchmark_agents.py --episodes 20
```

Current local result:

| Agent | Win Rate | Avg Reward | Avg Steps | Diversity | Adaptability |
|---|---:|---:|---:|---:|---:|
| scripted | 0.05 | 4.54 | 57.55 | 0.44 | 0.97 |
| opennpc_heuristic | 0.95 | 6.31 | 16.35 | 0.67 | 0.62 |

## Metrics

- **Win Rate:** fraction of episodes where the enemy defeats the player.
- **Avg Reward:** mean episode reward in `GridCombatEnv`.
- **Avg Steps:** mean episode length.
- **Diversity:** unique actions used divided by the combat action space.
- **Adaptability:** fraction of low-health moments where the agent chose a survival action.

## RL Checkpoints

PPO and DQN rows are included when a checkpoint is supplied:

```bash
PYTHONPATH=. python3 examples/benchmark_agents.py --episodes 20 --ppo artifacts/ppo_enemy.pt
PYTHONPATH=. python3 examples/benchmark_agents.py --episodes 20 --dqn artifacts/dqn_enemy.pt
```

This keeps published numbers honest: the repo does not invent trained-model results when no checkpoint is present.
