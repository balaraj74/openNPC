# OpenNPC — Demo Walkthrough

This guide walks through each runnable demo, explaining what it shows and how the SDK modules compose together.

---

## 1. Basic Decision (`examples/basic_decision.py`)

**What it demonstrates:** A single agent decision cycle — the simplest possible SDK usage.

**Pipeline:** `AgentConfig` → `GameState` → `DecisionEngine.decide()` → `ActionDecision`

```bash
python examples/basic_decision.py
```

**What to observe:**
- The engine selects `attack` because the enemy has high aggression (0.8), the target is close (1.0), and the target is weak (34 HP).
- The full `DecisionTrace` is printed, showing goal scores, personality influence, and memory summary.
- `fallback_used: false` confirms the heuristic policy produced a valid action.

---

## 2. Combat Simulation (`examples/run_simulation.py`)

**What it demonstrates:** A full combat loop in the `GridCombatEnv` — the enemy agent fights a scripted player on an 8x1 grid.

**Pipeline:** `GridCombatEnv` → per-tick `DecisionEngine.decide()` → `env.step()` → reward accumulation

```bash
python examples/run_simulation.py
```

**What to observe:**
- The grid is rendered each tick: `E` = enemy, `P` = player, `C` = cover.
- The enemy flanks to close distance, then switches to `attack` once in range.
- Positive rewards for damage dealt, negative for damage taken.
- `total_reward` at the end indicates overall combat performance.

---

## 3. Villain Adaptive Intelligence (`examples/villain_demo.py`)

**What it demonstrates:** The full strategic intelligence pipeline — a boss NPC adapts its behavior based on observed player patterns.

**Pipeline:** `PlayerPatternTracker.record()` → `VillainPlanner.plan()` → `DecisionEngine.decide()`

```bash
python examples/villain_demo.py
```

**What to observe:**
- **Strategy shifts** every 5 ticks: `defensive_attrition` → `exploit_predictability` → `retreat_and_rebuild`.
- The villain initially defends, then starts setting traps when it detects the player is predictable.
- At low HP, it switches to `seek_cover` and the planner recommends `retreat_and_rebuild`.
- The final **Pattern Analysis Summary** shows the tracker's assessment: aggression, predictability, and counter-recommendation.

**Key modules:** `opennpc.strategy.PlayerPatternTracker`, `opennpc.villain.VillainPlanner`

---

## 4. Village Social Simulation (`examples/village_demo.py`)

**What it demonstrates:** Non-combat NPC behavior — a civilian agent navigates a village, trades, talks, and responds to danger events.

**Pipeline:** `VillageEnv` → per-tick `DecisionEngine.decide()` → `env.step()` → reputation/gold tracking

```bash
python examples/village_demo.py
```

**What to observe:**
- The civilian alternates between `talk` and `trade` based on personality (high sociability).
- Reputation steadily increases from successful conversations.
- `⚠ DANGER!` events appear randomly — a well-tuned agent would `flee` or `hide`.
- End-of-day bonus rewards social interaction and trading.
- Memory summary shows important events the NPC remembered.

**Key modules:** `opennpc.simulation.village.VillageEnv`

---

## 5. LOD Scaling (`examples/lod_demo.py`)

**What it demonstrates:** The AI Level-of-Detail system assigning 50 agents to compute tiers.

**Pipeline:** `LODEngine.register()` → `LODEngine.evaluate_all()` → per-agent `LODProfile`

```bash
python examples/lod_demo.py
```

**What to observe:**
- Agents close to the camera get `FULL` tier (250ms interval, full policy, memory enabled).
- Distant invisible agents get `DORMANT` (5000ms interval, frozen).
- Combat agents always get `FULL` regardless of distance.
- High narrative importance pulls agents up a tier (effective distance reduction).
- The config modification example shows how `apply_to_config()` adjusts an agent's decision interval and policy.

**Key modules:** `opennpc.lod.LODEngine`, `opennpc.lod.LODTier`, `opennpc.lod.LODProfile`

---

## 6. Multi-Agent Coordination (`examples/coordination_demo.py`)

**What it demonstrates:** A squad of 3 raiders coordinates to avoid redundant behavior — only 1 attacks directly while others flank.

**Pipeline:** `MultiAgentCoordinator.coordinate()` → per-agent `DecisionEngine.decide()` → attack pressure balancing → role assignments

```bash
python examples/coordination_demo.py
```

**What to observe:**
- With `max_attackers_per_target=1`, only the highest-ranked attacker (raider_01, closest + most aggressive) gets the `attack` role.
- The other raiders are reassigned to `flank` with the `flanker` role.
- The **shared context** shows action counts, role counts, and target pressure across the squad.
- Coordination reasons explain why each agent was reassigned.

**Key modules:** `opennpc.coordination.MultiAgentCoordinator`, `opennpc.coordination.CoordinationPlan`

---

## 7. Combat Benchmark (`examples/benchmark_agents.py`)

**What it demonstrates:** Quantitative comparison of agent strategies across multiple episodes.

```bash
python examples/benchmark_agents.py --episodes 20
```

**What to observe:**
- **Win Rate:** OpenNPC heuristic agent wins ~95% vs the scripted baseline's ~5%.
- **Avg Reward:** Higher reward indicates better tactical decision-making.
- **Diversity:** Fraction of unique actions used — heuristic agents use more varied tactics.
- **Adaptability:** How often the agent chooses survival actions when at low health.

**With RL checkpoints:**
```bash
python examples/benchmark_agents.py --episodes 20 --ppo artifacts/ppo_enemy.pt
```

**Key modules:** `opennpc.training.evaluate.benchmark_combat`

---

## 8. PPO Training (`examples/train_enemy.py`)

**What it demonstrates:** Training an RL policy from scratch using PPO on the combat environment.

> **Requires:** `pip install -e ".[training]"` (PyTorch)

```bash
python examples/train_enemy.py
```

**What to observe:**
- Episode rewards should trend upward as the policy learns.
- Training uses the `GridCombatEnv` with the same state/action space as the heuristic demos.
- The trained policy can be loaded at runtime via `AgentConfig.metadata["model_path"]`.

**Key modules:** `opennpc.training.ppo`, `opennpc.training.evaluate`

---

## Running All Tests

```bash
python -m pytest tests/ -v
```

Expected: **61 tests, all passing.**

| Test File | Count | Coverage |
|---|---|---|
| `test_api.py` | 4 | FastAPI endpoints + dashboard |
| `test_async.py` | 3 | Async decision engine |
| `test_benchmark.py` | 2 | Combat benchmark harness |
| `test_coordination.py` | 4 | Multi-agent coordination |
| `test_decision_engine.py` | 8 | Core decisions + RL loading + rewards |
| `test_experience.py` | 2 | Runtime experience logger |
| `test_lod.py` | 7 | LOD tier assignment + config override |
| `test_memory.py` | 2 | In-memory + SQLite persistence |
| `test_rewards.py` | 3 | Reward shaping strategies |
| `test_simulation.py` | 2 | GridCombatEnv step + attack |
| `test_strategy.py` | 13 | Pattern tracker + villain planner |
| `test_training_logger.py` | 4 | JSONL training logger |
| `test_village.py` | 5 | VillageEnv social simulation |

---

## Debug Dashboard

Start the API server and open the dashboard in a browser:

```bash
uvicorn opennpc.api.service:app --port 8787
# then open http://localhost:8787/debug/dashboard
```

The dashboard shows:
- **Runtime summary** — total transitions, mean reward, terminal count
- **Recent decisions** — action, confidence, policy, fallback status, goal scores
- **Agent memory** — look up any agent's memory store and experience log
- **Player patterns** — aggression, predictability, counter-recommendation
- **LOD distribution** — current tier assignments across all agents

---

## Composing Modules in Your Game

A typical integration looks like this:

```python
from opennpc import (
    AgentConfig, AsyncDecisionEngine, DecisionCache, DecisionEngine,
    LODEngine, MultiAgentCoordinator,
    PlayerPatternTracker, VillainPlanner,
    RuntimeExperienceLogger,
)
from opennpc.training import ReplayConverter

# 1. Create core components
experience = RuntimeExperienceLogger(log_path="gameplay.jsonl")
engine = DecisionEngine(experience_logger=experience)
async_engine = AsyncDecisionEngine(max_workers=4)
lod = LODEngine()
cache = DecisionCache(ttl_seconds=2.0, tolerance=0.05)
tracker = PlayerPatternTracker()
planner = VillainPlanner(pattern_tracker=tracker)
coordinator = MultiAgentCoordinator(decision_engine=engine, max_attackers_per_target=2)

# 2. Per frame: update LOD
lod.update("boss_01", distance_to_camera=dist, is_visible=vis)
tier = lod.tier_for("boss_01")

# 3. Only decide when the LOD interval allows it
if should_decide(tier):
    # Record what the player did
    tracker.record(observed_player_action)

    # Generate strategic plan periodically
    if tick % 10 == 0:
        plan = planner.plan(config, state)

    # For squads, coordinate instead of individual decisions
    coord_plan = coordinator.coordinate(squad_configs, squad_states)

    # For single agents — check cache first, then async decide
    modified_config = lod.apply_to_config("boss_01", config)
    cached = cache.get(modified_config, state)
    if cached:
        decision = cached
    else:
        decision = await async_engine.decide(modified_config, state)
        cache.put(modified_config, state, decision)

    # 4. Apply to game world
    apply_action(decision.action)

# 5. Offline retraining from logged experience
converter = ReplayConverter.from_jsonl("gameplay.jsonl")
print(converter.summary())
# dqn_buffer = converter.to_dqn_replay_buffer()  # requires torch
```

For the inference API approach (Unreal/Unity/Godot), POST to `/decide`, `/batch/decide`, or `/coordinate` and the server handles all of this internally.
