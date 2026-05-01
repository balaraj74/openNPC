# OpenNPC Architecture

OpenNPC is organized into eight layers, each independently testable and composable:

## 1. Core SDK (`opennpc/`)

The foundation. Data models, memory, goals, policies, action validation, and decision traces.

- **`types.py`** — Shared data models: `AgentConfig`, `GameState`, `ActionDecision`, `DecisionTrace`, `Goal`, `Personality`, `MemoryEvent`. All serializable to/from dicts and JSON.
- **`decision.py`** — `DecisionEngine`: the central orchestrator. Takes config + state → produces a validated action with full trace. Supports dynamic RL policy loading, runtime reward feedback, and per-agent reward shaping.
- **`goals.py`** — `GoalScorer` / `GoalManager`: evaluates candidate actions against active goals, weighted by priority and personality.
- **`memory.py`** — `InMemoryMemoryStore` (prototyping) and `SQLiteMemoryStore` (production). Both implement store, recall, summarize, and forget.
- **`policy.py`** — Policy interfaces: `HeuristicPolicy` (personality-weighted), `ScriptedCombatPolicy` (rule-based), `RandomPolicy` (baseline). RL policies loaded dynamically from checkpoints.

## 2. Reward Shaping (`opennpc/rewards.py`)

Agent-type-specific reward strategies that break down composite rewards into named components.

- **`EnemyRewardStrategy`** — Balances damage dealt, damage taken, positioning, and survival.
- **`VillainRewardStrategy`** — Favors long-term pressure, area control, and trap setup. Penalizes reckless low-health behavior.
- **`CivilianRewardStrategy`** — Rewards reputation growth, productive social actions, and danger response. Penalizes ignoring danger.
- **`CompanionRewardStrategy`** — Prioritizes ally protection, staying close, and support actions.
- **`reward_strategy_for(agent_type)`** — Factory that returns the correct strategy by agent type.

## 3. Strategic Intelligence (`opennpc/strategy.py`, `villain.py`, `lod.py`)

Higher-level reasoning modules that sit above the per-tick decision engine.

- **`PlayerPatternTracker`** — Records player actions in a sliding window. Computes aggression estimate, predictability score, approach tendency, dominant action, and counter-recommendations.
- **`VillainPlanner`** — Consumes tracker data to generate multi-step strategic plans. Adapts between strategies: `defensive_attrition`, `exploit_predictability`, `aggressive_push`, `retreat_and_rebuild`, `area_denial`. Adjusts goal priorities dynamically.
- **`LODEngine`** — AI Level-of-Detail. Assigns agents to compute tiers (FULL → REDUCED → MINIMAL → DORMANT) based on camera distance, visibility, narrative importance, and combat status.

## 4. Multi-Agent Coordination (`opennpc/coordination.py`)

Squad-level tactical intelligence that ensures a group of agents doesn't all make the same decision.

- **`MultiAgentCoordinator`** — Runs individual `DecisionEngine` decisions, then applies encounter rules: limits attackers per target, reassigns excess agents to support roles (flank, set_trap, defend).
- **`CoordinationAssignment`** — Tactical role per agent: assault, flanker, area_denial, guard, escort, etc.
- **`CoordinationPlan`** — Full coordinated output: decisions, role assignments, and shared context (action counts, target pressure).

## 5. Runtime Experience (`opennpc/experience.py`)

Gameplay transition logging for online feedback and offline retraining.

- **`RuntimeExperienceLogger`** — Records `(state, action, reward, next_state)` transitions. Stores in memory with optional JSONL persistence. Supports per-agent filtering, summary statistics, and circular buffer eviction.
- **`ExperienceRecord`** — One gameplay transition suitable for replay buffers or offline retraining.

## 6. Async & Scaling (`opennpc/async_engine.py`)

- **`AsyncDecisionEngine`** — Wraps `DecisionEngine` in a `ThreadPoolExecutor` for non-blocking calls from game engine threads. Supports single decisions, batch decisions, and scheduled-interval decisions.

## 7. Simulation & Training

### Environments (`opennpc/simulation/`)
- **`GridCombatEnv`** — 8x1 grid combat sandbox with cover mechanics, flanking, and health tracking.
- **`VillageEnv`** — Social simulation with trading, talking, patrolling, and danger events.
- **`ScriptedCombatAgent`** — Deterministic baseline opponent for benchmarking.

### Training Pipelines (`opennpc/training/`)
- **`ppo.py`** — PPO with actor-critic network, GAE, and clipped surrogate loss.
- **`dqn.py`** — DQN with experience replay, target network, and epsilon-greedy exploration.
- **`evaluate.py`** — Evaluation harness for average reward, win rate, and episode length.
- **`logger.py`** — Structured JSONL logging for training runs.

## 8. Runtime API & Engine Adapters

### API (`opennpc/api/service.py`)
FastAPI service with endpoints:
- `POST /decide` — Single agent decision
- `POST /batch/decide` — Batch decisions
- `POST /coordinate` — Multi-agent coordinated decisions
- `GET /memory/{agent_id}` — Agent memory inspection
- `GET /debug/dashboard` — Browser-based debug dashboard
- `GET /debug/decisions` — Recent decision traces
- `GET /debug/experience` — Runtime experience summary
- `GET /debug/lod` — LOD tier distribution
- `GET /debug/patterns` — Player pattern analysis
- `POST /debug/villain/plan` — Villain strategic plan generation

### Engine Adapters (`adapters/`)
- **Unity** — `OpenNPCClient.cs` (REST client) + `OpenNPCDemoController.cs` (demo scaffold)
- **Unreal** — `OpenNPCClient.h/.cpp` (HTTP adapter, Blueprint-ready)
- **Godot** — `open_npc_client.gd` (GDScript, signal-based async)

---

## Decision Pipeline

```
GameState arrives
    ↓
1. Ingest recent events → MemoryStore
2. Compute reward from previous transition (if reward feedback enabled)
3. Log experience record to RuntimeExperienceLogger
4. Summarize short-term + long-term memories
5. Score active goals × valid actions × personality
6. Load RL policy from checkpoint (if available, else heuristic fallback)
7. Query policy → validate action against constraints
8. Store decision memory + trace
9. Return ActionDecision + DecisionTrace
```

## Coordination Pipeline

```
Multiple (config, state) pairs arrive
    ↓
1. Run individual DecisionEngine.decide() per agent
2. Assign tactical roles (assault, flanker, guard, etc.)
3. Count attackers per target
4. Reassign excess attackers to support actions (flank, set_trap, defend)
5. Record final coordinated decisions to memory
6. Return CoordinationPlan with decisions, roles, and shared context
```

## Performance Scaling

The LOD engine ensures that even with 1000+ agents, only a fraction run full AI each frame:

| Tier | Interval | Policy | Memory | Use Case |
|---|---|---|---|---|
| FULL | 250ms | Full RL/heuristic | Enabled | Player-facing, in-combat |
| REDUCED | 500ms | Heuristic only | Enabled | Nearby, visible |
| MINIMAL | 1000ms | Scripted fallback | Disabled | Background, distant |
| DORMANT | 5000ms | Frozen (last action) | Disabled | Off-screen, unimportant |

## Runtime Reliability

The decision engine keeps a heuristic fallback at all times. If a learned policy is missing, throws, or selects an invalid action, OpenNPC returns a validated fallback and marks `trace.fallback_used = True`. Missing RL checkpoints are handled gracefully without crashing.

## LLM Reasoning Layer

Prompt templates live in `opennpc/prompts.py`. They are intended for occasional high-level planning, dialogue, or strategy — never for frame-by-frame action selection. The core decision loop runs entirely locally without network calls.
