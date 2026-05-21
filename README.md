# OpenNPC — Autonomous NPC Intelligence SDK

> A Python-first, engine-agnostic framework for autonomous game characters.
> From heuristic villagers to RL-trained bosses — one unified pipeline.

OpenNPC provides a production-ready foundation for building intelligent NPCs that perceive the game world, form goals, remember interactions, make context-aware decisions, and adapt to player behavior — all without requiring an LLM at runtime.

---

## Features

| Layer | Capability |
|---|---|
| **Decision Engine** | Personality-weighted heuristic policy → RL policy → deterministic fallback |
| **Goal System** | Priority-scored goals with constraint validation and dynamic re-ranking |
| **Memory** | In-memory short-term + SQLite persistent long-term memory with forgetting curves |
| **Strategy** | `PlayerPatternTracker` + `VillainPlanner` for adaptive enemy intelligence |
| **LLM Brain** | Local small LLM (Qwen 0.5B) for NPC dialogue, personality text, and strategic planning |
| **LOD Engine** | AI Level-of-Detail that scales compute budget by distance/visibility/importance |
| **Async Engine** | Thread-pool backed async wrapper for non-blocking game engine integration |
| **Training** | PyTorch PPO + DQN pipelines with experience replay and target networks |
| **Simulation** | Grid combat env + village social env for offline training and evaluation |
| **Inference API** | FastAPI service with batch decisions, debug endpoints, and health checks |
| **Engine Adapters** | Unity C# REST client, Unreal C++ HTTP client, Minecraft Forge mod |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourusername/openNPC.git
cd openNPC
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,training,dev]"
```

### 2. Verify

```bash
pytest -q
```

### 3. Run Demos

```bash
# Basic single-decision example
python examples/basic_decision.py

# Combat simulation loop
python examples/run_simulation.py

# Villain adaptive intelligence (boss NPC vs player)
python examples/villain_demo.py

# Village social simulation (civilian NPCs)
python examples/village_demo.py

# LOD scaling with 50 agents
python examples/lod_demo.py

# LLM-enhanced dialogue and strategic planning
python examples/llm_dialogue_demo.py
```

### 4. Start the Inference API

```bash
uvicorn opennpc.api.service:app --reload --port 8787
```

POST decisions to `http://127.0.0.1:8787/decide`. Debug endpoints at `/debug/lod`, `/debug/patterns`, `/debug/villain/plan`.

---

## Minimal Usage

```python
from opennpc import (
    AgentConfig, AgentType, DecisionEngine,
    GameState, Goal, Personality,
)

engine = DecisionEngine()

config = AgentConfig(
    agent_id="guard_01",
    agent_type=AgentType.ENEMY,
    personality=Personality(aggression=0.8, caution=0.3, risk_tolerance=0.7),
    goals=[Goal("attack_target", 0.8), Goal("survive", 0.6)],
    allowed_actions=["move", "attack", "defend", "flee", "seek_cover", "flank"],
)

state = GameState(
    agent_id="guard_01",
    health=74,
    threat_level=0.6,
    distance_to_target=1.0,
)

decision = engine.decide(config, state)
print(decision.action, decision.confidence, decision.reason)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Game Engine                                │
│   (Unity / Unreal / Godot / Custom)                                │
│      ↓ HTTP/REST                                                    │
├─────────────────────────────────────────────────────────────────────┤
│  FastAPI Inference Service                                          │
│    /decide  /batch  /health  /debug/*                               │
├─────────────────────────────────────────────────────────────────────┤
│  AsyncDecisionEngine (thread-pool wrapper)                          │
├─────────────────────────────────────────────────────────────────────┤
│  LODEngine                   │ VillainPlanner + PatternTracker      │
│  (compute budget scaling)    │ (adaptive strategic planning)        │
├──────────────────────────────┼──────────────────────────────────────┤
│                   DecisionEngine                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ GoalScorer│  │ Policy   │  │ Memory   │  │ ActionValidator  │    │
│  │           │  │ (Heurist │  │ Store    │  │ (constraint      │    │
│  │ priority  │  │  /RL/    │  │ (RAM +   │  │  enforcement)    │    │
│  │ scoring   │  │  Script) │  │  SQLite) │  │                  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│  Types Layer: AgentConfig, GameState, ActionDecision, DecisionTrace │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Layout

```text
openNPC/
├── opennpc/                   Core SDK
│   ├── __init__.py            Public API surface
│   ├── types.py               Shared data models (AgentConfig, GameState, etc.)
│   ├── decision.py            DecisionEngine — main orchestration
│   ├── goals.py               GoalScorer — priority-weighted goal evaluation
│   ├── memory.py              InMemoryMemoryStore + SQLiteMemoryStore
│   ├── policy.py              HeuristicPolicy, ScriptedCombatPolicy, RandomPolicy
│   ├── llm.py                 LLMEngine — local small LLM backend (Qwen, SmolLM, Phi-3)
│   ├── dialogue.py            NPCDialogue — barks, conversation, personality text
│   ├── prompts.py             Prompt templates for LLM reasoning
│   ├── lod.py                 LODEngine — AI Level-of-Detail scaling
│   ├── strategy.py            PlayerPatternTracker — behavioral analysis
│   ├── villain.py             VillainPlanner — LLM-enhanced adaptive planning
│   ├── async_engine.py        AsyncDecisionEngine — non-blocking wrapper
│   ├── api/
│   │   └── service.py         FastAPI inference + debug + dialogue endpoints
│   ├── adapters/
│   │   └── adapter_utils.py   Python-side adapter helpers
│   ├── simulation/
│   │   ├── environment.py     GridCombatEnv — reference combat sim
│   │   ├── village.py         VillageEnv — social simulation
│   │   └── baseline.py        Scripted combat baselines
│   └── training/
│       ├── ppo.py             PPO training pipeline
│       ├── dqn.py             DQN training with replay + target net
│       ├── evaluate.py        Policy evaluation harness
│       └── logger.py          Structured training logger (JSONL)
├── adapters/
│   ├── unity/                 Unity C# REST client
│   ├── unreal/                Unreal C++ HTTP adapter (Blueprint-ready)
│   └── minecraft_forge/       Minecraft Forge 1.20.1 mod (Java)
├── configs/                   Example agent configs (enemy, civilian, companion, villain)
├── examples/                  Runnable demos
├── tests/                     Unit + integration tests
├── docs/                      Architecture notes
└── pyproject.toml             Package metadata & extras
```

---

## Module Reference

### DecisionEngine

The central orchestrator. Takes an `AgentConfig` + `GameState` → produces an `ActionDecision` with full `DecisionTrace`.

```python
engine = DecisionEngine()
decision = engine.decide(config, state)
# decision.action      → chosen action string
# decision.confidence  → 0.0–1.0 confidence score
# decision.reason      → human-readable explanation
# decision.trace       → full DecisionTrace with goal scores, memory, etc.
```

### LODEngine

Assigns agents to compute tiers based on distance, visibility, narrative importance, and combat status.

```python
from opennpc import LODEngine

lod = LODEngine()
lod.register("npc_01", distance_to_camera=50.0, is_visible=True, narrative_importance=0.2)
tier = lod.tier_for("npc_01")           # LODTier.MEDIUM
modified_config = lod.apply_to_config("npc_01", config)  # reduced interval/policy
```

| Tier | Decision Interval | Policy | Memory |
|---|---|---|---|
| CRITICAL | 100ms | Full RL | Enabled |
| HIGH | 250ms | Full heuristic | Enabled |
| MEDIUM | 500ms | Simplified | Disabled |
| LOW | 1000ms | Random | Disabled |
| DORMANT | Frozen | None | Disabled |

### PlayerPatternTracker + VillainPlanner

Adaptive enemy intelligence. The tracker records player actions and computes behavioral profiles. The planner generates strategic responses.

```python
from opennpc import PlayerPatternTracker, VillainPlanner, LLMEngine

tracker = PlayerPatternTracker(window_size=15)
tracker.record("attack")
tracker.record("attack")
tracker.record("flank")

print(tracker.aggression_estimate())    # 0.67
print(tracker.counter_recommendation()) # "defensive_positioning"

# With optional LLM for enriched strategic reasoning
llm = LLMEngine(model_name="qwen-0.5b")
planner = VillainPlanner(pattern_tracker=tracker, llm=llm)
plan = planner.plan(config, state)
print(plan.long_term_strategy)          # LLM-enriched strategy
print(plan.predicted_player_response)   # What the AI expects the player to do
print(plan.taunt)                       # In-character villain taunt
```

### LLMEngine + NPCDialogue

Local small LLM for dialogue, personality text, and strategic planning. Runs on GPU (CUDA) or CPU. Lazy-loaded on first use.

```python
from opennpc import LLMEngine, NPCDialogue

# Supported models (all fit in 4GB VRAM):
# "qwen-0.5b"    → Qwen2.5-0.5B-Instruct (default, best quality/speed)
# "smollm-360m"  → SmolLM2-360M-Instruct (fastest, smallest)
# "smollm-1.7b"  → SmolLM2-1.7B-Instruct (best quality, needs 2GB)
# "phi-3-mini"   → Phi-3-mini-4k-instruct (best reasoning, needs 3GB)

llm = LLMEngine(model_name="qwen-0.5b")  # auto-selects CUDA or CPU
dialogue = NPCDialogue(llm=llm)

# Combat barks — short, cached, fast (~150ms after warmup)
bark = dialogue.bark(enemy_config, category="combat_taunt")
print(bark.text)  # "Drop your weapons, you're toast!"

# NPC conversation — longer, context-aware
response = dialogue.converse(
    civilian_config, state,
    player_message="Can you help me find the temple?",
    conversation_history=[{"role": "player", "text": "Hello!"}],
)
print(response.text)  # "The temple is behind the ancient gate..."

# Personality description — for character sheets
desc = dialogue.personality_description(villain_config)
print(desc)  # Rich narrative description
```

### AsyncDecisionEngine

Non-blocking wrapper for game engine threads.

```python
import asyncio
from opennpc import AsyncDecisionEngine

async_engine = AsyncDecisionEngine(max_workers=4)
decision = await async_engine.decide(config, state)
```

### OpenNPCSDK

Recommended facade for engine integrations that need a compact sync/async API,
decision caching, and policy registration from one object.

```python
from opennpc import AgentConfig, GameState, Goal, OpenNPCSDK

with OpenNPCSDK() as npc:
    config = AgentConfig(
        agent_id="guard_01",
        goals=[Goal("survive", 1.0)],
        allowed_actions=["idle", "defend", "flee"],
    )
    state = GameState(agent_id="guard_01", health=42, threat_level=0.7)

    decision = npc.decide(config, state)
    print(decision.action)
```

### Production API Settings

The REST service is local-development friendly by default. For production,
set these environment variables before running `uvicorn`:

```bash
export OPENNPC_ENV=production
export OPENNPC_API_KEY="replace-with-a-long-random-token"
export OPENNPC_DEBUG=false
uvicorn opennpc.api.service:app --host 127.0.0.1 --port 8787
```

Clients can authenticate with `Authorization: Bearer <token>` or
`X-OpenNPC-Key: <token>`. Batch size, agent id length, event length, and debug
memory limits can be tuned with `OPENNPC_MAX_BATCH_SIZE`,
`OPENNPC_MAX_AGENT_ID_LENGTH`, `OPENNPC_MAX_EVENT_LENGTH`, and
`OPENNPC_MAX_MEMORY_LIMIT`.

### Memory

```python
from opennpc import InMemoryMemoryStore, SQLiteMemoryStore

# In-memory (for prototyping)
mem = InMemoryMemoryStore(max_events=200)

# Persistent (for production)
mem = SQLiteMemoryStore("npc_memory.db")

mem.store("guard_01", MemoryEvent(event_type="combat", description="Attacked intruder"))
recent = mem.recall("guard_01", limit=5)
summary = mem.summarize("guard_01")
```

---

## Configuration

Agent configs live in `configs/`. Four templates are provided:

| File | Agent Type | Focus |
|---|---|---|
| `enemy.json` | ENEMY | Combat-oriented attacker |
| `villain.json` | VILLAIN | Adaptive boss with strategic planning |
| `civilian.json` | CIVILIAN | Social NPC with pacifist constraint |
| `companion.json` | COMPANION | Player-support with protection goals |

Key config fields:

```json
{
  "agent_id": "guard_01",
  "agent_type": "ENEMY",
  "personality": {
    "aggression": 0.8,
    "caution": 0.3,
    "patience": 0.5,
    "risk_tolerance": 0.7,
    "loyalty": 0.6
  },
  "goals": [
    {"name": "attack_target", "priority": 0.8},
    {"name": "survive", "priority": 0.6}
  ],
  "constraints": [],
  "allowed_actions": ["move", "attack", "defend", "flee"],
  "rl_policy": "heuristic",
  "decision_interval_ms": 250
}
```

---

## Training

### PPO (reference combat environment)

```bash
python -m opennpc.training.ppo --episodes 500 --lr 3e-4 --gamma 0.99
```

### DQN (with experience replay)

```python
from opennpc.training.dqn import DQNTrainer

trainer = DQNTrainer(state_dim=8, action_dim=6, buffer_size=10000)
# Training loop: trainer.store_transition(...) → trainer.train_step()
```

### Evaluation

```python
from opennpc.training.evaluate import evaluate_policy

metrics = evaluate_policy(policy, env, episodes=100)
print(metrics)  # avg_reward, win_rate, avg_length
```

---

## API Reference

### POST `/decide`

```json
{
  "agent_id": "guard_01",
  "agent_type": "ENEMY",
  "health": 74,
  "threat_level": 0.6,
  "distance_to_target": 1.0
}
```

Response:

```json
{
  "action": "attack",
  "confidence": 0.85,
  "reason": "High aggression + close range + active attack goal",
  "fallback_used": false
}
```

### Debug Endpoints

| Endpoint | Description |
|---|---|
| `GET /debug/lod` | Current LOD tier assignments |
| `GET /debug/patterns` | Player pattern analysis |
| `GET /debug/villain/plan` | Latest villain strategic plan |

### Dialogue Endpoints

| Endpoint | Description |
|---|---|
| `POST /dialogue/bark` | Short in-character bark/flavor line (combat taunts, greetings) |
| `POST /dialogue/converse` | Conversational NPC dialogue response |
| `POST /dialogue/personality` | Rich personality description for NPC |
| `GET /llm/status` | LLM engine status (loaded, device, cache) |
| `POST /llm/load` | Pre-warm the LLM model |
| `GET /health` | Service health check |

---

## Game Engine Integration

### Unity (C#)

Drop `adapters/unity/OpenNPCClient.cs` into your Unity project. Uses `UnityWebRequest` to POST to the inference API.

```csharp
var client = gameObject.AddComponent<OpenNPCClient>();
client.serverUrl = "http://127.0.0.1:8787";
client.RequestDecision(gameState, decision => {
    Debug.Log($"Action: {decision.action}");
});
```

### Unreal Engine (C++)

Add `adapters/unreal/OpenNPCClient.h` and `.cpp` to your project. Requires `Http`, `Json`, and `JsonUtilities` modules in your `.Build.cs`.

```cpp
UOpenNPCClient* Client = NewObject<UOpenNPCClient>();
Client->ServerUrl = TEXT("http://127.0.0.1:8787");
Client->RequestDecision(GameState, [](const FOpenNPCDecision& Decision) {
    UE_LOG(LogTemp, Log, TEXT("Action: %s"), *Decision.Action);
});
```

### Godot / Custom Engines

Use any HTTP client to POST JSON to `/decide`. The API is engine-agnostic.

---

## Design Principles

1. **Local-first decisions.** Frequent decisions use lightweight policies. LLMs are optional, reserved for dialogue and high-level planning.
2. **Every action is validated.** Constraint enforcement and fallback policies guarantee a valid action is always returned.
3. **Full decision tracing.** Every decision records the policy used, goal scores, memory context, and fallback path.
4. **Engine-agnostic core.** The Python SDK knows nothing about Unity/Unreal/Godot. Integration happens through REST or native adapters.
5. **Scale by fidelity.** The LOD engine ensures you can run 1000+ agents without linearly scaling compute.

---

## Dependencies

| Extra | Packages |
|---|---|
| Core | `pydantic`, `numpy` |
| `[api]` | `fastapi`, `uvicorn` |
| `[training]` | `torch` |
| `[llm]` | `transformers`, `accelerate`, `torch` |
| `[dev]` | `pytest`, `httpx` |

---

## License

MIT — see `LICENSE` for details.
