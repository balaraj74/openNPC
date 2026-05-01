# OpenNPC Project Requirements

## 1. Project Name
**OpenNPC** — an open-source AI framework for autonomous game characters.

## 2. Vision
Build a plug-and-play SDK that game developers can integrate into their games to create:
- Non-scripted NPCs
- Adaptive enemies
- Strategic villains/bosses
- Persistent, personality-driven agents

The system should support learning from experience, memory, goals, and behavior variation across characters.

## 3. Problem Statement
Most game characters rely on scripted logic, behavior trees, or finite state machines. This leads to repetitive, predictable behavior. OpenNPC aims to replace or augment scripted behavior with a modular AI agent system that can:
- Observe game state
- Decide actions dynamically
- Learn from rewards and outcomes
- Adapt to player behavior
- Maintain character-specific traits and memory

## 4. Goals
### Primary goals
- Provide an SDK that game developers can integrate with minimal effort
- Support multiple character types: civilians, enemies, bosses, companions
- Enable non-repetitive behavior through memory and stochastic decision-making
- Support reinforcement learning for policy improvement
- Allow different personalities and goals per agent
- Scale to many agents efficiently

### Secondary goals
- Support Unity and Unreal integrations first
- Provide a Python core reference implementation
- Expose a REST/gRPC interface for engine-agnostic use
- Support offline training and runtime inference
- Provide debugging tools for inspecting agent decisions

## 5. Scope
### In scope
- Agent state ingestion
- Action selection
- Personality and goal configuration
- Memory storage and retrieval
- Reinforcement learning training pipeline
- Runtime inference service
- Game engine adapters
- Example demo environments

### Out of scope for initial release
- Full AAA production-ready anti-cheat-safe integration
- Real-time online training inside live commercial games
- Support for every game engine at launch
- Fully autonomous natural-language dialogue for all characters
- Massive MMO-scale distributed deployment

## 6. Target Users
- Game developers
- Indie studios
- Modding and simulation developers
- AI researchers
- Students building intelligent agents in games

## 7. Core Features
### 7.1 Agent Intelligence
- State-based decision making
- Reinforcement learning policy
- Rule constraints for safety and consistency
- Optional reasoning layer for higher-level planning

### 7.2 Memory System
- Short-term memory for recent events
- Long-term memory for significant events
- Retrieval of past interactions
- Memory summarization for scalability

### 7.3 Personality System
Each agent can have traits such as:
- Aggression
- Curiosity
- Sociability
- Fear
- Patience
- Risk tolerance

### 7.4 Goal System
Agents should support:
- Survival goals
- Combat goals
- Social goals
- Patrol goals
- Mission goals
- Villain strategy goals

### 7.5 Behavior Diversity
- Different initial seeds
- Personality-dependent choices
- Reward-shaping variations
- Memory-driven changes over time

### 7.6 Scalability
- Shared models across many agents
- Batch inference
- Level-of-detail AI
- Scheduled updates instead of per-frame reasoning
- Optional distributed service mode

## 8. Functional Requirements
### FR1: State ingestion
The SDK must accept game state from the engine or adapter layer.

### FR2: Action output
The SDK must return one or more valid actions compatible with the game.

### FR3: Agent configuration
Developers must be able to define personality, goals, and constraints.

### FR4: Memory persistence
The system must store agent memory across sessions when enabled.

### FR5: Policy inference
The system must produce an action using the learned policy.

### FR6: Training support
The system must support offline RL training in a simulation environment.

### FR7: Debug support
The system must expose logs or traces explaining why an action was selected.

### FR8: Engine adapters
The system must support adapters for Unity and Unreal as initial targets.

## 9. Non-Functional Requirements
### Performance
- Fast inference for real-time gameplay
- Batch processing for multiple agents
- Low memory overhead per agent

### Reliability
- Deterministic fallback behavior when AI fails
- Safe action validation before execution

### Maintainability
- Modular codebase
- Clear separation of core AI and engine adapters
- Clean documentation and examples

### Extensibility
- Easy to add new agent types
- Easy to add new RL algorithms or memory backends
- Easy to integrate additional game engines later

### Portability
- Core logic should be engine-agnostic
- Communication via API where possible

## 10. Technical Requirements
### Backend
- Python for core AI and training
- Optional FastAPI or gRPC service
- PyTorch for model training

### Game integration
- Unity C# adapter
- Unreal C++/Blueprint adapter later

### AI stack
- Reinforcement learning: PPO first, then DQN or multi-agent extensions
- Memory store: SQLite/PostgreSQL/Redis depending on scale
- Policy serialization: ONNX or native model files if suitable

### Observability
- Action logs
- State logs
- Reward logs
- Debug dashboard

## 11. Data Requirements
### Training data
- Simulated state-action-reward trajectories
- Example agent behavior logs
- Optional imitation learning data from scripted agents

### Runtime data
- Current state
- Nearby entities
- Historical memory summaries
- Goal context
- Personality vector

## 12. Quality Criteria
A good OpenNPC implementation should:
- Behave differently across agents
- React to past experiences
- Avoid repetitive loops
- Scale beyond a small demo
- Be easy for a developer to plug into a game

## 13. Risks
- Poor reward design causing broken behavior
- Overuse of expensive model calls
- Inconsistent real-time performance
- Hard-to-balance combat intelligence
- Too much complexity for developers to adopt

## 14. Success Metrics
- Number of agents supported in a demo scene
- Average inference latency
- Behavior diversity score
- Player-perceived realism
- Ease of integration
- Training improvement over baseline scripted AI

## 15. Deliverables
- Core SDK
- Reference RL agent
- Memory system
- Personality config system
- Engine adapters
- Demo environment
- Documentation
- Example prompts and configuration files

## 16. Suggested Release Plan
### Phase 1
Single NPC in a simulation environment with RL and memory

### Phase 2
Multiple agent types with personality and goals

### Phase 3
Unity integration and debugging tools

### Phase 4
Enemy and villain strategy system

### Phase 5
Optimization, scaling, and public open-source release
