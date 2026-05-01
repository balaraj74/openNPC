# OpenNPC Roadmap

This roadmap tracks the next high-leverage upgrades after the core SDK, simulations, training modules, API, adapters, and demos.

## Phase 1: Runtime RL Integration

Status: ✅ complete

- [x] Lazy-load PPO/DQN checkpoints from `AgentConfig.metadata["model_path"]`.
- [x] Cache loaded runtime policies inside `DecisionEngine`.
- [x] Mark missing, failing, or invalid learned policies as traced fallbacks.
- [x] Record runtime reward feedback through `RuntimeExperienceLogger`.
- [x] Add per-agent reward shaping strategies for enemy, civilian, villain, and companion agents.
- [x] Add a replay-to-training utility that converts runtime experience JSONL into PPO/DQN fine-tuning data (`opennpc/training/replay.py`).
- [ ] Add model metadata/version validation for shipped checkpoints.

## Phase 2: Engine Demo Scene

Status: planned

- [ ] Build a Unity scene with varied civilian, enemy, companion, and villain agents.
- [x] Add a Unity demo controller scaffold for varied NPC personalities.
- [ ] Show three enemies adapting to player behavior.
- [ ] Show one villain using pattern tracking and strategic planning.
- [ ] Capture a short demo video and setup instructions.

## Phase 3: Visual Debugging

Status: planned

- [x] Add a browser dashboard for action, confidence, fallback state, and policy source.
- [x] Show goal scores and personality influence from `DecisionTrace`.
- [x] Show recent memories and runtime experience summaries.
- [ ] Provide Unity overlay hooks.

## Phase 4: Performance Scaling

Status: planned

- [ ] Optimize batch inference by grouping shared model calls.
- [ ] Profile async worker throughput and queue pressure.
- [x] Add repeated-state caching with TTL (`opennpc/cache.py` — `DecisionCache` with configurable TTL, tolerance, eviction, and confidence decay).
- [ ] Add streaming or websocket updates if engine integrations need lower overhead.

## Phase 5: Public Launch

Status: planned

- [x] Add repeatable benchmark script and current combat benchmark results.
- [ ] Polish GitHub README and examples around a single best demo path.
- [ ] Publish docs site.
- [ ] Add release notes and contribution issues.
- [ ] Record demo video.
