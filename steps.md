# OpenNPC Build Steps

## Phase 0: Define the foundation
1. Finalize scope: NPCs, enemies, villains, and adaptive behavior.
2. Choose the first engine to support: Unity.
3. Decide the first runtime format: Python core + API.
4. Define the minimal actions and state fields.

## Phase 1: Build the simulation core
1. Create a small simulated environment.
2. Define agent state, action space, and reward function.
3. Create one baseline scripted agent.
4. Create one RL-based agent.
5. Compare behavior and tune the reward.

## Phase 2: Build the OpenNPC core
1. Implement the agent configuration model.
2. Implement personality vectors.
3. Implement short-term and long-term memory.
4. Implement goal management.
5. Implement a decision engine that combines:
   - state
   - personality
   - memory
   - goals
   - RL policy

## Phase 3: Build training pipeline
1. Implement offline training using simulation episodes.
2. Use PPO as the first algorithm.
3. Log rewards, states, and action history.
4. Save trained policies.
5. Evaluate against scripted baseline behavior.

## Phase 4: Build runtime inference
1. Create an inference service.
2. Add action validation before execution.
3. Add fallback logic if the model fails.
4. Add batching for multiple agents.
5. Add scheduled updates instead of per-frame inference.

## Phase 5: Build game adapter
1. Build a Unity adapter.
2. Define how the game sends state to OpenNPC.
3. Define how OpenNPC returns actions.
4. Execute actions inside the game.
5. Add debugging overlay to inspect decisions.

## Phase 6: Add enemy and villain intelligence
1. Define combat state fields.
2. Define tactical actions such as attack, retreat, flank, cover, trap.
3. Add enemy personality profiles.
4. Add strategy memory for player pattern tracking.
5. Add adaptation rules for changing difficulty and tactics.

## Phase 7: Scale the system
1. Share policies across many agents.
2. Use agent-specific personality and goal parameters.
3. Batch inference across agents.
4. Add AI level-of-detail rules.
5. Add asynchronous processing where possible.

## Phase 8: Package the open-source project
1. Write clean documentation.
2. Add installation instructions.
3. Add examples and demo scenes.
4. Add configuration templates.
5. Publish the repository and release notes.

## Phase 9: Improve after first release
1. Collect feedback from developers.
2. Improve reward design and stability.
3. Add more engine adapters.
4. Add multi-agent coordination.
5. Add optional LLM-based planning for high-level decisions.

## Suggested development order
1. Simulation environment
2. RL agent
3. Memory and personality
4. OpenNPC API
5. Unity demo
6. Enemy/villain adaptation
7. Scaling and optimization
8. Open-source release

## Milestone checklist
- [ ] One autonomous agent works
- [ ] Memory changes behavior
- [ ] Personality changes action choice
- [ ] RL policy improves over time
- [ ] Unity can send and receive state/action data
- [ ] Enemy logic adapts to player behavior
- [ ] Multiple agents can run without lag
- [ ] Project documentation is complete
