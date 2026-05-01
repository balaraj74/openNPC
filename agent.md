# OpenNPC Agent Design

## 1. Agent Concept
An OpenNPC agent is a game character controlled by a shared AI framework, with its own:
- Personality
- Goals
- Memory
- Policy
- Behavior constraints

## 2. Agent Types
### 2.1 Civilian NPC
- Lives in the world
- Talks, walks, works, explores
- Uses social and survival goals

### 2.2 Enemy
- Defends territory
- Attacks, retreats, flanks, uses cover
- Learns player patterns

### 2.3 Villain / Boss
- Uses strategy over time
- Changes tactics between phases
- Adapts to player behavior

### 2.4 Companion
- Assists the player
- Helps in combat or exploration
- May follow, protect, or gather resources

## 3. Agent State
A typical agent state may include:
- health
- stamina
- hunger
- location
- nearby entities
- threat level
- inventory
- objective status
- last action
- recent rewards

Example:
```json
{
  "agent_id": "npc_01",
  "type": "civilian",
  "health": 82,
  "location": "market",
  "nearby_entities": ["player", "shopkeeper"],
  "threat_level": 0.1,
  "current_goal": "trade"
}
```

## 4. Personality Vector
A personality vector influences decisions.

Example traits:
- aggression
- curiosity
- sociability
- caution
- patience
- loyalty
- risk tolerance

Example:
```json
{
  "aggression": 0.8,
  "curiosity": 0.4,
  "sociability": 0.7,
  "caution": 0.2,
  "patience": 0.5
}
```

## 5. Memory Model
### Short-term memory
Stores recent events such as:
- player attacked
- item was stolen
- goal failed
- enemy was seen

### Long-term memory
Stores important summaries such as:
- player is hostile
- this route is dangerous
- trading here is profitable

### Memory rules
- Important events are promoted to long-term memory
- Repeated events are summarized
- Old low-value memories expire

## 6. Goals
Goals guide the agent’s behavior.

### Civilian goals
- survive
- work
- explore
- interact

### Enemy goals
- defend
- ambush
- survive
- attack target

### Villain goals
- dominate area
- weaken player
- control resources
- trigger strategic traps

## 7. Decision Pipeline
1. Receive game state
2. Update memory
3. Evaluate goals
4. Apply personality influence
5. Query RL policy
6. Apply rule constraints
7. Select final action
8. Store outcome in memory

## 8. Action Set
Actions should be game-specific, but examples include:
- move
- idle
- talk
- trade
- follow
- flee
- attack
- defend
- patrol
- use_item
- hide
- seek_cover
- flank
- set_trap

## 9. Learning Behavior
The agent should improve by:
- receiving rewards
- avoiding repeated failure
- adapting to player behavior
- learning which actions work best in a context

## 10. Safety and Control
The agent must not be fully unconstrained.
Rules should limit:
- invalid actions
- impossible actions
- unsafe combat behavior
- performance-heavy behavior

## 11. Example Agent Configuration
```yaml
agent_id: villain_01
type: villain
personality:
  aggression: 0.9
  curiosity: 0.2
  caution: 0.6
goals:
  - weaken_player
  - control_area
  - survive
memory_enabled: true
rl_policy: ppo_v1
decision_interval_ms: 250
lod_level: high
```

## 12. Agent Lifecycle
- Spawn
- Initialize personality and goals
- Observe game state
- Decide
- Act
- Receive reward
- Update memory
- Repeat
- Persist important state if needed

## 13. Debugging Questions
When an agent behaves badly, ask:
- Was the reward wrong?
- Was the goal too broad?
- Did memory contain misleading summaries?
- Was the personality too extreme?
- Was the action space too large?

## 14. Design Principle
Each agent should feel:
- unique
- understandable
- adaptive
- believable
- bounded by rules

## 15. Recommended first agent
Start with a **single enemy agent** in a small simulation environment, then expand to civilians and villains after the learning loop is stable.
