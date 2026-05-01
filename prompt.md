# OpenNPC Prompt Design

## Purpose
This file defines the prompts, templates, and structured instructions used by OpenNPC when an LLM-based reasoning layer is enabled for planning, narration, or higher-level decision support.

## Important note
OpenNPC should not rely on LLM prompts for every frame or every action. Prompts should be used sparingly for:
- high-level planning
- dialogue
- mission reasoning
- strategic adaptation
- debugging explanations

The main real-time action selection should still be handled by the core agent policy.

## 1. System Prompt Template
Use this when the LLM acts as the reasoning layer:

```text
You are an autonomous game character controller inside the OpenNPC framework.
Your job is to choose the best next action based on the current state, personality, memory, goals, and constraints.

Rules:
- Always stay within the allowed action list.
- Prefer actions that help achieve current goals.
- Use memory to avoid repeating failed behavior.
- Match the character type: civilian, enemy, villain, or companion.
- Respect safety, game rules, and action constraints.
- Return only structured output in the required format.
```

## 2. Decision Prompt Template
```text
Character Type: {character_type}
Personality: {personality_json}
Goals: {goals_json}
Recent Memory: {memory_summary}
Current State: {state_json}
Allowed Actions: {action_list}
Constraints: {constraints}

Select the best next action and explain it briefly.
Return JSON only.
```

## 3. JSON Output Template
```json
{
  "action": "attack",
  "confidence": 0.87,
  "reason": "The target is weak and the enemy has cover nearby.",
  "memory_update": "Player rushed aggressively again."
}
```

## 4. Enemy Strategy Prompt
```text
You are controlling an enemy AI in a game.
The player’s recent pattern is: {player_pattern}
Your current health is: {health}
Your nearby cover options are: {cover_options}
Your objective is: {objective}

Choose a tactical action that improves your chance of success.
Prefer adaptive and non-repetitive tactics.
Return only JSON.
```

## 5. Villain Planning Prompt
```text
You are controlling a strategic villain.
Your goal is not just the next move, but the larger plan over time.

Use the following:
- current objectives
- player behavior history
- territory control status
- current resources
- memory of past encounters

Suggest:
1. next action
2. long-term plan
3. likely player counter-response
4. how to adapt next time

Return only JSON.
```

## 6. Debug Explanation Prompt
```text
Explain why the agent selected its action.
Use:
- current state
- personality influence
- memory influence
- goal influence
- reward history

Keep the explanation short and developer-friendly.
```

## 7. Prompting Rules
- Keep prompts short and structured
- Do not include unnecessary prose
- Use JSON output whenever possible
- Avoid asking the model to invent unsupported game actions
- Always constrain the action list
- Use memory summaries, not raw long logs
- Never let the LLM override engine constraints

## 8. Example Prompt Variables
Recommended variables:
- {character_type}
- {personality_json}
- {goals_json}
- {memory_summary}
- {state_json}
- {action_list}
- {constraints}
- {player_pattern}
- {objective}
- {health}
- {cover_options}

## 9. Prompt Phases
### Phase A: Planning
Use for high-level reasoning.

### Phase B: Tactical adaptation
Use for enemy or villain adjustments.

### Phase C: Dialogue generation
Use for optional conversation systems.

### Phase D: Debugging and explanations
Use for developer inspection.

## 10. Recommended usage policy
- RL policy handles the frequent decisions
- LLM handles occasional strategic prompts
- Memory summaries are refreshed periodically
- Prompts should be deterministic where possible

## 11. Prompt examples by character type
### Civilian
Focus on routines, social choices, work, and survival.

### Enemy
Focus on combat tactics, positioning, and adaptation.

### Villain
Focus on multi-step strategy, traps, escalation, and long-term control.

### Companion
Focus on assistance, support, and coordination.

## 12. Final best practice
Use prompts as a supporting reasoning layer, not the whole system. OpenNPC becomes strongest when:
- RL learns behavior
- memory preserves context
- personality differentiates agents
- prompts add strategic intelligence only when needed
