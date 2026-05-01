"""Prompt templates for optional high-level LLM reasoning."""

from __future__ import annotations

import json
from string import Template
from typing import Any

from opennpc.types import AgentConfig, GameState


SYSTEM_PROMPT = """You are an autonomous game character controller inside the OpenNPC framework.
Your job is to choose the best next action based on the current state, personality, memory, goals, and constraints.

Rules:
- Always stay within the allowed action list.
- Prefer actions that help achieve current goals.
- Use memory to avoid repeating failed behavior.
- Match the character type: civilian, enemy, villain, or companion.
- Respect safety, game rules, and action constraints.
- Return only structured output in the required format.
"""

DECISION_PROMPT_TEMPLATE = Template(
    """Character Type: $character_type
Personality: $personality_json
Goals: $goals_json
Recent Memory: $memory_summary
Current State: $state_json
Allowed Actions: $action_list
Constraints: $constraints

Select the best next action and explain it briefly.
Return JSON only.
"""
)

ENEMY_STRATEGY_PROMPT_TEMPLATE = Template(
    """You are controlling an enemy AI in a game.
The player's recent pattern is: $player_pattern
Your current health is: $health
Your nearby cover options are: $cover_options
Your objective is: $objective

Choose a tactical action that improves your chance of success.
Prefer adaptive and non-repetitive tactics.
Return only JSON.
"""
)

VILLAIN_PLANNING_PROMPT_TEMPLATE = Template(
    """You are controlling a strategic villain.
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
"""
)


def build_decision_prompt(
    config: AgentConfig,
    state: GameState,
    allowed_actions: list[str],
    memory_summary: str = "",
) -> str:
    return DECISION_PROMPT_TEMPLATE.substitute(
        character_type=config.agent_type.value,
        personality_json=json.dumps(config.personality.to_dict(), sort_keys=True),
        goals_json=json.dumps([goal.to_dict() for goal in config.goals], sort_keys=True),
        memory_summary=memory_summary,
        state_json=json.dumps(state.to_dict(), sort_keys=True),
        action_list=json.dumps(allowed_actions),
        constraints=json.dumps(config.constraints),
    )


def expected_json_shape() -> dict[str, Any]:
    return {
        "action": "attack",
        "confidence": 0.87,
        "reason": "The target is weak and the enemy has cover nearby.",
        "memory_update": "Player rushed aggressively again.",
    }
