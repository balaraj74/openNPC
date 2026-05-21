"""NPC dialogue and personality text system.

Uses a small local LLM to generate:
- In-character dialogue responses
- Personality-flavored bark/flavor text (combat taunts, greetings, etc.)
- Context-aware conversation with memory

This module is OPTIONAL. It enhances NPCs with natural language but
is not required for the core decision engine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from opennpc.llm import LLMEngine, LLMResponse
from opennpc.types import AgentConfig, AgentType, GameState, Personality

logger = logging.getLogger(__name__)


# ─── Personality Templates ──────────────────────────────────────────────────

PERSONALITY_PROFILES: dict[str, dict[str, str]] = {
    "aggressive_enemy": {
        "voice": "menacing, direct, threatening",
        "vocabulary": "crude, violent, intimidating",
        "speech_pattern": "short sentences, lots of threats",
        "example": "You dare enter MY territory? You'll regret that.",
    },
    "cautious_enemy": {
        "voice": "calculating, cold, analytical",
        "vocabulary": "precise, military, strategic",
        "speech_pattern": "measured, tactical observations",
        "example": "Interesting approach. Let's see how long you last.",
    },
    "friendly_civilian": {
        "voice": "warm, helpful, curious",
        "vocabulary": "casual, friendly, local dialect",
        "speech_pattern": "conversational, asks questions",
        "example": "Oh hey! Haven't seen you around here before. Need directions?",
    },
    "loyal_companion": {
        "voice": "supportive, encouraging, protective",
        "vocabulary": "tactical but caring",
        "speech_pattern": "team-oriented, warnings and encouragement",
        "example": "Watch your left! I've got your back.",
    },
    "villain_boss": {
        "voice": "theatrical, cunning, superior",
        "vocabulary": "dramatic, philosophical, mocking",
        "speech_pattern": "monologues, rhetorical questions, dark humor",
        "example": "You think you're the hero? Every villain is the hero of their own story.",
    },
}


def _personality_profile(config: AgentConfig) -> dict[str, str]:
    """Select the personality profile that best matches the agent config."""
    p = config.personality
    t = config.agent_type

    if t == AgentType.VILLAIN:
        return PERSONALITY_PROFILES["villain_boss"]
    elif t == AgentType.ENEMY:
        if p.aggression > 0.6:
            return PERSONALITY_PROFILES["aggressive_enemy"]
        return PERSONALITY_PROFILES["cautious_enemy"]
    elif t == AgentType.COMPANION:
        return PERSONALITY_PROFILES["loyal_companion"]
    else:
        return PERSONALITY_PROFILES["friendly_civilian"]


# ─── Bark / Flavor Text ────────────────────────────────────────────────────

BARK_SYSTEM_PROMPT = """You are a game NPC generating short in-character lines.
Rules:
- Stay in character based on the personality description
- Keep responses under 15 words
- Be expressive and memorable
- Never break character or mention being an AI
- Match the emotional tone to the situation"""

BARK_CATEGORIES = [
    "combat_taunt",
    "combat_victory",
    "combat_wounded",
    "combat_dying",
    "greeting",
    "idle_mutter",
    "alert",
    "retreat",
    "spot_player",
]


@dataclass(slots=True)
class DialogueLine:
    """A single line of NPC dialogue."""

    text: str
    category: str
    agent_id: str
    emotion: str = "neutral"
    latency_ms: float = 0.0
    cached: bool = False


# ─── Dialogue Engine ───────────────────────────────────────────────────────


class NPCDialogue:
    """Generates contextual NPC dialogue using a local LLM.

    Parameters
    ----------
    llm : LLMEngine
        The shared LLM engine instance.
    """

    def __init__(self, llm: LLMEngine) -> None:
        self.llm = llm

    def bark(
        self,
        config: AgentConfig,
        category: str = "combat_taunt",
        context: str = "",
    ) -> DialogueLine:
        """Generate a short in-character bark/flavor line.

        Fast, cached, meant for combat taunts, greetings, etc.
        Typically < 15 words.

        Parameters
        ----------
        config : AgentConfig
            The NPC's config (personality, type).
        category : str
            Type of bark (combat_taunt, greeting, idle_mutter, etc.).
        context : str
            Optional situation context (e.g., "player is running away").

        Returns
        -------
        DialogueLine
            The generated bark line.
        """
        profile = _personality_profile(config)
        emotion = _emotion_for_category(category)

        prompt = (
            f"Character type: {config.agent_type.value}\n"
            f"Personality voice: {profile['voice']}\n"
            f"Speech pattern: {profile['speech_pattern']}\n"
            f"Situation: {category.replace('_', ' ')}\n"
        )
        if context:
            prompt += f"Context: {context}\n"
        prompt += f"\nGenerate ONE short {category.replace('_', ' ')} line (max 15 words):"

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=BARK_SYSTEM_PROMPT,
            max_new_tokens=40,
            temperature=0.8,
        )

        # Clean up the response — strip quotes, extra whitespace
        text = response.text.strip().strip('"').strip("'").split("\n")[0]

        return DialogueLine(
            text=text,
            category=category,
            agent_id=config.agent_id,
            emotion=emotion,
            latency_ms=response.latency_ms,
            cached=response.cached,
        )

    def converse(
        self,
        config: AgentConfig,
        state: GameState,
        player_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        memory_summary: str = "",
    ) -> DialogueLine:
        """Generate a conversational dialogue response.

        Slower, richer, meant for direct player-NPC interaction.

        Parameters
        ----------
        config : AgentConfig
            The NPC's config.
        state : GameState
            Current game state.
        player_message : str
            What the player said to the NPC.
        conversation_history : list
            Previous messages [{"role": "player"/"npc", "text": "..."}].
        memory_summary : str
            Summary of past interactions with this player.

        Returns
        -------
        DialogueLine
            The NPC's response.
        """
        profile = _personality_profile(config)
        history = conversation_history or []

        system = (
            f"You are a {config.agent_type.value} NPC in a game.\n"
            f"Voice: {profile['voice']}\n"
            f"Vocabulary: {profile['vocabulary']}\n"
            f"Speech pattern: {profile['speech_pattern']}\n"
            f"Example of your style: \"{profile['example']}\"\n\n"
            f"Personality traits:\n"
            f"  Aggression: {config.personality.aggression:.1f}/1.0\n"
            f"  Caution: {config.personality.caution:.1f}/1.0\n"
            f"  Patience: {config.personality.patience:.1f}/1.0\n"
            f"  Sociability: {config.personality.sociability:.1f}/1.0\n\n"
            f"Rules:\n"
            f"- Stay in character at all times\n"
            f"- Keep responses under 50 words\n"
            f"- React based on current game situation\n"
            f"- Never mention being an AI or NPC\n"
            f"- Be memorable and expressive"
        )

        prompt_parts = []
        if memory_summary:
            prompt_parts.append(f"[Past memory: {memory_summary}]")

        prompt_parts.append(
            f"[Current situation: HP={state.health:.0f}, "
            f"Threat={state.threat_level:.1f}, "
            f"Location={state.location}]"
        )

        for msg in history[-4:]:  # Only last 4 exchanges for context window
            role = msg.get("role", "player")
            text = msg.get("text", "")
            prompt_parts.append(f"{'Player' if role == 'player' else 'You'}: {text}")

        prompt_parts.append(f"Player: {player_message}")
        prompt_parts.append("You:")

        prompt = "\n".join(prompt_parts)

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=system,
            max_new_tokens=80,
            temperature=0.75,
            use_cache=False,  # Conversations should feel unique
        )

        text = response.text.strip().strip('"').split("\n")[0]

        return DialogueLine(
            text=text,
            category="conversation",
            agent_id=config.agent_id,
            emotion=_emotion_from_state(state),
            latency_ms=response.latency_ms,
            cached=response.cached,
        )

    def personality_description(self, config: AgentConfig) -> str:
        """Generate a rich personality description for an NPC.

        Useful for tooltips, character sheets, or world-building.
        """
        profile = _personality_profile(config)
        p = config.personality

        prompt = (
            f"Write a 2-3 sentence character description for a {config.agent_type.value} NPC.\n"
            f"Traits: aggression={p.aggression:.1f}, caution={p.caution:.1f}, "
            f"patience={p.patience:.1f}, curiosity={p.curiosity:.1f}, "
            f"sociability={p.sociability:.1f}, loyalty={p.loyalty:.1f}\n"
            f"Voice style: {profile['voice']}\n"
            f"Write it as a game designer's note, vivid and concise."
        )

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a game narrative designer writing NPC descriptions.",
            max_new_tokens=100,
            temperature=0.7,
        )

        return response.text.strip()


# ─── Helpers ───────────────────────────────────────────────────────────────


def _emotion_for_category(category: str) -> str:
    """Map bark category to emotion."""
    mapping = {
        "combat_taunt": "aggressive",
        "combat_victory": "triumphant",
        "combat_wounded": "pained",
        "combat_dying": "desperate",
        "greeting": "friendly",
        "idle_mutter": "bored",
        "alert": "alarmed",
        "retreat": "fearful",
        "spot_player": "alert",
    }
    return mapping.get(category, "neutral")


def _emotion_from_state(state: GameState) -> str:
    """Infer emotion from game state."""
    health_ratio = state.health / 100.0 if state.health > 0 else 0.0
    if health_ratio < 0.2:
        return "desperate"
    if state.threat_level > 0.7:
        return "alarmed"
    if state.threat_level > 0.4:
        return "tense"
    return "calm"
