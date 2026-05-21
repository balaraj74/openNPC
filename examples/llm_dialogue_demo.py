"""Demo: LLM-enhanced NPC dialogue and strategic planning.

Shows the three main LLM use cases:
1. Personality-flavored barks (combat taunts, greetings)
2. NPC dialogue/conversation
3. VillainPlanner with LLM-enriched strategy

Run: python examples/llm_dialogue_demo.py
"""

from opennpc import (
    AgentConfig, AgentType, GameState, Goal, Personality,
    LLMEngine, NPCDialogue, VillainPlanner, PlayerPatternTracker,
)


def main() -> None:
    # ── Initialize LLM (loads Qwen2.5-0.5B-Instruct on first use) ──
    print("=" * 60)
    print("  OpenNPC — LLM-Enhanced NPC Intelligence Demo")
    print("=" * 60)

    llm = LLMEngine(model_name="qwen-0.5b")
    dialogue = NPCDialogue(llm=llm)

    # ── 1. Enemy NPC: Combat Barks ─────────────────────────────────
    print("\n🗡️  [1/4] Enemy Combat Barks")
    print("-" * 40)

    enemy_config = AgentConfig(
        agent_id="skeleton_archer_01",
        agent_type=AgentType.ENEMY,
        personality=Personality(aggression=0.9, caution=0.2, risk_tolerance=0.8),
        goals=[Goal("attack_target", 0.9), Goal("survive", 0.4)],
        allowed_actions=["attack", "move", "flee", "defend"],
    )

    for category in ["combat_taunt", "spot_player", "combat_wounded", "combat_victory"]:
        bark = dialogue.bark(enemy_config, category=category)
        print(f"  [{category:18s}] {bark.text}  ({bark.latency_ms:.0f}ms)")

    # ── 2. Civilian NPC: Conversation ──────────────────────────────
    print("\n🏘️  [2/4] Civilian NPC Conversation")
    print("-" * 40)

    civilian_config = AgentConfig(
        agent_id="village_elder_03",
        agent_type=AgentType.CIVILIAN,
        personality=Personality(
            aggression=0.1, caution=0.7, patience=0.9,
            sociability=0.8, curiosity=0.6, loyalty=0.9,
        ),
        goals=[Goal("help_travelers", 0.7), Goal("protect_village", 0.8)],
        allowed_actions=["talk", "give_quest", "trade", "flee"],
    )

    civilian_state = GameState(
        agent_id="village_elder_03",
        health=80,
        threat_level=0.1,
    )

    # Simulate a conversation
    messages = [
        "Hello there! What can you tell me about this place?",
        "Are there any dangerous creatures nearby?",
        "Can you help me find the hidden temple?",
    ]

    history: list[dict[str, str]] = []
    for msg in messages:
        print(f"  Player: {msg}")
        response = dialogue.converse(
            civilian_config, civilian_state, msg,
            conversation_history=history,
        )
        print(f"  Elder:  {response.text}  ({response.latency_ms:.0f}ms)")
        history.append({"role": "player", "text": msg})
        history.append({"role": "npc", "text": response.text})
        print()

    # ── 3. Personality Description ─────────────────────────────────
    print("\n📝  [3/4] Personality Description")
    print("-" * 40)

    boss_config = AgentConfig(
        agent_id="dragon_lord",
        agent_type=AgentType.VILLAIN,
        personality=Personality(
            aggression=0.7, caution=0.5, patience=0.8,
            curiosity=0.3, sociability=0.2, loyalty=0.1,
            risk_tolerance=0.6,
        ),
        goals=[
            Goal("dominate_realm", 0.9),
            Goal("collect_artifacts", 0.6),
            Goal("survive", 0.7),
        ],
        allowed_actions=["attack", "defend", "flank", "set_trap", "flee", "hide"],
    )

    desc = dialogue.personality_description(boss_config)
    print(f"  {desc}")

    # ── 4. LLM-Enhanced Villain Strategy ───────────────────────────
    print("\n🧠  [4/4] LLM-Enhanced Villain Strategy")
    print("-" * 40)

    tracker = PlayerPatternTracker()
    # Simulate player behavior patterns
    for action in ["attack", "attack", "flank", "attack", "flank", "attack"]:
        tracker.record(action)

    planner = VillainPlanner(pattern_tracker=tracker, llm=llm)

    boss_state = GameState(
        agent_id="dragon_lord",
        health=14,
        threat_level=0.8,
        distance_to_target=3.0,
        target_health=12.0,
        extras={"max_health": 20.0},
    )

    plan = planner.plan(boss_config, boss_state)
    print(f"  Strategy:   {plan.long_term_strategy}")
    print(f"  Next move:  {plan.next_action}")
    print(f"  Prediction: {plan.predicted_player_response}")
    print(f"  Adaptation: {plan.adaptation_note}")
    print(f"  Confidence: {plan.confidence:.1%}")
    if plan.taunt:
        print(f"  Taunt:      \"{plan.taunt}\"")
    print(f"  LLM used:   {plan.llm_enhanced}")

    print("\n" + "=" * 60)
    print(f"  LLM: {llm.status()['model']}")
    print(f"  Device: {llm.status()['device']}")
    print(f"  Cache: {llm.status()['cache_entries']} entries")
    print("=" * 60)


if __name__ == "__main__":
    main()
