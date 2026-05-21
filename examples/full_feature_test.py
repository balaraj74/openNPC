#!/usr/bin/env python3
"""OpenNPC — Full Feature Test Suite. Tests every subsystem with zombie villain."""

import sys, time
sys.path.insert(0, ".")

from opennpc.types import AgentConfig, AgentType, GameState, Personality, Goal

PASS, FAIL = "✅", "❌"
results = []

def test(name, fn):
    try:
        fn()
        results.append((name, True))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((name, False))
        print(f"  {FAIL} {name}: {e}")

# ── Zombie Villain Config ──
zombie = AgentConfig(
    agent_id="zombie_villain_01", agent_type=AgentType.VILLAIN,
    personality=Personality(aggression=0.75, caution=0.5, risk_tolerance=0.7,
                            patience=0.6, curiosity=0.4, sociability=0.1, loyalty=0.3),
    goals=[Goal("weaken_player", 0.9), Goal("survive", 0.6),
           Goal("control_area", 0.7), Goal("attack_target", 0.85)],
    allowed_actions=["patrol","attack","retreat","idle","move","flank","defend","set_trap","hide","seek_cover"],
)
combat = GameState(agent_id="zombie_villain_01", agent_type=AgentType.VILLAIN,
    health=16, threat_level=0.8, distance_to_target=3.0, target_health=12.0,
    nearby_entities=["player"], extras={"max_health": 20.0})
low_hp = GameState(agent_id="zombie_villain_01", agent_type=AgentType.VILLAIN,
    health=3, threat_level=0.9, distance_to_target=2.0, target_health=18.0,
    nearby_entities=["player"], extras={"max_health": 20.0})

print("=" * 60)
print("  OpenNPC — Full Feature Test (Zombie Villain)")
print("=" * 60)

# ── 1. Decision Engine ──
print("\n🧠 [1/10] Decision Engine")
from opennpc.decision import DecisionEngine
engine = DecisionEngine()

def t_decide():
    d = engine.decide(zombie, combat)
    if d.action not in zombie.allowed_actions and d.action != "idle":
        raise ValueError(f"Bad action: {d.action}")
    if not (0 < d.confidence <= 1.0):
        raise ValueError(f"Bad confidence: {d.confidence}")
test("Decide combat action", t_decide)

def t_decide_low():
    d = engine.decide(zombie, low_hp)
    if d.action not in zombie.allowed_actions and d.action != "idle":
        raise ValueError(f"Bad action: {d.action}")
test("Decide low-HP action", t_decide_low)

# ── 2. Goal System ──
print("\n🎯 [2/10] Goal System")
from opennpc.goals import GoalManager
goal_mgr = GoalManager()

def t_goals():
    ev = goal_mgr.evaluate(zombie, combat)
    if not ev.action_scores: raise ValueError("No scores")
    if not ev.active_goals: raise ValueError("No active goals")
test("Evaluate goals", t_goals)

# ── 3. Memory ──
print("\n💾 [3/10] Memory System")
from opennpc.memory import InMemoryMemoryStore, MemoryEvent
mem = InMemoryMemoryStore()

def t_memory():
    mem.add(MemoryEvent(agent_id="zombie_villain_01", text="player attacked from left"))
    mem.add(MemoryEvent(agent_id="zombie_villain_01", text="player used diamond sword"))
    r = mem.recent("zombie_villain_01", limit=5)
    if len(r) < 2: raise ValueError(f"Only {len(r)} memories")
test("Store and recall memory", t_memory)

# ── 4. Policy ──
print("\n⚔️ [4/10] Policy Engine")
from opennpc.policy import HeuristicPolicy, ScriptedCombatPolicy

def t_heuristic():
    d = HeuristicPolicy().select_action(zombie, combat, zombie.allowed_actions)
    if not d.action: raise ValueError("No action")
test("Heuristic policy", t_heuristic)

def t_scripted():
    d = ScriptedCombatPolicy().select_action(zombie, combat, zombie.allowed_actions)
    if not d.action: raise ValueError("No action")
test("Scripted combat policy", t_scripted)

# ── 5. LOD Engine ──
print("\n🔭 [5/10] LOD Engine")
from opennpc.lod import LODEngine

def t_lod():
    lod = LODEngine()
    lod.register("zombie_villain_01")
    lod.register("zombie_far")
    t1 = lod.update("zombie_villain_01", distance_to_camera=5.0, is_visible=True, narrative_importance=1.0)
    t2 = lod.update("zombie_far", distance_to_camera=50.0, is_visible=False, narrative_importance=0.2)
    if t1.name not in ("FULL", "HIGH"): raise ValueError(f"Close zombie: {t1.name}")
    if t2.name not in ("LOW", "DORMANT"): raise ValueError(f"Far zombie: {t2.name}")
test("Assign LOD tiers", t_lod)

# ── 6. Player Pattern Tracker ──
print("\n📊 [6/10] Player Pattern Tracker")
from opennpc.strategy import PlayerPatternTracker
tracker = PlayerPatternTracker(window_size=15)

def t_patterns():
    for a in ["attack","attack","attack","flank","attack","charge"]:
        tracker.record(a)
    if tracker.dominant_action() != "attack": raise ValueError("Bad dominant")
    if tracker.aggression_estimate() < 0.5: raise ValueError("Low aggression")
    if not tracker.counter_recommendation(): raise ValueError("No counter")
test("Track and analyze patterns", t_patterns)

# ── 7. Villain Planner (heuristic) ──
print("\n🏰 [7/10] Villain Planner (Heuristic)")
from opennpc.villain import VillainPlanner

def t_villain_plan():
    p = VillainPlanner(pattern_tracker=tracker)
    plan = p.plan(zombie, combat)
    if not plan.next_action: raise ValueError("No action")
    if not plan.long_term_strategy: raise ValueError("No strategy")
test("Plan vs aggressive player", t_villain_plan)

def t_villain_retreat():
    p = VillainPlanner(pattern_tracker=tracker)
    plan = p.plan(zombie, low_hp)
    if plan.long_term_strategy != "retreat_and_rebuild":
        raise ValueError(f"Strategy: {plan.long_term_strategy}")
test("Retreat at low HP", t_villain_retreat)

# ── 8. Async Engine ──
print("\n⚡ [8/10] Async Engine")
from opennpc.async_engine import AsyncDecisionEngine
import asyncio

def t_async():
    ae = AsyncDecisionEngine(engine)
    d = asyncio.get_event_loop().run_until_complete(ae.decide(zombie, combat))
    if not d.action: raise ValueError("No action")
test("Async decide", t_async)

# ── 9. LLM Engine + Dialogue ──
print("\n🤖 [9/10] LLM Engine + Dialogue")
from opennpc.llm import LLMEngine
from opennpc.dialogue import NPCDialogue

llm = LLMEngine(model_name="qwen-0.5b")
dialogue = NPCDialogue(llm=llm)

def t_status_before():
    s = llm.status()
    if s["loaded"]: raise ValueError("Should not be loaded yet")
test("LLM status (before load)", t_status_before)

def t_bark():
    t0 = time.time()
    bark = dialogue.bark(zombie, category="combat_taunt")
    elapsed = time.time() - t0
    if len(bark.text) < 5: raise ValueError(f"Short bark: '{bark.text}'")
    print(f"    ⏱️  First bark (incl. load): {elapsed:.1f}s → \"{bark.text}\"")
test("Combat bark", t_bark)

def t_cache():
    t0 = time.time()
    dialogue.bark(zombie, category="combat_taunt")
    if time.time() - t0 > 0.05: raise ValueError("Cache miss")
test("Bark cache hit (<50ms)", t_cache)

civilian = AgentConfig(agent_id="elder_01", agent_type=AgentType.CIVILIAN,
    personality=Personality(aggression=0.1, sociability=0.9, patience=0.8, caution=0.7))

def t_converse():
    t0 = time.time()
    r = dialogue.converse(civilian, combat, player_message="Where is the temple?")
    print(f"    ⏱️  Conversation: {time.time()-t0:.1f}s → \"{r.text[:80]}\"")
    if len(r.text) < 10: raise ValueError(f"Short: '{r.text}'")
test("NPC conversation", t_converse)

def t_personality():
    desc = dialogue.personality_description(zombie)
    if len(desc) < 30: raise ValueError(f"Short: '{desc}'")
    print(f"    📝 \"{desc[:80]}...\"")
test("Personality description", t_personality)

def t_status_after():
    s = llm.status()
    if not s["loaded"]: raise ValueError("Should be loaded")
    print(f"    🖥️  Device: {s['device']} | Cache: {s.get('cache_entries', 'N/A')} entries")
test("LLM status (after load)", t_status_after)

# ── 10. LLM-Enhanced Villain Planning ──
print("\n🧪 [10/10] LLM-Enhanced Villain Planning")
planner_llm = VillainPlanner(pattern_tracker=tracker, llm=llm)

def t_llm_plan():
    t0 = time.time()
    plan = planner_llm.plan(zombie, combat)
    elapsed = time.time() - t0
    if not plan.llm_enhanced: raise ValueError("LLM not used")
    print(f"    ⏱️  LLM plan: {elapsed:.1f}s")
    print(f"    📋 Strategy: {plan.long_term_strategy}")
    print(f"    🎯 Next: {plan.next_action} | Conf: {plan.confidence:.0%}")
    if hasattr(plan, 'taunt') and plan.taunt:
        print(f"    💬 Taunt: \"{plan.taunt}\"")
test("LLM-enriched strategy", t_llm_plan)

# ── Results ──
print("\n" + "=" * 60)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"  Results: {passed} passed, {failed} failed, {len(results)} total")
if failed:
    print("\n  Failed:")
    for name, ok in results:
        if not ok: print(f"    {FAIL} {name}")
print("=" * 60)
sys.exit(1 if failed else 0)
