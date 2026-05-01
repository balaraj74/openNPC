"""FastAPI runtime inference service for OpenNPC."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency import guard.
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    FastAPI = None
    HTMLResponse = None
    BaseModel = object
    Field = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from opennpc.coordination import MultiAgentCoordinator
from opennpc.decision import DecisionEngine
from opennpc.experience import RuntimeExperienceLogger
from opennpc.types import AgentConfig, GameState


def require_api_deps() -> None:
    if FastAPI is None:
        raise RuntimeError("OpenNPC API requires optional dependencies: fastapi, pydantic, uvicorn.") from _IMPORT_ERROR


if Field is not None:

    class DecisionRequest(BaseModel):
        config: dict[str, Any]
        state: dict[str, Any]
        available_actions: list[str] | None = None
        reward: float | None = None
        event: str | None = None

    class BatchDecisionRequest(BaseModel):
        requests: list[DecisionRequest] = Field(default_factory=list)

    class CoordinationRequest(BaseModel):
        configs: list[dict[str, Any]] = Field(default_factory=list)
        states: list[dict[str, Any]] = Field(default_factory=list)
        available_actions: dict[str, list[str]] | None = None
        max_attackers_per_target: int | None = None

else:  # pragma: no cover

    class DecisionRequest:  # type: ignore[no-redef]
        pass

    class BatchDecisionRequest:  # type: ignore[no-redef]
        pass

    class CoordinationRequest:  # type: ignore[no-redef]
        pass


require_api_deps()
app = FastAPI(title="OpenNPC Runtime", version="0.1.0")
experience_logger = RuntimeExperienceLogger()
engine = DecisionEngine(experience_logger=experience_logger)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "opennpc"}


@app.post("/decide")
def decide(request: DecisionRequest) -> dict[str, Any]:
    config = AgentConfig.from_dict(request.config)
    state = GameState.from_dict(request.state)
    decision = engine.decide(
        config,
        state,
        available_actions=request.available_actions,
        reward=request.reward,
        event=request.event,
    )
    return decision.to_dict()


@app.post("/batch/decide")
def decide_batch(request: BatchDecisionRequest) -> dict[str, list[dict[str, Any]]]:
    decisions = [
        engine.decide(
            AgentConfig.from_dict(item.config),
            GameState.from_dict(item.state),
            available_actions=item.available_actions,
            reward=item.reward,
            event=item.event,
        ).to_dict()
        for item in request.requests
    ]
    return {"decisions": decisions}


@app.post("/coordinate")
def coordinate(request: CoordinationRequest) -> dict[str, Any]:
    coordinator = MultiAgentCoordinator(
        decision_engine=engine,
        max_attackers_per_target=request.max_attackers_per_target or 2,
    )
    plan = coordinator.coordinate(
        configs=[AgentConfig.from_dict(config) for config in request.configs],
        states=[GameState.from_dict(state) for state in request.states],
        available_actions=request.available_actions,
    )
    return plan.to_dict()


@app.get("/memory/{agent_id}")
def memory(agent_id: str) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "summary": engine.memory_store.summarize(agent_id),
        "recent": [event.to_dict() for event in engine.memory_store.recent(agent_id)],
    }


@app.get("/debug/decisions")
def debug_decisions() -> dict[str, Any]:
    return {"decisions": engine.debug_decisions()}


@app.get("/debug/decisions/{agent_id}")
def debug_decision_agent(agent_id: str) -> dict[str, Any]:
    decision = engine.last_decision(agent_id)
    return {"agent_id": agent_id, "decision": decision.to_dict() if decision else None}


@app.get("/debug/experience")
def debug_experience() -> dict[str, Any]:
    return experience_logger.summary()


@app.get("/debug/experience/{agent_id}")
def debug_experience_agent(agent_id: str, limit: int = 20) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "recent": [item.to_dict() for item in experience_logger.recent(agent_id, limit=limit)],
    }


@app.get("/debug/dashboard", response_class=HTMLResponse)
def debug_dashboard() -> str:
    return DEBUG_DASHBOARD_HTML


from opennpc.lod import LODEngine
from opennpc.strategy import PlayerPatternTracker
from opennpc.villain import VillainPlanner

lod_engine = LODEngine()
pattern_tracker = PlayerPatternTracker()
villain_planner = VillainPlanner(pattern_tracker=pattern_tracker)


DEBUG_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenNPC Debug Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #5e6b76;
      --line: #d7dde2;
      --panel: #f7f9fb;
      --accent: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
      --good: #166534;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }
    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { margin: 0; font-size: 20px; font-weight: 700; }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
      min-height: calc(100vh - 70px);
    }
    aside {
      border-right: 1px solid var(--line);
      padding: 18px;
      background: var(--panel);
    }
    section { padding: 18px 22px; border-bottom: 1px solid var(--line); }
    h2 { margin: 0 0 12px; font-size: 15px; }
    button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 8px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      margin: 8px 0 12px;
      font: inherit;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      margin: 0;
      font-size: 12px;
      line-height: 1.45;
    }
    .metric {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 8px 0;
      border-bottom: 1px solid var(--line);
    }
    .metric span:first-child { color: var(--muted); }
    .decision-list {
      display: grid;
      gap: 10px;
    }
    .decision {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: #fff;
    }
    .decision strong { display: block; margin-bottom: 6px; }
    .bad { color: var(--bad); }
    .good { color: var(--good); }
    .muted { color: var(--muted); }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>OpenNPC Debug Dashboard</h1>
    <button onclick="refresh()">Refresh</button>
  </header>
  <main>
    <aside>
      <h2>Agent Lookup</h2>
      <input id="agentId" value="enemy_01" aria-label="Agent ID">
      <button onclick="loadAgent()">Load Agent</button>
      <section style="padding-left:0;padding-right:0;border-bottom:0">
        <h2>Runtime Summary</h2>
        <div id="metrics"></div>
      </section>
    </aside>
    <div>
      <section>
        <h2>Recent Decisions</h2>
        <div id="decisions" class="decision-list"></div>
      </section>
      <section>
        <h2>Agent Memory</h2>
        <pre id="memory">No agent loaded.</pre>
      </section>
      <section>
        <h2>Agent Experience</h2>
        <pre id="experience">No agent loaded.</pre>
      </section>
      <section>
        <h2>Player Patterns</h2>
        <pre id="patterns">Loading...</pre>
      </section>
      <section>
        <h2>LOD Distribution</h2>
        <pre id="lod">Loading...</pre>
      </section>
    </div>
  </main>
  <script>
    async function getJSON(path) {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json();
    }
    function pretty(value) {
      return JSON.stringify(value, null, 2);
    }
    function renderMetrics(exp) {
      const rows = [
        ["Transitions", exp.transitions ?? 0],
        ["Mean Reward", exp.reward_mean ?? "n/a"],
        ["Terminal Transitions", exp.terminal_transitions ?? 0],
      ];
      document.getElementById("metrics").innerHTML = rows.map(([k, v]) =>
        `<div class="metric"><span>${k}</span><strong>${v}</strong></div>`
      ).join("");
    }
    function renderDecisions(data) {
      const decisions = data.decisions || [];
      if (!decisions.length) {
        document.getElementById("decisions").innerHTML = '<p class="muted">No decisions yet. Call /decide or /coordinate.</p>';
        return;
      }
      document.getElementById("decisions").innerHTML = decisions.map(item => {
        const trace = item.trace || {};
        const cls = trace.fallback_used ? "bad" : "good";
        return `<article class="decision">
          <strong>${item.agent_id}: ${item.action} <span class="${cls}">${trace.fallback_used ? "fallback" : "ok"}</span></strong>
          <div class="muted">policy=${trace.policy_name || "unknown"} confidence=${Number(item.confidence || 0).toFixed(2)}</div>
          <div>${item.reason || ""}</div>
          <pre>${pretty({goal_scores: trace.goal_scores, personality: trace.personality_influence, memory: trace.memory_summary})}</pre>
        </article>`;
      }).join("");
    }
    async function loadAgent() {
      const id = document.getElementById("agentId").value.trim();
      if (!id) return;
      const [memory, experience] = await Promise.all([
        getJSON(`/memory/${encodeURIComponent(id)}`),
        getJSON(`/debug/experience/${encodeURIComponent(id)}`),
      ]);
      document.getElementById("memory").textContent = pretty(memory);
      document.getElementById("experience").textContent = pretty(experience);
    }
    async function refresh() {
      const [experience, decisions, patterns, lod] = await Promise.all([
        getJSON("/debug/experience"),
        getJSON("/debug/decisions"),
        getJSON("/debug/patterns"),
        getJSON("/debug/lod"),
      ]);
      renderMetrics(experience);
      renderDecisions(decisions);
      document.getElementById("patterns").textContent = pretty(patterns);
      document.getElementById("lod").textContent = pretty(lod);
    }
    refresh().catch(error => {
      document.getElementById("decisions").innerHTML = `<p class="bad">${error.message}</p>`;
    });
  </script>
</body>
</html>
"""


@app.get("/debug/lod")
def debug_lod() -> dict[str, Any]:
    """Return LOD distribution across all registered agents."""
    return lod_engine.summary()


@app.get("/debug/lod/{agent_id}")
def debug_lod_agent(agent_id: str) -> dict[str, Any]:
    """Return LOD profile for a specific agent."""
    profile = lod_engine.profile_for(agent_id)
    tier = lod_engine.tier_for(agent_id)
    return {
        "agent_id": agent_id,
        "tier": tier.name,
        "decision_interval_ms": profile.decision_interval_ms,
        "memory_enabled": profile.memory_enabled,
        "trace_enabled": profile.trace_enabled,
        "policy_override": profile.policy_override,
    }


@app.post("/debug/lod/{agent_id}")
def debug_lod_update(agent_id: str, distance: float = 10.0, visible: bool = True, importance: float = 0.5, in_combat: bool = False) -> dict[str, str]:
    """Update LOD parameters for an agent and return the new tier."""
    tier = lod_engine.update(agent_id, distance, visible, importance, in_combat)
    return {"agent_id": agent_id, "tier": tier.name}


@app.get("/debug/patterns")
def debug_patterns() -> dict[str, Any]:
    """Return current player pattern analysis."""
    return pattern_tracker.to_dict()


@app.post("/debug/patterns/record")
def debug_record_action(action: str) -> dict[str, str]:
    """Record a player action for pattern tracking."""
    pattern_tracker.record(action)
    return {"recorded": action, "total": str(pattern_tracker.total_observations)}


@app.post("/debug/villain/plan")
def debug_villain_plan(request: DecisionRequest) -> dict[str, Any]:
    """Generate a villain strategic plan from config + state."""
    config = AgentConfig.from_dict(request.config)
    state = GameState.from_dict(request.state)
    plan = villain_planner.plan(config, state)
    return {
        "next_action": plan.next_action,
        "strategy": plan.long_term_strategy,
        "predicted_player_response": plan.predicted_player_response,
        "adaptation": plan.adaptation_note,
        "confidence": plan.confidence,
        "adjusted_goals": [{"name": g.name, "priority": g.priority} for g in plan.adjusted_goals],
    }


def main() -> None:
    import uvicorn

    uvicorn.run("opennpc.api.service:app", host="127.0.0.1", port=8787, reload=False)


if __name__ == "__main__":
    main()
