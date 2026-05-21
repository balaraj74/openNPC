"""FastAPI runtime inference service for OpenNPC."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency import guard.
    from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, status
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    Depends = None
    FastAPI = None
    HTMLResponse = None
    Header = None
    HTTPException = None
    Path = None
    Query = None
    BaseModel = object
    Field = None
    status = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from opennpc.coordination import MultiAgentCoordinator
from opennpc.decision import DecisionEngine
from opennpc.experience import RuntimeExperienceLogger
from opennpc.security import RuntimeSettings, token_matches
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
        event: str | None = Field(default=None, max_length=512)

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
settings = RuntimeSettings.from_env()
app = FastAPI(title="OpenNPC Runtime", version="0.1.0")
experience_logger = RuntimeExperienceLogger()
engine = DecisionEngine(experience_logger=experience_logger)


def _extract_token(authorization: str | None, x_opennpc_key: str | None) -> str | None:
    if x_opennpc_key:
        return x_opennpc_key
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    return authorization


def require_api_key(
    authorization: str | None = Header(default=None),
    x_opennpc_key: str | None = Header(default=None),
) -> None:
    if token_matches(settings.api_key, _extract_token(authorization, x_opennpc_key)):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid OpenNPC API key is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_debug_access(_: None = Depends(require_api_key)) -> None:
    if settings.debug_enabled:
        return
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Debug endpoints are disabled.",
    )


def _validate_agent_id(agent_id: str) -> str:
    if not agent_id or len(agent_id) > settings.max_agent_id_length:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid agent_id length.")
    return agent_id


def _validate_decision_request(request: DecisionRequest) -> None:
    config_agent_id = str(request.config.get("agent_id", ""))
    state_agent_id = str(request.state.get("agent_id", config_agent_id))
    _validate_agent_id(config_agent_id)
    _validate_agent_id(state_agent_id)
    if request.event is not None and len(request.event) > settings.max_event_length:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Event text is too long.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "opennpc"}


@app.post("/decide", dependencies=[Depends(require_api_key)])
def decide(request: DecisionRequest) -> dict[str, Any]:
    _validate_decision_request(request)
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


@app.post("/batch/decide", dependencies=[Depends(require_api_key)])
def decide_batch(request: BatchDecisionRequest) -> dict[str, list[dict[str, Any]]]:
    if len(request.requests) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail="Batch request is too large.")
    for item in request.requests:
        _validate_decision_request(item)
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


@app.post("/coordinate", dependencies=[Depends(require_api_key)])
def coordinate(request: CoordinationRequest) -> dict[str, Any]:
    if len(request.configs) > settings.max_batch_size or len(request.states) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail="Coordination request is too large.",
        )
    for config in request.configs:
        _validate_agent_id(str(config.get("agent_id", "")))
    for state in request.states:
        _validate_agent_id(str(state.get("agent_id", "")))
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


@app.get("/memory/{agent_id}", dependencies=[Depends(require_api_key)])
def memory(agent_id: str = Path(min_length=1, max_length=settings.max_agent_id_length)) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "summary": engine.memory_store.summarize(agent_id),
        "recent": [event.to_dict() for event in engine.memory_store.recent(agent_id)],
    }


@app.get("/debug/decisions", dependencies=[Depends(require_debug_access)])
def debug_decisions() -> dict[str, Any]:
    return {"decisions": engine.debug_decisions()}


@app.get("/debug/decisions/{agent_id}", dependencies=[Depends(require_debug_access)])
def debug_decision_agent(agent_id: str = Path(min_length=1, max_length=settings.max_agent_id_length)) -> dict[str, Any]:
    decision = engine.last_decision(agent_id)
    return {"agent_id": agent_id, "decision": decision.to_dict() if decision else None}


@app.get("/debug/experience", dependencies=[Depends(require_debug_access)])
def debug_experience() -> dict[str, Any]:
    return experience_logger.summary()


@app.get("/debug/experience/{agent_id}", dependencies=[Depends(require_debug_access)])
def debug_experience_agent(
    agent_id: str = Path(min_length=1, max_length=settings.max_agent_id_length),
    limit: int = Query(default=20, ge=1, le=settings.max_memory_limit),
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "recent": [item.to_dict() for item in experience_logger.recent(agent_id, limit=limit)],
    }


@app.get("/debug/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_debug_access)])
def debug_dashboard() -> str:
    return DEBUG_DASHBOARD_HTML


from opennpc.lod import LODEngine
from opennpc.strategy import PlayerPatternTracker
from opennpc.villain import VillainPlanner

lod_engine = LODEngine()
pattern_tracker = PlayerPatternTracker()
villain_planner = VillainPlanner(pattern_tracker=pattern_tracker)


DEBUG_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenNPC — Debug Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0e1a;
  --surface: rgba(255,255,255,0.04);
  --surface-hover: rgba(255,255,255,0.07);
  --border: rgba(255,255,255,0.08);
  --accent: #6366f1;
  --accent2: #22d3ee;
  --good: #34d399;
  --warn: #fbbf24;
  --bad: #f87171;
  --text: #e2e8f0;
  --muted: #64748b;
  --glass: rgba(15,20,40,0.7);
  --radius: 12px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  background-image:
    radial-gradient(ellipse 80% 60% at 20% -10%, rgba(99,102,241,0.15) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 80% 110%, rgba(34,211,238,0.1) 0%, transparent 60%);
}
header {
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(12px);
  background: var(--glass);
  position: sticky; top: 0; z-index: 100;
}
.logo { display: flex; align-items: center; gap: 10px; }
.logo-icon {
  width: 32px; height: 32px; border-radius: 8px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 14px;
}
h1 { font-size: 16px; font-weight: 700; letter-spacing: -0.02em; }
.header-right { display: flex; align-items: center; gap: 12px; }
.status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--good); box-shadow: 0 0 8px var(--good);
  animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{ opacity:1 } 50%{ opacity:.4 } }
.status-text { font-size: 12px; color: var(--muted); }
.auto-label { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
toggle-wrap { display: flex; align-items: center; }
.toggle {
  position: relative; width: 36px; height: 20px; cursor: pointer;
  background: var(--accent); border-radius: 10px; border: none;
  transition: background .2s;
}
.toggle::after {
  content: ''; position: absolute; top: 2px; left: 2px;
  width: 16px; height: 16px; border-radius: 50%; background: white;
  transition: transform .2s;
}
.toggle.off { background: var(--border); }
.toggle.off::after { transform: translateX(0); }
.toggle.on::after { transform: translateX(16px); }
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text); font: 600 13px 'Inter', sans-serif;
  cursor: pointer; transition: background .15s, border-color .15s;
}
.btn:hover { background: var(--surface-hover); border-color: var(--accent); }
.btn-accent { background: var(--accent); border-color: var(--accent); }
.btn-accent:hover { background: #4f52d6; }
.layout {
  display: grid; grid-template-columns: 300px 1fr;
  min-height: calc(100vh - 65px);
}
aside {
  border-right: 1px solid var(--border);
  padding: 20px 16px;
  display: flex; flex-direction: column; gap: 16px;
}
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  backdrop-filter: blur(8px);
}
.card-title {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: var(--muted); margin-bottom: 12px;
}
.input-row { display: flex; gap: 8px; }
input[type=text] {
  flex: 1; padding: 8px 12px; border-radius: 8px;
  border: 1px solid var(--border); background: rgba(0,0,0,0.3);
  color: var(--text); font: 13px 'Inter', sans-serif;
  outline: none; transition: border-color .15s;
}
input[type=text]:focus { border-color: var(--accent); }
.metric-grid { display: flex; flex-direction: column; gap: 8px; }
.metric {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 0; border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.metric:last-child { border-bottom: 0; }
.metric-key { color: var(--muted); }
.metric-val { font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.main-content { padding: 20px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
.section-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.badge {
  font-size: 11px; padding: 2px 8px; border-radius: 99px;
  background: rgba(99,102,241,0.15); color: var(--accent);
  font-weight: 600;
}
.badge.good { background: rgba(52,211,153,0.15); color: var(--good); }
.badge.bad { background: rgba(248,113,113,0.15); color: var(--bad); }
.badge.warn { background: rgba(251,191,36,0.15); color: var(--warn); }
.decision-list { display: flex; flex-direction: column; gap: 10px; }
.decision-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
  display: grid; grid-template-columns: 1fr auto; gap: 8px;
  transition: border-color .15s, background .15s;
}
.decision-card:hover { background: var(--surface-hover); border-color: rgba(99,102,241,0.3); }
.decision-header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.agent-name { font-weight: 700; font-size: 13px; }
.action-name {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; font-weight: 600;
  color: var(--accent2); background: rgba(34,211,238,0.1);
  padding: 2px 8px; border-radius: 6px;
}
.decision-meta { font-size: 12px; color: var(--muted); margin-top: 4px; }
.confidence-bar-wrap { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); margin-top: 6px; }
.confidence-bar-bg { flex: 1; height: 4px; border-radius: 2px; background: rgba(255,255,255,0.08); }
.confidence-bar { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width .4s; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
pre.code {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 1.6;
  background: rgba(0,0,0,0.4); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px; overflow-x: auto; white-space: pre-wrap; word-break: break-word;
  color: var(--text); max-height: 320px; overflow-y: auto;
}
.empty-state { text-align: center; color: var(--muted); font-size: 13px; padding: 32px; }
.lod-tier { display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.lod-tier:last-child { border-bottom: 0; }
.tier-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.tier-FULL { background: var(--good); }
.tier-REDUCED { background: var(--warn); }
.tier-MINIMAL { background: var(--accent); }
.tier-DORMANT { background: var(--muted); }
.time-tag { font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }
#toastContainer { position: fixed; bottom: 24px; right: 24px; display: flex; flex-direction: column; gap: 8px; z-index: 999; }
.toast {
  padding: 10px 16px; border-radius: 8px; font-size: 13px; font-weight: 500;
  background: var(--glass); border: 1px solid var(--border); backdrop-filter: blur(12px);
  animation: slideIn .2s ease;
  max-width: 320px;
}
@keyframes slideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@media (max-width: 820px) {
  .layout { grid-template-columns: 1fr; }
  aside { border-right: 0; border-bottom: 1px solid var(--border); }
  .two-col { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">N</div>
    <h1>OpenNPC Debug Dashboard</h1>
  </div>
  <div class="header-right">
    <div class="status-dot" id="statusDot"></div>
    <span class="status-text" id="statusText">Connecting…</span>
    <label class="auto-label">
      Auto-refresh
      <button class="toggle on" id="autoToggle" onclick="toggleAuto()" aria-label="Toggle auto-refresh"></button>
    </label>
    <button class="btn btn-accent" onclick="refresh()">↻ Refresh</button>
  </div>
</header>

<div class="layout">
  <aside>
    <div class="card">
      <div class="card-title">Agent Lookup</div>
      <div class="input-row">
        <input type="text" id="agentId" value="enemy_01" aria-label="Agent ID" placeholder="Agent ID…">
        <button class="btn" onclick="loadAgent()">Load</button>
      </div>
    </div>
    <div class="card" id="metricsCard">
      <div class="card-title">Runtime Metrics</div>
      <div class="metric-grid" id="metrics">
        <div class="metric"><span class="metric-key">Transitions</span><span class="metric-val" id="m-trans">—</span></div>
        <div class="metric"><span class="metric-key">Mean Reward</span><span class="metric-val" id="m-reward">—</span></div>
        <div class="metric"><span class="metric-key">Terminal</span><span class="metric-val" id="m-term">—</span></div>
        <div class="metric"><span class="metric-key">Last Refresh</span><span class="metric-val" id="m-time">—</span></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">LOD Distribution</div>
      <div id="lodList"><p class="empty-state">Loading…</p></div>
    </div>
  </aside>

  <div class="main-content">
    <div class="card">
      <div class="section-title">Recent Decisions <span class="badge" id="decisionCount">0</span></div>
      <div class="decision-list" id="decisions">
        <div class="empty-state">No decisions yet — call /decide or /coordinate.</div>
      </div>
    </div>
    <div class="two-col">
      <div class="card">
        <div class="section-title">Agent Memory</div>
        <pre class="code" id="memory">No agent loaded.</pre>
      </div>
      <div class="card">
        <div class="section-title">Agent Experience</div>
        <pre class="code" id="experience">No agent loaded.</pre>
      </div>
    </div>
    <div class="card">
      <div class="section-title">Player Patterns</div>
      <pre class="code" id="patterns">Loading…</pre>
    </div>
  </div>
</div>

<div id="toastContainer"></div>

<script>
  let autoRefresh = true;
  let refreshTimer = null;
  const INTERVAL = 4000;

  function toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = 'toast';
    el.style.borderColor = type === 'error' ? 'rgba(248,113,113,0.4)' : 'rgba(99,102,241,0.3)';
    el.textContent = msg;
    document.getElementById('toastContainer').appendChild(el);
    setTimeout(() => el.remove(), 3500);
  }

  function toggleAuto() {
    autoRefresh = !autoRefresh;
    const btn = document.getElementById('autoToggle');
    btn.className = 'toggle ' + (autoRefresh ? 'on' : 'off');
    if (autoRefresh) scheduleRefresh();
    else clearTimeout(refreshTimer);
  }

  function scheduleRefresh() {
    clearTimeout(refreshTimer);
    if (autoRefresh) refreshTimer = setTimeout(() => { refresh(); }, INTERVAL);
  }

  async function getJSON(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  }

  function esc(v) {
    return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function pretty(v) { return JSON.stringify(v, null, 2); }

  function renderDecisions(data) {
    const list = data.decisions || [];
    const container = document.getElementById('decisions');
    document.getElementById('decisionCount').textContent = list.length;
    if (!list.length) {
      container.innerHTML = '<div class="empty-state">No decisions yet — call /decide or /coordinate.</div>';
      return;
    }
    container.innerHTML = list.map(item => {
      const trace = item.trace || {};
      const isFallback = trace.fallback_used;
      const conf = Number(item.confidence || 0);
      const pct = (conf * 100).toFixed(0);
      const badgeClass = isFallback ? 'bad' : 'good';
      const badgeText = isFallback ? 'fallback' : 'ok';
      return `<div class="decision-card">
        <div>
          <div class="decision-header">
            <span class="agent-name">${esc(item.agent_id)}</span>
            <span class="action-name">${esc(item.action)}</span>
            <span class="badge ${badgeClass}">${badgeText}</span>
          </div>
          <div class="decision-meta">policy: ${esc(trace.policy_name || 'rule-based')} &nbsp;|&nbsp; ${esc(item.reason || '')}</div>
          <div class="confidence-bar-wrap">
            <span>${pct}%</span>
            <div class="confidence-bar-bg"><div class="confidence-bar" style="width:${pct}%"></div></div>
          </div>
        </div>
        <span class="time-tag">${new Date().toLocaleTimeString()}</span>
      </div>`;
    }).join('');
  }

  function renderLOD(data) {
    const tiers = data.tiers || data;
    const el = document.getElementById('lodList');
    if (!tiers || typeof tiers !== 'object') { el.innerHTML = '<pre class="code">' + esc(pretty(data)) + '</pre>'; return; }
    const entries = Object.entries(tiers);
    if (!entries.length) { el.innerHTML = '<p class="empty-state">No LOD data.</p>'; return; }
    el.innerHTML = entries.map(([tier, count]) =>
      `<div class="lod-tier"><div class="tier-dot tier-${esc(tier)}"></div><span style="flex:1">${esc(tier)}</span><strong>${count}</strong></div>`
    ).join('');
  }

  function setStatus(ok) {
    document.getElementById('statusDot').style.background = ok ? 'var(--good)' : 'var(--bad)';
    document.getElementById('statusDot').style.boxShadow = ok ? '0 0 8px var(--good)' : '0 0 8px var(--bad)';
    document.getElementById('statusText').textContent = ok ? 'Live' : 'Error';
  }

  async function refresh() {
    try {
      const [exp, decisions, patterns, lod] = await Promise.all([
        getJSON('/debug/experience'),
        getJSON('/debug/decisions'),
        getJSON('/debug/patterns'),
        getJSON('/debug/lod'),
      ]);
      document.getElementById('m-trans').textContent = exp.transitions ?? 0;
      document.getElementById('m-reward').textContent = exp.reward_mean != null ? Number(exp.reward_mean).toFixed(3) : 'n/a';
      document.getElementById('m-term').textContent = exp.terminal_transitions ?? 0;
      document.getElementById('m-time').textContent = new Date().toLocaleTimeString();
      renderDecisions(decisions);
      document.getElementById('patterns').textContent = pretty(patterns);
      renderLOD(lod);
      setStatus(true);
    } catch (e) {
      setStatus(false);
      toast('Refresh failed: ' + e.message, 'error');
    } finally {
      scheduleRefresh();
    }
  }

  async function loadAgent() {
    const id = document.getElementById('agentId').value.trim();
    if (!id) return;
    try {
      const [mem, exp] = await Promise.all([
        getJSON('/memory/' + encodeURIComponent(id)),
        getJSON('/debug/experience/' + encodeURIComponent(id)),
      ]);
      document.getElementById('memory').textContent = pretty(mem);
      document.getElementById('experience').textContent = pretty(exp);
      toast('Loaded agent: ' + id);
    } catch (e) {
      toast('Agent load failed: ' + e.message, 'error');
    }
  }

  document.getElementById('agentId').addEventListener('keydown', e => { if (e.key === 'Enter') loadAgent(); });
  refresh();
</script>
</body>
</html>
"""



@app.get("/debug/lod", dependencies=[Depends(require_debug_access)])
def debug_lod() -> dict[str, Any]:
    """Return LOD distribution across all registered agents."""
    return lod_engine.summary()


@app.get("/debug/lod/{agent_id}", dependencies=[Depends(require_debug_access)])
def debug_lod_agent(agent_id: str = Path(min_length=1, max_length=settings.max_agent_id_length)) -> dict[str, Any]:
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


@app.post("/debug/lod/{agent_id}", dependencies=[Depends(require_debug_access)])
def debug_lod_update(
    agent_id: str = Path(min_length=1, max_length=settings.max_agent_id_length),
    distance: float = Query(default=10.0, ge=0.0),
    visible: bool = True,
    importance: float = Query(default=0.5, ge=0.0, le=1.0),
    in_combat: bool = False,
) -> dict[str, str]:
    """Update LOD parameters for an agent and return the new tier."""
    tier = lod_engine.update(agent_id, distance, visible, importance, in_combat)
    return {"agent_id": agent_id, "tier": tier.name}


@app.get("/debug/patterns", dependencies=[Depends(require_debug_access)])
def debug_patterns() -> dict[str, Any]:
    """Return current player pattern analysis."""
    return pattern_tracker.to_dict()


@app.post("/debug/patterns/record", dependencies=[Depends(require_debug_access)])
def debug_record_action(action: str = Query(min_length=1, max_length=64)) -> dict[str, str]:
    """Record a player action for pattern tracking."""
    pattern_tracker.record(action)
    return {"recorded": action, "total": str(pattern_tracker.total_observations)}


@app.post("/debug/villain/plan", dependencies=[Depends(require_debug_access)])
def debug_villain_plan(request: DecisionRequest) -> dict[str, Any]:
    """Generate a villain strategic plan from config + state."""
    _validate_decision_request(request)
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
