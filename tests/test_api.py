from fastapi.testclient import TestClient

from opennpc.api import service
from opennpc.api.service import app, experience_logger
from opennpc.security import RuntimeSettings


def test_api_decide_endpoint() -> None:
    client = TestClient(app)
    payload = {
        "config": {
            "agent_id": "enemy_01",
            "agent_type": "enemy",
            "personality": {"aggression": 0.9, "caution": 0.2, "risk_tolerance": 0.8},
            "goals": [{"name": "attack_target", "priority": 1.0}],
            "allowed_actions": ["idle", "move", "attack", "defend", "flank"],
        },
        "state": {
            "agent_id": "enemy_01",
            "health": 90,
            "threat_level": 0.2,
            "nearby_entities": ["player"],
            "target_health": 20,
            "distance_to_target": 1.0,
        },
    }

    response = client.post("/decide", json=payload)

    assert response.status_code == 200
    assert response.json()["action"] == "attack"


def test_api_coordinate_endpoint() -> None:
    client = TestClient(app)
    payload = {
        "max_attackers_per_target": 1,
        "configs": [
            {
                "agent_id": "enemy_a",
                "agent_type": "enemy",
                "personality": {"aggression": 0.9, "caution": 0.1, "risk_tolerance": 0.9},
                "goals": [{"name": "attack_target", "priority": 1.0}],
                "allowed_actions": ["idle", "move", "attack", "defend", "flank"],
            },
            {
                "agent_id": "enemy_b",
                "agent_type": "enemy",
                "personality": {"aggression": 0.8, "caution": 0.2, "risk_tolerance": 0.8},
                "goals": [{"name": "attack_target", "priority": 1.0}],
                "allowed_actions": ["idle", "move", "attack", "defend", "flank"],
            },
        ],
        "states": [
            {
                "agent_id": "enemy_a",
                "health": 90,
                "threat_level": 0.2,
                "nearby_entities": ["player"],
                "target_health": 25,
                "distance_to_target": 1.0,
            },
            {
                "agent_id": "enemy_b",
                "health": 90,
                "threat_level": 0.2,
                "nearby_entities": ["player"],
                "target_health": 25,
                "distance_to_target": 1.0,
            },
        ],
    }

    response = client.post("/coordinate", json=payload)

    assert response.status_code == 200
    data = response.json()
    actions = [decision["action"] for decision in data["decisions"].values()]
    assert actions.count("attack") == 1
    assert data["shared_context"]["agent_count"] == 2


def test_api_experience_debug_endpoint() -> None:
    experience_logger.clear()
    client = TestClient(app)

    response = client.get("/debug/experience")

    assert response.status_code == 200
    assert response.json() == {"transitions": 0}


def test_api_decision_debug_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/debug/decisions")

    assert response.status_code == 200
    assert "decisions" in response.json()


def test_api_debug_dashboard_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/debug/dashboard")

    assert response.status_code == 200
    assert "OpenNPC Debug Dashboard" in response.text
    assert "function esc(" in response.text or "escapeHTML" in response.text


def test_api_key_protects_runtime_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(service, "settings", RuntimeSettings(api_key="secret"))
    client = TestClient(app)
    payload = {
        "config": {
            "agent_id": "enemy_auth",
            "agent_type": "enemy",
            "goals": [{"name": "survive", "priority": 1.0}],
            "allowed_actions": ["idle", "defend"],
        },
        "state": {"agent_id": "enemy_auth", "health": 90},
    }

    assert client.post("/decide", json=payload).status_code == 401
    response = client.post("/decide", json=payload, headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json()["agent_id"] == "enemy_auth"


def test_debug_endpoints_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(service, "settings", RuntimeSettings(debug_enabled=False))
    client = TestClient(app)

    response = client.get("/debug/decisions")

    assert response.status_code == 404


def test_batch_limit_is_enforced(monkeypatch) -> None:
    monkeypatch.setattr(service, "settings", RuntimeSettings(max_batch_size=1))
    client = TestClient(app)
    item = {
        "config": {
            "agent_id": "enemy_batch",
            "agent_type": "enemy",
            "goals": [{"name": "survive", "priority": 1.0}],
            "allowed_actions": ["idle", "defend"],
        },
        "state": {"agent_id": "enemy_batch", "health": 90},
    }

    response = client.post("/batch/decide", json={"requests": [item, item]})

    assert response.status_code == 413
