from fastapi.testclient import TestClient

from opennpc.api.service import app, experience_logger


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
