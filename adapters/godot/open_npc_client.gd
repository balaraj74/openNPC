extends Node
class_name OpenNPCClient

signal decision_received(decision: Dictionary)
signal batch_received(decisions: Array)
signal coordination_received(plan: Dictionary)
signal request_failed(message: String)

@export var base_url: String = "http://127.0.0.1:8787"

var _http: HTTPRequest
var _pending_kind: String = ""


func _ready() -> void:
	_http = HTTPRequest.new()
	add_child(_http)
	_http.request_completed.connect(_on_request_completed)


func decide(config: Dictionary, state: Dictionary, available_actions: Array = []) -> void:
	_pending_kind = "decide"
	var payload := {
		"config": config,
		"state": state,
	}
	if not available_actions.is_empty():
		payload["available_actions"] = available_actions
	_post("/decide", payload)


func decide_batch(requests: Array) -> void:
	_pending_kind = "batch"
	_post("/batch/decide", {"requests": requests})


func coordinate(
	configs: Array,
	states: Array,
	available_actions_by_agent: Dictionary = {},
	max_attackers_per_target: int = 2
) -> void:
	_pending_kind = "coordinate"
	_post("/coordinate", {
		"configs": configs,
		"states": states,
		"available_actions": available_actions_by_agent,
		"max_attackers_per_target": max_attackers_per_target,
	})


func make_config(
	agent_id: String,
	agent_type: String = "enemy",
	personality: Dictionary = {},
	goals: Array = [],
	allowed_actions: Array = []
) -> Dictionary:
	var config := {
		"agent_id": agent_id,
		"agent_type": agent_type,
		"personality": personality,
		"goals": goals,
	}
	if not allowed_actions.is_empty():
		config["allowed_actions"] = allowed_actions
	return config


func make_state(
	agent_id: String,
	health: float = 100.0,
	threat_level: float = 0.0,
	nearby_entities: Array = [],
	target_health: float = -1.0,
	distance_to_target: float = -1.0,
	cover_available: bool = false
) -> Dictionary:
	var state := {
		"agent_id": agent_id,
		"health": health,
		"threat_level": threat_level,
		"nearby_entities": nearby_entities,
		"cover_available": cover_available,
	}
	if target_health >= 0.0:
		state["target_health"] = target_health
	if distance_to_target >= 0.0:
		state["distance_to_target"] = distance_to_target
	return state


func _post(path: String, payload: Dictionary) -> void:
	var body := JSON.stringify(payload)
	var error := _http.request(
		_url(path),
		["Content-Type: application/json"],
		HTTPClient.METHOD_POST,
		body
	)
	if error != OK:
		request_failed.emit("OpenNPC request failed to start: %s" % error)


func _url(path: String) -> String:
	var root := base_url
	while root.ends_with("/"):
		root = root.substr(0, root.length() - 1)
	return root + path


func _on_request_completed(result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS:
		request_failed.emit("OpenNPC request failed with result %s" % result)
		return
	if response_code < 200 or response_code >= 300:
		request_failed.emit("OpenNPC returned HTTP %s: %s" % [response_code, body.get_string_from_utf8()])
		return

	var parsed = JSON.parse_string(body.get_string_from_utf8())
	if typeof(parsed) != TYPE_DICTIONARY:
		request_failed.emit("OpenNPC returned invalid JSON.")
		return

	if _pending_kind == "batch":
		batch_received.emit(parsed.get("decisions", []))
	elif _pending_kind == "coordinate":
		coordination_received.emit(parsed)
	else:
		decision_received.emit(parsed)
