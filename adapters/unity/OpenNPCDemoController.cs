using System;
using System.Collections;
using UnityEngine;

[Serializable]
public class OpenNPCDemoAgent
{
    public string agentId = "enemy_01";
    public string agentType = "enemy";
    [Range(0f, 1f)] public float aggression = 0.75f;
    [Range(0f, 1f)] public float caution = 0.45f;
    [Range(0f, 1f)] public float sociability = 0.5f;
    [Range(0f, 1f)] public float riskTolerance = 0.65f;
    public Transform actor;
    public Renderer statusRenderer;
    [NonSerialized] public string lastAction = "idle";
    [NonSerialized] public float health = 100f;
    [NonSerialized] public float lastDecisionAt = -999f;
}

public class OpenNPCDemoController : MonoBehaviour
{
    [SerializeField] private OpenNPCClient client;
    [SerializeField] private Transform player;
    [SerializeField] private OpenNPCDemoAgent[] agents;
    [SerializeField] private float decisionInterval = 0.5f;
    [SerializeField] private float moveSpeed = 2.5f;

    private void Awake()
    {
        if (client == null)
        {
            client = GetComponent<OpenNPCClient>();
        }
        if (client == null)
        {
            client = gameObject.AddComponent<OpenNPCClient>();
        }
    }

    private void Update()
    {
        if (player == null || agents == null)
        {
            return;
        }

        foreach (var agent in agents)
        {
            if (agent.actor == null)
            {
                continue;
            }

            ApplyAction(agent, agent.lastAction);
            if (Time.time - agent.lastDecisionAt < decisionInterval)
            {
                continue;
            }
            agent.lastDecisionAt = Time.time;
            StartCoroutine(RequestAndApply(agent));
        }
    }

    private IEnumerator RequestAndApply(OpenNPCDemoAgent agent)
    {
        var configJson = BuildConfigJson(agent);
        var stateJson = BuildStateJson(agent);
        yield return client.Decide(
            configJson,
            stateJson,
            decision =>
            {
                agent.lastAction = decision.action;
            },
            error => Debug.LogWarning($"OpenNPC request failed for {agent.agentId}: {error}")
        );
    }

    private string BuildConfigJson(OpenNPCDemoAgent agent)
    {
        var goals = agent.agentType == "civilian"
            ? "[{\"name\":\"interact\",\"priority\":0.8},{\"name\":\"survive\",\"priority\":0.5}]"
            : "[{\"name\":\"attack_target\",\"priority\":0.8},{\"name\":\"survive\",\"priority\":0.6},{\"name\":\"weaken_player\",\"priority\":0.4}]";
        var actions = agent.agentType == "civilian"
            ? "[\"idle\",\"move\",\"talk\",\"trade\",\"flee\",\"hide\",\"patrol\"]"
            : "[\"idle\",\"move\",\"attack\",\"defend\",\"flee\",\"seek_cover\",\"flank\",\"set_trap\",\"patrol\"]";
        return "{" +
            $"\"agent_id\":\"{agent.agentId}\"," +
            $"\"agent_type\":\"{agent.agentType}\"," +
            "\"personality\":{" +
            $"\"aggression\":{agent.aggression:0.###}," +
            $"\"caution\":{agent.caution:0.###}," +
            $"\"sociability\":{agent.sociability:0.###}," +
            $"\"risk_tolerance\":{agent.riskTolerance:0.###}" +
            "}," +
            $"\"goals\":{goals}," +
            $"\"allowed_actions\":{actions}" +
            "}";
    }

    private string BuildStateJson(OpenNPCDemoAgent agent)
    {
        var distance = Vector3.Distance(agent.actor.position, player.position);
        var threat = Mathf.Clamp01(1f - distance / 12f);
        var targetHealth = Mathf.Max(1f, 100f - Time.time % 100f);
        return "{" +
            $"\"agent_id\":\"{agent.agentId}\"," +
            $"\"agent_type\":\"{agent.agentType}\"," +
            $"\"health\":{agent.health:0.###}," +
            $"\"threat_level\":{threat:0.###}," +
            $"\"target_health\":{targetHealth:0.###}," +
            $"\"distance_to_target\":{distance:0.###}," +
            $"\"cover_available\":{(distance > 4f ? "true" : "false")}," +
            "\"nearby_entities\":[\"player\"]," +
            $"\"last_action\":\"{agent.lastAction}\"" +
            "}";
    }

    private void ApplyAction(OpenNPCDemoAgent agent, string action)
    {
        if (action == "move" || action == "flank" || action == "attack")
        {
            agent.actor.position = Vector3.MoveTowards(agent.actor.position, player.position, moveSpeed * Time.deltaTime);
        }
        else if (action == "flee")
        {
            var away = (agent.actor.position - player.position).normalized;
            agent.actor.position += away * moveSpeed * Time.deltaTime;
        }
        else if (action == "patrol")
        {
            agent.actor.position += Vector3.right * Mathf.Sin(Time.time + agent.actor.GetInstanceID()) * moveSpeed * 0.25f * Time.deltaTime;
        }

        if (agent.statusRenderer != null)
        {
            agent.statusRenderer.material.color = ColorForAction(action);
        }
    }

    private Color ColorForAction(string action)
    {
        switch (action)
        {
            case "attack": return Color.red;
            case "flank": return new Color(1f, 0.55f, 0f);
            case "defend":
            case "seek_cover": return Color.blue;
            case "flee":
            case "hide": return Color.gray;
            case "talk":
            case "trade": return Color.green;
            default: return Color.white;
        }
    }
}
