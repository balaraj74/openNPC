using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

[Serializable]
public class OpenNPCDecisionRequest
{
    public string configJson;
    public string stateJson;

    public string ToJson()
    {
        return "{\"config\":" + configJson + ",\"state\":" + stateJson + "}";
    }
}

[Serializable]
public class OpenNPCDecisionResponse
{
    public string agent_id;
    public string action;
    public float confidence;
    public string reason;
    public string memory_update;
}

public class OpenNPCClient : MonoBehaviour
{
    [SerializeField] private string baseUrl = "http://127.0.0.1:8787";

    public void SetBaseUrl(string url)
    {
        baseUrl = string.IsNullOrWhiteSpace(url) ? "http://127.0.0.1:8787" : url.TrimEnd('/');
    }

    public IEnumerator Decide(string configJson, string stateJson, Action<OpenNPCDecisionResponse> onDecision, Action<string> onError = null)
    {
        var payload = new OpenNPCDecisionRequest
        {
            configJson = configJson,
            stateJson = stateJson
        }.ToJson();

        using var request = new UnityWebRequest(baseUrl.TrimEnd('/') + "/decide", "POST");
        byte[] body = Encoding.UTF8.GetBytes(payload);
        request.uploadHandler = new UploadHandlerRaw(body);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            onError?.Invoke(request.error);
            yield break;
        }

        var response = JsonUtility.FromJson<OpenNPCDecisionResponse>(request.downloadHandler.text);
        onDecision?.Invoke(response);
    }
}
