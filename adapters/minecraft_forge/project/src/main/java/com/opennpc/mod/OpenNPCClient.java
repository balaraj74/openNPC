package com.opennpc.mod;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

/**
 * HTTP client connecting Minecraft Forge mod to the OpenNPC Python server.
 * Supports decision requests, event reporting, and villain bark retrieval.
 */
public class OpenNPCClient {
    private final HttpClient httpClient;
    private final Gson gson;
    private String baseUrl = "http://127.0.0.1:8787";

    public OpenNPCClient() {
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(5))
                .build();
        this.gson = new Gson();
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
    }

    /**
     * Full decision response including optional villain bark.
     */
    public static class DecisionResponse {
        public String agent_id;
        public String action;
        public float confidence;
        public String reason;
        public String memory_update;
        // Villain-specific fields
        public String bark;
        public String strategy;
        public boolean llm_enhanced;
    }

    /**
     * Send a decision request asynchronously. Never blocks the server thread.
     */
    public CompletableFuture<DecisionResponse> decide(String configJson, String stateJson) {
        JsonObject payload = new JsonObject();
        payload.add("config", JsonParser.parseString(configJson).getAsJsonObject());
        payload.add("state", JsonParser.parseString(stateJson).getAsJsonObject());

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/decide"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(3))
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(payload)))
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return gson.fromJson(response.body(), DecisionResponse.class);
                    } else {
                        throw new RuntimeException("OpenNPC API error: " + response.statusCode() + " " + response.body());
                    }
                });
    }

    /**
     * Report a player action event for pattern tracking (fire-and-forget).
     */
    public void reportEvent(String agentId, String eventText) {
        JsonObject payload = new JsonObject();
        payload.addProperty("agent_id", agentId);
        payload.addProperty("event", eventText);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/event"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(2))
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(payload)))
                .build();

        // Fire and forget — don't block on result
        httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .exceptionally(ex -> null);
    }
}
