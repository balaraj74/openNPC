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
 * A Java client to connect a Minecraft Forge Mod (1.20.x / Java 17) 
 * to the OpenNPC python server.
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

    public static class DecisionResponse {
        public String agent_id;
        public String action;
        public float confidence;
        public String reason;
        public String memory_update;
    }

    /**
     * Sends a decision request asynchronously.
     * Use this so you do not block the Minecraft main server/client thread!
     */
    public CompletableFuture<DecisionResponse> decide(String configJson, String stateJson) {
        JsonObject payload = new JsonObject();
        payload.add("config", JsonParser.parseString(configJson).getAsJsonObject());
        payload.add("state", JsonParser.parseString(stateJson).getAsJsonObject());

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/decide"))
                .header("Content-Type", "application/json")
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
}
