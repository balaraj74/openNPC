// OpenNPC Unreal Engine Adapter — C++ Header
// 
// Drop this file (and the .cpp) into your UE project's Source directory.
// Requires: HTTP module enabled in your .Build.cs
//
// Usage in Blueprints or C++:
//   UOpenNPCClient* Client = NewObject<UOpenNPCClient>();
//   Client->SetBaseURL("http://127.0.0.1:8787");
//   Client->Decide(ConfigJson, StateJson);
//   // Result arrives via OnDecisionReceived delegate

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "OpenNPCClient.generated.h"

USTRUCT(BlueprintType)
struct FOpenNPCDecision
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "OpenNPC")
    FString AgentId;

    UPROPERTY(BlueprintReadOnly, Category = "OpenNPC")
    FString Action;

    UPROPERTY(BlueprintReadOnly, Category = "OpenNPC")
    float Confidence;

    UPROPERTY(BlueprintReadOnly, Category = "OpenNPC")
    FString Reason;

    UPROPERTY(BlueprintReadOnly, Category = "OpenNPC")
    FString MemoryUpdate;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnDecisionReceived, const FOpenNPCDecision&, Decision);

/**
 * UOpenNPCClient — Sends HTTP requests to the OpenNPC inference server.
 *
 * Non-blocking: fires OnDecisionReceived when the response arrives.
 */
UCLASS(BlueprintType, Blueprintable)
class YOURGAME_API UOpenNPCClient : public UObject
{
    GENERATED_BODY()

public:
    UOpenNPCClient();

    /** Base URL of the running OpenNPC API server. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OpenNPC")
    FString BaseURL;

    /** Fired when the server returns a decision. */
    UPROPERTY(BlueprintAssignable, Category = "OpenNPC")
    FOnDecisionReceived OnDecisionReceived;

    /** Set the base URL for the OpenNPC API. */
    UFUNCTION(BlueprintCallable, Category = "OpenNPC")
    void SetBaseURL(const FString& URL);

    /**
     * Request a decision from the inference server.
     *
     * @param ConfigJson  JSON string of agent config.
     * @param StateJson   JSON string of current game state.
     */
    UFUNCTION(BlueprintCallable, Category = "OpenNPC")
    void Decide(const FString& ConfigJson, const FString& StateJson);

    /**
     * Build a minimal config JSON from common parameters.
     * For full control, construct the JSON yourself.
     */
    UFUNCTION(BlueprintCallable, Category = "OpenNPC")
    static FString MakeConfigJson(
        const FString& AgentId,
        const FString& AgentType,
        float Aggression,
        float Caution,
        const TArray<FString>& AllowedActions
    );

    /**
     * Build a state JSON from common game-state values.
     */
    UFUNCTION(BlueprintCallable, Category = "OpenNPC")
    static FString MakeStateJson(
        const FString& AgentId,
        float Health,
        float ThreatLevel,
        float TargetHealth,
        float DistanceToTarget,
        bool CoverAvailable,
        const TArray<FString>& NearbyEntities
    );

private:
    void HandleResponse(const FString& ResponseBody);
};
