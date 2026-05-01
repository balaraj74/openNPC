// OpenNPC Unreal Engine Adapter — C++ Implementation
//
// Requires: HTTP, Json, JsonUtilities modules in your .Build.cs:
//   PublicDependencyModuleNames.AddRange(new string[] {
//       "Core", "CoreUObject", "Engine", "HTTP", "Json", "JsonUtilities"
//   });

#include "OpenNPCClient.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"

UOpenNPCClient::UOpenNPCClient()
    : BaseURL(TEXT("http://127.0.0.1:8787"))
{
}

void UOpenNPCClient::SetBaseURL(const FString& URL)
{
    BaseURL = URL;
    if (BaseURL.EndsWith(TEXT("/")))
    {
        BaseURL.RemoveFromEnd(TEXT("/"));
    }
}

void UOpenNPCClient::Decide(const FString& ConfigJson, const FString& StateJson)
{
    // Build the request payload
    TSharedPtr<FJsonObject> PayloadObj = MakeShareable(new FJsonObject());

    TSharedPtr<FJsonObject> ConfigObj;
    TSharedRef<TJsonReader<>> ConfigReader = TJsonReaderFactory<>::Create(ConfigJson);
    FJsonSerializer::Deserialize(ConfigReader, ConfigObj);

    TSharedPtr<FJsonObject> StateObj;
    TSharedRef<TJsonReader<>> StateReader = TJsonReaderFactory<>::Create(StateJson);
    FJsonSerializer::Deserialize(StateReader, StateObj);

    if (ConfigObj.IsValid()) PayloadObj->SetObjectField(TEXT("config"), ConfigObj);
    if (StateObj.IsValid()) PayloadObj->SetObjectField(TEXT("state"), StateObj);

    FString PayloadStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&PayloadStr);
    FJsonSerializer::Serialize(PayloadObj.ToSharedRef(), Writer);

    // Fire the HTTP request
    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
    Request->SetURL(BaseURL + TEXT("/decide"));
    Request->SetVerb(TEXT("POST"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Request->SetContentAsString(PayloadStr);

    Request->OnProcessRequestComplete().BindLambda(
        [this](FHttpRequestPtr Req, FHttpResponsePtr Resp, bool bSuccess)
        {
            if (bSuccess && Resp.IsValid() && Resp->GetResponseCode() == 200)
            {
                HandleResponse(Resp->GetContentAsString());
            }
            else
            {
                // Fallback: return idle so the NPC doesn't freeze
                FOpenNPCDecision Fallback;
                Fallback.AgentId = TEXT("unknown");
                Fallback.Action = TEXT("idle");
                Fallback.Confidence = 0.1f;
                Fallback.Reason = TEXT("OpenNPC server unreachable — using idle fallback.");
                OnDecisionReceived.Broadcast(Fallback);
            }
        });

    Request->ProcessRequest();
}

void UOpenNPCClient::HandleResponse(const FString& ResponseBody)
{
    TSharedPtr<FJsonObject> JsonObj;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseBody);

    if (!FJsonSerializer::Deserialize(Reader, JsonObj) || !JsonObj.IsValid())
    {
        return;
    }

    FOpenNPCDecision Decision;
    Decision.AgentId = JsonObj->GetStringField(TEXT("agent_id"));
    Decision.Action = JsonObj->GetStringField(TEXT("action"));
    Decision.Confidence = static_cast<float>(JsonObj->GetNumberField(TEXT("confidence")));
    Decision.Reason = JsonObj->GetStringField(TEXT("reason"));
    Decision.MemoryUpdate = JsonObj->HasField(TEXT("memory_update"))
        ? JsonObj->GetStringField(TEXT("memory_update"))
        : TEXT("");

    OnDecisionReceived.Broadcast(Decision);
}

FString UOpenNPCClient::MakeConfigJson(
    const FString& AgentId,
    const FString& AgentType,
    float Aggression,
    float Caution,
    const TArray<FString>& AllowedActions)
{
    TSharedPtr<FJsonObject> Obj = MakeShareable(new FJsonObject());
    Obj->SetStringField(TEXT("agent_id"), AgentId);
    Obj->SetStringField(TEXT("agent_type"), AgentType);

    TSharedPtr<FJsonObject> Personality = MakeShareable(new FJsonObject());
    Personality->SetNumberField(TEXT("aggression"), Aggression);
    Personality->SetNumberField(TEXT("caution"), Caution);
    Obj->SetObjectField(TEXT("personality"), Personality);

    TArray<TSharedPtr<FJsonValue>> ActionsArray;
    for (const FString& Action : AllowedActions)
    {
        ActionsArray.Add(MakeShareable(new FJsonValueString(Action)));
    }
    Obj->SetArrayField(TEXT("allowed_actions"), ActionsArray);

    FString Output;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
    FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);
    return Output;
}

FString UOpenNPCClient::MakeStateJson(
    const FString& AgentId,
    float Health,
    float ThreatLevel,
    float TargetHealth,
    float DistanceToTarget,
    bool CoverAvailable,
    const TArray<FString>& NearbyEntities)
{
    TSharedPtr<FJsonObject> Obj = MakeShareable(new FJsonObject());
    Obj->SetStringField(TEXT("agent_id"), AgentId);
    Obj->SetNumberField(TEXT("health"), Health);
    Obj->SetNumberField(TEXT("threat_level"), ThreatLevel);
    Obj->SetNumberField(TEXT("target_health"), TargetHealth);
    Obj->SetNumberField(TEXT("distance_to_target"), DistanceToTarget);
    Obj->SetBoolField(TEXT("cover_available"), CoverAvailable);

    TArray<TSharedPtr<FJsonValue>> EntitiesArray;
    for (const FString& Entity : NearbyEntities)
    {
        EntitiesArray.Add(MakeShareable(new FJsonValueString(Entity)));
    }
    Obj->SetArrayField(TEXT("nearby_entities"), EntitiesArray);

    FString Output;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
    FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);
    return Output;
}
