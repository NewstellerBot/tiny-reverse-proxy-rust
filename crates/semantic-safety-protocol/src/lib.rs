pub mod semantic_safety {
    tonic::include_proto!("semantic_safety.v1");
}

pub use semantic_safety::semantic_safety_service_client::SemanticSafetyServiceClient;
pub use semantic_safety::semantic_safety_service_server::{
    SemanticSafetyService, SemanticSafetyServiceServer,
};
pub use semantic_safety::{
    Chunk, DeleteProjectPolicyRequest, DeleteProjectPolicyResponse, EvaluateRequest,
    EvaluateResponse, Finding, GetProjectSyncStatusRequest, GetProjectSyncStatusResponse,
    HealthRequest, HealthResponse, IndexState, ListProjectSyncStatesRequest,
    ListProjectSyncStatesResponse, ProjectSemanticPolicy, ProjectSyncState, SemanticEntity,
    SemanticTopic, UpsertProjectPolicyRequest, UpsertProjectPolicyResponse,
};

pub fn index_state_name(value: i32) -> &'static str {
    match IndexState::try_from(value).ok() {
        Some(IndexState::Ready) => "ready",
        Some(IndexState::Missing) => "missing",
        Some(IndexState::Stale) => "stale",
        Some(IndexState::Disabled) => "disabled",
        Some(IndexState::Degraded) => "degraded",
        _ => "unspecified",
    }
}
