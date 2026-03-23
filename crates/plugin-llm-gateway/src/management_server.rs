use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::AtomicU64;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use proxy_auth::{AuthContext, Permission, ProjectId, Role};
use proxy_core::config::{
    ProviderDataCollectionMode, ProviderFamily, ProviderFamilyConfig, ProviderKeyConfig,
    ProviderSurfaceCatalog,
};
use semantic_safety_protocol::{ProjectSemanticPolicy, SemanticEntity, SemanticTopic};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::api::LlmGatewayApi;
use crate::evals::{ProjectEvalRunComparisonGateRequest, ProjectEvalRunRequest};
use crate::governance::current_timestamp_string;
use crate::semantic_safety::{generate_semantic_policy_version, proto_to_record, record_to_proto};
use crate::session_recovery::{
    reconcile_session_record, session_cancel_is_pending, session_handoff_is_pending,
    session_owner_is_stale, session_recovery_reason, session_recovery_required,
    SessionReconcileAction, SESSION_RECOVERY_REQUIRED_REASON,
};
use crate::store::{
    ManagedProviderRecord, ProjectDatasetItemRecord, ProjectDatasetRecord,
    ProjectEvalRunItemRecord, ProjectEvalRunRecord, ProjectPolicyRecord, ProjectPromptRecord,
    ProjectPromptRolloutRecord, ProjectRolloutPolicyRecord, ProjectSemanticPolicyRecord,
    ProjectToolRecord, RequestLogEntry, RequestLogQuery, RoutingRuleRecord, SafetyPolicyRecord,
    SessionEventRecord, SessionListQuery, SessionRecord, VirtualKeyRecord,
};
use crate::tool_runtime::{ToolRuntimeAudit, ToolRuntimeMcpServerSnapshot};
use crate::virtual_keys::{
    effective_tool_approval_policy, normalize_tool_approval_mode_value, validate_string_array_json,
    VirtualKeyLookupError,
};

#[derive(Debug, Deserialize)]
struct SemanticPolicyPayload {
    version: Option<String>,
    enabled: Option<bool>,
    entities: Option<Vec<SemanticEntity>>,
    topics: Option<Vec<SemanticTopic>>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ProviderRoutingMetadataPayload {
    data_collection: Option<String>,
    zdr: Option<bool>,
    distillable_text: Option<bool>,
    quantizations: Option<Vec<String>>,
    supported_parameter_families: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ProviderPayload {
    name: Option<String>,
    enabled: Option<bool>,
    api_key_env: Option<String>,
    base_url: Option<String>,
    models: Option<Vec<String>>,
    api_key_header: Option<String>,
    timeout_secs: Option<u64>,
    family: Option<ProviderFamily>,
    surfaces: Option<ProviderSurfaceCatalog>,
    routing_metadata: Option<ProviderRoutingMetadataPayload>,
}

#[derive(Debug, Deserialize)]
struct ProjectToolPayload {
    description: Option<String>,
    input_schema: Option<serde_json::Value>,
    executor_kind: Option<String>,
    executor_config: Option<serde_json::Value>,
    enabled: Option<bool>,
    timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ProjectPromptPayload {
    environment: Option<String>,
    description: Option<String>,
    target: Option<String>,
    template_text: Option<String>,
    variables_schema: Option<serde_json::Value>,
    rollout_metadata: Option<serde_json::Value>,
    active: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ProjectRolloutPolicyPayload {
    description: Option<String>,
    gate: Option<ProjectEvalRunComparisonGateRequest>,
    canary: Option<RolloutCanaryPolicyPayload>,
    target_environment: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RolloutCanaryPolicyPayload {
    steps: Option<Vec<u8>>,
    auto_promote_final: Option<bool>,
    auto_advance_on_pass: Option<bool>,
    auto_rollback_on_fail: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct PromptPromotionPayload {
    candidate_version: Option<String>,
    baseline_run_id: Option<String>,
    candidate_run_id: Option<String>,
    policy_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProjectPromptRolloutPayload {
    candidate_version: Option<String>,
    baseline_run_id: Option<String>,
    candidate_run_id: Option<String>,
    policy_name: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct PromptRolloutApplyPayload {
    mode: Option<String>,
    traffic_percent: Option<u8>,
}

#[derive(Debug, Deserialize, Default)]
struct PromptRolloutAdvancePayload {
    traffic_percent: Option<u8>,
}

#[derive(Debug, Deserialize)]
struct PromptRolloutEvaluatePayload {
    baseline_run_id: Option<String>,
    candidate_run_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProjectDatasetPayload {
    description: Option<String>,
    schema: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ProjectDatasetItemPayload {
    input: Option<serde_json::Value>,
    expected_output: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ProjectEvalRunPayload {
    dataset_name: Option<String>,
    target_url: Option<String>,
    headers: Option<HashMap<String, String>>,
    timeout_ms: Option<u64>,
    judge_url: Option<String>,
    judge_kind: Option<String>,
    judge_model: Option<String>,
    judge_headers: Option<HashMap<String, String>>,
    judge_timeout_ms: Option<u64>,
    prompt_name: Option<String>,
    prompt_version: Option<String>,
    provider_name: Option<String>,
    model: Option<String>,
    route_path: Option<String>,
    safety_profile: Option<String>,
    #[serde(rename = "async")]
    run_async: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionStatePayload {
    project_id: Option<String>,
    status: Option<String>,
    state: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionTransitionPayload {
    project_id: Option<String>,
    status: Option<String>,
    reason: Option<String>,
    state: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
    lease_ttl_secs: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionHeartbeatPayload {
    project_id: Option<String>,
    owner_id: Option<String>,
    state: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
    lease_ttl_secs: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionClaimPayload {
    project_id: Option<String>,
    owner_id: Option<String>,
    lease_ttl_secs: Option<u64>,
    state: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionReleasePayload {
    project_id: Option<String>,
    owner_id: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionCancelPayload {
    project_id: Option<String>,
    requested_by: Option<String>,
    reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionHandoffPayload {
    project_id: Option<String>,
    owner_id: Option<String>,
    target_owner_id: Option<String>,
    reason: Option<String>,
    state: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionAcceptPayload {
    project_id: Option<String>,
    owner_id: Option<String>,
    lease_ttl_secs: Option<u64>,
    state: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionTakeoverPayload {
    project_id: Option<String>,
    owner_id: Option<String>,
    lease_ttl_secs: Option<u64>,
    force: Option<bool>,
    reason: Option<String>,
    state: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
struct SessionReconcilePayload {
    project_id: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct McpOauthCallbackPayload {
    state: Option<String>,
    code: Option<String>,
}

static PROMPT_ROLLOUT_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static ROUTING_RULE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Serialize)]
struct FailedProviderResponse {
    failed: Vec<FailedProviderResponseEntry>,
}

#[derive(Debug, Serialize)]
struct FailedProviderResponseEntry {
    name: String,
    failed_ago_secs: u64,
    cooldown_remaining_secs: u64,
    reason: String,
}

#[derive(Debug, Serialize)]
struct ProviderHealthResponse {
    providers: Vec<ProviderHealthResponseEntry>,
}

#[derive(Debug, Serialize)]
struct ProviderHealthResponseEntry {
    name: String,
    eligible: bool,
    cooldown_remaining_secs: u64,
    cooldown_reason: Option<String>,
    active_requests: u32,
    samples: u64,
    ewma_latency_ms: f64,
    ewma_error_rate: f64,
    ewma_timeout_rate: f64,
    ewma_rate_limit_rate: f64,
    adaptive_penalties: ProviderPenaltyResponse,
    adaptive_penalty_total: f64,
}

#[derive(Debug, Serialize)]
struct ProviderPenaltyResponse {
    active_requests: f64,
    latency: f64,
    error: f64,
    timeout: f64,
    rate_limit: f64,
}

struct PreparedPromptRolloutDecision {
    policy: ProjectRolloutPolicyRecord,
    candidate_prompt: ProjectPromptRecord,
    comparison: crate::evals::ProjectEvalRunComparison,
}

/// Start the management HTTP server on the given port.
pub async fn start_management_server(port: u16, api: LlmGatewayApi) {
    start_management_server_with_auth(port, api, None).await;
}

/// Start the management HTTP server on localhost with optional bearer auth.
pub async fn start_management_server_with_auth(
    port: u16,
    api: LlmGatewayApi,
    _auth_token: Option<String>,
) {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("failed to bind management API server");

    tracing::info!(
        port,
        auth_enabled = api.auth_required(),
        "LLM gateway management API listening on 127.0.0.1"
    );

    loop {
        let (stream, _addr) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                tracing::error!(error = %e, "failed to accept management connection");
                continue;
            }
        };

        let api = api.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let service = service_fn(move |req: Request<Incoming>| {
                let api = api.clone();
                async move { handle_request_with_auth(req, api, None).await }
            });

            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                tracing::error!(error = %e, "management connection error");
            }
        });
    }
}

pub async fn handle_request(
    req: Request<Incoming>,
    api: LlmGatewayApi,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    handle_request_with_auth(req, api, None).await
}

pub async fn handle_request_with_auth(
    req: Request<Incoming>,
    api: LlmGatewayApi,
    auth_token: Option<String>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let auth_ctx = match authenticate_request(&api, req.headers(), auth_token.as_deref()).await {
        Ok(auth) => auth,
        Err(resp) => return Ok(resp),
    };

    let method = req.method().clone();
    let path = req.uri().path().to_string();

    let resp = match (method, path.as_str()) {
        (Method::GET, "/api/v1/status") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewStatus, None)
            {
                resp
            } else {
                handle_status(&api)
            }
        }
        (Method::GET, "/api/v1/tool-runtime/status") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewStatus, None)
            {
                resp
            } else {
                handle_tool_runtime_status(&api)
            }
        }
        (Method::GET, "/api/v1/prompt-cache/status") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewStatus, None)
            {
                resp
            } else {
                handle_prompt_cache_status(&api)
            }
        }
        (Method::GET, "/api/v1/semantic-cache/status") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewStatus, None)
            {
                resp
            } else {
                handle_semantic_cache_status(&api)
            }
        }
        (Method::GET, "/api/v1/cost/usage") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewUsage, None)
            {
                resp
            } else {
                handle_cost_usage(&api)
            }
        }
        (Method::GET, "/api/v1/cost/usage/by-model") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewUsage, None)
            {
                resp
            } else {
                handle_cost_usage_by_model(&api)
            }
        }
        (Method::GET, "/api/v1/cost/models") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewModelCosts, None)
            {
                resp
            } else {
                handle_cost_models(&api)
            }
        }
        (Method::DELETE, "/api/v1/cost/usage") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ManageUsage, None)
            {
                resp
            } else {
                handle_delete_all_usage(&api).await
            }
        }
        (Method::GET, "/api/v1/rate-limiter/status") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewRateLimiter, None)
            {
                resp
            } else {
                handle_rate_limiter_status(&api)
            }
        }
        (Method::GET, "/api/v1/providers") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewProviders, None)
            {
                resp
            } else {
                handle_providers(&api)
            }
        }
        (Method::POST, "/api/v1/providers") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ManageProviders, None)
            {
                resp
            } else {
                handle_create_provider(&api, req).await
            }
        }
        (Method::GET, "/api/v1/providers/health") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewProviders, None)
            {
                resp
            } else {
                handle_provider_health(&api)
            }
        }
        (Method::GET, "/api/v1/providers/failed") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ViewProviders, None)
            {
                resp
            } else {
                handle_failed_providers(&api)
            }
        }
        (Method::DELETE, "/api/v1/providers/failed") => {
            if let Some(resp) =
                ensure_permission(&api, auth_ctx.as_ref(), Permission::ManageProviders, None)
            {
                resp
            } else {
                handle_clear_all_failed(&api)
            }
        }
        (Method::POST, "/api/v1/keys") => handle_create_key(&api, req, auth_ctx.as_ref()).await,
        (Method::GET, "/api/v1/keys") => handle_list_keys(&api, auth_ctx.as_ref()),
        (Method::GET, "/api/v1/logs") => handle_request_logs(&api, &req, auth_ctx.as_ref()).await,
        (Method::GET, "/api/v1/sessions") => {
            handle_sessions_list(&api, &req, auth_ctx.as_ref()).await
        }
        (Method::GET, "/api/v1/projects") => handle_list_projects(&api, auth_ctx.as_ref()),
        (Method::POST, "/api/v1/projects") => {
            handle_create_project(&api, req, auth_ctx.as_ref()).await
        }
        (Method::GET, "/api/v1/principals") => handle_list_principals(&api, auth_ctx.as_ref()),
        (Method::POST, "/api/v1/principals") => {
            handle_create_principal(&api, req, auth_ctx.as_ref()).await
        }
        (Method::GET, "/api/v1/roles") => handle_list_roles(&api, auth_ctx.as_ref()),
        (Method::GET, "/api/v1/role-bindings") => {
            handle_list_role_bindings(&api, &req, auth_ctx.as_ref())
        }
        (Method::POST, "/api/v1/role-bindings") => {
            handle_create_role_binding(&api, req, auth_ctx.as_ref()).await
        }
        (Method::GET, "/api/v1/tokens") => handle_list_tokens(&api, &req, auth_ctx.as_ref()),
        _ => {
            // Path-parameter routes.
            if let Some(hash_prefix) = path.strip_prefix("/api/v1/keys/") {
                if !hash_prefix.is_empty() {
                    match *req.method() {
                        Method::GET => handle_get_key(&api, hash_prefix, auth_ctx.as_ref()),
                        Method::PATCH => {
                            handle_update_key(&api, hash_prefix, req, auth_ctx.as_ref()).await
                        }
                        Method::DELETE => {
                            handle_delete_key(&api, hash_prefix, auth_ctx.as_ref()).await
                        }
                        _ => method_not_allowed(),
                    }
                } else {
                    not_found()
                }
            } else if let Some(key) = path.strip_prefix("/api/v1/cost/usage/") {
                if !key.is_empty() {
                    match req.method() {
                        &Method::DELETE => {
                            if let Some(resp) = ensure_permission(
                                &api,
                                auth_ctx.as_ref(),
                                Permission::ManageUsage,
                                None,
                            ) {
                                resp
                            } else {
                                handle_delete_usage(&api, key).await
                            }
                        }
                        _ => method_not_allowed(),
                    }
                } else {
                    not_found()
                }
            } else if let Some(name) = path.strip_prefix("/api/v1/cost/models/") {
                if !name.is_empty() {
                    match *req.method() {
                        Method::PUT => {
                            if let Some(resp) = ensure_permission(
                                &api,
                                auth_ctx.as_ref(),
                                Permission::ManageModelCosts,
                                None,
                            ) {
                                resp
                            } else {
                                handle_put_model_cost(&api, name, req).await
                            }
                        }
                        Method::DELETE => {
                            if let Some(resp) = ensure_permission(
                                &api,
                                auth_ctx.as_ref(),
                                Permission::ManageModelCosts,
                                None,
                            ) {
                                resp
                            } else {
                                handle_delete_model_cost(&api, name).await
                            }
                        }
                        _ => method_not_allowed(),
                    }
                } else {
                    not_found()
                }
            } else if let Some(name) = path.strip_prefix("/api/v1/providers/failed/") {
                if !name.is_empty() {
                    match req.method() {
                        &Method::DELETE => {
                            if let Some(resp) = ensure_permission(
                                &api,
                                auth_ctx.as_ref(),
                                Permission::ManageProviders,
                                None,
                            ) {
                                resp
                            } else {
                                handle_clear_failed_provider(&api, name)
                            }
                        }
                        _ => method_not_allowed(),
                    }
                } else {
                    not_found()
                }
            } else if let Some(name) = path.strip_prefix("/api/v1/providers/") {
                if !name.is_empty() {
                    match *req.method() {
                        Method::GET => {
                            if let Some(resp) = ensure_permission(
                                &api,
                                auth_ctx.as_ref(),
                                Permission::ViewProviders,
                                None,
                            ) {
                                resp
                            } else {
                                handle_get_provider(&api, name)
                            }
                        }
                        Method::PUT => {
                            if let Some(resp) = ensure_permission(
                                &api,
                                auth_ctx.as_ref(),
                                Permission::ManageProviders,
                                None,
                            ) {
                                resp
                            } else {
                                handle_put_provider(&api, name, req).await
                            }
                        }
                        Method::PATCH => {
                            if let Some(resp) = ensure_permission(
                                &api,
                                auth_ctx.as_ref(),
                                Permission::ManageProviders,
                                None,
                            ) {
                                resp
                            } else {
                                handle_patch_provider(&api, name, req).await
                            }
                        }
                        Method::DELETE => {
                            if let Some(resp) = ensure_permission(
                                &api,
                                auth_ctx.as_ref(),
                                Permission::ManageProviders,
                                None,
                            ) {
                                resp
                            } else {
                                handle_delete_provider(&api, name).await
                            }
                        }
                        _ => method_not_allowed(),
                    }
                } else {
                    not_found()
                }
            } else if let Some(server_path) = path.strip_prefix("/api/v1/tool-runtime/mcp/") {
                handle_tool_runtime_mcp_subroutes(&api, server_path, req, auth_ctx.as_ref()).await
            } else if let Some(project_id) = path.strip_prefix("/api/v1/projects/") {
                handle_project_subroutes(&api, project_id, req, auth_ctx.as_ref()).await
            } else if let Some(session_path) = path.strip_prefix("/api/v1/sessions/") {
                handle_session_subroutes(&api, session_path, req, auth_ctx.as_ref()).await
            } else if let Some(principal_id) = path.strip_prefix("/api/v1/principals/") {
                handle_principal_subroutes(&api, principal_id, req, auth_ctx.as_ref()).await
            } else if let Some(token_hash) = path.strip_prefix("/api/v1/tokens/") {
                handle_delete_token(&api, token_hash, auth_ctx.as_ref()).await
            } else if let Some(binding_id) = path.strip_prefix("/api/v1/role-bindings/") {
                handle_delete_role_binding(&api, binding_id, auth_ctx.as_ref()).await
            } else {
                not_found()
            }
        }
    };

    Ok(resp)
}

fn extract_bearer_token(headers: &hyper::HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(ToString::to_string)
}

#[cfg(test)]
fn is_authorized(headers: &hyper::HeaderMap, expected_token: &str) -> bool {
    extract_bearer_token(headers).as_deref() == Some(expected_token)
}

fn json_response(status: StatusCode, body: String) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

fn unauthorized() -> Response<Full<Bytes>> {
    json_response(
        StatusCode::UNAUTHORIZED,
        r#"{"error":"unauthorized"}"#.to_string(),
    )
}

fn forbidden() -> Response<Full<Bytes>> {
    json_response(
        StatusCode::FORBIDDEN,
        r#"{"error":"forbidden"}"#.to_string(),
    )
}

fn not_found() -> Response<Full<Bytes>> {
    json_response(
        StatusCode::NOT_FOUND,
        r#"{"error":"not found"}"#.to_string(),
    )
}

fn method_not_allowed() -> Response<Full<Bytes>> {
    json_response(
        StatusCode::METHOD_NOT_ALLOWED,
        r#"{"error":"method not allowed"}"#.to_string(),
    )
}

fn virtual_key_lookup_response(err: &VirtualKeyLookupError) -> Response<Full<Bytes>> {
    match err {
        VirtualKeyLookupError::AmbiguousPrefix(prefix) => json_response(
            StatusCode::CONFLICT,
            format!(
                r#"{{"error":"hash prefix matches multiple keys: {}"}}"#,
                prefix
            ),
        ),
    }
}

async fn authenticate_request(
    api: &LlmGatewayApi,
    headers: &hyper::HeaderMap,
    _legacy_token: Option<&str>,
) -> Result<Option<AuthContext>, Response<Full<Bytes>>> {
    if !api.auth_required() {
        return Ok(None);
    }
    let token = extract_bearer_token(headers).ok_or_else(unauthorized)?;
    api.authenticate_bearer(&token)
        .await
        .map(Some)
        .ok_or_else(unauthorized)
}

fn ensure_permission(
    api: &LlmGatewayApi,
    auth: Option<&AuthContext>,
    permission: Permission,
    project_id: Option<&str>,
) -> Option<Response<Full<Bytes>>> {
    if !api.auth_required() {
        return None;
    }
    let auth = match auth {
        Some(auth) => auth,
        None => return Some(unauthorized()),
    };
    let project = project_id.map(|project_id| ProjectId(project_id.to_string()));
    if api.is_allowed(auth, permission, project.as_ref()) {
        None
    } else {
        Some(forbidden())
    }
}

fn is_instance_admin(auth: Option<&AuthContext>) -> bool {
    auth.map(|auth| {
        auth.assignments
            .iter()
            .any(|assignment| assignment.role == Role::InstanceAdmin)
    })
    .unwrap_or(false)
}

fn scoped_project_query(
    api: &LlmGatewayApi,
    auth: Option<&AuthContext>,
) -> Result<Option<String>, Response<Full<Bytes>>> {
    if is_instance_admin(auth) {
        return Ok(None);
    }
    let auth = auth.ok_or_else(unauthorized)?;
    let projects = api.accessible_projects(auth);
    if projects.len() == 1 {
        Ok(Some(projects[0].0.clone()))
    } else {
        Err(forbidden())
    }
}

fn bad_request(message: &str) -> Response<Full<Bytes>> {
    json_response(
        StatusCode::BAD_REQUEST,
        format!(r#"{{"error":"{}"}}"#, message),
    )
}

fn resolve_runtime_key_project(
    api: &LlmGatewayApi,
    auth: Option<&AuthContext>,
    requested_project_id: Option<String>,
) -> Result<String, Response<Full<Bytes>>> {
    if let Some(project_id) = requested_project_id {
        if project_id.trim().is_empty() {
            return Err(bad_request("project_id must not be empty"));
        }
        return Ok(project_id);
    }

    let auth = auth.ok_or_else(unauthorized)?;
    let projects = api.accessible_projects(auth);
    if projects.len() == 1 {
        Ok(projects[0].0.clone())
    } else {
        Err(bad_request(
            "project_id is required unless the caller is scoped to exactly one project",
        ))
    }
}

/// Mask an API key for display (show first 8 chars + "...").
fn mask_key(key: &str) -> String {
    if key.len() > 8 {
        format!("{}...", &key[..8])
    } else {
        key.to_string()
    }
}

// --- Handlers ---

fn handle_status(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let cost_keys = api.cost_usage().map(|u| u.len()).unwrap_or(0);
    let rate_keys = api.rate_limiter_tracked_keys().unwrap_or(0);
    let failed_count = api.failed_providers().map(|f| f.len()).unwrap_or(0);

    let body = format!(
        r#"{{"cost_tracker_enabled":{},"rate_limiter_enabled":{},"provider_failover_enabled":{},"tracked_api_keys":{},"rate_limiter_tracked_keys":{},"failed_providers_count":{}}}"#,
        api.cost_tracker_enabled(),
        api.rate_limiter_enabled(),
        api.provider_failover_enabled(),
        cost_keys,
        rate_keys,
        failed_count,
    );
    json_response(StatusCode::OK, body)
}

fn handle_cost_usage(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let usage = match api.cost_usage() {
        Some(u) => u,
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"cost_tracker not enabled"}"#.to_string(),
            );
        }
    };

    let budget = api.budget_limit().unwrap_or(0.0);

    let entries: Vec<String> = usage
        .iter()
        .map(|(key, u)| {
            format!(
                r#"{{"api_key":"{}","total_input_tokens":{},"total_output_tokens":{},"total_cost":{:.6}}}"#,
                mask_key(key),
                u.total_input_tokens,
                u.total_output_tokens,
                u.total_cost,
            )
        })
        .collect();

    let body = format!(
        r#"{{"budget_limit":{:.6},"usage":[{}]}}"#,
        budget,
        entries.join(","),
    );
    json_response(StatusCode::OK, body)
}

fn handle_cost_models(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let models = match api.model_costs() {
        Some(m) => m,
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"cost_tracker not enabled"}"#.to_string(),
            );
        }
    };

    let entries: Vec<String> = models
        .iter()
        .map(|(name, cost)| {
            format!(
                r#"{{"model":"{}","input_cost_per_1k":{:.6},"output_cost_per_1k":{:.6}}}"#,
                name, cost.input, cost.output,
            )
        })
        .collect();

    let body = format!(r#"{{"models":[{}]}}"#, entries.join(","));
    json_response(StatusCode::OK, body)
}

async fn handle_delete_all_usage(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    match api.reset_all_cost_usage().await {
        Some(()) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"cost_tracker not enabled"}"#.to_string(),
        ),
    }
}

async fn handle_delete_usage(api: &LlmGatewayApi, key: &str) -> Response<Full<Bytes>> {
    match api.reset_cost_usage(key).await {
        Some(true) => json_response(StatusCode::OK, r#"{"ok":true,"deleted":true}"#.to_string()),
        Some(false) => json_response(
            StatusCode::NOT_FOUND,
            r#"{"ok":false,"deleted":false}"#.to_string(),
        ),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"cost_tracker not enabled"}"#.to_string(),
        ),
    }
}

async fn handle_put_model_cost(
    api: &LlmGatewayApi,
    model: &str,
    req: Request<Incoming>,
) -> Response<Full<Bytes>> {
    use http_body_util::BodyExt;

    // Read body to get input/output costs.
    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"failed to read body"}"#.to_string(),
            );
        }
    };

    let body_str = match std::str::from_utf8(&body_bytes) {
        Ok(s) => s,
        Err(_) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"body is not valid UTF-8"}"#.to_string(),
            );
        }
    };

    // Parse input_cost_per_1k and output_cost_per_1k from JSON body.
    // Simple extraction without serde_json.
    let input = extract_json_float(body_str, "input_cost_per_1k");
    let output = extract_json_float(body_str, "output_cost_per_1k");

    match (input, output) {
        (Some(i), Some(o)) => match api.set_model_cost(model, i, o).await {
            Some(()) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
            None => json_response(
                StatusCode::OK,
                r#"{"error":"cost_tracker not enabled"}"#.to_string(),
            ),
        },
        _ => json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"body must contain input_cost_per_1k and output_cost_per_1k as numbers"}"#
                .to_string(),
        ),
    }
}

async fn handle_delete_model_cost(api: &LlmGatewayApi, model: &str) -> Response<Full<Bytes>> {
    match api.delete_model_cost(model).await {
        Some(true) => json_response(StatusCode::OK, r#"{"ok":true,"deleted":true}"#.to_string()),
        Some(false) => json_response(
            StatusCode::NOT_FOUND,
            r#"{"ok":false,"deleted":false}"#.to_string(),
        ),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"cost_tracker not enabled"}"#.to_string(),
        ),
    }
}

fn handle_rate_limiter_status(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    match api.rate_limiter_config() {
        Some((rate, burst)) => {
            let tracked = api.rate_limiter_tracked_keys().unwrap_or(0);
            let body = format!(
                r#"{{"rate_per_second":{:.2},"burst":{:.0},"tracked_keys":{}}}"#,
                rate, burst, tracked,
            );
            json_response(StatusCode::OK, body)
        }
        None => json_response(
            StatusCode::OK,
            r#"{"error":"rate_limiter not enabled"}"#.to_string(),
        ),
    }
}

fn provider_source_name(
    static_provider: Option<&ProviderKeyConfig>,
    managed_provider: Option<&ManagedProviderRecord>,
) -> &'static str {
    match (static_provider.is_some(), managed_provider.is_some()) {
        (true, true) => "static+overlay",
        (true, false) => "static",
        (false, true) => "managed",
        (false, false) => "unknown",
    }
}

fn provider_json(
    api: &LlmGatewayApi,
    name: &str,
    pattern: Option<String>,
    configured_provider: Option<&ProviderKeyConfig>,
    static_provider: Option<&ProviderKeyConfig>,
    managed_provider: Option<&ManagedProviderRecord>,
    capability: Option<&crate::tool_runtime::ToolRuntimeProviderSnapshot>,
    prompt_cache: Option<&crate::prompt_cache::PromptCacheProviderSnapshot>,
) -> serde_json::Value {
    let (preview_provider, configuration_error) = match api.provider_preview(name) {
        Some(Ok(provider)) => (provider, None),
        Some(Err(error)) => (None, Some(error)),
        None => (None, None),
    };
    let visible_provider = configured_provider
        .cloned()
        .or(preview_provider)
        .or_else(|| static_provider.cloned());
    let provider_family = visible_provider
        .as_ref()
        .map(|provider| provider.family_kind().as_str().to_string())
        .or_else(|| managed_provider.and_then(|provider| provider.family.clone()));
    let provider_surfaces = visible_provider
        .as_ref()
        .and_then(|provider| serde_json::to_value(provider.surfaces()).ok())
        .or_else(|| {
            managed_provider.and_then(|provider| {
                provider
                    .surfaces_json
                    .as_deref()
                    .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            })
        });

    let runtime_semantics = capability
        .map(|provider| provider.semantics.clone())
        .or_else(|| {
            visible_provider
                .as_ref()
                .map(|provider| provider.runtime_semantics())
        })
        .unwrap_or_default();
    let prompt_cache_semantics = prompt_cache
        .map(|provider| provider.prompt_cache.clone())
        .or_else(|| {
            visible_provider
                .as_ref()
                .map(|provider| provider.prompt_cache_semantics())
        })
        .unwrap_or_default();
    let data_collection = visible_provider
        .as_ref()
        .and_then(|provider| provider.routing_metadata.data_collection.as_ref())
        .map(|value| value.as_str().to_string());
    let zdr = visible_provider
        .as_ref()
        .map(|provider| provider.routing_metadata.zdr)
        .unwrap_or(false);
    let distillable_text = visible_provider
        .as_ref()
        .map(|provider| provider.routing_metadata.distillable_text)
        .unwrap_or(false);
    let quantizations = visible_provider
        .as_ref()
        .map(|provider| provider.routing_metadata.quantizations.clone())
        .unwrap_or_default();
    let supported_parameter_families = visible_provider
        .as_ref()
        .map(|provider| {
            provider
                .routing_metadata
                .supported_parameter_families
                .clone()
        })
        .unwrap_or_default();
    let enabled = managed_provider
        .map(|provider| provider.enabled)
        .unwrap_or_else(|| visible_provider.is_some() || static_provider.is_some());
    let timeout_secs = capability
        .and_then(|provider| provider.timeout_secs)
        .or_else(|| {
            visible_provider
                .as_ref()
                .and_then(|provider| provider.timeout_secs)
        });
    let routing_metadata = serde_json::json!({
        "data_collection": data_collection.clone(),
        "zdr": zdr,
        "distillable_text": distillable_text,
        "quantizations": quantizations.clone(),
        "supported_parameter_families": supported_parameter_families.clone(),
    });
    let mut object = serde_json::Map::new();
    object.insert("name".to_string(), serde_json::json!(name));
    object.insert("enabled".to_string(), serde_json::json!(enabled));
    object.insert(
        "source".to_string(),
        serde_json::json!(provider_source_name(static_provider, managed_provider)),
    );
    object.insert(
        "api_key_env".to_string(),
        serde_json::json!(managed_provider.and_then(|provider| provider.api_key_env.clone())),
    );
    object.insert(
        "created_at".to_string(),
        serde_json::json!(managed_provider.map(|provider| provider.created_at.clone())),
    );
    object.insert(
        "updated_at".to_string(),
        serde_json::json!(managed_provider.map(|provider| provider.updated_at.clone())),
    );
    object.insert(
        "configuration_error".to_string(),
        serde_json::json!(configuration_error),
    );
    object.insert("pattern".to_string(), serde_json::json!(pattern));
    object.insert(
        "base_url".to_string(),
        serde_json::json!(visible_provider
            .as_ref()
            .map(|provider| provider.base_url.clone())),
    );
    object.insert("family".to_string(), serde_json::json!(provider_family));
    object.insert(
        "surfaces".to_string(),
        provider_surfaces.unwrap_or(serde_json::Value::Null),
    );
    if let Ok(serde_json::Value::Object(semantics)) = serde_json::to_value(&runtime_semantics) {
        object.extend(semantics);
    }
    object.insert("routing_metadata".to_string(), routing_metadata);
    object.insert(
        "data_collection".to_string(),
        serde_json::json!(data_collection),
    );
    object.insert("zdr".to_string(), serde_json::json!(zdr));
    object.insert(
        "distillable_text".to_string(),
        serde_json::json!(distillable_text),
    );
    object.insert(
        "quantizations".to_string(),
        serde_json::json!(quantizations),
    );
    object.insert(
        "supported_parameter_families".to_string(),
        serde_json::json!(supported_parameter_families),
    );
    object.insert(
        "capabilities".to_string(),
        serde_json::json!(runtime_semantics.capabilities),
    );
    object.insert(
        "prompt_cache_protocol".to_string(),
        serde_json::json!(prompt_cache_semantics.prompt_cache_protocol),
    );
    object.insert(
        "supports_prompt_cache".to_string(),
        serde_json::json!(prompt_cache_semantics.supports_prompt_cache),
    );
    object.insert(
        "prompt_cache_request_controls_supported".to_string(),
        serde_json::json!(prompt_cache_semantics.request_controls_supported),
    );
    object.insert(
        "models".to_string(),
        serde_json::json!(visible_provider
            .as_ref()
            .map(|provider| provider.models.clone())
            .unwrap_or_default()),
    );
    object.insert("timeout_secs".to_string(), serde_json::json!(timeout_secs));
    serde_json::Value::Object(object)
}

fn handle_providers(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let providers = api.providers();
    let configured_providers = api.configured_providers();
    let static_providers = api.static_providers();
    let managed_providers = api.managed_providers();
    let tool_runtime = api.tool_runtime_status();
    let prompt_cache = api.prompt_cache_status();

    if providers.is_none()
        && configured_providers.is_none()
        && static_providers.is_none()
        && managed_providers.is_none()
        && tool_runtime.is_none()
        && prompt_cache.is_none()
    {
        return json_response(
            StatusCode::OK,
            r#"{"error":"provider data not available"}"#.to_string(),
        );
    }

    let cooldown = api.provider_cooldown().map(|d| d.as_secs()).unwrap_or(0);
    let patterns = providers
        .unwrap_or_default()
        .into_iter()
        .map(|provider| (provider.name, provider.pattern))
        .collect::<std::collections::HashMap<_, _>>();
    let configured = configured_providers
        .unwrap_or_default()
        .into_iter()
        .map(|provider| (provider.name.clone(), provider))
        .collect::<std::collections::HashMap<_, _>>();
    let static_provider_map = static_providers
        .unwrap_or_default()
        .into_iter()
        .map(|provider| (provider.name.clone(), provider))
        .collect::<std::collections::HashMap<_, _>>();
    let managed_provider_map = managed_providers
        .unwrap_or_default()
        .into_iter()
        .map(|provider| (provider.name.clone(), provider))
        .collect::<std::collections::HashMap<_, _>>();
    let capabilities = tool_runtime
        .map(|status| {
            status
                .providers
                .into_iter()
                .map(|provider| (provider.name.clone(), provider))
                .collect::<std::collections::HashMap<_, _>>()
        })
        .unwrap_or_default();
    let prompt_cache_capabilities = prompt_cache
        .map(|status| {
            status
                .providers
                .into_iter()
                .map(|provider| (provider.name.clone(), provider))
                .collect::<std::collections::HashMap<_, _>>()
        })
        .unwrap_or_default();

    let provider_names = patterns
        .keys()
        .cloned()
        .chain(static_provider_map.keys().cloned())
        .chain(managed_provider_map.keys().cloned())
        .chain(configured.keys().cloned())
        .chain(capabilities.keys().cloned())
        .chain(prompt_cache_capabilities.keys().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let providers = provider_names
        .iter()
        .map(|name| {
            provider_json(
                api,
                name,
                patterns.get(name).cloned(),
                configured.get(name),
                static_provider_map.get(name),
                managed_provider_map.get(name),
                capabilities.get(name),
                prompt_cache_capabilities.get(name),
            )
        })
        .collect::<Vec<_>>();

    json_response(
        StatusCode::OK,
        serde_json::json!({
            "cooldown_secs": cooldown,
            "providers": providers,
        })
        .to_string(),
    )
}

#[derive(Clone, Copy)]
enum ProviderWriteMode {
    Create,
    Replace,
    Patch,
}

fn empty_managed_provider_record(name: &str, created_at: String) -> ManagedProviderRecord {
    ManagedProviderRecord {
        name: name.to_string(),
        enabled: true,
        api_key_env: None,
        base_url: None,
        models_json: None,
        api_key_header: None,
        timeout_secs: None,
        family: None,
        surfaces_json: None,
        routing_metadata_json: None,
        created_at: created_at.clone(),
        updated_at: created_at,
    }
}

fn serialize_provider_routing_metadata_payload(
    payload: ProviderRoutingMetadataPayload,
) -> Result<String, String> {
    let data_collection = payload
        .data_collection
        .map(|value| {
            ProviderDataCollectionMode::parse(value.as_str()).ok_or_else(|| {
                format!(
                    "routing_metadata.data_collection must be one of allow, deny (got '{value}')"
                )
            })
        })
        .transpose()?;
    serde_json::to_string(&serde_json::json!({
        "data_collection": data_collection.map(|value| value.as_str().to_string()),
        "zdr": payload.zdr.unwrap_or(false),
        "distillable_text": payload.distillable_text.unwrap_or(false),
        "quantizations": payload.quantizations.unwrap_or_default(),
        "supported_parameter_families": payload.supported_parameter_families.unwrap_or_default(),
    }))
    .map_err(|error| format!("failed to serialize routing_metadata: {error}"))
}

fn serialize_provider_surfaces(surfaces: &ProviderSurfaceCatalog) -> Result<String, String> {
    serde_json::to_string(surfaces)
        .map_err(|error| format!("failed to serialize provider surfaces: {error}"))
}

fn parse_provider_surfaces_json(
    raw: Option<&str>,
) -> Result<Option<ProviderSurfaceCatalog>, String> {
    raw.map(|value| {
        serde_json::from_str::<ProviderSurfaceCatalog>(value)
            .map_err(|error| format!("surfaces must be a valid provider surface catalog: {error}"))
    })
    .transpose()
}

fn parse_provider_family_name(raw: Option<&str>) -> Result<Option<ProviderFamily>, String> {
    raw.map(|value| {
        ProviderFamily::parse(value).ok_or_else(|| format!("invalid provider family '{value}'"))
    })
    .transpose()
}

fn build_managed_provider_record(
    name: &str,
    payload: ProviderPayload,
    existing_managed: Option<&ManagedProviderRecord>,
    base_provider: Option<&ProviderKeyConfig>,
    write_mode: ProviderWriteMode,
) -> Result<ManagedProviderRecord, String> {
    if let Some(payload_name) = payload.name.as_deref() {
        if payload_name != name {
            return Err(format!(
                "provider payload name '{}' does not match route '{}'",
                payload_name, name
            ));
        }
    }

    let now = current_timestamp_string();
    let mut record = match write_mode {
        ProviderWriteMode::Create => empty_managed_provider_record(name, now.clone()),
        ProviderWriteMode::Replace => {
            let created_at = existing_managed
                .map(|record| record.created_at.clone())
                .unwrap_or_else(|| now.clone());
            empty_managed_provider_record(name, created_at)
        }
        ProviderWriteMode::Patch => existing_managed
            .cloned()
            .unwrap_or_else(|| empty_managed_provider_record(name, now.clone())),
    };

    if let Some(enabled) = payload.enabled {
        record.enabled = enabled;
    }
    if let Some(api_key_env) = payload.api_key_env {
        record.api_key_env = Some(api_key_env);
    }
    if let Some(base_url) = payload.base_url {
        record.base_url = Some(base_url);
    }
    if let Some(models) = payload.models {
        record.models_json = Some(
            serde_json::to_string(&models)
                .map_err(|error| format!("failed to serialize models: {error}"))?,
        );
    }
    if let Some(api_key_header) = payload.api_key_header {
        record.api_key_header = Some(api_key_header);
    }
    if let Some(timeout_secs) = payload.timeout_secs {
        record.timeout_secs = Some(timeout_secs);
    }
    if let Some(routing_metadata) = payload.routing_metadata {
        record.routing_metadata_json = Some(serialize_provider_routing_metadata_payload(
            routing_metadata,
        )?);
    }

    let stored_surfaces = parse_provider_surfaces_json(record.surfaces_json.as_deref())?;
    let surfaces = payload
        .surfaces
        .or(stored_surfaces)
        .or_else(|| base_provider.map(|provider| provider.surfaces().clone()))
        .unwrap_or_default();
    let family = payload
        .family
        .or(parse_provider_family_name(record.family.as_deref())?)
        .or_else(|| base_provider.map(|provider| provider.family_kind()));
    let family_config = ProviderFamilyConfig::from_optional_parts(name, family, surfaces)?;
    let surfaces = family_config.surfaces();
    record.family = Some(family_config.family().as_str().to_string());
    record.surfaces_json = Some(serialize_provider_surfaces(surfaces)?);
    record.updated_at = now;
    Ok(record)
}

fn provider_lookup_maps(
    api: &LlmGatewayApi,
) -> (
    HashMap<String, ProviderKeyConfig>,
    HashMap<String, ProviderKeyConfig>,
    HashMap<String, ManagedProviderRecord>,
) {
    let static_providers = api
        .static_providers()
        .unwrap_or_default()
        .into_iter()
        .map(|provider| (provider.name.clone(), provider))
        .collect::<HashMap<_, _>>();
    let configured_providers = api
        .configured_providers()
        .unwrap_or_default()
        .into_iter()
        .map(|provider| (provider.name.clone(), provider))
        .collect::<HashMap<_, _>>();
    let managed_providers = api
        .managed_providers()
        .unwrap_or_default()
        .into_iter()
        .map(|provider| (provider.name.clone(), provider))
        .collect::<HashMap<_, _>>();
    (static_providers, configured_providers, managed_providers)
}

fn handle_get_provider(api: &LlmGatewayApi, name: &str) -> Response<Full<Bytes>> {
    let patterns = api
        .providers()
        .unwrap_or_default()
        .into_iter()
        .map(|provider| (provider.name, provider.pattern))
        .collect::<HashMap<_, _>>();
    let (static_providers, configured_providers, managed_providers) = provider_lookup_maps(api);
    if !patterns.contains_key(name)
        && !static_providers.contains_key(name)
        && !configured_providers.contains_key(name)
        && !managed_providers.contains_key(name)
    {
        return json_response(
            StatusCode::NOT_FOUND,
            serde_json::json!({ "error": "provider not found" }).to_string(),
        );
    }

    let provider = provider_json(
        api,
        name,
        patterns.get(name).cloned(),
        configured_providers.get(name),
        static_providers.get(name),
        managed_providers.get(name),
        None,
        None,
    );
    json_response(StatusCode::OK, provider.to_string())
}

async fn handle_create_provider(
    api: &LlmGatewayApi,
    req: Request<Incoming>,
) -> Response<Full<Bytes>> {
    let body = match read_body_string(req).await {
        Ok(body) => body,
        Err(resp) => return resp,
    };
    let payload: ProviderPayload = match serde_json::from_str(&body) {
        Ok(payload) => payload,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": format!("invalid JSON body: {error}") }).to_string(),
            );
        }
    };
    let Some(name) = payload.name.clone() else {
        return json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({ "error": "provider name is required" }).to_string(),
        );
    };
    let (static_providers, configured_providers, managed_providers) = provider_lookup_maps(api);
    if static_providers.contains_key(&name)
        || configured_providers.contains_key(&name)
        || managed_providers.contains_key(&name)
    {
        return json_response(
            StatusCode::CONFLICT,
            serde_json::json!({ "error": "provider already exists; use PUT or PATCH to manage it" })
                .to_string(),
        );
    }
    let record = match build_managed_provider_record(
        &name,
        payload,
        None,
        None,
        ProviderWriteMode::Create,
    ) {
        Ok(record) => record,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": error }).to_string(),
            );
        }
    };
    match api.upsert_managed_provider(record).await {
        Some(Ok(())) => handle_get_provider(api, &name),
        Some(Err(error)) => json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({ "error": error.to_string() }).to_string(),
        ),
        None => json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({ "error": "provider runtime not enabled" }).to_string(),
        ),
    }
}

async fn handle_put_provider(
    api: &LlmGatewayApi,
    name: &str,
    req: Request<Incoming>,
) -> Response<Full<Bytes>> {
    let body = match read_body_string(req).await {
        Ok(body) => body,
        Err(resp) => return resp,
    };
    let payload: ProviderPayload = match serde_json::from_str(&body) {
        Ok(payload) => payload,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": format!("invalid JSON body: {error}") }).to_string(),
            );
        }
    };
    let (static_providers, configured_providers, managed_providers) = provider_lookup_maps(api);
    let record = match build_managed_provider_record(
        name,
        payload,
        managed_providers.get(name),
        static_providers
            .get(name)
            .or_else(|| configured_providers.get(name)),
        ProviderWriteMode::Replace,
    ) {
        Ok(record) => record,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": error }).to_string(),
            );
        }
    };
    match api.upsert_managed_provider(record).await {
        Some(Ok(())) => handle_get_provider(api, name),
        Some(Err(error)) => json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({ "error": error.to_string() }).to_string(),
        ),
        None => json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({ "error": "provider runtime not enabled" }).to_string(),
        ),
    }
}

async fn handle_patch_provider(
    api: &LlmGatewayApi,
    name: &str,
    req: Request<Incoming>,
) -> Response<Full<Bytes>> {
    let body = match read_body_string(req).await {
        Ok(body) => body,
        Err(resp) => return resp,
    };
    let payload: ProviderPayload = match serde_json::from_str(&body) {
        Ok(payload) => payload,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": format!("invalid JSON body: {error}") }).to_string(),
            );
        }
    };
    let (static_providers, configured_providers, managed_providers) = provider_lookup_maps(api);
    if !static_providers.contains_key(name)
        && !configured_providers.contains_key(name)
        && !managed_providers.contains_key(name)
    {
        return json_response(
            StatusCode::NOT_FOUND,
            serde_json::json!({ "error": "provider not found" }).to_string(),
        );
    }
    let record = match build_managed_provider_record(
        name,
        payload,
        managed_providers.get(name),
        static_providers
            .get(name)
            .or_else(|| configured_providers.get(name)),
        ProviderWriteMode::Patch,
    ) {
        Ok(record) => record,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": error }).to_string(),
            );
        }
    };
    match api.upsert_managed_provider(record).await {
        Some(Ok(())) => handle_get_provider(api, name),
        Some(Err(error)) => json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({ "error": error.to_string() }).to_string(),
        ),
        None => json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({ "error": "provider runtime not enabled" }).to_string(),
        ),
    }
}

async fn handle_delete_provider(api: &LlmGatewayApi, name: &str) -> Response<Full<Bytes>> {
    match api.delete_managed_provider(name).await {
        Some(Ok(true)) => json_response(
            StatusCode::OK,
            serde_json::json!({ "ok": true, "deleted": true }).to_string(),
        ),
        Some(Ok(false)) => json_response(
            StatusCode::NOT_FOUND,
            serde_json::json!({ "error": "managed provider override not found" }).to_string(),
        ),
        Some(Err(error)) => json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({ "error": error.to_string() }).to_string(),
        ),
        None => json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({ "error": "provider runtime not enabled" }).to_string(),
        ),
    }
}

fn handle_failed_providers(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let failed = match api.failed_providers() {
        Some(f) => f,
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"provider_failover not enabled"}"#.to_string(),
            );
        }
    };

    let body = FailedProviderResponse {
        failed: failed
            .into_iter()
            .map(|status| FailedProviderResponseEntry {
                name: status.name,
                failed_ago_secs: status.failed_ago_secs,
                cooldown_remaining_secs: status.cooldown_remaining_secs,
                reason: status.reason,
            })
            .collect(),
    };
    json_response(StatusCode::OK, serde_json::to_string(&body).unwrap())
}

fn handle_provider_health(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let health = match api.provider_health() {
        Some(health) => health,
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"provider health not available"}"#.to_string(),
            );
        }
    };

    let body = ProviderHealthResponse {
        providers: health
            .into_iter()
            .map(|provider| ProviderHealthResponseEntry {
                name: provider.name,
                eligible: provider.eligible,
                cooldown_remaining_secs: provider.cooldown_remaining_secs,
                cooldown_reason: provider.cooldown_reason,
                active_requests: provider.active_requests,
                samples: provider.samples,
                ewma_latency_ms: provider.ewma_latency_ms,
                ewma_error_rate: provider.ewma_error_rate,
                ewma_timeout_rate: provider.ewma_timeout_rate,
                ewma_rate_limit_rate: provider.ewma_rate_limit_rate,
                adaptive_penalties: ProviderPenaltyResponse {
                    active_requests: provider.adaptive_penalty_active_requests,
                    latency: provider.adaptive_penalty_latency,
                    error: provider.adaptive_penalty_error,
                    timeout: provider.adaptive_penalty_timeout,
                    rate_limit: provider.adaptive_penalty_rate_limit,
                },
                adaptive_penalty_total: provider.adaptive_penalty_total,
            })
            .collect(),
    };
    json_response(StatusCode::OK, serde_json::to_string(&body).unwrap())
}

fn handle_tool_runtime_status(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let Some(status) = api.tool_runtime_status() else {
        return json_response(
            StatusCode::OK,
            r#"{"error":"tool_runtime not enabled"}"#.to_string(),
        );
    };

    json_response(
        StatusCode::OK,
        serde_json::json!({
            "default_timeout_ms": status.default_timeout_ms,
            "max_round_trips": status.max_round_trips,
            "responses_stream_mode": status.responses_stream_mode,
            "supported_executors": status.supported_executors,
            "web_search_backends": status.web_search_backends.into_iter().map(|backend| {
                serde_json::json!({
                    "name": backend.name,
                    "url": backend.url,
                    "method": backend.method,
                })
            }).collect::<Vec<_>>(),
            "mcp_servers": status.mcp_servers.into_iter().map(tool_runtime_mcp_server_json).collect::<Vec<_>>(),
            "arxiv_base_url": status.arxiv_base_url,
            "arxiv_default_max_results": status.arxiv_default_max_results,
            "providers": status.providers.into_iter().map(|provider| {
                serde_json::to_value(provider).unwrap_or(serde_json::Value::Null)
            }).collect::<Vec<_>>(),
            "registered_tools_total": status.registered_tools_total,
            "enabled_tools_total": status.enabled_tools_total,
            "executors": status.executors.into_iter().map(|executor| {
                serde_json::json!({
                    "executor_kind": executor.executor_kind,
                    "total": executor.total,
                    "enabled": executor.enabled,
                })
            }).collect::<Vec<_>>(),
        })
        .to_string(),
    )
}

fn tool_runtime_mcp_server_json(server: ToolRuntimeMcpServerSnapshot) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    object.insert("name".to_string(), serde_json::json!(server.name));
    object.insert("transport".to_string(), serde_json::json!(server.transport));
    object.insert("auth_mode".to_string(), serde_json::json!(server.auth_mode));
    object.insert(
        "auth_status".to_string(),
        serde_json::json!(server.auth_status),
    );
    object.insert(
        "auth_last_error".to_string(),
        serde_json::json!(server.auth_last_error),
    );
    object.insert(
        "auth_last_discovery_error".to_string(),
        serde_json::json!(server.auth_last_discovery_error),
    );
    object.insert(
        "auth_refreshes".to_string(),
        serde_json::json!(server.auth_refreshes),
    );
    object.insert(
        "auth_last_refreshed_at".to_string(),
        serde_json::json!(server.auth_last_refreshed_at),
    );
    object.insert(
        "auth_token_expires_at_unix_ms".to_string(),
        serde_json::json!(server.auth_token_expires_at_unix_ms),
    );
    object.insert(
        "auth_resource".to_string(),
        serde_json::json!(server.auth_resource),
    );
    object.insert(
        "auth_authorization_url".to_string(),
        serde_json::json!(server.auth_authorization_url),
    );
    object.insert(
        "auth_token_url".to_string(),
        serde_json::json!(server.auth_token_url),
    );
    object.insert(
        "auth_authorization_server_url".to_string(),
        serde_json::json!(server.auth_authorization_server_url),
    );
    object.insert(
        "auth_pending_authorization".to_string(),
        serde_json::json!(server.auth_pending_authorization),
    );
    object.insert(
        "auth_pending_authorization_expires_at_unix_ms".to_string(),
        serde_json::json!(server.auth_pending_authorization_expires_at_unix_ms),
    );
    object.insert("url".to_string(), serde_json::json!(server.url));
    object.insert("method".to_string(), serde_json::json!(server.method));
    object.insert("command".to_string(), serde_json::json!(server.command));
    object.insert("args".to_string(), serde_json::json!(server.args));
    object.insert("cwd".to_string(), serde_json::json!(server.cwd));
    object.insert(
        "timeout_ms".to_string(),
        serde_json::json!(server.timeout_ms),
    );
    object.insert(
        "max_retries".to_string(),
        serde_json::json!(server.max_retries),
    );
    object.insert(
        "max_calls_per_request".to_string(),
        serde_json::json!(server.max_calls_per_request),
    );
    object.insert(
        "max_total_time_ms".to_string(),
        serde_json::json!(server.max_total_time_ms),
    );
    object.insert(
        "max_output_tokens".to_string(),
        serde_json::json!(server.max_output_tokens),
    );
    object.insert(
        "operator_state".to_string(),
        serde_json::json!(server.operator_state),
    );
    object.insert(
        "operator_state_at".to_string(),
        serde_json::json!(server.operator_state_at),
    );
    object.insert(
        "operator_state_actor".to_string(),
        serde_json::json!(server.operator_state_actor),
    );
    object.insert(
        "operator_state_reason".to_string(),
        serde_json::json!(server.operator_state_reason),
    );
    object.insert(
        "health_state".to_string(),
        serde_json::json!(server.health_state),
    );
    object.insert(
        "health_reason".to_string(),
        serde_json::json!(server.health_reason),
    );
    object.insert(
        "recommended_action".to_string(),
        serde_json::json!(server.recommended_action),
    );
    object.insert("reachable".to_string(), serde_json::json!(server.reachable));
    object.insert(
        "protocol_version".to_string(),
        serde_json::json!(server.protocol_version),
    );
    object.insert(
        "session_id_present".to_string(),
        serde_json::json!(server.session_id_present),
    );
    object.insert(
        "discovered_tools".to_string(),
        serde_json::json!(server.discovered_tools),
    );
    object.insert(
        "last_error".to_string(),
        serde_json::json!(server.last_error),
    );
    object.insert(
        "discovery_refreshes".to_string(),
        serde_json::json!(server.discovery_refreshes),
    );
    object.insert(
        "last_discovery_at".to_string(),
        serde_json::json!(server.last_discovery_at),
    );
    object.insert(
        "last_discovery_status".to_string(),
        serde_json::json!(server.last_discovery_status),
    );
    object.insert(
        "last_discovery_error".to_string(),
        serde_json::json!(server.last_discovery_error),
    );
    object.insert(
        "total_calls".to_string(),
        serde_json::json!(server.total_calls),
    );
    object.insert(
        "successful_calls".to_string(),
        serde_json::json!(server.successful_calls),
    );
    object.insert(
        "failed_calls".to_string(),
        serde_json::json!(server.failed_calls),
    );
    object.insert(
        "retried_calls".to_string(),
        serde_json::json!(server.retried_calls),
    );
    object.insert(
        "session_reinitializations".to_string(),
        serde_json::json!(server.session_reinitializations),
    );
    object.insert(
        "last_session_reinitialized_at".to_string(),
        serde_json::json!(server.last_session_reinitialized_at),
    );
    object.insert(
        "last_recovery_error".to_string(),
        serde_json::json!(server.last_recovery_error),
    );
    object.insert(
        "budget_exceeded_calls".to_string(),
        serde_json::json!(server.budget_exceeded_calls),
    );
    object.insert(
        "last_budget_exceeded_at".to_string(),
        serde_json::json!(server.last_budget_exceeded_at),
    );
    object.insert(
        "last_budget_exceeded_error".to_string(),
        serde_json::json!(server.last_budget_exceeded_error),
    );
    object.insert(
        "last_session_reset_at".to_string(),
        serde_json::json!(server.last_session_reset_at),
    );
    object.insert(
        "last_session_reset_status".to_string(),
        serde_json::json!(server.last_session_reset_status),
    );
    object.insert(
        "last_session_reset_error".to_string(),
        serde_json::json!(server.last_session_reset_error),
    );
    object.insert(
        "last_session_reset_http_status".to_string(),
        serde_json::json!(server.last_session_reset_http_status),
    );
    object.insert(
        "last_call_at".to_string(),
        serde_json::json!(server.last_call_at),
    );
    object.insert(
        "last_call_tool".to_string(),
        serde_json::json!(server.last_call_tool),
    );
    object.insert(
        "last_call_status".to_string(),
        serde_json::json!(server.last_call_status),
    );
    object.insert(
        "last_call_error".to_string(),
        serde_json::json!(server.last_call_error),
    );
    object.insert(
        "last_call_http_status".to_string(),
        serde_json::json!(server.last_call_http_status),
    );
    serde_json::Value::Object(object)
}

async fn handle_tool_runtime_mcp_subroutes(
    api: &LlmGatewayApi,
    server_path: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let Some((server_name, action)) = server_path.split_once('/') else {
        return not_found();
    };
    if server_name.is_empty() || action.is_empty() {
        return not_found();
    }

    match (req.method(), action) {
        (&Method::POST, "refresh") => {
            if let Some(resp) = ensure_permission(api, auth, Permission::ManageProviders, None) {
                resp
            } else {
                handle_refresh_tool_runtime_mcp_server(api, server_name).await
            }
        }
        (&Method::POST, "disable") => {
            if let Some(resp) = ensure_permission(api, auth, Permission::ManageProviders, None) {
                resp
            } else {
                handle_disable_tool_runtime_mcp_server(api, server_name, req).await
            }
        }
        (&Method::POST, "enable") => {
            if let Some(resp) = ensure_permission(api, auth, Permission::ManageProviders, None) {
                resp
            } else {
                handle_enable_tool_runtime_mcp_server(api, server_name, req).await
            }
        }
        (&Method::DELETE, "session") => {
            if let Some(resp) = ensure_permission(api, auth, Permission::ManageProviders, None) {
                resp
            } else {
                handle_reset_tool_runtime_mcp_session(api, server_name).await
            }
        }
        (&Method::POST, "oauth/authorize") => {
            if let Some(resp) = ensure_permission(api, auth, Permission::ManageProviders, None) {
                resp
            } else {
                handle_begin_tool_runtime_mcp_oauth_authorization(api, server_name).await
            }
        }
        (&Method::POST, "oauth/callback") => {
            if let Some(resp) = ensure_permission(api, auth, Permission::ManageProviders, None) {
                resp
            } else {
                handle_complete_tool_runtime_mcp_oauth_authorization(api, server_name, req).await
            }
        }
        _ => method_not_allowed(),
    }
}

async fn handle_refresh_tool_runtime_mcp_server(
    api: &LlmGatewayApi,
    server_name: &str,
) -> Response<Full<Bytes>> {
    let Some(result) = api.refresh_tool_runtime_mcp_server(server_name).await else {
        return json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"tool_runtime not enabled"}"#.to_string(),
        );
    };
    match result {
        Ok(snapshot) => json_response(
            StatusCode::OK,
            tool_runtime_mcp_server_json(snapshot).to_string(),
        ),
        Err(error) => {
            let message = error.to_string();
            let status = if message.starts_with("unknown mcp server ") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_GATEWAY
            };
            json_response(status, serde_json::json!({ "error": message }).to_string())
        }
    }
}

async fn handle_reset_tool_runtime_mcp_session(
    api: &LlmGatewayApi,
    server_name: &str,
) -> Response<Full<Bytes>> {
    let Some(result) = api.reset_tool_runtime_mcp_session(server_name).await else {
        return json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"tool_runtime not enabled"}"#.to_string(),
        );
    };
    match result {
        Ok(snapshot) => json_response(
            StatusCode::OK,
            tool_runtime_mcp_server_json(snapshot).to_string(),
        ),
        Err(error) => {
            let message = error.to_string();
            let status = if message.starts_with("unknown mcp server ") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_GATEWAY
            };
            json_response(status, serde_json::json!({ "error": message }).to_string())
        }
    }
}

async fn handle_disable_tool_runtime_mcp_server(
    api: &LlmGatewayApi,
    server_name: &str,
    req: Request<Incoming>,
) -> Response<Full<Bytes>> {
    let body = match read_body_string(req).await {
        Ok(body) => body,
        Err(response) => return response,
    };
    let actor = extract_json_optional_string(&body, "actor_id");
    let reason = extract_json_optional_string(&body, "reason");
    let Some(result) = api
        .disable_tool_runtime_mcp_server(server_name, actor, reason)
        .await
    else {
        return json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"tool_runtime not enabled"}"#.to_string(),
        );
    };
    match result {
        Ok(snapshot) => json_response(
            StatusCode::OK,
            tool_runtime_mcp_server_json(snapshot).to_string(),
        ),
        Err(error) => {
            let message = error.to_string();
            let status = if message.starts_with("unknown mcp server ") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_GATEWAY
            };
            json_response(status, serde_json::json!({ "error": message }).to_string())
        }
    }
}

async fn handle_enable_tool_runtime_mcp_server(
    api: &LlmGatewayApi,
    server_name: &str,
    req: Request<Incoming>,
) -> Response<Full<Bytes>> {
    let body = match read_body_string(req).await {
        Ok(body) => body,
        Err(response) => return response,
    };
    let actor = extract_json_optional_string(&body, "actor_id");
    let reason = extract_json_optional_string(&body, "reason");
    let Some(result) = api
        .enable_tool_runtime_mcp_server(server_name, actor, reason)
        .await
    else {
        return json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"tool_runtime not enabled"}"#.to_string(),
        );
    };
    match result {
        Ok(snapshot) => json_response(
            StatusCode::OK,
            tool_runtime_mcp_server_json(snapshot).to_string(),
        ),
        Err(error) => {
            let message = error.to_string();
            let status = if message.starts_with("unknown mcp server ") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_GATEWAY
            };
            json_response(status, serde_json::json!({ "error": message }).to_string())
        }
    }
}

async fn handle_begin_tool_runtime_mcp_oauth_authorization(
    api: &LlmGatewayApi,
    server_name: &str,
) -> Response<Full<Bytes>> {
    let Some(result) = api
        .begin_tool_runtime_mcp_oauth_authorization(server_name)
        .await
    else {
        return json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"tool_runtime not enabled"}"#.to_string(),
        );
    };
    match result {
        Ok(flow) => json_response(
            StatusCode::OK,
            serde_json::json!({
                "server_name": flow.server_name,
                "authorization_url": flow.authorization_url,
                "redirect_uri": flow.redirect_uri,
                "state": flow.state,
                "expires_at_unix_ms": flow.expires_at_unix_ms,
            })
            .to_string(),
        ),
        Err(error) => {
            let message = error.to_string();
            let status = if message.starts_with("unknown mcp server ") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_REQUEST
            };
            json_response(status, serde_json::json!({ "error": message }).to_string())
        }
    }
}

async fn handle_complete_tool_runtime_mcp_oauth_authorization(
    api: &LlmGatewayApi,
    server_name: &str,
    req: Request<Incoming>,
) -> Response<Full<Bytes>> {
    let body = match read_body_string(req).await {
        Ok(body) => body,
        Err(response) => return response,
    };
    let payload: McpOauthCallbackPayload = match serde_json::from_str(&body) {
        Ok(payload) => payload,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": format!("invalid JSON body: {error}") }).to_string(),
            );
        }
    };
    let Some(state) = payload
        .state
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    else {
        return json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"state is required"}"#.to_string(),
        );
    };
    let Some(code) = payload
        .code
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    else {
        return json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"code is required"}"#.to_string(),
        );
    };
    let Some(result) = api
        .complete_tool_runtime_mcp_oauth_authorization(server_name, state, code)
        .await
    else {
        return json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"tool_runtime not enabled"}"#.to_string(),
        );
    };
    match result {
        Ok(snapshot) => json_response(
            StatusCode::OK,
            tool_runtime_mcp_server_json(snapshot).to_string(),
        ),
        Err(error) => {
            let message = error.to_string();
            let status = if message.starts_with("unknown mcp server ") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_REQUEST
            };
            json_response(status, serde_json::json!({ "error": message }).to_string())
        }
    }
}

fn handle_prompt_cache_status(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let Some(status) = api.prompt_cache_status() else {
        return json_response(
            StatusCode::OK,
            r#"{"error":"prompt_cache not enabled"}"#.to_string(),
        );
    };

    json_response(
        StatusCode::OK,
        serde_json::json!({
            "anthropic_default_scope": status.anthropic_default_scope,
            "store_backed": status.store_backed,
            "routing_hint_persistence_enabled": status.routing_hint_persistence_enabled,
            "routing_flush_interval_ms": status.routing_flush_interval_ms,
            "routing_prune_interval_secs": status.routing_prune_interval_secs,
            "warmed_route_count": status.warmed_route_count,
            "negative_route_count": status.negative_route_count,
            "pending_route_updates": status.pending_route_updates,
            "last_route_flush_unix_ms": status.last_route_flush_unix_ms,
            "providers": status.providers.into_iter().map(|provider| {
                serde_json::to_value(provider).unwrap_or(serde_json::Value::Null)
            }).collect::<Vec<_>>(),
        })
        .to_string(),
    )
}

fn handle_semantic_cache_status(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let Some(status) = api.semantic_cache_status() else {
        return json_response(
            StatusCode::OK,
            r#"{"error":"semantic_cache not enabled"}"#.to_string(),
        );
    };

    json_response(
        StatusCode::OK,
        serde_json::json!({
            "default_ttl_secs": status.default_ttl_secs,
            "default_similarity_threshold": status.default_similarity_threshold,
            "max_entries": status.max_entries,
            "store_backed": status.store_backed,
            "entry_count": status.entry_count,
            "hits": status.hits,
            "misses": status.misses,
            "stores": status.stores,
            "skips": status.skips,
            "saved_prompt_tokens": status.saved_prompt_tokens,
        })
        .to_string(),
    )
}

fn handle_clear_all_failed(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    match api.clear_all_failed_providers() {
        Some(()) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"provider_failover not enabled"}"#.to_string(),
        ),
    }
}

fn handle_clear_failed_provider(api: &LlmGatewayApi, name: &str) -> Response<Full<Bytes>> {
    match api.clear_failed_provider(name) {
        Some(true) => json_response(StatusCode::OK, r#"{"ok":true,"cleared":true}"#.to_string()),
        Some(false) => json_response(
            StatusCode::NOT_FOUND,
            r#"{"ok":false,"cleared":false}"#.to_string(),
        ),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"provider_failover not enabled"}"#.to_string(),
        ),
    }
}

// --- Virtual key handlers ---

async fn handle_create_key(
    api: &LlmGatewayApi,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    use http_body_util::BodyExt;

    if !api.virtual_keys_enabled() {
        return json_response(
            StatusCode::OK,
            r#"{"error":"virtual_keys not enabled"}"#.to_string(),
        );
    }

    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"failed to read body"}"#.to_string(),
            );
        }
    };

    let body_str = match std::str::from_utf8(&body_bytes) {
        Ok(s) => s,
        Err(_) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"body is not valid UTF-8"}"#.to_string(),
            );
        }
    };

    let name = match extract_json_string(body_str, "name") {
        Some(n) => n,
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"name is required"}"#.to_string(),
            );
        }
    };

    let provider_name = match extract_json_string(body_str, "provider_name") {
        Some(p) => p,
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"provider_name is required"}"#.to_string(),
            );
        }
    };

    let budget_limit = extract_json_float(body_str, "budget_limit");
    let budget_duration = extract_json_string(body_str, "budget_duration");
    let rpm_limit = extract_json_float(body_str, "rpm_limit").map(|v| v as u32);
    let tpm_limit = extract_json_float(body_str, "tpm_limit").map(|v| v as u32);
    let timeout_secs = extract_json_float(body_str, "timeout_secs").map(|v| v as u64);
    let expires_at = extract_json_string(body_str, "expires_at");
    let allowed_models = extract_json_string_array(body_str, "allowed_models");
    let raw_allowed_tools = if body_str.contains("\"allowed_tools\"") {
        extract_json_raw(body_str, "allowed_tools")
    } else {
        None
    };
    let allowed_tools = extract_json_string_array_allow_empty(body_str, "allowed_tools");
    let tool_approval_mode = match normalize_tool_approval_mode_value(
        extract_json_string(body_str, "tool_approval_mode").as_deref(),
        raw_allowed_tools.as_deref(),
    ) {
        Ok(mode) => mode,
        Err(error) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                format!(r#"{{"error":"{}"}}"#, error),
            );
        }
    };
    let project_id =
        match resolve_runtime_key_project(api, auth, extract_json_string(body_str, "project_id")) {
            Ok(project_id) => project_id,
            Err(resp) => return resp,
        };

    if let Some(resp) = ensure_permission(
        api,
        auth,
        Permission::ManageRuntimeKeys,
        Some(project_id.as_str()),
    ) {
        return resp;
    }

    match api
        .create_virtual_key_with_runtime_policy(
            Some(project_id.as_str()),
            &name,
            &provider_name,
            budget_limit,
            budget_duration,
            rpm_limit,
            tpm_limit,
            allowed_models,
            expires_at,
            timeout_secs,
            tool_approval_mode.clone(),
            allowed_tools.clone(),
        )
        .await
    {
        Some(Ok((plaintext, key_hash))) => json_response(
            StatusCode::CREATED,
            serde_json::json!({
                "key": plaintext,
                "key_hash": key_hash,
                "project_id": project_id,
                "name": name,
                "provider_name": provider_name,
                "timeout_secs": timeout_secs,
                "tool_approval_mode": tool_approval_mode,
                "allowed_tools": allowed_tools,
            })
            .to_string(),
        ),
        Some(Err(e)) => json_response(StatusCode::BAD_REQUEST, format!(r#"{{"error":"{}"}}"#, e)),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"virtual_keys not enabled"}"#.to_string(),
        ),
    }
}

fn handle_list_keys(api: &LlmGatewayApi, auth: Option<&AuthContext>) -> Response<Full<Bytes>> {
    if let Some(resp) = ensure_permission(api, auth, Permission::ViewRuntimeKeys, None) {
        return resp;
    }

    let keys = match if is_instance_admin(auth) {
        api.list_virtual_keys()
    } else {
        api.list_virtual_keys_for_projects(&api.accessible_projects(auth.unwrap()))
    } {
        Some(k) => k,
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"virtual_keys not enabled"}"#.to_string(),
            );
        }
    };

    let entries: Vec<String> = keys
        .iter()
        .map(|k| {
            let hash_preview = if k.key_hash.len() > 12 {
                &k.key_hash[..12]
            } else {
                &k.key_hash
            };
            serde_json::json!({
                "key_hash": k.key_hash,
                "key_hash_preview": format!("{hash_preview}..."),
                "project_id": k.project_id,
                "name": k.name,
                "provider_name": k.provider_name,
                "active": k.active,
                "budget_limit": k.budget_limit,
                "rpm_limit": k.rpm_limit,
                "tpm_limit": k.tpm_limit,
                "timeout_secs": k.timeout_secs,
                "tool_approval_mode": k.tool_approval_mode,
                "allowed_tools": optional_json_value(k.allowed_tools.as_deref()),
            })
            .to_string()
        })
        .collect();

    let body = format!(r#"{{"keys":[{}]}}"#, entries.join(","));
    json_response(StatusCode::OK, body)
}

fn handle_get_key(
    api: &LlmGatewayApi,
    hash_prefix: &str,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    match api.get_virtual_key(hash_prefix) {
        Some(Ok(Some(k))) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewRuntimeKeys, Some(&k.project_id))
            {
                return resp;
            }
            json_response(
                StatusCode::OK,
                serde_json::json!({
                    "key_hash": k.key_hash,
                    "project_id": k.project_id,
                    "name": k.name,
                    "provider_name": k.provider_name,
                    "active": k.active,
                    "budget_limit": k.budget_limit,
                    "budget_duration": k.budget_duration,
                    "rpm_limit": k.rpm_limit,
                    "tpm_limit": k.tpm_limit,
                    "allowed_models": optional_json_value(k.allowed_models.as_deref()),
                    "timeout_secs": k.timeout_secs,
                    "tool_approval_mode": k.tool_approval_mode,
                    "allowed_tools": optional_json_value(k.allowed_tools.as_deref()),
                    "created_at": k.created_at,
                    "expires_at": k.expires_at,
                })
                .to_string(),
            )
        }
        Some(Ok(None)) => json_response(
            StatusCode::NOT_FOUND,
            r#"{"error":"key not found"}"#.to_string(),
        ),
        Some(Err(err)) => virtual_key_lookup_response(&err),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"virtual_keys not enabled"}"#.to_string(),
        ),
    }
}

async fn handle_update_key(
    api: &LlmGatewayApi,
    hash_prefix: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    use http_body_util::BodyExt;

    if !api.virtual_keys_enabled() {
        return json_response(
            StatusCode::OK,
            r#"{"error":"virtual_keys not enabled"}"#.to_string(),
        );
    }

    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"failed to read body"}"#.to_string(),
            );
        }
    };

    let body_str = match std::str::from_utf8(&body_bytes) {
        Ok(s) => s,
        Err(_) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"body is not valid UTF-8"}"#.to_string(),
            );
        }
    };

    // Parse optional fields
    let budget_limit = if body_str.contains("\"budget_limit\"") {
        Some(extract_json_float(body_str, "budget_limit"))
    } else {
        None
    };
    let rpm_limit = if body_str.contains("\"rpm_limit\"") {
        Some(extract_json_float(body_str, "rpm_limit").map(|v| v as u32))
    } else {
        None
    };
    let tpm_limit = if body_str.contains("\"tpm_limit\"") {
        Some(extract_json_float(body_str, "tpm_limit").map(|v| v as u32))
    } else {
        None
    };
    let active = extract_json_bool(body_str, "active");
    let allowed_models = if body_str.contains("\"allowed_models\"") {
        Some(extract_json_string_array(body_str, "allowed_models"))
    } else {
        None
    };
    let expires_at = if body_str.contains("\"expires_at\"") {
        Some(extract_json_string(body_str, "expires_at"))
    } else {
        None
    };
    let timeout_secs = if body_str.contains("\"timeout_secs\"") {
        Some(extract_json_float(body_str, "timeout_secs").map(|v| v as u64))
    } else {
        None
    };
    let raw_allowed_tools = if body_str.contains("\"allowed_tools\"") {
        extract_json_raw(body_str, "allowed_tools")
    } else {
        None
    };
    let allowed_tools = if body_str.contains("\"allowed_tools\"") {
        Some(extract_json_string_array_allow_empty(
            body_str,
            "allowed_tools",
        ))
    } else {
        None
    };
    let tool_approval_mode = if body_str.contains("\"tool_approval_mode\"") {
        let raw_mode = extract_json_optional_string(body_str, "tool_approval_mode");
        match normalize_tool_approval_mode_value(raw_mode.as_deref(), raw_allowed_tools.as_deref())
        {
            Ok(mode) => Some(mode),
            Err(error) => {
                return json_response(
                    StatusCode::BAD_REQUEST,
                    format!(r#"{{"error":"{}"}}"#, error),
                );
            }
        }
    } else {
        None
    };

    let existing = match api.get_virtual_key(hash_prefix) {
        Some(Ok(Some(record))) => record,
        Some(Ok(None)) => {
            return json_response(
                StatusCode::NOT_FOUND,
                r#"{"error":"key not found"}"#.to_string(),
            );
        }
        Some(Err(err)) => return virtual_key_lookup_response(&err),
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"virtual_keys not enabled"}"#.to_string(),
            );
        }
    };
    if let Some(resp) = ensure_permission(
        api,
        auth,
        Permission::ManageRuntimeKeys,
        Some(&existing.project_id),
    ) {
        return resp;
    }

    match api
        .update_virtual_key_with_runtime_policy(
            hash_prefix,
            budget_limit,
            rpm_limit,
            tpm_limit,
            active,
            allowed_models,
            expires_at,
            timeout_secs,
            tool_approval_mode,
            allowed_tools,
        )
        .await
    {
        Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
        Some(Ok(false)) => json_response(
            StatusCode::NOT_FOUND,
            r#"{"error":"key not found"}"#.to_string(),
        ),
        Some(Err(e)) => {
            if let Some(err) = e.downcast_ref::<VirtualKeyLookupError>() {
                virtual_key_lookup_response(err)
            } else {
                json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, e),
                )
            }
        }
        None => json_response(
            StatusCode::OK,
            r#"{"error":"virtual_keys not enabled"}"#.to_string(),
        ),
    }
}

fn handle_cost_usage_by_model(api: &LlmGatewayApi) -> Response<Full<Bytes>> {
    let entries = match api.model_usage_breakdown() {
        Some(e) => e,
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"cost_tracker not enabled"}"#.to_string(),
            );
        }
    };

    let items: Vec<String> = entries
        .iter()
        .map(|((api_key, model), usage)| {
            format!(
                r#"{{"api_key":"{}","model":"{}","total_input_tokens":{},"total_output_tokens":{},"total_cost":{:.6}}}"#,
                mask_key(api_key),
                model,
                usage.total_input_tokens,
                usage.total_output_tokens,
                usage.total_cost,
            )
        })
        .collect();

    json_response(
        StatusCode::OK,
        format!(r#"{{"data":[{}]}}"#, items.join(",")),
    )
}

async fn handle_request_logs(
    api: &LlmGatewayApi,
    req: &Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let query = req.uri().query().unwrap_or("");
    let api_key = extract_query_param(query, "api_key");
    let model = extract_query_param(query, "model");
    let session_id = extract_query_param(query, "session_id");
    let project_id = match extract_query_param(query, "project_id") {
        Some(project_id) => Some(project_id),
        None => match scoped_project_query(api, auth) {
            Ok(project_id) => project_id,
            Err(resp) => return resp,
        },
    };
    if let Some(ref project_id) = project_id {
        if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, Some(project_id)) {
            return resp;
        }
    } else if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, None) {
        return resp;
    }
    let limit = match parse_u32_query_param(query, "limit") {
        Ok(Some(value)) if value > 0 => value,
        Ok(_) => 100,
        Err(error) => return bad_request(&error),
    };
    let metadata_key = normalize_nonempty_string(extract_query_param(query, "metadata_key"));
    let metadata_value = normalize_nonempty_string(extract_query_param(query, "metadata_value"));
    let has_custom_cost = match parse_bool_query_param(query, "has_custom_cost") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    let custom_cost_applied = match parse_bool_query_param(query, "custom_cost_applied") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    if metadata_key.is_none() && metadata_value.is_some() {
        return bad_request("metadata_value requires metadata_key");
    }
    if session_id.is_some() && (api_key.is_some() || model.is_some()) {
        return bad_request("session_id cannot be combined with api_key or model filters");
    }

    let query = RequestLogQuery {
        api_key,
        model,
        project_id,
        session_id,
        metadata_key,
        metadata_value,
        has_custom_cost,
        custom_cost_applied,
        limit,
    };
    let result = api.query_request_logs(&query).await;

    match result {
        Some(Ok(logs)) => request_logs_response(&logs),
        Some(Err(e)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, e),
        ),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"store not configured"}"#.to_string(),
        ),
    }
}

fn request_logs_response(logs: &[RequestLogEntry]) -> Response<Full<Bytes>> {
    let items = logs.iter().map(request_log_entry_json).collect::<Vec<_>>();
    json_response(
        StatusCode::OK,
        serde_json::json!({ "data": items }).to_string(),
    )
}

fn request_log_entry_json(entry: &RequestLogEntry) -> serde_json::Value {
    serde_json::json!({
        "timestamp": entry.timestamp_unix,
        "api_key": mask_key(&entry.api_key),
        "project_id": entry.project_id.clone(),
        "session_id": entry.session_id.clone(),
        "metadata": optional_json_value(entry.metadata_json.as_deref()),
        "custom_cost": optional_json_value(entry.custom_cost_json.as_deref()),
        "custom_cost_applied": entry.custom_cost_applied,
        "provider_name": entry.provider_name.clone(),
        "prompt_name": entry.prompt_name.clone(),
        "prompt_version": entry.prompt_version.clone(),
        "prompt_environment": entry.prompt_environment.clone(),
        "model": entry.model.clone(),
        "input_tokens": entry.input_tokens,
        "output_tokens": entry.output_tokens,
        "cost": entry.cost,
        "is_streaming": entry.is_streaming,
        "safety_mode": entry.safety_mode.clone(),
        "safety_matches": optional_json_value(entry.safety_matches.as_deref()),
        "semantic_policy_version": entry.semantic_policy_version.clone(),
        "semantic_index_state": entry.semantic_index_state.clone(),
        "semantic_degraded_reason": entry.semantic_degraded_reason.clone(),
        "semantic_findings": optional_json_value(entry.semantic_findings.as_deref()),
        "tool_trace": optional_json_value(entry.tool_trace.as_deref()),
    })
}

async fn handle_session_subroutes(
    api: &LlmGatewayApi,
    session_path: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let (session_id, suffix) = match session_path.split_once('/') {
        Some((session_id, suffix)) => (session_id, Some(suffix)),
        None => (session_path, None),
    };
    if session_id.is_empty() {
        return not_found();
    }

    match (req.method(), suffix) {
        (&Method::GET, None) => handle_session_summary(api, session_id, &req, auth).await,
        (&Method::PUT, None) => handle_session_state_upsert(api, session_id, req, auth).await,
        (&Method::POST, Some("transition")) => {
            handle_session_transition(api, session_id, req, auth).await
        }
        (&Method::POST, Some("heartbeat")) => {
            handle_session_heartbeat(api, session_id, req, auth).await
        }
        (&Method::POST, Some("claim")) => handle_session_claim(api, session_id, req, auth).await,
        (&Method::POST, Some("handoff")) => {
            handle_session_handoff(api, session_id, req, auth).await
        }
        (&Method::POST, Some("accept")) => handle_session_accept(api, session_id, req, auth).await,
        (&Method::POST, Some("takeover")) => {
            handle_session_takeover(api, session_id, req, auth).await
        }
        (&Method::GET, Some("events")) => handle_session_events(api, session_id, &req, auth).await,
        (&Method::GET, Some("wait")) => handle_session_wait(api, session_id, &req, auth).await,
        (&Method::POST, Some("reconcile")) => {
            handle_session_reconcile(api, session_id, req, auth).await
        }
        (&Method::POST, Some("release")) => {
            handle_session_release(api, session_id, req, auth).await
        }
        (&Method::POST, Some("cancel")) => handle_session_cancel(api, session_id, req, auth).await,
        (&Method::GET, Some("logs")) => handle_session_logs(api, session_id, &req, auth).await,
        (&Method::GET, Some(_)) => not_found(),
        _ => method_not_allowed(),
    }
}

async fn handle_sessions_list(
    api: &LlmGatewayApi,
    req: &Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let query = req.uri().query().unwrap_or("");
    let project_id = match extract_query_param(query, "project_id") {
        Some(project_id) => Some(project_id),
        None => match scoped_project_query(api, auth) {
            Ok(project_id) => project_id,
            Err(resp) => return resp,
        },
    };
    if let Some(ref project_id) = project_id {
        if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, Some(project_id)) {
            return resp;
        }
    } else if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, None) {
        return resp;
    }

    let status = extract_query_param(query, "status")
        .and_then(|value| normalize_nonempty_str(&value).map(|value| value.to_ascii_lowercase()));
    let owner_id = normalize_nonempty_string(extract_query_param(query, "owner_id"));
    let updated_after_unix = match parse_i64_query_param(query, "updated_after_unix") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    let recovery_required = match parse_bool_query_param(query, "recovery_required") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    let handoff_pending = match parse_bool_query_param(query, "handoff_pending") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    let cancel_pending = match parse_bool_query_param(query, "cancel_pending") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    let owner_stale = match parse_bool_query_param(query, "owner_stale") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    let limit = match parse_u32_query_param(query, "limit") {
        Ok(Some(value)) if value > 0 => value,
        Ok(_) => 100,
        Err(error) => return bad_request(&error),
    };
    let advanced_filter_requested = recovery_required.is_some()
        || handoff_pending.is_some()
        || cancel_pending.is_some()
        || owner_stale.is_some();
    let fetch_limit = if advanced_filter_requested {
        limit.saturating_mul(5).min(1000).max(limit)
    } else {
        limit
    };

    let store_query = SessionListQuery {
        project_id: project_id.clone(),
        status,
        owner_id,
        updated_after_unix,
        limit: fetch_limit,
    };
    let sessions = match api.list_sessions(&store_query).await {
        Some(Ok(records)) => records,
        Some(Err(error)) => {
            return json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(r#"{{"error":"{}"}}"#, error),
            )
        }
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"store not configured"}"#.to_string(),
            )
        }
    };

    let now = current_unix_time();
    let mut filtered = sessions
        .into_iter()
        .filter(|record| session_record_matches_project(record, project_id.as_deref()))
        .filter(|record| {
            recovery_required
                .map(|expected| session_recovery_required(record, now) == expected)
                .unwrap_or(true)
        })
        .filter(|record| {
            handoff_pending
                .map(|expected| session_handoff_is_pending(record) == expected)
                .unwrap_or(true)
        })
        .filter(|record| {
            cancel_pending
                .map(|expected| session_cancel_is_pending(record) == expected)
                .unwrap_or(true)
        })
        .filter(|record| {
            owner_stale
                .map(|expected| session_owner_is_stale(record, now) == expected)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    let truncated = filtered.len() > limit as usize;
    filtered.truncate(limit as usize);

    json_response(
        StatusCode::OK,
        serde_json::json!({
            "data": filtered
                .iter()
                .map(session_summary_json_from_record)
                .collect::<Vec<_>>(),
            "count": filtered.len(),
            "limit": limit,
            "truncated": truncated,
        })
        .to_string(),
    )
}

async fn handle_session_summary(
    api: &LlmGatewayApi,
    session_id: &str,
    req: &Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let query = req.uri().query().unwrap_or("");
    let project_id = match extract_query_param(query, "project_id") {
        Some(project_id) => Some(project_id),
        None => match scoped_project_query(api, auth) {
            Ok(project_id) => project_id,
            Err(resp) => return resp,
        },
    };
    if let Some(ref project_id) = project_id {
        if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, Some(project_id)) {
            return resp;
        }
    } else if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, None) {
        return resp;
    }

    let limit: u32 = extract_query_param(query, "limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let session_record = match api.get_session(session_id).await {
        Some(Ok(record)) => record,
        Some(Err(error)) => {
            return json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(r#"{{"error":"{}"}}"#, error),
            )
        }
        None => None,
    };
    if let Some(record) = session_record.as_ref() {
        if session_record_matches_project(record, project_id.as_deref()) {
            return json_response(
                StatusCode::OK,
                session_summary_json_from_record(record).to_string(),
            );
        }
    }
    let result = api
        .get_request_logs_for_session(session_id, project_id.as_deref(), limit)
        .await;

    match result {
        Some(Ok(logs)) => json_response(
            StatusCode::OK,
            session_summary_json_from_logs(session_id, limit, &logs).to_string(),
        ),
        Some(Err(error)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, error),
        ),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"store not configured"}"#.to_string(),
        ),
    }
}

async fn load_session_record_for_view(
    api: &LlmGatewayApi,
    session_id: &str,
    auth: Option<&AuthContext>,
    requested_project_id: Option<String>,
) -> Result<(Option<SessionRecord>, Option<String>), Response<Full<Bytes>>> {
    let existing = match api.get_session(session_id).await {
        Some(Ok(record)) => record,
        Some(Err(error)) => {
            return Err(json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(r#"{{"error":"{}"}}"#, error),
            ))
        }
        None => None,
    };

    let effective_project_id = requested_project_id.clone().or_else(|| {
        existing
            .as_ref()
            .and_then(|record| record.project_id.clone())
            .or_else(|| scoped_project_query(api, auth).ok().flatten())
    });

    if let Some(ref project_id) = effective_project_id {
        if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, Some(project_id)) {
            return Err(resp);
        }
    } else if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, None) {
        return Err(resp);
    }

    Ok((existing, effective_project_id))
}

async fn handle_session_events(
    api: &LlmGatewayApi,
    session_id: &str,
    req: &Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let query = req.uri().query().unwrap_or("");
    let project_id = extract_query_param(query, "project_id");
    let (existing, effective_project_id) =
        match load_session_record_for_view(api, session_id, auth, project_id).await {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    if let Some(record) = existing.as_ref() {
        if !session_record_matches_project(record, effective_project_id.as_deref()) {
            return not_found();
        }
    } else {
        return not_found();
    }

    let after_seq = match parse_i64_query_param(query, "after_seq") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    let limit = match parse_u32_query_param(query, "limit") {
        Ok(Some(value)) if value > 0 => value,
        Ok(_) => 100,
        Err(error) => return bad_request(&error),
    };

    let events = match api.get_session_events(session_id, after_seq, limit).await {
        Some(Ok(records)) => records,
        Some(Err(error)) => {
            return json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(r#"{{"error":"{}"}}"#, error),
            )
        }
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"store not configured"}"#.to_string(),
            )
        }
    };

    let truncated = events.len() >= limit as usize;
    let latest_event_seq = events
        .last()
        .map(|event| event.event_seq)
        .or(after_seq)
        .unwrap_or(0);

    json_response(
        StatusCode::OK,
        serde_json::json!({
            "session_id": session_id,
            "after_seq": after_seq,
            "latest_event_seq": latest_event_seq,
            "count": events.len(),
            "limit": limit,
            "truncated": truncated,
            "data": events.iter().map(session_event_json_from_record).collect::<Vec<_>>(),
        })
        .to_string(),
    )
}

async fn handle_session_wait(
    api: &LlmGatewayApi,
    session_id: &str,
    req: &Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let query = req.uri().query().unwrap_or("");
    let project_id = extract_query_param(query, "project_id");
    let (existing, effective_project_id) =
        match load_session_record_for_view(api, session_id, auth, project_id).await {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    if let Some(record) = existing.as_ref() {
        if !session_record_matches_project(record, effective_project_id.as_deref()) {
            return not_found();
        }
    } else {
        return not_found();
    }

    let after_seq = match parse_i64_query_param(query, "after_seq") {
        Ok(Some(value)) => value,
        Ok(None) => 0,
        Err(error) => return bad_request(&error),
    };
    let timeout_secs = match parse_u32_query_param(query, "timeout_secs") {
        Ok(Some(value)) if value > 0 => value.min(30),
        Ok(_) => 15,
        Err(error) => return bad_request(&error),
    };
    let limit = match parse_u32_query_param(query, "limit") {
        Ok(Some(value)) if value > 0 => value,
        Ok(_) => 100,
        Err(error) => return bad_request(&error),
    };
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs as u64);

    loop {
        let events = match api
            .get_session_events(session_id, Some(after_seq), limit)
            .await
        {
            Some(Ok(records)) => records,
            Some(Err(error)) => {
                return json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                )
            }
            None => {
                return json_response(
                    StatusCode::OK,
                    r#"{"error":"store not configured"}"#.to_string(),
                )
            }
        };
        if !events.is_empty() {
            let session = match api.get_session(session_id).await {
                Some(Ok(record)) => record,
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
                None => None,
            };
            let session_json = session
                .as_ref()
                .map(session_summary_json_from_record)
                .unwrap_or(serde_json::Value::Null);
            return json_response(
                StatusCode::OK,
                serde_json::json!({
                    "session_id": session_id,
                    "after_seq": after_seq,
                    "wait_timed_out": false,
                    "count": events.len(),
                    "latest_event_seq": events.last().map(|event| event.event_seq).unwrap_or(after_seq),
                    "session": session_json,
                    "events": events.iter().map(session_event_json_from_record).collect::<Vec<_>>(),
                })
                .to_string(),
            );
        }

        if tokio::time::Instant::now() >= deadline {
            let session = match api.get_session(session_id).await {
                Some(Ok(record)) => record,
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
                None => None,
            };
            let session_json = session
                .as_ref()
                .map(session_summary_json_from_record)
                .unwrap_or(serde_json::Value::Null);
            return json_response(
                StatusCode::OK,
                serde_json::json!({
                    "session_id": session_id,
                    "after_seq": after_seq,
                    "wait_timed_out": true,
                    "count": 0,
                    "latest_event_seq": after_seq,
                    "session": session_json,
                    "events": [],
                })
                .to_string(),
            );
        }

        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn load_session_record_for_mutation(
    api: &LlmGatewayApi,
    session_id: &str,
    auth: Option<&AuthContext>,
    requested_project_id: Option<String>,
) -> Result<(Option<SessionRecord>, Option<String>), Response<Full<Bytes>>> {
    let existing = match api.get_session(session_id).await {
        Some(Ok(record)) => record,
        Some(Err(error)) => {
            return Err(json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(r#"{{"error":"{}"}}"#, error),
            ))
        }
        None => None,
    };
    let existing_project_id = existing
        .as_ref()
        .and_then(|record| record.project_id.clone());
    let scoped_project_id: Option<String> =
        if requested_project_id.is_none() && existing_project_id.is_none() {
            match scoped_project_query(api, auth) {
                Ok(project_id) => project_id,
                Err(resp) => return Err(resp),
            }
        } else {
            None
        };
    let effective_project_id = requested_project_id
        .clone()
        .or(existing_project_id)
        .or(scoped_project_id);

    if let Some(resp) = ensure_permission(
        api,
        auth,
        Permission::ManageProjectPolicy,
        effective_project_id.as_deref(),
    ) {
        return Err(resp);
    }
    Ok((existing, effective_project_id))
}

fn empty_session_state_record(session_id: &str) -> SessionRecord {
    SessionRecord {
        session_id: session_id.to_string(),
        project_id: None,
        project_ids_json: None,
        first_request_unix: None,
        last_request_unix: None,
        updated_at_unix: current_unix_time(),
        request_count: 0,
        streaming_request_count: 0,
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_cost: 0.0,
        providers_json: None,
        models_json: None,
        prompt_names_json: None,
        prompt_versions_json: None,
        tool_names_json: None,
        latest_request_json: None,
        safety_event_count: 0,
        semantic_event_count: 0,
        semantic_degraded_count: 0,
        tool_call_count: 0,
        tool_error_count: 0,
        status: None,
        owner_id: None,
        owner_acquired_at_unix: None,
        last_transition_at_unix: None,
        last_transition_reason: None,
        last_heartbeat_unix: None,
        lease_expires_at_unix: None,
        cancel_requested_at_unix: None,
        cancel_requested_by: None,
        cancel_reason: None,
        handoff_target_owner_id: None,
        handoff_requested_at_unix: None,
        handoff_reason: None,
        state_json: None,
        metadata_json: None,
    }
}

fn session_event_payload_value(record: &SessionRecord) -> serde_json::Value {
    serde_json::json!({
        "status": record.status,
        "owner_id": record.owner_id,
        "lease_expires_at_unix": record.lease_expires_at_unix,
        "last_transition_at_unix": record.last_transition_at_unix,
        "last_transition_reason": record.last_transition_reason,
        "cancel_requested_at_unix": record.cancel_requested_at_unix,
        "cancel_requested_by": record.cancel_requested_by,
        "cancel_reason": record.cancel_reason,
        "handoff_target_owner_id": record.handoff_target_owner_id,
        "handoff_requested_at_unix": record.handoff_requested_at_unix,
        "handoff_reason": record.handoff_reason,
        "state": parse_json_value(record.state_json.as_deref()),
        "metadata": parse_json_value(record.metadata_json.as_deref()),
    })
}

async fn persist_session_record(
    api: &LlmGatewayApi,
    record: SessionRecord,
    event_kind: Option<&str>,
    actor_id: Option<String>,
    reason: Option<String>,
    payload: Option<serde_json::Value>,
    created_at_unix: Option<i64>,
) -> Response<Full<Bytes>> {
    let upsert_outcome: Result<Option<()>, String> = match api.upsert_session(record.clone()).await
    {
        Some(Ok(())) => Ok(Some(())),
        Some(Err(error)) => Err(error.to_string()),
        None => Ok(None),
    };
    match upsert_outcome {
        Ok(Some(())) => {
            if let Some(event_kind) = event_kind {
                let event = SessionEventRecord {
                    event_seq: 0,
                    session_id: record.session_id.clone(),
                    project_id: record.project_id.clone(),
                    event_kind: event_kind.to_string(),
                    actor_id,
                    reason,
                    payload_json: payload.map(|value| value.to_string()),
                    created_at_unix: created_at_unix.unwrap_or_else(current_unix_time),
                };
                let append_outcome: Result<Option<i64>, String> =
                    match api.append_session_event(&event).await {
                        Some(Ok(seq)) => Ok(Some(seq)),
                        Some(Err(error)) => Err(error.to_string()),
                        None => Ok(None),
                    };
                if let Err(error) = append_outcome {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    );
                }
            }
            json_response(
                StatusCode::OK,
                session_summary_json_from_record(&record).to_string(),
            )
        }
        Err(error) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, error),
        ),
        Ok(None) => json_response(
            StatusCode::OK,
            r#"{"error":"store not configured"}"#.to_string(),
        ),
    }
}

async fn handle_session_state_upsert(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload: SessionStatePayload = if body.is_empty() {
        SessionStatePayload::default()
    } else {
        match serde_json::from_slice::<SessionStatePayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }

    let (existing, effective_project_id) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    if payload.status.is_none()
        && payload.state.is_none()
        && payload.metadata.is_none()
        && effective_project_id.is_none()
        && existing.is_none()
    {
        return bad_request("session update must include project_id, status, state, or metadata");
    }

    let mut record = existing.unwrap_or_else(|| empty_session_state_record(session_id));
    let now = current_unix_time();

    if let Some(project_id) = effective_project_id {
        let mut project_ids = parse_string_array_json(record.project_ids_json.as_deref());
        project_ids.insert(project_id.clone());
        record.project_id = if project_ids.len() == 1 {
            Some(project_id)
        } else {
            None
        };
        record.project_ids_json = serialize_string_array(project_ids);
    }
    if let Some(status) = payload.status {
        record.status = normalize_session_status(status);
        record.last_transition_at_unix = Some(now);
        record.last_transition_reason = None;
        if record
            .status
            .as_deref()
            .map(session_status_is_terminal)
            .unwrap_or(false)
        {
            record.owner_id = None;
            record.owner_acquired_at_unix = None;
            record.lease_expires_at_unix = None;
            record.cancel_requested_at_unix = None;
            record.cancel_requested_by = None;
            record.cancel_reason = None;
            clear_session_handoff(&mut record);
        }
    }
    if let Some(state) = payload.state {
        record.state_json = Some(state.to_string());
    }
    if let Some(metadata) = payload.metadata {
        record.metadata_json = Some(metadata.to_string());
    }
    record.updated_at_unix = now;

    persist_session_record(
        api,
        record.clone(),
        Some("state_upsert"),
        None,
        None,
        Some(session_event_payload_value(&record)),
        Some(now),
    )
    .await
}

async fn handle_session_claim(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload = if body.is_empty() {
        SessionClaimPayload::default()
    } else {
        match serde_json::from_slice::<SessionClaimPayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }
    let Some(owner_id) = normalize_nonempty_string(payload.owner_id) else {
        return bad_request("owner_id is required");
    };
    let (existing, effective_project_id) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    if existing.is_none() && effective_project_id.is_none() {
        return bad_request("session claim must include project_id for new sessions");
    }

    let now = current_unix_time();
    let lease_ttl_secs = payload.lease_ttl_secs.unwrap_or(60);
    let mut record = existing.unwrap_or_else(|| empty_session_state_record(session_id));
    if record
        .status
        .as_deref()
        .map(session_status_is_terminal)
        .unwrap_or(false)
    {
        return json_response(
            StatusCode::CONFLICT,
            r#"{"error":"cannot claim a terminal session"}"#.to_string(),
        );
    }
    if let Some(current_owner) = record.owner_id.as_deref() {
        if current_owner != owner_id && session_owner_is_active(&record, now) {
            return json_response(
                StatusCode::CONFLICT,
                serde_json::json!({
                    "error": format!("session is already claimed by '{}'", current_owner)
                })
                .to_string(),
            );
        }
    }

    if let Some(project_id) = effective_project_id {
        let mut project_ids = parse_string_array_json(record.project_ids_json.as_deref());
        project_ids.insert(project_id.clone());
        record.project_id = if project_ids.len() == 1 {
            Some(project_id)
        } else {
            None
        };
        record.project_ids_json = serialize_string_array(project_ids);
    }
    if record.owner_id.as_deref() != Some(owner_id.as_str()) {
        record.owner_acquired_at_unix = Some(now);
    }
    if record.status.is_none() {
        record.status = Some("active".to_string());
        record.last_transition_at_unix = Some(now);
        record.last_transition_reason = Some("claimed".to_string());
    }
    record.owner_id = Some(owner_id.to_string());
    record.last_heartbeat_unix = Some(now);
    record.lease_expires_at_unix = Some(now.saturating_add(lease_ttl_secs as i64));
    clear_session_handoff(&mut record);
    if let Some(state) = payload.state {
        record.state_json = Some(state.to_string());
    }
    if let Some(metadata) = payload.metadata {
        record.metadata_json = Some(metadata.to_string());
    }
    record.updated_at_unix = now;

    persist_session_record(
        api,
        record.clone(),
        Some("claimed"),
        Some(owner_id),
        Some("claimed".to_string()),
        Some(session_event_payload_value(&record)),
        Some(now),
    )
    .await
}

async fn handle_session_handoff(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload = if body.is_empty() {
        SessionHandoffPayload::default()
    } else {
        match serde_json::from_slice::<SessionHandoffPayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }
    let Some(requested_owner) = normalize_nonempty_string(payload.owner_id) else {
        return bad_request("owner_id is required");
    };
    let Some(target_owner_id) = normalize_nonempty_string(payload.target_owner_id) else {
        return bad_request("target_owner_id is required");
    };
    if requested_owner == target_owner_id {
        return bad_request("target_owner_id must differ from owner_id");
    }

    let (existing, _) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    let Some(mut record) = existing else {
        return not_found();
    };
    if record
        .status
        .as_deref()
        .map(session_status_is_terminal)
        .unwrap_or(false)
    {
        return json_response(
            StatusCode::CONFLICT,
            r#"{"error":"cannot hand off a terminal session"}"#.to_string(),
        );
    }

    let now = current_unix_time();
    let Some(current_owner) = record.owner_id.as_deref() else {
        return json_response(
            StatusCode::CONFLICT,
            r#"{"error":"session is not actively claimed; claim it before handing off"}"#
                .to_string(),
        );
    };
    if !session_owner_is_active(&record, now) {
        return json_response(
            StatusCode::CONFLICT,
            serde_json::json!({
                "error": format!(
                    "session owner '{}' lease expired; reconcile or claim again before handoff",
                    current_owner
                )
            })
            .to_string(),
        );
    }
    if requested_owner != current_owner {
        return json_response(
            StatusCode::CONFLICT,
            serde_json::json!({
                "error": format!("owner_id '{}' is required to hand off this session", current_owner)
            })
            .to_string(),
        );
    }

    record.handoff_target_owner_id = Some(target_owner_id);
    record.handoff_requested_at_unix = Some(now);
    record.handoff_reason = payload
        .reason
        .and_then(|value| normalize_nonempty_str(&value).map(ToString::to_string));
    if let Some(state) = payload.state {
        record.state_json = Some(state.to_string());
    }
    if let Some(metadata) = payload.metadata {
        record.metadata_json = Some(metadata.to_string());
    }
    record.updated_at_unix = now;

    persist_session_record(
        api,
        record.clone(),
        Some("handoff_requested"),
        Some(requested_owner),
        record.handoff_reason.clone(),
        Some(session_event_payload_value(&record)),
        Some(now),
    )
    .await
}

async fn handle_session_accept(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload = if body.is_empty() {
        SessionAcceptPayload::default()
    } else {
        match serde_json::from_slice::<SessionAcceptPayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }
    let Some(owner_id) = normalize_nonempty_string(payload.owner_id) else {
        return bad_request("owner_id is required");
    };

    let (existing, _) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    let Some(mut record) = existing else {
        return not_found();
    };
    if record
        .status
        .as_deref()
        .map(session_status_is_terminal)
        .unwrap_or(false)
    {
        return json_response(
            StatusCode::CONFLICT,
            r#"{"error":"cannot accept handoff for a terminal session"}"#.to_string(),
        );
    }

    let Some(expected_owner) = record.handoff_target_owner_id.as_deref() else {
        return json_response(
            StatusCode::CONFLICT,
            r#"{"error":"session does not have a pending handoff"}"#.to_string(),
        );
    };
    if owner_id != expected_owner {
        return json_response(
            StatusCode::CONFLICT,
            serde_json::json!({
                "error": format!("owner_id '{}' is required to accept this handoff", expected_owner)
            })
            .to_string(),
        );
    }

    let now = current_unix_time();
    let lease_ttl_secs = payload.lease_ttl_secs.unwrap_or(60);
    record.owner_id = Some(owner_id);
    record.owner_acquired_at_unix = Some(now);
    record.last_heartbeat_unix = Some(now);
    record.lease_expires_at_unix = Some(now.saturating_add(lease_ttl_secs as i64));
    if record.status.is_none()
        || (record.status.as_deref() == Some("paused")
            && record.last_transition_reason.as_deref() == Some(SESSION_RECOVERY_REQUIRED_REASON))
    {
        record.status = Some("active".to_string());
    }
    record.last_transition_at_unix = Some(now);
    record.last_transition_reason = Some("handoff accepted".to_string());
    clear_session_handoff(&mut record);
    if let Some(state) = payload.state {
        record.state_json = Some(state.to_string());
    }
    if let Some(metadata) = payload.metadata {
        record.metadata_json = Some(metadata.to_string());
    }
    record.updated_at_unix = now;

    persist_session_record(
        api,
        record.clone(),
        Some("handoff_accepted"),
        record.owner_id.clone(),
        record.last_transition_reason.clone(),
        Some(session_event_payload_value(&record)),
        Some(now),
    )
    .await
}

async fn handle_session_takeover(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload = if body.is_empty() {
        SessionTakeoverPayload::default()
    } else {
        match serde_json::from_slice::<SessionTakeoverPayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }
    let Some(owner_id) = normalize_nonempty_string(payload.owner_id) else {
        return bad_request("owner_id is required");
    };

    let (existing, _) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    let Some(mut record) = existing else {
        return not_found();
    };
    if record
        .status
        .as_deref()
        .map(session_status_is_terminal)
        .unwrap_or(false)
    {
        return json_response(
            StatusCode::CONFLICT,
            r#"{"error":"cannot take over a terminal session"}"#.to_string(),
        );
    }

    let now = current_unix_time();
    let force = payload.force.unwrap_or(false);
    let handoff_target_owner = record.handoff_target_owner_id.clone();
    if let Some(current_owner) = record.owner_id.as_deref() {
        let owner_active = session_owner_is_active(&record, now);
        if current_owner != owner_id && owner_active && !force {
            if let Some(target_owner) = handoff_target_owner.as_deref() {
                if target_owner == owner_id {
                    return json_response(
                        StatusCode::CONFLICT,
                        serde_json::json!({
                            "error": format!(
                                "session is still actively owned by '{}'; wait for '{}' to accept the handoff or use force=true",
                                current_owner, owner_id
                            )
                        })
                        .to_string(),
                    );
                }
                return json_response(
                    StatusCode::CONFLICT,
                    serde_json::json!({
                        "error": format!(
                            "session is actively owned by '{}' and has a pending handoff to '{}'; use force=true to override",
                            current_owner, target_owner
                        )
                    })
                    .to_string(),
                );
            }
            return json_response(
                StatusCode::CONFLICT,
                serde_json::json!({
                    "error": format!(
                        "session is actively owned by '{}'; request a handoff, wait for lease expiry, or use force=true",
                        current_owner
                    )
                })
                .to_string(),
            );
        }
    }
    if !force {
        if let Some(target_owner) = handoff_target_owner.as_deref() {
            if target_owner != owner_id {
                return json_response(
                    StatusCode::CONFLICT,
                    serde_json::json!({
                        "error": format!(
                            "session has a pending handoff to '{}'; wait for acceptance, cancel the handoff, or use force=true",
                            target_owner
                        )
                    })
                    .to_string(),
                );
            }
        }
    }

    let lease_ttl_secs = payload.lease_ttl_secs.unwrap_or(60);
    if record.owner_id.as_deref() != Some(owner_id.as_str())
        || record.owner_acquired_at_unix.is_none()
    {
        record.owner_acquired_at_unix = Some(now);
    }
    record.owner_id = Some(owner_id);
    record.last_heartbeat_unix = Some(now);
    record.lease_expires_at_unix = Some(now.saturating_add(lease_ttl_secs as i64));
    record.status = Some("active".to_string());
    record.last_transition_at_unix = Some(now);
    record.last_transition_reason = payload
        .reason
        .and_then(|value| normalize_nonempty_str(&value).map(ToString::to_string))
        .or_else(|| {
            Some(if force {
                "force takeover".to_string()
            } else {
                "taken over".to_string()
            })
        });
    clear_session_handoff(&mut record);
    if let Some(state) = payload.state {
        record.state_json = Some(state.to_string());
    }
    if let Some(metadata) = payload.metadata {
        record.metadata_json = Some(metadata.to_string());
    }
    record.updated_at_unix = now;

    persist_session_record(
        api,
        record.clone(),
        Some("taken_over"),
        record.owner_id.clone(),
        record.last_transition_reason.clone(),
        Some(session_event_payload_value(&record)),
        Some(now),
    )
    .await
}

async fn handle_session_release(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload = if body.is_empty() {
        SessionReleasePayload::default()
    } else {
        match serde_json::from_slice::<SessionReleasePayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }
    let requested_owner = normalize_nonempty_string(payload.owner_id);

    let (existing, _) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    let Some(mut record) = existing else {
        return not_found();
    };

    let now = current_unix_time();
    if let Some(current_owner) = record.owner_id.as_deref() {
        if session_owner_is_active(&record, now)
            && requested_owner.as_deref() != Some(current_owner)
        {
            return json_response(
                StatusCode::CONFLICT,
                serde_json::json!({
                    "error": format!("session is currently owned by '{}'", current_owner)
                })
                .to_string(),
            );
        }
    }

    record.owner_id = None;
    record.owner_acquired_at_unix = None;
    record.lease_expires_at_unix = None;
    clear_session_handoff(&mut record);
    record.updated_at_unix = now;

    persist_session_record(
        api,
        record.clone(),
        Some("released"),
        requested_owner.or_else(|| record.owner_id.clone()),
        Some("released".to_string()),
        Some(session_event_payload_value(&record)),
        Some(now),
    )
    .await
}

async fn handle_session_cancel(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload = if body.is_empty() {
        SessionCancelPayload::default()
    } else {
        match serde_json::from_slice::<SessionCancelPayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }

    let (existing, _) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    let Some(mut record) = existing else {
        return not_found();
    };
    if record
        .status
        .as_deref()
        .map(session_status_is_terminal)
        .unwrap_or(false)
    {
        return json_response(
            StatusCode::CONFLICT,
            r#"{"error":"cannot request cancellation for a terminal session"}"#.to_string(),
        );
    }

    let now = current_unix_time();
    record.cancel_requested_at_unix = Some(now);
    record.cancel_requested_by = normalize_nonempty_string(payload.requested_by);
    record.cancel_reason = normalize_nonempty_string(payload.reason);
    record.updated_at_unix = now;

    persist_session_record(
        api,
        record.clone(),
        Some("cancel_requested"),
        record.cancel_requested_by.clone(),
        record.cancel_reason.clone(),
        Some(session_event_payload_value(&record)),
        Some(now),
    )
    .await
}

async fn handle_session_transition(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload = if body.is_empty() {
        SessionTransitionPayload::default()
    } else {
        match serde_json::from_slice::<SessionTransitionPayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }
    let Some(next_status) = payload.status.and_then(normalize_session_status) else {
        return bad_request("status is required");
    };
    if !is_supported_session_status(&next_status) {
        return bad_request("unsupported session status");
    }

    let (existing, effective_project_id) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    if existing.is_none() && effective_project_id.is_none() {
        return bad_request("session transition must include project_id for new sessions");
    }

    let mut record = existing.unwrap_or_else(|| empty_session_state_record(session_id));
    if let Err(error) = validate_session_transition(record.status.as_deref(), &next_status) {
        return json_response(
            StatusCode::CONFLICT,
            serde_json::json!({ "error": error }).to_string(),
        );
    }

    let now = current_unix_time();
    if let Some(project_id) = effective_project_id {
        let mut project_ids = parse_string_array_json(record.project_ids_json.as_deref());
        project_ids.insert(project_id.clone());
        record.project_id = if project_ids.len() == 1 {
            Some(project_id)
        } else {
            None
        };
        record.project_ids_json = serialize_string_array(project_ids);
    }
    record.status = Some(next_status.clone());
    record.last_transition_at_unix = Some(now);
    record.last_transition_reason = payload
        .reason
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if let Some(state) = payload.state {
        record.state_json = Some(state.to_string());
    }
    if let Some(metadata) = payload.metadata {
        record.metadata_json = Some(metadata.to_string());
    }
    if let Some(lease_ttl_secs) = payload.lease_ttl_secs {
        record.lease_expires_at_unix = Some(now.saturating_add(lease_ttl_secs as i64));
    } else if session_status_is_terminal(&next_status) {
        record.owner_id = None;
        record.owner_acquired_at_unix = None;
        record.lease_expires_at_unix = None;
        record.cancel_requested_at_unix = None;
        record.cancel_requested_by = None;
        record.cancel_reason = None;
        clear_session_handoff(&mut record);
    }
    record.updated_at_unix = now;

    persist_session_record(
        api,
        record.clone(),
        Some("transitioned"),
        None,
        record.last_transition_reason.clone(),
        Some(session_event_payload_value(&record)),
        Some(now),
    )
    .await
}

async fn handle_session_heartbeat(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload = if body.is_empty() {
        SessionHeartbeatPayload::default()
    } else {
        match serde_json::from_slice::<SessionHeartbeatPayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }

    let (existing, effective_project_id) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    if existing.is_none() && effective_project_id.is_none() {
        return bad_request("session heartbeat must include project_id for new sessions");
    }

    let mut record = existing.unwrap_or_else(|| empty_session_state_record(session_id));
    if record
        .status
        .as_deref()
        .map(session_status_is_terminal)
        .unwrap_or(false)
    {
        return json_response(
            StatusCode::CONFLICT,
            r#"{"error":"cannot heartbeat a terminal session"}"#.to_string(),
        );
    }

    let now = current_unix_time();
    if let Some(current_owner) = record.owner_id.as_deref() {
        if !session_owner_is_active(&record, now) {
            return json_response(
                StatusCode::CONFLICT,
                serde_json::json!({
                    "error": format!(
                        "session owner '{}' lease expired; claim the session again",
                        current_owner
                    )
                })
                .to_string(),
            );
        }
        match payload.owner_id.as_deref().and_then(normalize_nonempty_str) {
            Some(owner_id) if owner_id == current_owner => {}
            _ => {
                return json_response(
                    StatusCode::CONFLICT,
                    serde_json::json!({
                        "error": format!(
                            "owner_id '{}' is required to heartbeat this session",
                            current_owner
                        )
                    })
                    .to_string(),
                )
            }
        }
    } else if payload
        .owner_id
        .as_deref()
        .and_then(normalize_nonempty_str)
        .is_some()
    {
        return json_response(
            StatusCode::CONFLICT,
            r#"{"error":"session is not claimed; use claim before heartbeating with owner_id"}"#
                .to_string(),
        );
    }
    if let Some(project_id) = effective_project_id {
        let mut project_ids = parse_string_array_json(record.project_ids_json.as_deref());
        project_ids.insert(project_id.clone());
        record.project_id = if project_ids.len() == 1 {
            Some(project_id)
        } else {
            None
        };
        record.project_ids_json = serialize_string_array(project_ids);
    }
    if record.status.is_none() {
        record.status = Some("active".to_string());
    }
    record.last_heartbeat_unix = Some(now);
    if let Some(lease_ttl_secs) = payload.lease_ttl_secs {
        record.lease_expires_at_unix = Some(now.saturating_add(lease_ttl_secs as i64));
    }
    if let Some(state) = payload.state {
        record.state_json = Some(state.to_string());
    }
    if let Some(metadata) = payload.metadata {
        record.metadata_json = Some(metadata.to_string());
    }
    record.updated_at_unix = now;

    persist_session_record(
        api,
        record.clone(),
        Some("heartbeat"),
        record.owner_id.clone(),
        None,
        Some(session_event_payload_value(&record)),
        Some(now),
    )
    .await
}

async fn handle_session_reconcile(
    api: &LlmGatewayApi,
    session_id: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(error) => return bad_request(&format!("failed to read request body: {error}")),
    };
    let payload = if body.is_empty() {
        SessionReconcilePayload::default()
    } else {
        match serde_json::from_slice::<SessionReconcilePayload>(&body) {
            Ok(value) => value,
            Err(_) => return bad_request("invalid json"),
        }
    };
    if payload.project_id.as_deref().map(str::trim) == Some("") {
        return bad_request("project_id must not be empty");
    }

    let (existing, _) =
        match load_session_record_for_mutation(api, session_id, auth, payload.project_id.clone())
            .await
        {
            Ok(value) => value,
            Err(resp) => return resp,
        };
    let Some(mut record) = existing else {
        return not_found();
    };

    let now = current_unix_time();
    let action = reconcile_session_record(&mut record, now);
    match action {
        Some(action) => match persist_session_record(
            api,
            record.clone(),
            Some("reconciled"),
            None,
            Some(action.as_str().to_string()),
            Some(serde_json::json!({
                "reconciled_action": action.as_str(),
                "session": session_event_payload_value(&record),
            })),
            Some(now),
        )
        .await
        .status()
        {
            StatusCode::OK => session_reconcile_response(StatusCode::OK, &record, Some(action)),
            status => json_response(
                status,
                serde_json::json!({
                    "error": format!("failed to persist reconciled session state")
                })
                .to_string(),
            ),
        },
        None => session_reconcile_response(StatusCode::OK, &record, None),
    }
}

async fn handle_session_logs(
    api: &LlmGatewayApi,
    session_id: &str,
    req: &Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let query = req.uri().query().unwrap_or("");
    let project_id = match extract_query_param(query, "project_id") {
        Some(project_id) => Some(project_id),
        None => match scoped_project_query(api, auth) {
            Ok(project_id) => project_id,
            Err(resp) => return resp,
        },
    };
    if let Some(ref project_id) = project_id {
        if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, Some(project_id)) {
            return resp;
        }
    } else if let Some(resp) = ensure_permission(api, auth, Permission::ViewLogs, None) {
        return resp;
    }

    let limit = match parse_u32_query_param(query, "limit") {
        Ok(Some(value)) if value > 0 => value,
        Ok(_) => 100,
        Err(error) => return bad_request(&error),
    };
    let metadata_key = normalize_nonempty_string(extract_query_param(query, "metadata_key"));
    let metadata_value = normalize_nonempty_string(extract_query_param(query, "metadata_value"));
    let has_custom_cost = match parse_bool_query_param(query, "has_custom_cost") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    let custom_cost_applied = match parse_bool_query_param(query, "custom_cost_applied") {
        Ok(value) => value,
        Err(error) => return bad_request(&error),
    };
    if metadata_key.is_none() && metadata_value.is_some() {
        return bad_request("metadata_value requires metadata_key");
    }
    let query = RequestLogQuery {
        api_key: None,
        model: None,
        project_id,
        session_id: Some(session_id.to_string()),
        metadata_key,
        metadata_value,
        has_custom_cost,
        custom_cost_applied,
        limit,
    };
    let result = api.query_request_logs(&query).await;

    match result {
        Some(Ok(logs)) => request_logs_response(&logs),
        Some(Err(error)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, error),
        ),
        None => json_response(
            StatusCode::OK,
            r#"{"error":"store not configured"}"#.to_string(),
        ),
    }
}

fn session_summary_json_from_logs(
    session_id: &str,
    limit: u32,
    logs: &[RequestLogEntry],
) -> serde_json::Value {
    let mut project_ids = BTreeSet::new();
    let mut providers = BTreeSet::new();
    let mut models = BTreeSet::new();
    let mut prompt_names = BTreeSet::new();
    let mut prompt_versions = BTreeSet::new();
    let mut tool_names = BTreeSet::new();
    let mut total_input_tokens = 0u64;
    let mut total_output_tokens = 0u64;
    let mut total_cost = 0.0f64;
    let mut streaming_request_count = 0usize;
    let mut safety_event_count = 0usize;
    let mut semantic_event_count = 0usize;
    let mut semantic_degraded_count = 0usize;
    let mut tool_call_count = 0usize;
    let mut tool_error_count = 0usize;
    let mut first_request_unix: Option<i64> = None;
    let mut last_request_unix: Option<i64> = None;
    let mut latest_request: Option<&RequestLogEntry> = None;

    for entry in logs {
        if let Some(project_id) = entry.project_id.as_ref() {
            project_ids.insert(project_id.clone());
        }
        if let Some(provider_name) = entry.provider_name.as_ref() {
            providers.insert(provider_name.clone());
        }
        if let Some(model) = entry.model.as_ref() {
            models.insert(model.clone());
        }
        if let Some(prompt_name) = entry.prompt_name.as_ref() {
            prompt_names.insert(prompt_name.clone());
        }
        if let (Some(prompt_name), Some(prompt_version)) =
            (entry.prompt_name.as_ref(), entry.prompt_version.as_ref())
        {
            prompt_versions.insert(format!("{prompt_name}@{prompt_version}"));
        }
        total_input_tokens = total_input_tokens.saturating_add(entry.input_tokens);
        total_output_tokens = total_output_tokens.saturating_add(entry.output_tokens);
        total_cost += entry.cost;
        first_request_unix = Some(
            first_request_unix
                .map(|current| current.min(entry.timestamp_unix))
                .unwrap_or(entry.timestamp_unix),
        );
        last_request_unix = Some(
            last_request_unix
                .map(|current| current.max(entry.timestamp_unix))
                .unwrap_or(entry.timestamp_unix),
        );
        if latest_request
            .map(|current| entry.timestamp_unix >= current.timestamp_unix)
            .unwrap_or(true)
        {
            latest_request = Some(entry);
        }
        if entry.is_streaming {
            streaming_request_count += 1;
        }
        if entry.safety_mode.is_some()
            || entry
                .safety_matches
                .as_deref()
                .map(|value| value != "[]" && !value.is_empty())
                .unwrap_or(false)
        {
            safety_event_count += 1;
        }
        if entry.semantic_policy_version.is_some()
            || entry.semantic_index_state.is_some()
            || entry.semantic_degraded_reason.is_some()
            || entry
                .semantic_findings
                .as_deref()
                .map(|value| value != "[]" && !value.is_empty())
                .unwrap_or(false)
        {
            semantic_event_count += 1;
        }
        if entry.semantic_degraded_reason.is_some()
            || entry.semantic_index_state.as_deref() == Some("degraded")
        {
            semantic_degraded_count += 1;
        }
        if let Some(audit) = parse_tool_runtime_audit(entry.tool_trace.as_deref()) {
            tool_call_count += audit.calls.len();
            tool_error_count += audit
                .calls
                .iter()
                .filter(|call| call.status == "error")
                .count();
            for call in audit.calls {
                tool_names.insert(call.tool_name);
            }
        }
    }

    let project_id = if project_ids.len() == 1 {
        project_ids.iter().next().cloned()
    } else {
        None
    };
    let latest_request_value = latest_request
        .map(|entry| {
            serde_json::json!({
                "timestamp_unix": entry.timestamp_unix,
                "provider_name": entry.provider_name.clone(),
                "model": entry.model.clone(),
                "prompt_name": entry.prompt_name.clone(),
                "prompt_version": entry.prompt_version.clone(),
                "prompt_environment": entry.prompt_environment.clone(),
                "safety_mode": entry.safety_mode.clone(),
                "semantic_index_state": entry.semantic_index_state.clone(),
                "semantic_degraded_reason": entry.semantic_degraded_reason.clone(),
            })
        })
        .unwrap_or(serde_json::Value::Null);

    let mut summary = serde_json::Map::new();
    summary.insert("session_id".to_string(), serde_json::json!(session_id));
    summary.insert("project_id".to_string(), serde_json::json!(project_id));
    summary.insert(
        "project_ids".to_string(),
        serde_json::json!(project_ids.into_iter().collect::<Vec<_>>()),
    );
    summary.insert(
        "first_request_unix".to_string(),
        serde_json::json!(first_request_unix),
    );
    summary.insert("request_count".to_string(), serde_json::json!(logs.len()));
    summary.insert(
        "streaming_request_count".to_string(),
        serde_json::json!(streaming_request_count),
    );
    summary.insert(
        "total_input_tokens".to_string(),
        serde_json::json!(total_input_tokens),
    );
    summary.insert(
        "total_output_tokens".to_string(),
        serde_json::json!(total_output_tokens),
    );
    summary.insert("total_cost".to_string(), serde_json::json!(total_cost));
    summary.insert(
        "providers".to_string(),
        serde_json::json!(providers.into_iter().collect::<Vec<_>>()),
    );
    summary.insert(
        "models".to_string(),
        serde_json::json!(models.into_iter().collect::<Vec<_>>()),
    );
    summary.insert(
        "prompt_names".to_string(),
        serde_json::json!(prompt_names.into_iter().collect::<Vec<_>>()),
    );
    summary.insert(
        "prompt_versions".to_string(),
        serde_json::json!(prompt_versions.into_iter().collect::<Vec<_>>()),
    );
    summary.insert(
        "last_request_unix".to_string(),
        serde_json::json!(last_request_unix),
    );
    summary.insert("latest_request".to_string(), latest_request_value);
    summary.insert(
        "safety_event_count".to_string(),
        serde_json::json!(safety_event_count),
    );
    summary.insert(
        "semantic_event_count".to_string(),
        serde_json::json!(semantic_event_count),
    );
    summary.insert(
        "semantic_degraded_count".to_string(),
        serde_json::json!(semantic_degraded_count),
    );
    summary.insert(
        "tool_names".to_string(),
        serde_json::json!(tool_names.into_iter().collect::<Vec<_>>()),
    );
    summary.insert(
        "tool_call_count".to_string(),
        serde_json::json!(tool_call_count),
    );
    summary.insert(
        "tool_error_count".to_string(),
        serde_json::json!(tool_error_count),
    );
    summary.insert("status".to_string(), serde_json::Value::Null);
    summary.insert("owner_id".to_string(), serde_json::Value::Null);
    summary.insert(
        "owner_acquired_at_unix".to_string(),
        serde_json::Value::Null,
    );
    summary.insert("owner_active".to_string(), serde_json::json!(false));
    summary.insert("owner_stale".to_string(), serde_json::json!(false));
    summary.insert(
        "last_transition_at_unix".to_string(),
        serde_json::Value::Null,
    );
    summary.insert(
        "last_transition_reason".to_string(),
        serde_json::Value::Null,
    );
    summary.insert("last_heartbeat_unix".to_string(), serde_json::Value::Null);
    summary.insert("lease_expires_at_unix".to_string(), serde_json::Value::Null);
    summary.insert(
        "cancel_requested_at_unix".to_string(),
        serde_json::Value::Null,
    );
    summary.insert("cancel_requested_by".to_string(), serde_json::Value::Null);
    summary.insert("cancel_reason".to_string(), serde_json::Value::Null);
    summary.insert(
        "handoff_target_owner_id".to_string(),
        serde_json::Value::Null,
    );
    summary.insert(
        "handoff_requested_at_unix".to_string(),
        serde_json::Value::Null,
    );
    summary.insert("handoff_reason".to_string(), serde_json::Value::Null);
    summary.insert("handoff_pending".to_string(), serde_json::json!(false));
    summary.insert("cancel_pending".to_string(), serde_json::json!(false));
    summary.insert("recovery_required".to_string(), serde_json::json!(false));
    summary.insert("recovery_reason".to_string(), serde_json::Value::Null);
    summary.insert("state".to_string(), serde_json::Value::Null);
    summary.insert("metadata".to_string(), serde_json::Value::Null);
    summary.insert(
        "updated_at_unix".to_string(),
        serde_json::json!(last_request_unix),
    );
    summary.insert(
        "truncated".to_string(),
        serde_json::json!(logs.len() as u32 >= limit),
    );
    serde_json::Value::Object(summary)
}

fn session_summary_json_from_record(record: &SessionRecord) -> serde_json::Value {
    let now = current_unix_time();
    let mut summary = serde_json::Map::new();
    summary.insert(
        "session_id".to_string(),
        serde_json::json!(record.session_id),
    );
    summary.insert(
        "project_id".to_string(),
        serde_json::json!(record.project_id),
    );
    summary.insert(
        "project_ids".to_string(),
        parse_string_array_value(record.project_ids_json.as_deref()),
    );
    summary.insert(
        "first_request_unix".to_string(),
        serde_json::json!(record.first_request_unix),
    );
    summary.insert(
        "request_count".to_string(),
        serde_json::json!(record.request_count),
    );
    summary.insert(
        "streaming_request_count".to_string(),
        serde_json::json!(record.streaming_request_count),
    );
    summary.insert(
        "total_input_tokens".to_string(),
        serde_json::json!(record.total_input_tokens),
    );
    summary.insert(
        "total_output_tokens".to_string(),
        serde_json::json!(record.total_output_tokens),
    );
    summary.insert(
        "total_cost".to_string(),
        serde_json::json!(record.total_cost),
    );
    summary.insert(
        "providers".to_string(),
        parse_string_array_value(record.providers_json.as_deref()),
    );
    summary.insert(
        "models".to_string(),
        parse_string_array_value(record.models_json.as_deref()),
    );
    summary.insert(
        "prompt_names".to_string(),
        parse_string_array_value(record.prompt_names_json.as_deref()),
    );
    summary.insert(
        "prompt_versions".to_string(),
        parse_string_array_value(record.prompt_versions_json.as_deref()),
    );
    summary.insert(
        "last_request_unix".to_string(),
        serde_json::json!(record.last_request_unix),
    );
    summary.insert(
        "latest_request".to_string(),
        parse_json_value(record.latest_request_json.as_deref()),
    );
    summary.insert(
        "safety_event_count".to_string(),
        serde_json::json!(record.safety_event_count),
    );
    summary.insert(
        "semantic_event_count".to_string(),
        serde_json::json!(record.semantic_event_count),
    );
    summary.insert(
        "semantic_degraded_count".to_string(),
        serde_json::json!(record.semantic_degraded_count),
    );
    summary.insert(
        "tool_names".to_string(),
        parse_string_array_value(record.tool_names_json.as_deref()),
    );
    summary.insert(
        "tool_call_count".to_string(),
        serde_json::json!(record.tool_call_count),
    );
    summary.insert(
        "tool_error_count".to_string(),
        serde_json::json!(record.tool_error_count),
    );
    summary.insert("status".to_string(), serde_json::json!(record.status));
    summary.insert("owner_id".to_string(), serde_json::json!(record.owner_id));
    summary.insert(
        "owner_acquired_at_unix".to_string(),
        serde_json::json!(record.owner_acquired_at_unix),
    );
    summary.insert(
        "owner_active".to_string(),
        serde_json::json!(session_owner_is_active(record, now)),
    );
    summary.insert(
        "owner_stale".to_string(),
        serde_json::json!(session_owner_is_stale(record, now)),
    );
    summary.insert(
        "last_transition_at_unix".to_string(),
        serde_json::json!(record.last_transition_at_unix),
    );
    summary.insert(
        "last_transition_reason".to_string(),
        serde_json::json!(record.last_transition_reason),
    );
    summary.insert(
        "last_heartbeat_unix".to_string(),
        serde_json::json!(record.last_heartbeat_unix),
    );
    summary.insert(
        "lease_expires_at_unix".to_string(),
        serde_json::json!(record.lease_expires_at_unix),
    );
    summary.insert(
        "cancel_requested_at_unix".to_string(),
        serde_json::json!(record.cancel_requested_at_unix),
    );
    summary.insert(
        "cancel_requested_by".to_string(),
        serde_json::json!(record.cancel_requested_by),
    );
    summary.insert(
        "cancel_reason".to_string(),
        serde_json::json!(record.cancel_reason),
    );
    summary.insert(
        "handoff_target_owner_id".to_string(),
        serde_json::json!(record.handoff_target_owner_id),
    );
    summary.insert(
        "handoff_requested_at_unix".to_string(),
        serde_json::json!(record.handoff_requested_at_unix),
    );
    summary.insert(
        "handoff_reason".to_string(),
        serde_json::json!(record.handoff_reason),
    );
    summary.insert(
        "handoff_pending".to_string(),
        serde_json::json!(session_handoff_is_pending(record)),
    );
    summary.insert(
        "cancel_pending".to_string(),
        serde_json::json!(session_cancel_is_pending(record)),
    );
    summary.insert(
        "recovery_required".to_string(),
        serde_json::json!(session_recovery_required(record, now)),
    );
    summary.insert(
        "recovery_reason".to_string(),
        serde_json::json!(session_recovery_reason(record, now)),
    );
    summary.insert(
        "state".to_string(),
        parse_json_value(record.state_json.as_deref()),
    );
    summary.insert(
        "metadata".to_string(),
        parse_json_value(record.metadata_json.as_deref()),
    );
    summary.insert(
        "updated_at_unix".to_string(),
        serde_json::json!(record.updated_at_unix),
    );
    summary.insert("truncated".to_string(), serde_json::json!(false));
    serde_json::Value::Object(summary)
}

fn session_event_json_from_record(record: &SessionEventRecord) -> serde_json::Value {
    serde_json::json!({
        "event_seq": record.event_seq,
        "session_id": record.session_id,
        "project_id": record.project_id,
        "event_kind": record.event_kind,
        "actor_id": record.actor_id,
        "reason": record.reason,
        "payload": parse_json_value(record.payload_json.as_deref()),
        "created_at_unix": record.created_at_unix,
    })
}

fn session_reconcile_response(
    status: StatusCode,
    record: &SessionRecord,
    action: Option<SessionReconcileAction>,
) -> Response<Full<Bytes>> {
    let mut payload = session_summary_json_from_record(record);
    if let serde_json::Value::Object(ref mut map) = payload {
        map.insert(
            "reconciled_action".to_string(),
            action
                .map(|action| serde_json::Value::String(action.as_str().to_string()))
                .unwrap_or(serde_json::Value::Null),
        );
    }
    json_response(status, payload.to_string())
}

fn parse_tool_runtime_audit(value: Option<&str>) -> Option<ToolRuntimeAudit> {
    value.and_then(|value| serde_json::from_str(value).ok())
}

fn session_record_matches_project(record: &SessionRecord, project_id: Option<&str>) -> bool {
    let Some(project_id) = project_id else {
        return true;
    };
    if record.project_id.as_deref() == Some(project_id) {
        return true;
    }
    parse_string_array_json(record.project_ids_json.as_deref()).contains(project_id)
}

fn parse_string_array_json(value: Option<&str>) -> BTreeSet<String> {
    value
        .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
        .unwrap_or_default()
        .into_iter()
        .collect()
}

fn extract_query_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key && !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn parse_f64_query_param(query: &str, key: &str) -> Result<Option<f64>, String> {
    let Some(value) = extract_query_param(query, key) else {
        return Ok(None);
    };
    value
        .parse::<f64>()
        .map(Some)
        .map_err(|_| format!("{key} must be a number"))
}

fn parse_u32_query_param(query: &str, key: &str) -> Result<Option<u32>, String> {
    let Some(value) = extract_query_param(query, key) else {
        return Ok(None);
    };
    value
        .parse::<u32>()
        .map(Some)
        .map_err(|_| format!("{key} must be an unsigned integer"))
}

fn parse_i64_query_param(query: &str, key: &str) -> Result<Option<i64>, String> {
    let Some(value) = extract_query_param(query, key) else {
        return Ok(None);
    };
    value
        .parse::<i64>()
        .map(Some)
        .map_err(|_| format!("{key} must be an integer"))
}

fn parse_bool_query_param(query: &str, key: &str) -> Result<Option<bool>, String> {
    let Some(value) = extract_query_param(query, key) else {
        return Ok(None);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(Some(true)),
        "false" | "0" | "no" => Ok(Some(false)),
        _ => Err(format!("{key} must be a boolean")),
    }
}

fn parse_eval_comparison_gate_request(
    query: &str,
) -> Result<Option<ProjectEvalRunComparisonGateRequest>, String> {
    let preset = match extract_query_param(query, "preset") {
        Some(value) => Some(
            crate::evals::ProjectEvalRunComparisonGatePreset::from_str(&value).ok_or_else(
                || "preset must be one of: strict, balanced, exploratory".to_string(),
            )?,
        ),
        None => None,
    };
    let request = ProjectEvalRunComparisonGateRequest {
        preset,
        max_regressions: parse_u32_query_param(query, "max_regressions")?,
        min_candidate_pass_rate: parse_f64_query_param(query, "min_candidate_pass_rate")?,
        min_pass_rate_delta: parse_f64_query_param(query, "min_pass_rate_delta")?,
        max_latency_increase_ms: parse_f64_query_param(query, "max_latency_increase_ms")?,
        max_cost_increase: parse_f64_query_param(query, "max_cost_increase")?,
    };
    if request.is_empty() {
        Ok(None)
    } else {
        Ok(Some(request))
    }
}

fn parse_rollout_policy_name(query: &str) -> Option<String> {
    extract_query_param(query, "policy_name").and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn deserialize_rollout_policy_gate(
    record: &ProjectRolloutPolicyRecord,
) -> Result<ProjectEvalRunComparisonGateRequest, String> {
    let value = serde_json::from_str::<serde_json::Value>(&record.gate_config_json)
        .map_err(|error| format!("invalid rollout policy gate config: {error}"))?;
    let gate_value = value.get("gate").cloned().unwrap_or_else(|| value.clone());
    serde_json::from_value(gate_value)
        .map_err(|error| format!("invalid rollout policy gate config: {error}"))
}

#[derive(Clone, Debug)]
struct RolloutCanaryPolicyConfig {
    steps: Vec<u8>,
    auto_promote_final: bool,
    auto_advance_on_pass: bool,
    auto_rollback_on_fail: bool,
}

fn deserialize_rollout_policy_canary(
    record: &ProjectRolloutPolicyRecord,
) -> Result<Option<RolloutCanaryPolicyConfig>, String> {
    let value = serde_json::from_str::<serde_json::Value>(&record.gate_config_json)
        .map_err(|error| format!("invalid rollout policy gate config: {error}"))?;
    let Some(canary_value) = value.get("canary") else {
        return Ok(None);
    };
    let canary = canary_value
        .as_object()
        .ok_or_else(|| "rollout policy canary config must be an object".to_string())?;
    let steps = canary
        .get("steps")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "rollout policy canary.steps must be an array".to_string())?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| "rollout policy canary.steps entries must be integers".to_string())
                .and_then(|value| {
                    let step = u8::try_from(value).map_err(|_| {
                        "rollout policy canary.steps entries must be between 1 and 100".to_string()
                    })?;
                    if step == 0 {
                        Err(
                            "rollout policy canary.steps entries must be between 1 and 100"
                                .to_string(),
                        )
                    } else {
                        Ok(step)
                    }
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if steps.is_empty() {
        return Err("rollout policy canary.steps must not be empty".to_string());
    }
    if steps.windows(2).any(|window| window[0] >= window[1]) {
        return Err("rollout policy canary.steps must be strictly increasing".to_string());
    }
    Ok(Some(RolloutCanaryPolicyConfig {
        steps,
        auto_promote_final: canary
            .get("auto_promote_final")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        auto_advance_on_pass: canary
            .get("auto_advance_on_pass")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        auto_rollback_on_fail: canary
            .get("auto_rollback_on_fail")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
    }))
}

fn encode_rollout_policy_config(
    gate: &ProjectEvalRunComparisonGateRequest,
    canary: Option<&RolloutCanaryPolicyConfig>,
) -> Result<String, String> {
    let gate_value = serde_json::to_value(gate)
        .map_err(|error| format!("failed to encode rollout policy gate: {error}"))?;
    let mut value = serde_json::json!({ "gate": gate_value });
    if let Some(canary) = canary {
        value["canary"] = serde_json::json!({
            "steps": canary.steps,
            "auto_promote_final": canary.auto_promote_final,
            "auto_advance_on_pass": canary.auto_advance_on_pass,
            "auto_rollback_on_fail": canary.auto_rollback_on_fail,
        });
    }
    serde_json::to_string(&value)
        .map_err(|error| format!("failed to encode rollout policy gate: {error}"))
}

fn query_param_is_truthy(query: &str, key: &str) -> bool {
    matches!(
        extract_query_param(query, key).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn optional_json_value(value: Option<&str>) -> serde_json::Value {
    match value {
        Some(value) => serde_json::from_str(value)
            .unwrap_or_else(|_| serde_json::Value::String(value.to_string())),
        None => serde_json::Value::Null,
    }
}

fn json_diff_entry(before: serde_json::Value, after: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "before": before,
        "after": after,
    })
}

fn diff_json_values(
    before: &serde_json::Value,
    after: &serde_json::Value,
) -> Option<serde_json::Value> {
    if before == after {
        return None;
    }

    match (before, after) {
        (serde_json::Value::Object(before_map), serde_json::Value::Object(after_map)) => {
            let mut keys = std::collections::BTreeSet::new();
            keys.extend(before_map.keys().cloned());
            keys.extend(after_map.keys().cloned());

            let mut diff = serde_json::Map::new();
            for key in keys {
                let before_value = before_map.get(&key).unwrap_or(&serde_json::Value::Null);
                let after_value = after_map.get(&key).unwrap_or(&serde_json::Value::Null);
                if let Some(child_diff) = diff_json_values(before_value, after_value) {
                    diff.insert(key, child_diff);
                }
            }

            if diff.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(diff))
            }
        }
        _ => Some(json_diff_entry(before.clone(), after.clone())),
    }
}

fn optional_json_diff(before: Option<&str>, after: Option<&str>) -> serde_json::Value {
    let before_value = optional_json_value(before);
    let after_value = optional_json_value(after);
    diff_json_values(&before_value, &after_value).unwrap_or(serde_json::Value::Null)
}

fn parse_json_map_u64(value: Option<&str>) -> std::collections::HashMap<String, u64> {
    value
        .and_then(|raw| serde_json::from_str::<std::collections::HashMap<String, u64>>(raw).ok())
        .unwrap_or_default()
}

fn parse_json_map_u32(value: Option<&str>) -> std::collections::HashMap<String, u32> {
    value
        .and_then(|raw| serde_json::from_str::<std::collections::HashMap<String, u32>>(raw).ok())
        .unwrap_or_default()
}

fn parse_json_map_f64(value: Option<&str>) -> std::collections::HashMap<String, f64> {
    value
        .and_then(|raw| serde_json::from_str::<std::collections::HashMap<String, f64>>(raw).ok())
        .unwrap_or_default()
}

fn project_policy_value(policy: &ProjectPolicyRecord) -> serde_json::Value {
    serde_json::json!({
        "project_id": policy.project_id,
        "budget_limit": policy.budget_limit,
        "budget_duration": policy.budget_duration,
        "rpm_limit": policy.rpm_limit,
        "tpm_limit": policy.tpm_limit,
        "fallback_order": optional_json_value(policy.fallback_order.as_deref()),
        "adaptive_enabled": policy.adaptive_enabled,
        "timeout_secs": policy.timeout_secs,
        "provider_rpm_limits": optional_json_value(policy.provider_rpm_limits.as_deref()),
        "provider_tpm_limits": optional_json_value(policy.provider_tpm_limits.as_deref()),
        "provider_timeouts": optional_json_value(policy.provider_timeouts.as_deref()),
        "provider_input_costs": optional_json_value(policy.provider_input_costs.as_deref()),
        "provider_output_costs": optional_json_value(policy.provider_output_costs.as_deref()),
        "semantic_cache_enabled": policy.semantic_cache_enabled,
        "semantic_cache_ttl_secs": policy.semantic_cache_ttl_secs,
        "semantic_cache_similarity_threshold": policy.semantic_cache_similarity_threshold,
        "tool_approval_mode": policy.tool_approval_mode,
        "allowed_tools": optional_json_value(policy.allowed_tools.as_deref()),
    })
}

fn virtual_key_value(record: &VirtualKeyRecord) -> serde_json::Value {
    serde_json::json!({
        "key_hash": record.key_hash,
        "project_id": record.project_id,
        "name": record.name,
        "provider_name": record.provider_name,
        "budget_limit": record.budget_limit,
        "budget_duration": record.budget_duration,
        "rpm_limit": record.rpm_limit,
        "tpm_limit": record.tpm_limit,
        "allowed_models": optional_json_value(record.allowed_models.as_deref()),
        "timeout_secs": record.timeout_secs,
        "tool_approval_mode": record.tool_approval_mode,
        "allowed_tools": optional_json_value(record.allowed_tools.as_deref()),
        "active": record.active,
        "expires_at": record.expires_at,
    })
}

fn source_value(source: Option<&str>) -> serde_json::Value {
    source
        .map(|value| serde_json::Value::String(value.to_string()))
        .unwrap_or(serde_json::Value::Null)
}

fn effective_optional<T: Copy>(
    key_value: Option<T>,
    project_value: Option<T>,
) -> (Option<T>, Option<&'static str>) {
    if let Some(value) = key_value {
        (Some(value), Some("virtual_key"))
    } else if let Some(value) = project_value {
        (Some(value), Some("project_policy"))
    } else {
        (None, None)
    }
}

fn effective_optional_with_provider<T: Copy>(
    key_value: Option<T>,
    provider_value: Option<T>,
    project_value: Option<T>,
) -> (Option<T>, Option<&'static str>) {
    if let Some(value) = key_value {
        (Some(value), Some("virtual_key"))
    } else if let Some(value) = provider_value {
        (Some(value), Some("project_provider_policy"))
    } else if let Some(value) = project_value {
        (Some(value), Some("project_policy"))
    } else {
        (None, None)
    }
}

fn effective_timeout_value(
    key_timeout_secs: Option<u64>,
    project_provider_timeout_secs: Option<u64>,
    project_timeout_secs: Option<u64>,
    provider_default_timeout_secs: Option<u64>,
) -> (Option<u64>, Option<&'static str>) {
    if let Some(value) = key_timeout_secs {
        (Some(value), Some("virtual_key"))
    } else if let Some(value) = project_provider_timeout_secs {
        (Some(value), Some("project_provider_timeout"))
    } else if let Some(value) = project_timeout_secs {
        (Some(value), Some("project_policy"))
    } else if let Some(value) = provider_default_timeout_secs {
        (Some(value), Some("provider_default"))
    } else {
        (None, None)
    }
}

fn effective_string_value(
    key_value: Option<&String>,
    project_value: Option<&String>,
) -> (Option<String>, Option<&'static str>) {
    if let Some(value) = key_value {
        (Some(value.clone()), Some("virtual_key"))
    } else if let Some(value) = project_value {
        (Some(value.clone()), Some("project_policy"))
    } else {
        (None, None)
    }
}

fn effective_json_value(
    project_value: Option<&str>,
    source_name: &'static str,
) -> (serde_json::Value, serde_json::Value) {
    match project_value {
        Some(value) => (
            optional_json_value(Some(value)),
            serde_json::Value::String(source_name.to_string()),
        ),
        None => (serde_json::Value::Null, serde_json::Value::Null),
    }
}

fn effective_runtime_policy_response(
    api: &LlmGatewayApi,
    project_id: &str,
    project_policy: Option<&ProjectPolicyRecord>,
    provider_name: Option<&str>,
    virtual_key: Option<&VirtualKeyRecord>,
) -> serde_json::Value {
    let provider_rpm_limits =
        parse_json_map_u32(project_policy.and_then(|policy| policy.provider_rpm_limits.as_deref()));
    let provider_tpm_limits =
        parse_json_map_u32(project_policy.and_then(|policy| policy.provider_tpm_limits.as_deref()));
    let provider_timeouts =
        parse_json_map_u64(project_policy.and_then(|policy| policy.provider_timeouts.as_deref()));
    let provider_input_costs = parse_json_map_f64(
        project_policy.and_then(|policy| policy.provider_input_costs.as_deref()),
    );
    let provider_output_costs = parse_json_map_f64(
        project_policy.and_then(|policy| policy.provider_output_costs.as_deref()),
    );

    let project_provider_rpm_limit =
        provider_name.and_then(|provider| provider_rpm_limits.get(provider).copied());
    let project_provider_tpm_limit =
        provider_name.and_then(|provider| provider_tpm_limits.get(provider).copied());
    let project_provider_timeout_secs =
        provider_name.and_then(|provider| provider_timeouts.get(provider).copied());
    let provider_input_cost =
        provider_name.and_then(|provider| provider_input_costs.get(provider).copied());
    let provider_output_cost =
        provider_name.and_then(|provider| provider_output_costs.get(provider).copied());
    let provider_default_timeout_secs =
        provider_name.and_then(|provider| api.provider_timeout_secs(provider));

    let (budget_limit, budget_limit_source) = effective_optional(
        virtual_key.and_then(|record| record.budget_limit),
        project_policy.and_then(|policy| policy.budget_limit),
    );
    let (budget_duration, budget_duration_source) = effective_string_value(
        virtual_key.and_then(|record| record.budget_duration.as_ref()),
        project_policy.and_then(|policy| policy.budget_duration.as_ref()),
    );
    let (rpm_limit, rpm_limit_source) = effective_optional_with_provider(
        virtual_key.and_then(|record| record.rpm_limit),
        project_provider_rpm_limit,
        project_policy.and_then(|policy| policy.rpm_limit),
    );
    let (tpm_limit, tpm_limit_source) = effective_optional_with_provider(
        virtual_key.and_then(|record| record.tpm_limit),
        project_provider_tpm_limit,
        project_policy.and_then(|policy| policy.tpm_limit),
    );
    let (timeout_secs, timeout_source) = effective_timeout_value(
        virtual_key.and_then(|record| record.timeout_secs),
        project_provider_timeout_secs,
        project_policy.and_then(|policy| policy.timeout_secs),
        provider_default_timeout_secs,
    );
    let (tool_approval_mode, allowed_tools, tool_policy_source) = effective_tool_approval_policy(
        virtual_key.and_then(|record| record.tool_approval_mode.as_deref()),
        virtual_key.and_then(|record| record.allowed_tools.as_deref()),
        project_policy.and_then(|policy| policy.tool_approval_mode.as_deref()),
        project_policy.and_then(|policy| policy.allowed_tools.as_deref()),
    )
    .unwrap_or({
        (
            crate::virtual_keys::ToolApprovalMode::AllowAll,
            None,
            "default",
        )
    });
    let (fallback_order, fallback_order_source) = effective_json_value(
        project_policy.and_then(|policy| policy.fallback_order.as_deref()),
        "project_policy",
    );
    let semantic_cache_status = api.semantic_cache_status();
    let (semantic_cache_enabled, semantic_cache_enabled_source) =
        match project_policy.and_then(|policy| policy.semantic_cache_enabled) {
            Some(value) => (Some(value), Some("project_policy")),
            None if semantic_cache_status.is_some() => (Some(false), Some("request_opt_in")),
            None => (None, None),
        };
    let (semantic_cache_ttl_secs, semantic_cache_ttl_secs_source) =
        match project_policy.and_then(|policy| policy.semantic_cache_ttl_secs) {
            Some(value) => (Some(value), Some("project_policy")),
            None => (
                semantic_cache_status
                    .as_ref()
                    .map(|status| status.default_ttl_secs),
                semantic_cache_status.as_ref().map(|_| "plugin_default"),
            ),
        };
    let (semantic_cache_similarity_threshold, semantic_cache_similarity_threshold_source) =
        match project_policy.and_then(|policy| policy.semantic_cache_similarity_threshold) {
            Some(value) => (Some(value), Some("project_policy")),
            None => (
                semantic_cache_status
                    .as_ref()
                    .map(|status| status.default_similarity_threshold),
                semantic_cache_status.as_ref().map(|_| "plugin_default"),
            ),
        };

    let (adaptive_enabled, adaptive_enabled_source) = match project_policy {
        Some(policy) => (policy.adaptive_enabled, Some("project_policy")),
        None => (true, Some("default")),
    };

    serde_json::json!({
        "project_id": project_id,
        "provider_name": provider_name,
        "virtual_key": virtual_key.map(virtual_key_value).unwrap_or(serde_json::Value::Null),
        "project_policy": project_policy.map(project_policy_value).unwrap_or(serde_json::Value::Null),
        "effective": {
            "budget_limit": budget_limit,
            "budget_duration": budget_duration,
            "rpm_limit": rpm_limit,
            "tpm_limit": tpm_limit,
            "timeout_secs": timeout_secs,
            "tool_approval_mode": tool_approval_mode.as_str(),
            "allowed_tools": allowed_tools,
            "fallback_order": fallback_order,
            "adaptive_enabled": adaptive_enabled,
            "provider_input_cost": provider_input_cost,
            "provider_output_cost": provider_output_cost,
            "semantic_cache_enabled": semantic_cache_enabled,
            "semantic_cache_ttl_secs": semantic_cache_ttl_secs,
            "semantic_cache_similarity_threshold": semantic_cache_similarity_threshold,
        },
        "sources": {
            "budget_limit": source_value(budget_limit_source),
            "budget_duration": source_value(budget_duration_source),
            "rpm_limit": source_value(rpm_limit_source),
            "tpm_limit": source_value(tpm_limit_source),
            "timeout_secs": source_value(timeout_source),
            "tool_approval_mode": source_value(Some(tool_policy_source)),
            "allowed_tools": source_value((allowed_tools.is_some()).then_some(tool_policy_source)),
            "fallback_order": fallback_order_source,
            "adaptive_enabled": source_value(adaptive_enabled_source),
            "provider_input_cost": if provider_input_cost.is_some() {
                serde_json::Value::String("project_policy".to_string())
            } else {
                serde_json::Value::Null
            },
            "provider_output_cost": if provider_output_cost.is_some() {
                serde_json::Value::String("project_policy".to_string())
            } else {
                serde_json::Value::Null
            },
            "semantic_cache_enabled": source_value(semantic_cache_enabled_source),
            "semantic_cache_ttl_secs": source_value(semantic_cache_ttl_secs_source),
            "semantic_cache_similarity_threshold": source_value(semantic_cache_similarity_threshold_source),
        },
        "provider_context": {
            "project_provider_rpm_limit": project_provider_rpm_limit,
            "project_provider_tpm_limit": project_provider_tpm_limit,
            "provider_input_cost": provider_input_cost,
            "provider_output_cost": provider_output_cost,
        },
        "timeout_context": {
            "project_provider_timeout_secs": project_provider_timeout_secs,
            "project_timeout_secs": project_policy.and_then(|policy| policy.timeout_secs),
            "provider_default_timeout_secs": provider_default_timeout_secs,
        },
        "notes": [
            "Routing rule overrides are request-dependent and are not included in this snapshot."
        ],
    })
}

fn prompt_record_json(record: &ProjectPromptRecord) -> serde_json::Value {
    serde_json::json!({
        "project_id": record.project_id.clone(),
        "prompt_name": record.prompt_name.clone(),
        "version": record.version.clone(),
        "environment": record.environment.clone(),
        "description": record.description.clone(),
        "target": record.target.clone(),
        "template_text": record.template_text.clone(),
        "variables_schema": record
            .variables_schema_json
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "rollout_metadata": record
            .rollout_metadata_json
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "active": record.active,
        "updated_at": record.updated_at.clone(),
    })
}

fn rollout_policy_record_json(record: &ProjectRolloutPolicyRecord) -> serde_json::Value {
    let gate_config = serde_json::from_str::<serde_json::Value>(&record.gate_config_json)
        .unwrap_or(serde_json::Value::Null);
    let gate = gate_config
        .get("gate")
        .cloned()
        .unwrap_or_else(|| gate_config.clone());
    let canary = gate_config
        .get("canary")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "project_id": record.project_id.clone(),
        "policy_name": record.policy_name.clone(),
        "description": record.description.clone(),
        "gate": gate,
        "canary": canary,
        "target_environment": record.target_environment.clone(),
        "updated_at": record.updated_at.clone(),
    })
}

fn prompt_rollout_record_json(record: &ProjectPromptRolloutRecord) -> serde_json::Value {
    serde_json::json!({
        "project_id": record.project_id.clone(),
        "prompt_name": record.prompt_name.clone(),
        "rollout_id": record.rollout_id.clone(),
        "policy_name": record.policy_name.clone(),
        "baseline_version": record.baseline_version.clone(),
        "candidate_version": record.candidate_version.clone(),
        "baseline_run_id": record.baseline_run_id.clone(),
        "candidate_run_id": record.candidate_run_id.clone(),
        "target_environment": record.target_environment.clone(),
        "status": record.status.clone(),
        "recommendation_action": record.recommendation_action.clone(),
        "comparison": optional_json_value(Some(record.comparison_json.as_str())),
        "latest_canary_evaluation": prompt_rollout_latest_canary_evaluation_json(&record.comparison_json),
        "runtime_rollout": prompt_rollout_runtime_json(&record.comparison_json),
        "created_at": record.created_at.clone(),
        "applied_at": record.applied_at.clone(),
    })
}

fn prompt_rollout_latest_canary_evaluation_json(comparison_json: &str) -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(comparison_json)
        .ok()
        .and_then(|value| value.get("latest_canary_evaluation").cloned())
        .unwrap_or(serde_json::Value::Null)
}

fn prompt_rollout_runtime_json(comparison_json: &str) -> serde_json::Value {
    parse_prompt_rollout_runtime_config(comparison_json)
        .map(|config| {
            serde_json::json!({
                "mode": config.mode,
                "traffic_percent": config.traffic_percent,
                "current_step_index": config.current_step_index,
            })
        })
        .unwrap_or(serde_json::Value::Null)
}

#[derive(Clone, Debug)]
struct PromptRolloutRuntimeConfig {
    mode: String,
    traffic_percent: u8,
    current_step_index: Option<usize>,
}

fn parse_prompt_rollout_runtime_config(raw: &str) -> Option<PromptRolloutRuntimeConfig> {
    let value = serde_json::from_str::<serde_json::Value>(raw).ok()?;
    let runtime = value.get("runtime_rollout")?.as_object()?;
    let mode = runtime.get("mode")?.as_str()?.trim();
    if mode.is_empty() {
        return None;
    }
    let traffic_percent = runtime
        .get("traffic_percent")
        .and_then(|value| value.as_u64())
        .and_then(|value| u8::try_from(value).ok())?;
    Some(PromptRolloutRuntimeConfig {
        mode: mode.to_string(),
        traffic_percent,
        current_step_index: runtime
            .get("current_step_index")
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok()),
    })
}

fn update_prompt_rollout_runtime_config(
    raw: &str,
    runtime: Option<&PromptRolloutRuntimeConfig>,
) -> String {
    let mut value = serde_json::from_str::<serde_json::Value>(raw)
        .unwrap_or_else(|_| serde_json::json!({ "comparison": raw }));
    if !value.is_object() {
        value = serde_json::json!({ "comparison": raw });
    }
    let object = value.as_object_mut().expect("comparison object");
    match runtime {
        Some(runtime) => {
            object.insert(
                "runtime_rollout".to_string(),
                serde_json::json!({
                    "mode": runtime.mode,
                    "traffic_percent": runtime.traffic_percent,
                    "current_step_index": runtime.current_step_index,
                }),
            );
        }
        None => {
            object.remove("runtime_rollout");
        }
    }
    value.to_string()
}

fn update_prompt_rollout_canary_evaluation(
    raw: &str,
    runtime: Option<&PromptRolloutRuntimeConfig>,
    baseline_run_id: &str,
    candidate_run_id: &str,
    comparison: serde_json::Value,
    action: &str,
    applied: bool,
    reason: Option<&str>,
    evaluated_at: &str,
) -> String {
    update_prompt_rollout_comparison(
        raw,
        runtime,
        Some(serde_json::json!({
            "evaluated_at": evaluated_at,
            "baseline_run_id": baseline_run_id,
            "candidate_run_id": candidate_run_id,
            "action": action,
            "applied": applied,
            "reason": reason,
            "comparison": comparison,
        })),
    )
}

fn update_prompt_rollout_comparison(
    raw: &str,
    runtime: Option<&PromptRolloutRuntimeConfig>,
    latest_canary_evaluation: Option<serde_json::Value>,
) -> String {
    let mut value = serde_json::from_str::<serde_json::Value>(raw)
        .unwrap_or_else(|_| serde_json::json!({ "comparison": raw }));
    if !value.is_object() {
        value = serde_json::json!({ "comparison": raw });
    }
    let object = value.as_object_mut().expect("comparison object");
    match runtime {
        Some(runtime) => {
            object.insert(
                "runtime_rollout".to_string(),
                serde_json::json!({
                    "mode": runtime.mode,
                    "traffic_percent": runtime.traffic_percent,
                    "current_step_index": runtime.current_step_index,
                }),
            );
        }
        None => {
            object.remove("runtime_rollout");
        }
    }
    match latest_canary_evaluation {
        Some(latest_canary_evaluation) => {
            object.insert(
                "latest_canary_evaluation".to_string(),
                latest_canary_evaluation,
            );
        }
        None => {
            object.remove("latest_canary_evaluation");
        }
    }
    value.to_string()
}

enum PromptRolloutAutomationAction {
    Hold(String),
    Advance(PromptRolloutRuntimeConfig),
    Promote,
    Rollback,
}

impl PromptRolloutAutomationAction {
    fn action_name(&self) -> &'static str {
        match self {
            Self::Hold(_) => "hold",
            Self::Advance(_) => "advance",
            Self::Promote => "promote",
            Self::Rollback => "rollback",
        }
    }

    fn reason(&self) -> Option<&str> {
        match self {
            Self::Hold(reason) => Some(reason.as_str()),
            _ => None,
        }
    }
}

fn dataset_record_json(record: &ProjectDatasetRecord) -> serde_json::Value {
    serde_json::json!({
        "project_id": record.project_id.clone(),
        "dataset_name": record.dataset_name.clone(),
        "description": record.description.clone(),
        "schema": optional_json_value(record.schema_json.as_deref()),
        "updated_at": record.updated_at.clone(),
    })
}

fn dataset_item_record_json(record: &ProjectDatasetItemRecord) -> serde_json::Value {
    serde_json::json!({
        "project_id": record.project_id.clone(),
        "dataset_name": record.dataset_name.clone(),
        "item_id": record.item_id.clone(),
        "input": optional_json_value(Some(record.input_json.as_str())),
        "expected_output": optional_json_value(record.expected_output_json.as_deref()),
        "metadata": optional_json_value(record.metadata_json.as_deref()),
        "updated_at": record.updated_at.clone(),
    })
}

fn eval_run_record_json(record: &ProjectEvalRunRecord) -> serde_json::Value {
    serde_json::json!({
        "run_id": record.run_id.clone(),
        "project_id": record.project_id.clone(),
        "dataset_name": record.dataset_name.clone(),
        "target_url": record.target_url.clone(),
        "status": record.status.clone(),
        "total_items": record.total_items,
        "passed_items": record.passed_items,
        "failed_items": record.failed_items,
        "total_input_tokens": record.total_input_tokens,
        "total_output_tokens": record.total_output_tokens,
        "total_cost": record.total_cost,
        "average_latency_ms": record.average_latency_ms,
        "summary": optional_json_value(record.summary_json.as_deref()),
        "created_at": record.created_at.clone(),
        "completed_at": record.completed_at.clone(),
    })
}

fn eval_run_item_record_json(record: &ProjectEvalRunItemRecord) -> serde_json::Value {
    serde_json::json!({
        "run_id": record.run_id.clone(),
        "project_id": record.project_id.clone(),
        "dataset_name": record.dataset_name.clone(),
        "item_id": record.item_id.clone(),
        "passed": record.passed,
        "status_code": record.status_code,
        "latency_ms": record.latency_ms,
        "output_text": record.output_text.clone(),
        "evaluation": optional_json_value(record.evaluation_json.as_deref()),
        "error": record.error.clone(),
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "cost": record.cost,
        "created_at": record.created_at.clone(),
    })
}

fn eval_run_comparison_json_with_rollout_policy(
    comparison: &crate::evals::ProjectEvalRunComparison,
    rollout_policy: Option<&ProjectRolloutPolicyRecord>,
) -> serde_json::Value {
    serde_json::json!({
        "baseline_run": eval_run_record_json(&comparison.baseline_run),
        "candidate_run": eval_run_record_json(&comparison.candidate_run),
        "rollout_policy": rollout_policy.map(rollout_policy_record_json).unwrap_or(serde_json::Value::Null),
        "context": {
            "baseline": comparison.context.baseline,
            "candidate": comparison.context.candidate,
            "changed_fields": comparison.context.changed_fields,
        },
        "summary": {
            "baseline_pass_rate": comparison.summary.baseline_pass_rate,
            "candidate_pass_rate": comparison.summary.candidate_pass_rate,
            "delta_pass_rate": comparison.summary.delta_pass_rate,
            "baseline_average_latency_ms": comparison.summary.baseline_average_latency_ms,
            "candidate_average_latency_ms": comparison.summary.candidate_average_latency_ms,
            "delta_average_latency_ms": comparison.summary.delta_average_latency_ms,
            "baseline_total_cost": comparison.summary.baseline_total_cost,
            "candidate_total_cost": comparison.summary.candidate_total_cost,
            "delta_total_cost": comparison.summary.delta_total_cost,
            "improved_items": comparison.summary.improved_items,
            "regressed_items": comparison.summary.regressed_items,
            "unchanged_items": comparison.summary.unchanged_items,
        },
        "gate": comparison.gate.as_ref().map(|gate| serde_json::json!({
            "passed": gate.passed,
            "preset": gate.preset.map(|preset| preset.as_str()),
            "reasons": gate.reasons,
            "thresholds": {
                "preset": gate.thresholds.preset.map(|preset| preset.as_str()),
                "max_regressions": gate.thresholds.max_regressions,
                "min_candidate_pass_rate": gate.thresholds.min_candidate_pass_rate,
                "min_pass_rate_delta": gate.thresholds.min_pass_rate_delta,
                "max_latency_increase_ms": gate.thresholds.max_latency_increase_ms,
                "max_cost_increase": gate.thresholds.max_cost_increase,
            },
            "recommendation": {
                "action": gate.recommendation.action.as_str(),
                "summary": gate.recommendation.summary,
                "changed_context_fields": gate.recommendation.changed_context_fields,
            }
        })).unwrap_or(serde_json::Value::Null),
        "items": comparison.items.iter().map(|item| {
            serde_json::json!({
                "item_id": item.item_id,
                "baseline_passed": item.baseline_passed,
                "candidate_passed": item.candidate_passed,
                "improved": item.improved,
                "regressed": item.regressed,
                "changed": item.changed,
                "baseline_output_text": item.baseline_output_text,
                "candidate_output_text": item.candidate_output_text,
                "baseline_evaluation": optional_json_value(item.baseline_evaluation_json.as_deref()),
                "candidate_evaluation": optional_json_value(item.candidate_evaluation_json.as_deref()),
            })
        }).collect::<Vec<_>>(),
    })
}

fn prompt_context_field(run: &ProjectEvalRunRecord, key: &str) -> Option<String> {
    run.summary_json
        .as_deref()
        .and_then(|summary| serde_json::from_str::<serde_json::Value>(summary).ok())
        .and_then(|summary| summary.get("context").cloned())
        .and_then(|context| context.get(key).cloned())
        .and_then(|value| value.as_str().map(ToString::to_string))
}

fn validate_prompt_promotion_context(
    prompt_name: &str,
    candidate_version: &str,
    comparison: &crate::evals::ProjectEvalRunComparison,
) -> Result<(), String> {
    for run in [&comparison.baseline_run, &comparison.candidate_run] {
        if let Some(context_prompt_name) = prompt_context_field(run, "prompt_name") {
            if context_prompt_name != prompt_name {
                return Err(format!(
                    "eval run '{}' was executed for prompt '{}' instead of '{}'",
                    run.run_id, context_prompt_name, prompt_name
                ));
            }
        }
    }

    if let Some(context_candidate_version) =
        prompt_context_field(&comparison.candidate_run, "prompt_version")
    {
        if context_candidate_version != candidate_version {
            return Err(format!(
                "candidate eval run '{}' used prompt version '{}' instead of '{}'",
                comparison.candidate_run.run_id, context_candidate_version, candidate_version
            ));
        }
    }

    Ok(())
}

fn validate_prompt_rollout_evaluation_context(
    prompt_name: &str,
    baseline_version: Option<&str>,
    candidate_version: &str,
    comparison: &crate::evals::ProjectEvalRunComparison,
) -> Result<(), String> {
    validate_prompt_promotion_context(prompt_name, candidate_version, comparison)?;

    if let Some(expected_baseline_version) = baseline_version {
        if let Some(actual_baseline_version) =
            prompt_context_field(&comparison.baseline_run, "prompt_version")
        {
            if actual_baseline_version != expected_baseline_version {
                return Err(format!(
                    "baseline eval run '{}' used prompt version '{}' instead of '{}'",
                    comparison.baseline_run.run_id,
                    actual_baseline_version,
                    expected_baseline_version
                ));
            }
        }
    }

    Ok(())
}

fn generate_prompt_rollout_id() -> String {
    crate::governance::generate_unique_id("rollout", &PROMPT_ROLLOUT_SEQUENCE)
}

fn generate_routing_rule_id() -> String {
    crate::governance::generate_unique_id("rr", &ROUTING_RULE_SEQUENCE)
}

async fn supersede_live_prompt_canaries(
    api: &LlmGatewayApi,
    project_id: &str,
    prompt_name: &str,
    keep_rollout_id: &str,
    target_environment: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(Ok(rollouts)) = api
        .list_project_prompt_rollouts(project_id, prompt_name)
        .await
    else {
        return Ok(());
    };
    for mut rollout in rollouts {
        if rollout.rollout_id == keep_rollout_id || rollout.status != "applied_canary" {
            continue;
        }
        if let Some(environment) = target_environment {
            if rollout.target_environment.as_deref() != Some(environment) {
                continue;
            }
        }
        rollout.status = "superseded".to_string();
        api.upsert_project_prompt_rollout(rollout)
            .await
            .transpose()?
            .unwrap_or(());
    }
    Ok(())
}

async fn load_rollout_canary_policy(
    api: &LlmGatewayApi,
    project_id: &str,
    policy_name: &str,
) -> Result<Option<RolloutCanaryPolicyConfig>, Response<Full<Bytes>>> {
    let policy = match api
        .get_project_rollout_policy(project_id, policy_name)
        .await
    {
        Some(Ok(Some(policy))) => policy,
        Some(Ok(None)) => {
            return Err(json_response(
                StatusCode::NOT_FOUND,
                r#"{"error":"rollout policy not found"}"#.to_string(),
            ))
        }
        Some(Err(error)) => {
            return Err(json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(r#"{{"error":"{}"}}"#, error),
            ))
        }
        None => {
            return Err(json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                r#"{"error":"store not enabled"}"#.to_string(),
            ))
        }
    };
    deserialize_rollout_policy_canary(&policy).map_err(|message| bad_request(&message))
}

fn canary_apply_runtime_config(
    requested_traffic_percent: Option<u8>,
    policy: Option<&RolloutCanaryPolicyConfig>,
) -> Result<PromptRolloutRuntimeConfig, String> {
    if let Some(traffic_percent) = requested_traffic_percent {
        if traffic_percent == 0 || traffic_percent > 100 {
            return Err("canary traffic_percent must be between 1 and 100".to_string());
        }
        let current_step_index = policy.and_then(|policy| {
            policy
                .steps
                .iter()
                .position(|step| *step == traffic_percent)
        });
        return Ok(PromptRolloutRuntimeConfig {
            mode: "canary".to_string(),
            traffic_percent,
            current_step_index,
        });
    }
    let Some(policy) = policy else {
        return Err(
            "canary traffic_percent is required when the rollout policy has no canary steps"
                .to_string(),
        );
    };
    Ok(PromptRolloutRuntimeConfig {
        mode: "canary".to_string(),
        traffic_percent: policy.steps[0],
        current_step_index: Some(0),
    })
}

fn next_canary_runtime_config(
    current: &PromptRolloutRuntimeConfig,
    requested_traffic_percent: Option<u8>,
    policy: Option<&RolloutCanaryPolicyConfig>,
) -> Result<PromptRolloutRuntimeConfig, String> {
    if current.mode != "canary" {
        return Err("prompt rollout is not in canary mode".to_string());
    }
    if let Some(traffic_percent) = requested_traffic_percent {
        if traffic_percent == 0 || traffic_percent > 100 {
            return Err("canary traffic_percent must be between 1 and 100".to_string());
        }
        if traffic_percent <= current.traffic_percent {
            return Err(
                "canary traffic_percent must be greater than the current value".to_string(),
            );
        }
        let current_step_index = policy.and_then(|policy| {
            policy
                .steps
                .iter()
                .position(|step| *step == traffic_percent)
        });
        return Ok(PromptRolloutRuntimeConfig {
            mode: "canary".to_string(),
            traffic_percent,
            current_step_index,
        });
    }
    let Some(policy) = policy else {
        return Err(
            "traffic_percent is required when the rollout policy has no canary steps".to_string(),
        );
    };
    let current_index = current.current_step_index.or_else(|| {
        policy
            .steps
            .iter()
            .position(|step| *step == current.traffic_percent)
    });
    let Some(current_index) = current_index else {
        return Err(
            "current canary traffic does not match any rollout policy stage; provide traffic_percent explicitly"
                .to_string(),
        );
    };
    let next_index = current_index + 1;
    let Some(&traffic_percent) = policy.steps.get(next_index) else {
        return Err("no further canary stages remain for this rollout policy".to_string());
    };
    Ok(PromptRolloutRuntimeConfig {
        mode: "canary".to_string(),
        traffic_percent,
        current_step_index: Some(next_index),
    })
}

fn determine_prompt_rollout_automation_action(
    current: &PromptRolloutRuntimeConfig,
    gate_passed: bool,
    policy: Option<&RolloutCanaryPolicyConfig>,
) -> PromptRolloutAutomationAction {
    let Some(policy) = policy else {
        return PromptRolloutAutomationAction::Hold(
            "rollout policy does not define canary automation".to_string(),
        );
    };

    if gate_passed {
        if !policy.auto_advance_on_pass {
            return PromptRolloutAutomationAction::Hold(
                "policy does not enable auto-advance on pass".to_string(),
            );
        }
        return match next_canary_runtime_config(current, None, Some(policy)) {
            Ok(next_runtime)
                if policy.auto_promote_final && next_runtime.traffic_percent == 100 =>
            {
                PromptRolloutAutomationAction::Promote
            }
            Ok(next_runtime) => PromptRolloutAutomationAction::Advance(next_runtime),
            Err(message) => PromptRolloutAutomationAction::Hold(message),
        };
    }

    if policy.auto_rollback_on_fail {
        PromptRolloutAutomationAction::Rollback
    } else {
        PromptRolloutAutomationAction::Hold(
            "policy does not enable auto-rollback on fail".to_string(),
        )
    }
}

async fn prepare_prompt_rollout_decision(
    api: &LlmGatewayApi,
    project_id: &str,
    prompt_name: &str,
    candidate_version: &str,
    baseline_run_id: &str,
    candidate_run_id: &str,
    policy_name: &str,
) -> Result<PreparedPromptRolloutDecision, Response<Full<Bytes>>> {
    let policy = match api
        .get_project_rollout_policy(project_id, policy_name)
        .await
    {
        Some(Ok(Some(policy))) => policy,
        Some(Ok(None)) => {
            return Err(json_response(
                StatusCode::NOT_FOUND,
                r#"{"error":"rollout policy not found"}"#.to_string(),
            ))
        }
        Some(Err(error)) => {
            return Err(json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(r#"{{"error":"{}"}}"#, error),
            ))
        }
        None => {
            return Err(json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                r#"{"error":"store not enabled"}"#.to_string(),
            ))
        }
    };
    let candidate_prompt = match api.get_project_prompt(project_id, prompt_name, candidate_version)
    {
        Some(record) => record,
        None => {
            return Err(json_response(
                StatusCode::NOT_FOUND,
                r#"{"error":"candidate prompt version not found"}"#.to_string(),
            ))
        }
    };
    if let Some(target_environment) = policy.target_environment.as_deref() {
        if candidate_prompt.environment != target_environment {
            return Err(bad_request(
                "candidate prompt environment does not match rollout policy target_environment",
            ));
        }
    }
    let gate_request = match deserialize_rollout_policy_gate(&policy) {
        Ok(gate) if !gate.is_empty() => gate,
        Ok(_) => return Err(bad_request("rollout policy gate is empty")),
        Err(message) => return Err(bad_request(&message)),
    };
    let comparison = match api
        .compare_project_eval_runs(
            project_id,
            baseline_run_id,
            candidate_run_id,
            Some(gate_request),
        )
        .await
    {
        Some(Ok(comparison)) => comparison,
        Some(Err(error)) => {
            return Err(json_response(
                StatusCode::BAD_REQUEST,
                serde_json::json!({ "error": error.to_string() }).to_string(),
            ))
        }
        None => {
            return Err(json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                r#"{"error":"store not enabled"}"#.to_string(),
            ))
        }
    };
    if let Err(message) =
        validate_prompt_promotion_context(prompt_name, candidate_version, &comparison)
    {
        return Err(bad_request(&message));
    }
    if comparison.gate.is_none() {
        return Err(json_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":"rollout policy did not produce a gate result"}"#.to_string(),
        ));
    }
    Ok(PreparedPromptRolloutDecision {
        policy,
        candidate_prompt,
        comparison,
    })
}

async fn read_body_string(req: Request<Incoming>) -> Result<String, Response<Full<Bytes>>> {
    use http_body_util::BodyExt;

    let body_bytes = req
        .into_body()
        .collect()
        .await
        .map_err(|_| {
            json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"failed to read body"}"#.to_string(),
            )
        })?
        .to_bytes();

    std::str::from_utf8(&body_bytes)
        .map(ToString::to_string)
        .map_err(|_| {
            json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"body is not valid UTF-8"}"#.to_string(),
            )
        })
}

async fn handle_delete_key(
    api: &LlmGatewayApi,
    hash_prefix: &str,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let existing = match api.get_virtual_key(hash_prefix) {
        Some(Ok(Some(record))) => record,
        Some(Ok(None)) => {
            return json_response(
                StatusCode::NOT_FOUND,
                r#"{"ok":false,"deleted":false}"#.to_string(),
            );
        }
        Some(Err(err)) => return virtual_key_lookup_response(&err),
        None => {
            return json_response(
                StatusCode::OK,
                r#"{"error":"virtual_keys not enabled"}"#.to_string(),
            );
        }
    };
    if let Some(resp) = ensure_permission(
        api,
        auth,
        Permission::ManageRuntimeKeys,
        Some(&existing.project_id),
    ) {
        return resp;
    }

    match api.delete_virtual_key(hash_prefix).await {
        Some(Ok(true)) => {
            json_response(StatusCode::OK, r#"{"ok":true,"deleted":true}"#.to_string())
        }
        Some(Ok(false)) => json_response(
            StatusCode::NOT_FOUND,
            r#"{"ok":false,"deleted":false}"#.to_string(),
        ),
        Some(Err(e)) => {
            if let Some(err) = e.downcast_ref::<VirtualKeyLookupError>() {
                virtual_key_lookup_response(err)
            } else {
                json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, e),
                )
            }
        }
        None => json_response(
            StatusCode::OK,
            r#"{"error":"virtual_keys not enabled"}"#.to_string(),
        ),
    }
}

fn handle_list_projects(api: &LlmGatewayApi, auth: Option<&AuthContext>) -> Response<Full<Bytes>> {
    if let Some(resp) = ensure_permission(api, auth, Permission::ViewProjects, None) {
        return resp;
    }

    let projects = match api.list_projects() {
        Some(projects) => {
            if is_instance_admin(auth) {
                projects
            } else {
                let allowed = api
                    .accessible_projects(auth.expect("auth required when filtering projects"))
                    .into_iter()
                    .map(|project| project.0)
                    .collect::<Vec<_>>();
                projects
                    .into_iter()
                    .filter(|project| {
                        allowed
                            .iter()
                            .any(|allowed_id| allowed_id == &project.project_id)
                    })
                    .collect()
            }
        }
        None => return json_response(StatusCode::OK, r#"{"projects":[]}"#.to_string()),
    };

    let body = format!(
        r#"{{"projects":[{}]}}"#,
        projects
            .iter()
            .map(|project| {
                format!(
                    r#"{{"project_id":"{}","name":"{}","description":{},"active":{},"created_at":"{}"}}"#,
                    project.project_id,
                    project.name,
                    project
                        .description
                        .as_ref()
                        .map(|value| format!("\"{}\"", value))
                        .unwrap_or_else(|| "null".to_string()),
                    project.active,
                    project.created_at,
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    );
    json_response(StatusCode::OK, body)
}

async fn handle_create_project(
    api: &LlmGatewayApi,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    if let Some(resp) = ensure_permission(api, auth, Permission::ManageProjects, None) {
        return resp;
    }

    let body = match read_body_string(req).await {
        Ok(body) => body,
        Err(resp) => return resp,
    };
    let project_id = match extract_json_string(&body, "project_id") {
        Some(project_id) => project_id,
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"project_id is required"}"#.to_string(),
            );
        }
    };
    let name = extract_json_string(&body, "name").unwrap_or_else(|| project_id.clone());
    let description = extract_json_string(&body, "description");

    match api.create_project(&project_id, &name, description).await {
        Some(Ok(project)) => json_response(
            StatusCode::CREATED,
            format!(
                r#"{{"project_id":"{}","name":"{}"}}"#,
                project.project_id, project.name
            ),
        ),
        Some(Err(error)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, error),
        ),
        None => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":"auth service not enabled"}"#.to_string(),
        ),
    }
}

fn handle_project_effective_runtime_policy(
    api: &LlmGatewayApi,
    project_id: &str,
    req: &Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    if let Some(resp) =
        ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
    {
        return resp;
    }

    let query = req.uri().query().unwrap_or("");
    let key_prefix =
        extract_query_param(query, "key_hash").or_else(|| extract_query_param(query, "key"));
    let provider_name_query = extract_query_param(query, "provider_name");

    let virtual_key = if let Some(prefix) = key_prefix.as_deref() {
        if let Some(resp) =
            ensure_permission(api, auth, Permission::ViewRuntimeKeys, Some(project_id))
        {
            return resp;
        }
        match api.get_virtual_key(prefix) {
            Some(Ok(Some(record))) => {
                if record.project_id != project_id {
                    return json_response(
                        StatusCode::NOT_FOUND,
                        r#"{"error":"key not found"}"#.to_string(),
                    );
                }
                Some(record)
            }
            Some(Ok(None)) => {
                return json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"key not found"}"#.to_string(),
                );
            }
            Some(Err(err)) => return virtual_key_lookup_response(&err),
            None => {
                return json_response(
                    StatusCode::OK,
                    r#"{"error":"virtual_keys not enabled"}"#.to_string(),
                );
            }
        }
    } else {
        None
    };

    if let (Some(provider_name), Some(record)) =
        (provider_name_query.as_deref(), virtual_key.as_ref())
    {
        if record.provider_name != provider_name {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"provider_name does not match virtual key provider"}"#.to_string(),
            );
        }
    }

    let provider_name = provider_name_query.or_else(|| {
        virtual_key
            .as_ref()
            .map(|record| record.provider_name.clone())
    });
    let project_policy = api.get_project_policy(project_id);

    json_response(
        StatusCode::OK,
        effective_runtime_policy_response(
            api,
            project_id,
            project_policy.as_ref(),
            provider_name.as_deref(),
            virtual_key.as_ref(),
        )
        .to_string(),
    )
}

async fn handle_project_subroutes(
    api: &LlmGatewayApi,
    tail: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let mut parts = tail.splitn(2, '/');
    let project_id = parts.next().unwrap_or_default();
    let remainder = parts.next();
    if project_id.is_empty() {
        return not_found();
    }

    match (req.method(), remainder) {
        (&Method::DELETE, None) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjects, Some(project_id))
            {
                return resp;
            }
            match api.delete_project(project_id).await {
                Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Ok(false)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"project not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"auth service not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("history")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let query = req.uri().query().unwrap_or("");
            let resource_type = extract_query_param(query, "resource_type");
            let include_diff = query_param_is_truthy(query, "include_diff");
            let limit = extract_query_param(query, "limit")
                .and_then(|value| value.parse::<u32>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(100)
                .min(500);
            match api
                .get_governance_changes(project_id, resource_type.as_deref(), limit)
                .await
            {
                Some(Ok(changes)) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": project_id,
                        "changes": changes.into_iter().map(|change| {
                            serde_json::json!({
                                "change_id": change.change_id,
                                "project_id": change.project_id,
                                "resource_type": change.resource_type,
                                "resource_id": change.resource_id,
                                "action": change.action,
                                "before": optional_json_value(change.before_json.as_deref()),
                                "after": optional_json_value(change.after_json.as_deref()),
                                "diff": if include_diff {
                                    optional_json_diff(
                                        change.before_json.as_deref(),
                                        change.after_json.as_deref(),
                                    )
                                } else {
                                    serde_json::Value::Null
                                },
                                "changed_at": change.changed_at,
                            })
                        }).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("effective-runtime-policy")) => {
            handle_project_effective_runtime_policy(api, project_id, &req, auth)
        }
        (&Method::GET, Some("policy")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let body = api
                .get_project_policy(project_id)
                .map(|policy| project_policy_value(&policy).to_string())
                .unwrap_or_else(|| serde_json::json!({ "project_id": project_id }).to_string());
            json_response(StatusCode::OK, body)
        }
        (&Method::PUT, Some("policy")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            if let Err(error) = validate_string_array_json(
                extract_json_raw(&body, "allowed_tools").as_deref(),
                "allowed_tools",
            ) {
                return json_response(
                    StatusCode::BAD_REQUEST,
                    format!(r#"{{"error":"{}"}}"#, error),
                );
            }
            let semantic_cache_similarity_threshold =
                extract_json_optional_float(&body, "semantic_cache_similarity_threshold");
            if let Some(value) = semantic_cache_similarity_threshold {
                if !(0.0..=1.0).contains(&value) {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        r#"{"error":"semantic_cache_similarity_threshold must be between 0.0 and 1.0"}"#.to_string(),
                    );
                }
            }
            let record = ProjectPolicyRecord {
                project_id: project_id.to_string(),
                budget_limit: extract_json_optional_float(&body, "budget_limit"),
                budget_duration: extract_json_optional_string(&body, "budget_duration"),
                rpm_limit: extract_json_optional_float(&body, "rpm_limit").map(|v| v as u32),
                tpm_limit: extract_json_optional_float(&body, "tpm_limit").map(|v| v as u32),
                fallback_order: extract_json_raw(&body, "fallback_order"),
                adaptive_enabled: extract_json_bool(&body, "adaptive_enabled").unwrap_or(true),
                timeout_secs: extract_json_optional_float(&body, "timeout_secs").map(|v| v as u64),
                provider_rpm_limits: extract_json_raw(&body, "provider_rpm_limits"),
                provider_tpm_limits: extract_json_raw(&body, "provider_tpm_limits"),
                provider_timeouts: extract_json_raw(&body, "provider_timeouts"),
                provider_input_costs: extract_json_raw(&body, "provider_input_costs"),
                provider_output_costs: extract_json_raw(&body, "provider_output_costs"),
                semantic_cache_enabled: extract_json_bool(&body, "semantic_cache_enabled"),
                semantic_cache_ttl_secs: extract_json_optional_float(
                    &body,
                    "semantic_cache_ttl_secs",
                )
                .map(|v| v as u64),
                semantic_cache_similarity_threshold,
                tool_approval_mode: match normalize_tool_approval_mode_value(
                    extract_json_optional_string(&body, "tool_approval_mode").as_deref(),
                    extract_json_raw(&body, "allowed_tools").as_deref(),
                ) {
                    Ok(mode) => mode,
                    Err(error) => {
                        return json_response(
                            StatusCode::BAD_REQUEST,
                            format!(r#"{{"error":"{}"}}"#, error),
                        );
                    }
                },
                allowed_tools: extract_json_raw(&body, "allowed_tools"),
                updated_at: current_timestamp_string(),
            };
            match api.upsert_project_policy(record).await {
                Some(Ok(())) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::DELETE, Some("policy")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            match api.delete_project_policy(project_id).await {
                Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Ok(false)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"policy not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("routing-rules")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewRoutingRules, Some(project_id))
            {
                return resp;
            }
            let rules = api.list_routing_rules(Some(project_id)).unwrap_or_default();
            let body = format!(
                r#"{{"rules":[{}]}}"#,
                rules
                    .iter()
                    .map(|rule| {
                        format!(
                            r#"{{"rule_id":"{}","project_id":"{}","name":"{}","priority":{},"enabled":{},"match_path":{},"match_model":{},"match_streaming":{},"match_role":{},"match_headers":{},"min_prompt_tokens":{},"max_prompt_tokens":{},"deny_reason":{},"provider_order":{},"provider_weights":{},"timeout_secs":{}}}"#,
                            rule.rule_id,
                            rule.project_id,
                            rule.name,
                            rule.priority,
                            rule.enabled,
                            opt_json_string(&rule.match_path),
                            opt_json_string(&rule.match_model),
                            opt_json_bool(rule.match_streaming),
                            opt_json_string(&rule.match_role),
                            rule.match_headers.clone().unwrap_or_else(|| "null".to_string()),
                            opt_json_u32(rule.min_prompt_tokens),
                            opt_json_u32(rule.max_prompt_tokens),
                            opt_json_string(&rule.deny_reason),
                            rule.provider_order.clone().unwrap_or_else(|| "null".to_string()),
                            rule.provider_weights.clone().unwrap_or_else(|| "null".to_string()),
                            rule.timeout_secs.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            );
            json_response(StatusCode::OK, body)
        }
        (&Method::POST, Some("routing-rules")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageRoutingRules, Some(project_id))
            {
                return resp;
            }
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let record = RoutingRuleRecord {
                rule_id: extract_json_string(&body, "rule_id")
                    .unwrap_or_else(generate_routing_rule_id),
                project_id: project_id.to_string(),
                name: extract_json_string(&body, "name")
                    .unwrap_or_else(|| "routing rule".to_string()),
                priority: extract_json_float(&body, "priority").unwrap_or(0.0) as i32,
                enabled: extract_json_bool(&body, "enabled").unwrap_or(true),
                match_path: extract_json_string(&body, "match_path"),
                match_model: extract_json_string(&body, "match_model"),
                match_streaming: extract_json_bool(&body, "match_streaming"),
                match_role: extract_json_string(&body, "match_role"),
                match_headers: extract_json_raw(&body, "match_headers"),
                min_prompt_tokens: extract_json_float(&body, "min_prompt_tokens").map(|v| v as u32),
                max_prompt_tokens: extract_json_float(&body, "max_prompt_tokens").map(|v| v as u32),
                deny_reason: extract_json_string(&body, "deny_reason"),
                provider_order: extract_json_raw(&body, "provider_order"),
                provider_weights: extract_json_raw(&body, "provider_weights"),
                timeout_secs: extract_json_float(&body, "timeout_secs").map(|v| v as u64),
                created_at: current_timestamp_string(),
            };
            match api.upsert_routing_rule(record).await {
                Some(Ok(())) => json_response(StatusCode::CREATED, r#"{"ok":true}"#.to_string()),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::DELETE, Some(remainder)) if remainder.starts_with("routing-rules/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageRoutingRules, Some(project_id))
            {
                return resp;
            }
            let rule_id = remainder.trim_start_matches("routing-rules/");
            match api.delete_routing_rule(rule_id).await {
                Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Ok(false)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"rule not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("datasets")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            match api.list_project_datasets(Some(project_id)).await {
                Some(Ok(datasets)) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": project_id,
                        "datasets": datasets.iter().map(dataset_record_json).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some(remainder))
            if remainder.starts_with("datasets/") && remainder.contains("/items/") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let path = remainder.trim_start_matches("datasets/");
            let Some((dataset_name, item_id)) = path.split_once("/items/") else {
                return not_found();
            };
            match api
                .get_project_dataset_item(project_id, dataset_name, item_id)
                .await
            {
                Some(Ok(Some(item))) => {
                    json_response(StatusCode::OK, dataset_item_record_json(&item).to_string())
                }
                Some(Ok(None)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"dataset item not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::PUT, Some(remainder))
            if remainder.starts_with("datasets/") && remainder.contains("/items/") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let path = remainder.trim_start_matches("datasets/");
            let Some((dataset_name, item_id)) = path.split_once("/items/") else {
                return not_found();
            };
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload: ProjectDatasetItemPayload = match serde_json::from_str(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(r#"{{"error":"invalid dataset item payload: {}"}}"#, error),
                    )
                }
            };
            let input = match payload.input {
                Some(value) => value.to_string(),
                None => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        r#"{"error":"input is required"}"#.to_string(),
                    )
                }
            };
            let record = ProjectDatasetItemRecord {
                project_id: project_id.to_string(),
                dataset_name: dataset_name.to_string(),
                item_id: item_id.to_string(),
                input_json: input,
                expected_output_json: payload.expected_output.map(|value| value.to_string()),
                metadata_json: payload.metadata.map(|value| value.to_string()),
                updated_at: current_timestamp_string(),
            };
            match api.upsert_project_dataset_item(record).await {
                Some(Ok(())) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::DELETE, Some(remainder))
            if remainder.starts_with("datasets/") && remainder.contains("/items/") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let path = remainder.trim_start_matches("datasets/");
            let Some((dataset_name, item_id)) = path.split_once("/items/") else {
                return not_found();
            };
            match api
                .delete_project_dataset_item(project_id, dataset_name, item_id)
                .await
            {
                Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Ok(false)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"dataset item not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some(remainder))
            if remainder.starts_with("datasets/") && remainder.ends_with("/items") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let dataset_name = remainder
                .trim_start_matches("datasets/")
                .trim_end_matches("/items");
            match api.get_project_dataset(project_id, dataset_name).await {
                Some(Ok(Some(_))) => {}
                Some(Ok(None)) => {
                    return json_response(
                        StatusCode::NOT_FOUND,
                        r#"{"error":"dataset not found"}"#.to_string(),
                    )
                }
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
                None => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"store not enabled"}"#.to_string(),
                    )
                }
            }
            match api
                .list_project_dataset_items(project_id, dataset_name)
                .await
            {
                Some(Ok(items)) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": project_id,
                        "dataset_name": dataset_name,
                        "items": items.iter().map(dataset_item_record_json).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some(remainder)) if remainder.starts_with("datasets/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let dataset_name = remainder.trim_start_matches("datasets/");
            match api.get_project_dataset(project_id, dataset_name).await {
                Some(Ok(Some(dataset))) => {
                    json_response(StatusCode::OK, dataset_record_json(&dataset).to_string())
                }
                Some(Ok(None)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"dataset not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::PUT, Some(remainder)) if remainder.starts_with("datasets/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let dataset_name = remainder.trim_start_matches("datasets/");
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload: ProjectDatasetPayload = match serde_json::from_str(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(r#"{{"error":"invalid dataset payload: {}"}}"#, error),
                    )
                }
            };
            let record = ProjectDatasetRecord {
                project_id: project_id.to_string(),
                dataset_name: dataset_name.to_string(),
                description: payload.description,
                schema_json: payload.schema.map(|value| value.to_string()),
                updated_at: current_timestamp_string(),
            };
            match api.upsert_project_dataset(record).await {
                Some(Ok(())) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::DELETE, Some(remainder)) if remainder.starts_with("datasets/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let dataset_name = remainder.trim_start_matches("datasets/");
            match api.delete_project_dataset(project_id, dataset_name).await {
                Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Ok(false)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"dataset not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some(remainder))
            if remainder.starts_with("eval-runs/") && remainder.ends_with("/items") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let run_id = remainder
                .trim_start_matches("eval-runs/")
                .trim_end_matches("/items");
            match api.get_project_eval_run(project_id, run_id).await {
                Some(Ok(Some(_))) => {}
                Some(Ok(None)) => {
                    return json_response(
                        StatusCode::NOT_FOUND,
                        r#"{"error":"eval run not found"}"#.to_string(),
                    )
                }
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
                None => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"store not enabled"}"#.to_string(),
                    )
                }
            }
            match api.list_project_eval_run_items(project_id, run_id).await {
                Some(Ok(items)) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": project_id,
                        "run_id": run_id,
                        "items": items.iter().map(eval_run_item_record_json).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some(remainder)) if remainder.starts_with("eval-runs/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            if remainder == "eval-runs/compare" {
                let query = req.uri().query().unwrap_or("");
                let Some(baseline_run_id) = extract_query_param(query, "baseline_run_id") else {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        r#"{"error":"baseline_run_id is required"}"#.to_string(),
                    );
                };
                let Some(candidate_run_id) = extract_query_param(query, "candidate_run_id") else {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        r#"{"error":"candidate_run_id is required"}"#.to_string(),
                    );
                };
                let gate_overrides = match parse_eval_comparison_gate_request(query) {
                    Ok(request) => request,
                    Err(message) => return bad_request(&message),
                };
                let rollout_policy = match parse_rollout_policy_name(query) {
                    Some(policy_name) => match api
                        .get_project_rollout_policy(project_id, &policy_name)
                        .await
                    {
                        Some(Ok(Some(policy))) => Some(policy),
                        Some(Ok(None)) => {
                            return json_response(
                                StatusCode::NOT_FOUND,
                                r#"{"error":"rollout policy not found"}"#.to_string(),
                            )
                        }
                        Some(Err(error)) => {
                            return json_response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!(r#"{{"error":"{}"}}"#, error),
                            )
                        }
                        None => {
                            return json_response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                r#"{"error":"store not enabled"}"#.to_string(),
                            )
                        }
                    },
                    None => None,
                };
                let gate_request = match rollout_policy.as_ref() {
                    Some(policy) => match deserialize_rollout_policy_gate(policy) {
                        Ok(gate) => Some(match gate_overrides.as_ref() {
                            Some(overrides) => gate.merged_with(overrides),
                            None => gate,
                        }),
                        Err(message) => return bad_request(&message),
                    },
                    None => gate_overrides,
                };
                return match api
                    .compare_project_eval_runs(
                        project_id,
                        &baseline_run_id,
                        &candidate_run_id,
                        gate_request,
                    )
                    .await
                {
                    Some(Ok(comparison)) => json_response(
                        StatusCode::OK,
                        eval_run_comparison_json_with_rollout_policy(
                            &comparison,
                            rollout_policy.as_ref(),
                        )
                        .to_string(),
                    ),
                    Some(Err(error)) => json_response(
                        StatusCode::BAD_REQUEST,
                        serde_json::json!({ "error": error.to_string() }).to_string(),
                    ),
                    None => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"store not enabled"}"#.to_string(),
                    ),
                };
            }
            let run_id = remainder.trim_start_matches("eval-runs/");
            match api.get_project_eval_run(project_id, run_id).await {
                Some(Ok(Some(run))) => {
                    json_response(StatusCode::OK, eval_run_record_json(&run).to_string())
                }
                Some(Ok(None)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"eval run not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::POST, Some("eval-runs")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload: ProjectEvalRunPayload = match serde_json::from_str(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(r#"{{"error":"invalid eval run payload: {}"}}"#, error),
                    )
                }
            };
            let Some(dataset_name) = payload.dataset_name else {
                return json_response(
                    StatusCode::BAD_REQUEST,
                    r#"{"error":"dataset_name is required"}"#.to_string(),
                );
            };
            let Some(target_url) = payload.target_url else {
                return json_response(
                    StatusCode::BAD_REQUEST,
                    r#"{"error":"target_url is required"}"#.to_string(),
                );
            };
            let run_async = payload.run_async.unwrap_or(false);
            let request = ProjectEvalRunRequest {
                dataset_name,
                target_url,
                headers: payload.headers.unwrap_or_default(),
                timeout_ms: payload.timeout_ms,
                judge_url: payload.judge_url,
                judge_kind: payload.judge_kind,
                judge_model: payload.judge_model,
                judge_headers: payload.judge_headers.unwrap_or_default(),
                judge_timeout_ms: payload.judge_timeout_ms,
                prompt_name: payload.prompt_name,
                prompt_version: payload.prompt_version,
                provider_name: payload.provider_name,
                model: payload.model,
                route_path: payload.route_path,
                safety_profile: payload.safety_profile,
            };
            if run_async {
                match api.queue_project_eval_run(project_id, request).await {
                    Some(Ok(run)) => json_response(
                        StatusCode::ACCEPTED,
                        serde_json::json!({
                            "run": eval_run_record_json(&run),
                            "queued": true,
                        })
                        .to_string(),
                    ),
                    Some(Err(error)) => json_response(
                        StatusCode::BAD_REQUEST,
                        serde_json::json!({ "error": error.to_string() }).to_string(),
                    ),
                    None => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"store not enabled"}"#.to_string(),
                    ),
                }
            } else {
                match api.execute_project_eval_run(project_id, request).await {
                    Some(Ok(execution)) => json_response(
                        StatusCode::OK,
                        serde_json::json!({
                            "run": eval_run_record_json(&execution.run),
                            "items": execution
                                .items
                                .iter()
                                .map(eval_run_item_record_json)
                                .collect::<Vec<_>>(),
                        })
                        .to_string(),
                    ),
                    Some(Err(error)) => json_response(
                        StatusCode::BAD_REQUEST,
                        serde_json::json!({ "error": error.to_string() }).to_string(),
                    ),
                    None => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"store not enabled"}"#.to_string(),
                    ),
                }
            }
        }
        (&Method::GET, Some("eval-runs")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let dataset_name = extract_query_param(req.uri().query().unwrap_or(""), "dataset_name");
            match api
                .list_project_eval_runs(project_id, dataset_name.as_deref())
                .await
            {
                Some(Ok(runs)) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": project_id,
                        "runs": runs.iter().map(eval_run_record_json).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("tools")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let tools = api.list_project_tools(Some(project_id)).unwrap_or_default();
            let body = serde_json::json!({
                "project_id": project_id,
                "tools": tools.into_iter().map(|tool| {
                    serde_json::json!({
                        "project_id": tool.project_id,
                        "tool_name": tool.tool_name,
                        "description": tool.description,
                        "input_schema": serde_json::from_str::<serde_json::Value>(&tool.input_schema_json).unwrap_or(serde_json::Value::Null),
                        "executor_kind": tool.executor_kind,
                        "executor_config": tool.executor_config_json.as_ref().and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok()).unwrap_or(serde_json::Value::Null),
                        "enabled": tool.enabled,
                        "timeout_ms": tool.timeout_ms,
                        "updated_at": tool.updated_at,
                    })
                }).collect::<Vec<_>>(),
            });
            json_response(StatusCode::OK, body.to_string())
        }
        (&Method::GET, Some(remainder)) if remainder.starts_with("tools/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let tool_name = remainder.trim_start_matches("tools/");
            match api.get_project_tool(project_id, tool_name) {
                Some(tool) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": tool.project_id,
                        "tool_name": tool.tool_name,
                        "description": tool.description,
                        "input_schema": serde_json::from_str::<serde_json::Value>(&tool.input_schema_json).unwrap_or(serde_json::Value::Null),
                        "executor_kind": tool.executor_kind,
                        "executor_config": tool.executor_config_json.as_ref().and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok()).unwrap_or(serde_json::Value::Null),
                        "enabled": tool.enabled,
                        "timeout_ms": tool.timeout_ms,
                        "updated_at": tool.updated_at,
                    })
                    .to_string(),
                ),
                None => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"tool not found"}"#.to_string(),
                ),
            }
        }
        (&Method::PUT, Some(remainder)) if remainder.starts_with("tools/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let tool_name = remainder.trim_start_matches("tools/");
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload: ProjectToolPayload = match serde_json::from_str(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(r#"{{"error":"invalid tool payload: {}"}}"#, error),
                    )
                }
            };
            let input_schema = match payload.input_schema {
                Some(schema) => schema.to_string(),
                None => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        r#"{"error":"input_schema is required"}"#.to_string(),
                    )
                }
            };
            let executor_kind = match payload.executor_kind {
                Some(kind) if !kind.trim().is_empty() => kind,
                _ => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        r#"{"error":"executor_kind is required"}"#.to_string(),
                    )
                }
            };
            let record = ProjectToolRecord {
                project_id: project_id.to_string(),
                tool_name: tool_name.to_string(),
                description: payload.description,
                input_schema_json: input_schema,
                executor_kind,
                executor_config_json: payload.executor_config.map(|value| value.to_string()),
                enabled: payload.enabled.unwrap_or(true),
                timeout_ms: payload.timeout_ms,
                updated_at: current_timestamp_string(),
            };
            match api.upsert_project_tool(record).await {
                Some(Ok(())) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::DELETE, Some(remainder)) if remainder.starts_with("tools/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let tool_name = remainder.trim_start_matches("tools/");
            match api.delete_project_tool(project_id, tool_name).await {
                Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Ok(false)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"tool not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("rollout-policies")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            match api.list_project_rollout_policies(Some(project_id)).await {
                Some(Ok(policies)) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": project_id,
                        "policies": policies.iter().map(rollout_policy_record_json).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some(remainder)) if remainder.starts_with("rollout-policies/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let policy_name = remainder.trim_start_matches("rollout-policies/");
            match api
                .get_project_rollout_policy(project_id, policy_name)
                .await
            {
                Some(Ok(Some(record))) => json_response(
                    StatusCode::OK,
                    rollout_policy_record_json(&record).to_string(),
                ),
                Some(Ok(None)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"rollout policy not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::PUT, Some(remainder)) if remainder.starts_with("rollout-policies/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let policy_name = remainder.trim_start_matches("rollout-policies/").trim();
            if policy_name.is_empty() {
                return bad_request("policy_name is required");
            }
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload: ProjectRolloutPolicyPayload = match serde_json::from_str(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(r#"{{"error":"invalid rollout policy payload: {}"}}"#, error),
                    )
                }
            };
            let Some(gate) = payload.gate.filter(|gate| !gate.is_empty()) else {
                return bad_request("gate is required");
            };
            let canary_config = match payload.canary {
                Some(canary) => {
                    let steps = canary.steps.unwrap_or_default();
                    if steps.is_empty() {
                        return bad_request("rollout policy canary.steps must not be empty");
                    }
                    if steps.contains(&0) {
                        return bad_request(
                            "rollout policy canary.steps entries must be between 1 and 100",
                        );
                    }
                    if steps.windows(2).any(|window| window[0] >= window[1]) {
                        return bad_request(
                            "rollout policy canary.steps must be strictly increasing",
                        );
                    }
                    Some(RolloutCanaryPolicyConfig {
                        steps,
                        auto_promote_final: canary.auto_promote_final.unwrap_or(false),
                        auto_advance_on_pass: canary.auto_advance_on_pass.unwrap_or(false),
                        auto_rollback_on_fail: canary.auto_rollback_on_fail.unwrap_or(false),
                    })
                }
                None => None,
            };
            let gate_config_json = match encode_rollout_policy_config(&gate, canary_config.as_ref())
            {
                Ok(json) => json,
                Err(message) => return bad_request(&message),
            };
            let record = ProjectRolloutPolicyRecord {
                project_id: project_id.to_string(),
                policy_name: policy_name.to_string(),
                description: payload.description,
                gate_config_json,
                target_environment: payload
                    .target_environment
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string),
                updated_at: current_timestamp_string(),
            };
            match api.upsert_project_rollout_policy(record).await {
                Some(Ok(())) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::DELETE, Some(remainder)) if remainder.starts_with("rollout-policies/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let policy_name = remainder.trim_start_matches("rollout-policies/");
            match api
                .delete_project_rollout_policy(project_id, policy_name)
                .await
            {
                Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Ok(false)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"rollout policy not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some(remainder))
            if remainder.starts_with("prompts/") && remainder.ends_with("/rollouts") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some(prompt_name) = remainder
                .strip_prefix("prompts/")
                .and_then(|value| value.strip_suffix("/rollouts"))
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return not_found();
            };
            match api.list_project_prompt_rollouts(project_id, prompt_name).await {
                Some(Ok(rollouts)) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": project_id,
                        "prompt_name": prompt_name,
                        "rollouts": rollouts.iter().map(prompt_rollout_record_json).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some(remainder))
            if remainder.starts_with("prompts/") && remainder.contains("/rollouts/") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some(path) = remainder.strip_prefix("prompts/") else {
                return not_found();
            };
            let Some((prompt_name, rollout_id)) = path.split_once("/rollouts/") else {
                return not_found();
            };
            if prompt_name.trim().is_empty() || rollout_id.trim().is_empty() {
                return not_found();
            }
            match api
                .get_project_prompt_rollout(project_id, prompt_name, rollout_id)
                .await
            {
                Some(Ok(Some(record))) => json_response(
                    StatusCode::OK,
                    prompt_rollout_record_json(&record).to_string(),
                ),
                Some(Ok(None)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"prompt rollout not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::POST, Some(remainder))
            if remainder.starts_with("prompts/") && remainder.ends_with("/rollouts") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some(prompt_name) = remainder
                .strip_prefix("prompts/")
                .and_then(|value| value.strip_suffix("/rollouts"))
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return not_found();
            };
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload: ProjectPromptRolloutPayload = match serde_json::from_str(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(r#"{{"error":"invalid prompt rollout payload: {}"}}"#, error),
                    )
                }
            };
            let Some(candidate_version) = payload
                .candidate_version
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return bad_request("candidate_version is required");
            };
            let Some(baseline_run_id) = payload
                .baseline_run_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return bad_request("baseline_run_id is required");
            };
            let Some(candidate_run_id) = payload
                .candidate_run_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return bad_request("candidate_run_id is required");
            };
            let Some(policy_name) = payload
                .policy_name
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return bad_request("policy_name is required");
            };
            let decision = match prepare_prompt_rollout_decision(
                api,
                project_id,
                prompt_name,
                candidate_version,
                baseline_run_id,
                candidate_run_id,
                policy_name,
            )
            .await
            {
                Ok(decision) => decision,
                Err(resp) => return resp,
            };
            let comparison_json = eval_run_comparison_json_with_rollout_policy(
                &decision.comparison,
                Some(&decision.policy),
            );
            let gate = decision.comparison.gate.as_ref().expect("gate checked");
            let rollout = ProjectPromptRolloutRecord {
                project_id: project_id.to_string(),
                prompt_name: prompt_name.to_string(),
                rollout_id: generate_prompt_rollout_id(),
                policy_name: decision.policy.policy_name.clone(),
                baseline_version: prompt_context_field(
                    &decision.comparison.baseline_run,
                    "prompt_version",
                ),
                candidate_version: decision.candidate_prompt.version.clone(),
                baseline_run_id: decision.comparison.baseline_run.run_id.clone(),
                candidate_run_id: decision.comparison.candidate_run.run_id.clone(),
                target_environment: decision
                    .policy
                    .target_environment
                    .clone()
                    .or(Some(decision.candidate_prompt.environment.clone())),
                status: if gate.passed { "ready" } else { "blocked" }.to_string(),
                recommendation_action: Some(gate.recommendation.action.as_str().to_string()),
                comparison_json: comparison_json.to_string(),
                created_at: current_timestamp_string(),
                applied_at: None,
            };
            match api.upsert_project_prompt_rollout(rollout.clone()).await {
                Some(Ok(())) => json_response(
                    StatusCode::CREATED,
                    serde_json::json!({
                        "created": true,
                        "rollout": prompt_rollout_record_json(&rollout),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"store not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::POST, Some(remainder))
            if remainder.starts_with("prompts/") && remainder.ends_with("/apply") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some(path) = remainder.strip_prefix("prompts/") else {
                return not_found();
            };
            let Some((prompt_name, rollout_id)) = path
                .strip_suffix("/apply")
                .and_then(|value| value.split_once("/rollouts/"))
            else {
                return not_found();
            };
            if prompt_name.trim().is_empty() || rollout_id.trim().is_empty() {
                return not_found();
            }
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let apply_payload = if body.trim().is_empty() {
                PromptRolloutApplyPayload::default()
            } else {
                match serde_json::from_str::<PromptRolloutApplyPayload>(&body) {
                    Ok(payload) => payload,
                    Err(error) => {
                        return json_response(
                            StatusCode::BAD_REQUEST,
                            format!(
                                r#"{{"error":"invalid prompt rollout apply payload: {}"}}"#,
                                error
                            ),
                        )
                    }
                }
            };
            let apply_mode = apply_payload
                .mode
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("promote");
            if apply_mode != "promote" && apply_mode != "canary" {
                return bad_request("prompt rollout apply mode must be 'promote' or 'canary'");
            }
            let mut rollout = match api
                .get_project_prompt_rollout(project_id, prompt_name, rollout_id)
                .await
            {
                Some(Ok(Some(record))) => record,
                Some(Ok(None)) => {
                    return json_response(
                        StatusCode::NOT_FOUND,
                        r#"{"error":"prompt rollout not found"}"#.to_string(),
                    )
                }
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
                None => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"store not enabled"}"#.to_string(),
                    )
                }
            };
            let mut candidate_prompt =
                match api.get_project_prompt(project_id, prompt_name, &rollout.candidate_version) {
                    Some(record) => record,
                    None => {
                        return json_response(
                            StatusCode::NOT_FOUND,
                            r#"{"error":"candidate prompt version not found"}"#.to_string(),
                        )
                    }
                };
            if rollout.status == "blocked" {
                return json_response(
                    StatusCode::CONFLICT,
                    serde_json::json!({
                        "applied": false,
                        "rollout": prompt_rollout_record_json(&rollout),
                    })
                    .to_string(),
                );
            }
            if rollout.status == "applied" && apply_mode == "promote" {
                return json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "applied": true,
                        "mode": "promote",
                        "prompt": prompt_record_json(&candidate_prompt),
                        "rollout": prompt_rollout_record_json(&rollout),
                    })
                    .to_string(),
                );
            }
            if apply_mode == "canary" {
                let canary_policy = if apply_payload.traffic_percent.is_none() {
                    match load_rollout_canary_policy(api, project_id, &rollout.policy_name).await {
                        Ok(policy) => policy,
                        Err(resp) => return resp,
                    }
                } else {
                    None
                };
                let runtime_config = match canary_apply_runtime_config(
                    apply_payload.traffic_percent,
                    canary_policy.as_ref(),
                ) {
                    Ok(runtime_config) => runtime_config,
                    Err(message) => return bad_request(&message),
                };
                let Some(baseline_version) = rollout
                    .baseline_version
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                else {
                    return bad_request(
                        "prompt rollout is missing baseline_version for canary apply",
                    );
                };
                let baseline_prompt =
                    match api.get_project_prompt(project_id, prompt_name, baseline_version) {
                        Some(record) => record,
                        None => {
                            return json_response(
                                StatusCode::NOT_FOUND,
                                r#"{"error":"baseline prompt version not found"}"#.to_string(),
                            )
                        }
                    };
                if baseline_prompt.environment != candidate_prompt.environment {
                    return bad_request(
                        "baseline and candidate prompt environments must match for canary apply",
                    );
                }
                if !baseline_prompt.active {
                    return json_response(
                        StatusCode::CONFLICT,
                        serde_json::json!({
                            "applied": false,
                            "error": "baseline prompt version is not currently active",
                            "rollout": prompt_rollout_record_json(&rollout),
                        })
                        .to_string(),
                    );
                }
                let applied_at = current_timestamp_string();
                rollout.status = "applied_canary".to_string();
                rollout.applied_at = Some(applied_at);
                rollout.comparison_json = update_prompt_rollout_runtime_config(
                    &rollout.comparison_json,
                    Some(&runtime_config),
                );
                if let Err(error) = supersede_live_prompt_canaries(
                    api,
                    project_id,
                    prompt_name,
                    &rollout.rollout_id,
                    rollout.target_environment.as_deref(),
                )
                .await
                {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    );
                }
                match api.upsert_project_prompt_rollout(rollout.clone()).await {
                    Some(Ok(())) => json_response(
                        StatusCode::OK,
                        serde_json::json!({
                            "applied": true,
                            "mode": "canary",
                            "baseline_prompt": prompt_record_json(&baseline_prompt),
                            "prompt": prompt_record_json(&candidate_prompt),
                            "rollout": prompt_rollout_record_json(&rollout),
                        })
                        .to_string(),
                    ),
                    Some(Err(error)) => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    ),
                    None => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"governance not enabled"}"#.to_string(),
                    ),
                }
            } else {
                let applied_at = current_timestamp_string();
                candidate_prompt.active = true;
                candidate_prompt.updated_at = applied_at.clone();
                match api.upsert_project_prompt(candidate_prompt.clone()).await {
                    Some(Ok(())) => {}
                    Some(Err(error)) => {
                        return json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!(r#"{{"error":"{}"}}"#, error),
                        )
                    }
                    None => {
                        return json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            r#"{"error":"governance not enabled"}"#.to_string(),
                        )
                    }
                }
                if let Err(error) = supersede_live_prompt_canaries(
                    api,
                    project_id,
                    prompt_name,
                    &rollout.rollout_id,
                    rollout.target_environment.as_deref(),
                )
                .await
                {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    );
                }
                rollout.status = "applied".to_string();
                rollout.applied_at = Some(applied_at);
                rollout.comparison_json =
                    update_prompt_rollout_runtime_config(&rollout.comparison_json, None);
                match api.upsert_project_prompt_rollout(rollout.clone()).await {
                    Some(Ok(())) => json_response(
                        StatusCode::OK,
                        serde_json::json!({
                            "applied": true,
                            "mode": "promote",
                            "prompt": prompt_record_json(&candidate_prompt),
                            "rollout": prompt_rollout_record_json(&rollout),
                        })
                        .to_string(),
                    ),
                    Some(Err(error)) => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    ),
                    None => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"governance not enabled"}"#.to_string(),
                    ),
                }
            }
        }
        (&Method::POST, Some(remainder))
            if remainder.starts_with("prompts/") && remainder.ends_with("/advance") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some(path) = remainder.strip_prefix("prompts/") else {
                return not_found();
            };
            let Some((prompt_name, rollout_id)) = path
                .strip_suffix("/advance")
                .and_then(|value| value.split_once("/rollouts/"))
            else {
                return not_found();
            };
            if prompt_name.trim().is_empty() || rollout_id.trim().is_empty() {
                return not_found();
            }
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload = if body.trim().is_empty() {
                PromptRolloutAdvancePayload::default()
            } else {
                match serde_json::from_str::<PromptRolloutAdvancePayload>(&body) {
                    Ok(payload) => payload,
                    Err(error) => {
                        return json_response(
                            StatusCode::BAD_REQUEST,
                            format!(
                                r#"{{"error":"invalid prompt rollout advance payload: {}"}}"#,
                                error
                            ),
                        )
                    }
                }
            };
            let mut rollout = match api
                .get_project_prompt_rollout(project_id, prompt_name, rollout_id)
                .await
            {
                Some(Ok(Some(record))) => record,
                Some(Ok(None)) => {
                    return json_response(
                        StatusCode::NOT_FOUND,
                        r#"{"error":"prompt rollout not found"}"#.to_string(),
                    )
                }
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
                None => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"governance not enabled"}"#.to_string(),
                    )
                }
            };
            if rollout.status != "applied_canary" {
                return json_response(
                    StatusCode::CONFLICT,
                    serde_json::json!({
                        "advanced": false,
                        "error": "prompt rollout is not currently in canary mode",
                        "rollout": prompt_rollout_record_json(&rollout),
                    })
                    .to_string(),
                );
            }
            let current_runtime =
                match parse_prompt_rollout_runtime_config(&rollout.comparison_json) {
                    Some(runtime) => runtime,
                    None => {
                        return bad_request("prompt rollout is missing runtime canary metadata")
                    }
                };
            let canary_policy = if payload.traffic_percent.is_none() {
                match load_rollout_canary_policy(api, project_id, &rollout.policy_name).await {
                    Ok(policy) => policy,
                    Err(resp) => return resp,
                }
            } else {
                None
            };
            let next_runtime = match next_canary_runtime_config(
                &current_runtime,
                payload.traffic_percent,
                canary_policy.as_ref(),
            ) {
                Ok(runtime) => runtime,
                Err(message) => return bad_request(&message),
            };
            let auto_promote_final = canary_policy
                .as_ref()
                .map(|policy| policy.auto_promote_final)
                .unwrap_or(false)
                && next_runtime.traffic_percent == 100;
            let mut candidate_prompt =
                match api.get_project_prompt(project_id, prompt_name, &rollout.candidate_version) {
                    Some(record) => record,
                    None => {
                        return json_response(
                            StatusCode::NOT_FOUND,
                            r#"{"error":"candidate prompt version not found"}"#.to_string(),
                        )
                    }
                };
            if auto_promote_final {
                let applied_at = current_timestamp_string();
                candidate_prompt.active = true;
                candidate_prompt.updated_at = applied_at.clone();
                match api.upsert_project_prompt(candidate_prompt.clone()).await {
                    Some(Ok(())) => {}
                    Some(Err(error)) => {
                        return json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!(r#"{{"error":"{}"}}"#, error),
                        )
                    }
                    None => {
                        return json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            r#"{"error":"governance not enabled"}"#.to_string(),
                        )
                    }
                }
                rollout.status = "applied".to_string();
                rollout.applied_at = Some(applied_at);
                rollout.comparison_json =
                    update_prompt_rollout_runtime_config(&rollout.comparison_json, None);
                match api.upsert_project_prompt_rollout(rollout.clone()).await {
                    Some(Ok(())) => json_response(
                        StatusCode::OK,
                        serde_json::json!({
                            "advanced": true,
                            "promoted": true,
                            "mode": "promote",
                            "prompt": prompt_record_json(&candidate_prompt),
                            "rollout": prompt_rollout_record_json(&rollout),
                        })
                        .to_string(),
                    ),
                    Some(Err(error)) => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    ),
                    None => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"governance not enabled"}"#.to_string(),
                    ),
                }
            } else {
                rollout.comparison_json = update_prompt_rollout_runtime_config(
                    &rollout.comparison_json,
                    Some(&next_runtime),
                );
                match api.upsert_project_prompt_rollout(rollout.clone()).await {
                    Some(Ok(())) => json_response(
                        StatusCode::OK,
                        serde_json::json!({
                            "advanced": true,
                            "promoted": false,
                            "mode": "canary",
                            "rollout": prompt_rollout_record_json(&rollout),
                        })
                        .to_string(),
                    ),
                    Some(Err(error)) => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    ),
                    None => json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"governance not enabled"}"#.to_string(),
                    ),
                }
            }
        }
        (&Method::POST, Some(remainder))
            if remainder.starts_with("prompts/") && remainder.ends_with("/evaluate") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some(path) = remainder.strip_prefix("prompts/") else {
                return not_found();
            };
            let Some((prompt_name, rollout_id)) = path
                .strip_suffix("/evaluate")
                .and_then(|value| value.split_once("/rollouts/"))
            else {
                return not_found();
            };
            if prompt_name.trim().is_empty() || rollout_id.trim().is_empty() {
                return not_found();
            }
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload = match serde_json::from_str::<PromptRolloutEvaluatePayload>(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(
                            r#"{{"error":"invalid prompt rollout evaluate payload: {}"}}"#,
                            error
                        ),
                    )
                }
            };
            let baseline_run_id = payload
                .baseline_run_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string);
            let candidate_run_id = payload
                .candidate_run_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string);
            let (Some(baseline_run_id), Some(candidate_run_id)) =
                (baseline_run_id, candidate_run_id)
            else {
                return bad_request("baseline_run_id and candidate_run_id are required");
            };
            let mut rollout = match api
                .get_project_prompt_rollout(project_id, prompt_name, rollout_id)
                .await
            {
                Some(Ok(Some(record))) => record,
                Some(Ok(None)) => {
                    return json_response(
                        StatusCode::NOT_FOUND,
                        r#"{"error":"prompt rollout not found"}"#.to_string(),
                    )
                }
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
                None => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"governance not enabled"}"#.to_string(),
                    )
                }
            };
            if rollout.status != "applied_canary" {
                return json_response(
                    StatusCode::CONFLICT,
                    serde_json::json!({
                        "evaluated": false,
                        "error": "prompt rollout is not currently in canary mode",
                        "rollout": prompt_rollout_record_json(&rollout),
                    })
                    .to_string(),
                );
            }
            let current_runtime =
                match parse_prompt_rollout_runtime_config(&rollout.comparison_json) {
                    Some(runtime) => runtime,
                    None => {
                        return bad_request("prompt rollout is missing runtime canary metadata")
                    }
                };
            let PreparedPromptRolloutDecision {
                policy, comparison, ..
            } = match prepare_prompt_rollout_decision(
                api,
                project_id,
                prompt_name,
                &rollout.candidate_version,
                &baseline_run_id,
                &candidate_run_id,
                &rollout.policy_name,
            )
            .await
            {
                Ok(decision) => decision,
                Err(resp) => return resp,
            };
            if let Err(message) = validate_prompt_rollout_evaluation_context(
                prompt_name,
                rollout.baseline_version.as_deref(),
                &rollout.candidate_version,
                &comparison,
            ) {
                return bad_request(&message);
            }
            let canary_policy =
                match load_rollout_canary_policy(api, project_id, &rollout.policy_name).await {
                    Ok(policy) => policy,
                    Err(resp) => return resp,
                };
            let Some(gate) = comparison.gate.as_ref() else {
                return json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"rollout policy did not produce a gate result"}"#.to_string(),
                );
            };
            let automation_action = determine_prompt_rollout_automation_action(
                &current_runtime,
                gate.passed,
                canary_policy.as_ref(),
            );
            let action_name = automation_action.action_name().to_string();
            let action_reason = automation_action.reason().map(ToString::to_string);
            let comparison_json =
                eval_run_comparison_json_with_rollout_policy(&comparison, Some(&policy));
            let evaluated_at = current_timestamp_string();

            rollout.baseline_run_id = baseline_run_id.clone();
            rollout.candidate_run_id = candidate_run_id.clone();
            rollout.recommendation_action = Some(gate.recommendation.action.as_str().to_string());

            match automation_action {
                PromptRolloutAutomationAction::Hold(_) => {
                    rollout.comparison_json = update_prompt_rollout_canary_evaluation(
                        &rollout.comparison_json,
                        Some(&current_runtime),
                        &baseline_run_id,
                        &candidate_run_id,
                        comparison_json.clone(),
                        &action_name,
                        false,
                        action_reason.as_deref(),
                        &evaluated_at,
                    );
                    match api.upsert_project_prompt_rollout(rollout.clone()).await {
                        Some(Ok(())) => json_response(
                            StatusCode::OK,
                            serde_json::json!({
                                "evaluated": true,
                                "gate_passed": gate.passed,
                                "action": action_name,
                                "applied": false,
                                "reason": action_reason,
                                "comparison": comparison_json,
                                "rollout": prompt_rollout_record_json(&rollout),
                            })
                            .to_string(),
                        ),
                        Some(Err(error)) => json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!(r#"{{"error":"{}"}}"#, error),
                        ),
                        None => json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            r#"{"error":"governance not enabled"}"#.to_string(),
                        ),
                    }
                }
                PromptRolloutAutomationAction::Advance(next_runtime) => {
                    rollout.comparison_json = update_prompt_rollout_canary_evaluation(
                        &rollout.comparison_json,
                        Some(&next_runtime),
                        &baseline_run_id,
                        &candidate_run_id,
                        comparison_json.clone(),
                        &action_name,
                        true,
                        None,
                        &evaluated_at,
                    );
                    match api.upsert_project_prompt_rollout(rollout.clone()).await {
                        Some(Ok(())) => json_response(
                            StatusCode::OK,
                            serde_json::json!({
                                "evaluated": true,
                                "gate_passed": gate.passed,
                                "action": action_name,
                                "applied": true,
                                "mode": "canary",
                                "comparison": comparison_json,
                                "rollout": prompt_rollout_record_json(&rollout),
                            })
                            .to_string(),
                        ),
                        Some(Err(error)) => json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!(r#"{{"error":"{}"}}"#, error),
                        ),
                        None => json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            r#"{"error":"governance not enabled"}"#.to_string(),
                        ),
                    }
                }
                PromptRolloutAutomationAction::Promote => {
                    let mut candidate_prompt = match api.get_project_prompt(
                        project_id,
                        prompt_name,
                        &rollout.candidate_version,
                    ) {
                        Some(record) => record,
                        None => {
                            return json_response(
                                StatusCode::NOT_FOUND,
                                r#"{"error":"candidate prompt version not found"}"#.to_string(),
                            )
                        }
                    };
                    candidate_prompt.active = true;
                    candidate_prompt.updated_at = evaluated_at.clone();
                    match api.upsert_project_prompt(candidate_prompt.clone()).await {
                        Some(Ok(())) => {}
                        Some(Err(error)) => {
                            return json_response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!(r#"{{"error":"{}"}}"#, error),
                            )
                        }
                        None => {
                            return json_response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                r#"{"error":"governance not enabled"}"#.to_string(),
                            )
                        }
                    }
                    rollout.status = "applied".to_string();
                    rollout.applied_at = Some(evaluated_at.clone());
                    rollout.comparison_json = update_prompt_rollout_canary_evaluation(
                        &rollout.comparison_json,
                        None,
                        &baseline_run_id,
                        &candidate_run_id,
                        comparison_json.clone(),
                        &action_name,
                        true,
                        None,
                        &evaluated_at,
                    );
                    match api.upsert_project_prompt_rollout(rollout.clone()).await {
                        Some(Ok(())) => json_response(
                            StatusCode::OK,
                            serde_json::json!({
                                "evaluated": true,
                                "gate_passed": gate.passed,
                                "action": action_name,
                                "applied": true,
                                "mode": "promote",
                                "comparison": comparison_json,
                                "prompt": prompt_record_json(&candidate_prompt),
                                "rollout": prompt_rollout_record_json(&rollout),
                            })
                            .to_string(),
                        ),
                        Some(Err(error)) => json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!(r#"{{"error":"{}"}}"#, error),
                        ),
                        None => json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            r#"{"error":"governance not enabled"}"#.to_string(),
                        ),
                    }
                }
                PromptRolloutAutomationAction::Rollback => {
                    rollout.status = "rolled_back".to_string();
                    rollout.comparison_json = update_prompt_rollout_canary_evaluation(
                        &rollout.comparison_json,
                        None,
                        &baseline_run_id,
                        &candidate_run_id,
                        comparison_json.clone(),
                        &action_name,
                        true,
                        None,
                        &evaluated_at,
                    );
                    match api.upsert_project_prompt_rollout(rollout.clone()).await {
                        Some(Ok(())) => json_response(
                            StatusCode::OK,
                            serde_json::json!({
                                "evaluated": true,
                                "gate_passed": gate.passed,
                                "action": action_name,
                                "applied": true,
                                "mode": "rollback",
                                "comparison": comparison_json,
                                "rollout": prompt_rollout_record_json(&rollout),
                            })
                            .to_string(),
                        ),
                        Some(Err(error)) => json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!(r#"{{"error":"{}"}}"#, error),
                        ),
                        None => json_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            r#"{"error":"governance not enabled"}"#.to_string(),
                        ),
                    }
                }
            }
        }
        (&Method::POST, Some(remainder))
            if remainder.starts_with("prompts/") && remainder.ends_with("/rollback") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some(path) = remainder.strip_prefix("prompts/") else {
                return not_found();
            };
            let Some((prompt_name, rollout_id)) = path
                .strip_suffix("/rollback")
                .and_then(|value| value.split_once("/rollouts/"))
            else {
                return not_found();
            };
            if prompt_name.trim().is_empty() || rollout_id.trim().is_empty() {
                return not_found();
            }
            let mut rollout = match api
                .get_project_prompt_rollout(project_id, prompt_name, rollout_id)
                .await
            {
                Some(Ok(Some(record))) => record,
                Some(Ok(None)) => {
                    return json_response(
                        StatusCode::NOT_FOUND,
                        r#"{"error":"prompt rollout not found"}"#.to_string(),
                    )
                }
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
                None => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"governance not enabled"}"#.to_string(),
                    )
                }
            };
            if rollout.status == "rolled_back" {
                return json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "rolled_back": true,
                        "rollout": prompt_rollout_record_json(&rollout),
                    })
                    .to_string(),
                );
            }
            if rollout.status != "applied_canary" {
                return json_response(
                    StatusCode::CONFLICT,
                    serde_json::json!({
                        "rolled_back": false,
                        "error": "prompt rollout is not currently in canary mode",
                        "rollout": prompt_rollout_record_json(&rollout),
                    })
                    .to_string(),
                );
            }
            rollout.status = "rolled_back".to_string();
            rollout.comparison_json =
                update_prompt_rollout_runtime_config(&rollout.comparison_json, None);
            match api.upsert_project_prompt_rollout(rollout.clone()).await {
                Some(Ok(())) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "rolled_back": true,
                        "rollout": prompt_rollout_record_json(&rollout),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("prompts")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let prompts = api
                .list_project_prompts(Some(project_id), None)
                .unwrap_or_default();
            let body = serde_json::json!({
                "project_id": project_id,
                "prompts": prompts.iter().map(prompt_record_json).collect::<Vec<_>>(),
            });
            json_response(StatusCode::OK, body.to_string())
        }
        (&Method::GET, Some(remainder))
            if remainder.starts_with("prompts/") && !remainder.contains("/versions/") =>
        {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let prompt_name = remainder.trim_start_matches("prompts/");
            let prompts = api
                .list_project_prompts(Some(project_id), Some(prompt_name))
                .unwrap_or_default();
            if prompts.is_empty() {
                json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"prompt not found"}"#.to_string(),
                )
            } else {
                json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": project_id,
                        "prompt_name": prompt_name,
                        "versions": prompts.iter().map(prompt_record_json).collect::<Vec<_>>(),
                    })
                    .to_string(),
                )
            }
        }
        (&Method::POST, Some(remainder)) if remainder.starts_with("prompts/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some(prompt_name) = remainder
                .trim_start_matches("prompts/")
                .strip_suffix("/promote")
            else {
                return not_found();
            };
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload: PromptPromotionPayload = match serde_json::from_str(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(
                            r#"{{"error":"invalid prompt promotion payload: {}"}}"#,
                            error
                        ),
                    )
                }
            };
            let Some(candidate_version) = payload
                .candidate_version
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return bad_request("candidate_version is required");
            };
            let Some(baseline_run_id) = payload
                .baseline_run_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return bad_request("baseline_run_id is required");
            };
            let Some(candidate_run_id) = payload
                .candidate_run_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return bad_request("candidate_run_id is required");
            };
            let Some(policy_name) = payload
                .policy_name
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return bad_request("policy_name is required");
            };
            let policy = match api
                .get_project_rollout_policy(project_id, policy_name)
                .await
            {
                Some(Ok(Some(policy))) => policy,
                Some(Ok(None)) => {
                    return json_response(
                        StatusCode::NOT_FOUND,
                        r#"{"error":"rollout policy not found"}"#.to_string(),
                    )
                }
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
                None => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"store not enabled"}"#.to_string(),
                    )
                }
            };
            let mut candidate_prompt =
                match api.get_project_prompt(project_id, prompt_name, candidate_version) {
                    Some(record) => record,
                    None => {
                        return json_response(
                            StatusCode::NOT_FOUND,
                            r#"{"error":"candidate prompt version not found"}"#.to_string(),
                        )
                    }
                };
            if let Some(target_environment) = policy.target_environment.as_deref() {
                if candidate_prompt.environment != target_environment {
                    return bad_request("candidate prompt environment does not match rollout policy target_environment");
                }
            }
            let gate_request = match deserialize_rollout_policy_gate(&policy) {
                Ok(gate) if !gate.is_empty() => gate,
                Ok(_) => return bad_request("rollout policy gate is empty"),
                Err(message) => return bad_request(&message),
            };
            let comparison = match api
                .compare_project_eval_runs(
                    project_id,
                    baseline_run_id,
                    candidate_run_id,
                    Some(gate_request),
                )
                .await
            {
                Some(Ok(comparison)) => comparison,
                Some(Err(error)) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        serde_json::json!({ "error": error.to_string() }).to_string(),
                    )
                }
                None => {
                    return json_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        r#"{"error":"store not enabled"}"#.to_string(),
                    )
                }
            };
            if let Err(message) =
                validate_prompt_promotion_context(prompt_name, candidate_version, &comparison)
            {
                return bad_request(&message);
            }
            let comparison_json =
                eval_run_comparison_json_with_rollout_policy(&comparison, Some(&policy));
            let Some(gate) = comparison.gate.as_ref() else {
                return json_response(
                    StatusCode::BAD_REQUEST,
                    r#"{"error":"rollout policy did not produce a gate result"}"#.to_string(),
                );
            };
            if !gate.passed {
                return json_response(
                    StatusCode::CONFLICT,
                    serde_json::json!({
                        "promoted": false,
                        "prompt": prompt_record_json(&candidate_prompt),
                        "comparison": comparison_json,
                    })
                    .to_string(),
                );
            }
            candidate_prompt.active = true;
            candidate_prompt.updated_at = current_timestamp_string();
            match api.upsert_project_prompt(candidate_prompt.clone()).await {
                Some(Ok(())) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "promoted": true,
                        "prompt": prompt_record_json(&candidate_prompt),
                        "comparison": comparison_json,
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some(remainder)) if remainder.starts_with("prompts/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some((prompt_name, version)) = remainder
                .trim_start_matches("prompts/")
                .split_once("/versions/")
            else {
                return not_found();
            };
            match api.get_project_prompt(project_id, prompt_name, version) {
                Some(record) => {
                    json_response(StatusCode::OK, prompt_record_json(&record).to_string())
                }
                None => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"prompt version not found"}"#.to_string(),
                ),
            }
        }
        (&Method::PUT, Some(remainder)) if remainder.starts_with("prompts/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some((prompt_name, version)) = remainder
                .trim_start_matches("prompts/")
                .split_once("/versions/")
            else {
                return not_found();
            };
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload: ProjectPromptPayload = match serde_json::from_str(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(r#"{{"error":"invalid prompt payload: {}"}}"#, error),
                    )
                }
            };
            let environment = payload
                .environment
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("prod")
                .to_string();
            let target = payload
                .target
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("system")
                .to_string();
            let template_text = match payload.template_text {
                Some(template_text) if !template_text.trim().is_empty() => template_text,
                _ => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        r#"{"error":"template_text is required"}"#.to_string(),
                    )
                }
            };
            let record = ProjectPromptRecord {
                project_id: project_id.to_string(),
                prompt_name: prompt_name.to_string(),
                version: version.to_string(),
                environment,
                description: payload.description,
                target,
                template_text,
                variables_schema_json: payload.variables_schema.map(|value| value.to_string()),
                rollout_metadata_json: payload.rollout_metadata.map(|value| value.to_string()),
                active: payload.active.unwrap_or(true),
                updated_at: current_timestamp_string(),
            };
            match api.upsert_project_prompt(record).await {
                Some(Ok(())) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::DELETE, Some(remainder)) if remainder.starts_with("prompts/") => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let Some((prompt_name, version)) = remainder
                .trim_start_matches("prompts/")
                .split_once("/versions/")
            else {
                return not_found();
            };
            match api
                .delete_project_prompt(project_id, prompt_name, version)
                .await
            {
                Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Ok(false)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"prompt version not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("safety/detectors")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            match api.list_safety_detectors(Some(project_id)) {
                Some(Ok(detectors)) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "project_id": project_id,
                        "detectors": detectors.into_iter().map(|detector| {
                            serde_json::json!({
                                "detector_class": detector.detector_class,
                                "display_name": detector.display_name,
                                "provider": detector.provider,
                                "category": detector.category,
                                "source": detector.source,
                                "source_url": detector.source_url,
                                "effective_action": detector.effective_action,
                                "action_source": detector.action_source,
                                "verification_mode": detector.verification_mode,
                                "verifier_kind": detector.verifier_kind,
                                "remote_verifier_kind": detector.remote_verifier_kind,
                                "replacement": detector.replacement,
                                "path_patterns": detector.path_patterns,
                                "allowlist_patterns": detector.allowlist_patterns,
                            })
                        }).collect::<Vec<_>>(),
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("safety")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let policy = api.get_safety_policy(project_id);
            let body = policy
                .map(|policy| {
                    format!(
                        r#"{{"project_id":"{}","mode":"{}","rules":{}}}"#,
                        policy.project_id,
                        policy.mode,
                        policy.rules_json.unwrap_or_else(|| "[]".to_string()),
                    )
                })
                .unwrap_or_else(|| {
                    format!(
                        r#"{{"project_id":"{}","mode":"redact_and_forward","rules":[]}}"#,
                        project_id
                    )
                });
            json_response(StatusCode::OK, body)
        }
        (&Method::PUT, Some("safety")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let record = SafetyPolicyRecord {
                project_id: project_id.to_string(),
                mode: extract_json_string(&body, "mode")
                    .unwrap_or_else(|| "redact_and_forward".to_string()),
                rules_json: extract_json_raw(&body, "rules"),
                updated_at: current_timestamp_string(),
            };
            match api.upsert_safety_policy(record).await {
                Some(Ok(())) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::DELETE, Some("safety")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            match api.delete_safety_policy(project_id).await {
                Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
                Some(Ok(false)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"safety policy not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("semantic-safety")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let record = api.get_semantic_policy(project_id);
            let body = record
                .as_ref()
                .and_then(|record| record_to_proto(record).ok())
                .map(|policy| serde_json::to_string(&policy).unwrap_or_else(|_| "{}".to_string()))
                .unwrap_or_else(|| {
                    serde_json::json!({
                        "project_id": project_id,
                        "version": "",
                        "enabled": false,
                        "entities": [],
                        "topics": [],
                        "updated_at": "",
                    })
                    .to_string()
                });
            json_response(StatusCode::OK, body)
        }
        (&Method::PUT, Some("semantic-safety")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let body = match read_body_string(req).await {
                Ok(body) => body,
                Err(resp) => return resp,
            };
            let payload: SemanticPolicyPayload = match serde_json::from_str(&body) {
                Ok(payload) => payload,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(
                            r#"{{"error":"invalid semantic policy payload: {}"}}"#,
                            error
                        ),
                    )
                }
            };
            let policy = ProjectSemanticPolicy {
                project_id: project_id.to_string(),
                version: payload
                    .version
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(generate_semantic_policy_version),
                enabled: payload.enabled.unwrap_or(true),
                entities: payload.entities.unwrap_or_default(),
                topics: payload.topics.unwrap_or_default(),
                updated_at: current_timestamp_string(),
            };
            let record: ProjectSemanticPolicyRecord = match proto_to_record(&policy) {
                Ok(record) => record,
                Err(error) => {
                    return json_response(
                        StatusCode::BAD_REQUEST,
                        format!(r#"{{"error":"{}"}}"#, error),
                    )
                }
            };
            match api.upsert_semantic_policy(record).await {
                Some(Ok(result)) => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "ok": true,
                        "policy_version": result.policy_version,
                        "synced": result.synced,
                        "sync_error": result.sync_error,
                    })
                    .to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::DELETE, Some("semantic-safety")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ManageProjectPolicy, Some(project_id))
            {
                return resp;
            }
            match api.delete_semantic_policy(project_id).await {
                Some(Ok(result)) if result.existed => json_response(
                    StatusCode::OK,
                    serde_json::json!({
                        "ok": true,
                        "synced": result.synced,
                        "sync_error": result.sync_error,
                    })
                    .to_string(),
                ),
                Some(Ok(_)) => json_response(
                    StatusCode::NOT_FOUND,
                    r#"{"error":"semantic policy not found"}"#.to_string(),
                ),
                Some(Err(error)) => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(r#"{{"error":"{}"}}"#, error),
                ),
                None => json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    r#"{"error":"governance not enabled"}"#.to_string(),
                ),
            }
        }
        (&Method::GET, Some("semantic-safety/status")) => {
            if let Some(resp) =
                ensure_permission(api, auth, Permission::ViewProjectPolicy, Some(project_id))
            {
                return resp;
            }
            let local = api.get_semantic_policy(project_id);
            let sync = api.get_semantic_policy_sync_status(project_id).await;
            let expected_index_state =
                local
                    .as_ref()
                    .map(|record| if record.enabled { "ready" } else { "disabled" });
            let body = serde_json::json!({
                "project_id": project_id,
                "local_policy_version": local.as_ref().map(|record| record.version.clone()),
                "local_enabled": local.as_ref().map(|record| record.enabled).unwrap_or(false),
                "service": sync.as_ref().map(|status| {
                    serde_json::json!({
                        "available": status.available,
                        "ready": status.ready,
                        "backend": status.backend,
                        "message": status.message,
                        "policy_version": status.policy_version,
                        "index_state": status.index_state,
                        "updated_at": status.updated_at,
                        "stored_exemplar_count": status.stored_exemplar_count,
                        "error": status.error,
                    })
                }),
                "synced": matches!((&local, &sync, expected_index_state), (Some(record), Some(status), Some(expected_state)) if status.available && status.policy_version == record.version && status.index_state == expected_state),
            });
            json_response(StatusCode::OK, body.to_string())
        }
        _ => not_found(),
    }
}

fn handle_list_principals(
    api: &LlmGatewayApi,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    if let Some(resp) = ensure_permission(api, auth, Permission::ViewPrincipals, None) {
        return resp;
    }
    let principals = api.list_principals().unwrap_or_default();
    json_response(
        StatusCode::OK,
        format!(
            r#"{{"principals":[{}]}}"#,
            principals
                .iter()
                .map(|principal| {
                    format!(
                        r#"{{"principal_id":"{}","name":"{}","active":{},"created_at":"{}"}}"#,
                        principal.principal_id,
                        principal.name,
                        principal.active,
                        principal.created_at
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        ),
    )
}

async fn handle_create_principal(
    api: &LlmGatewayApi,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match read_body_string(req).await {
        Ok(body) => body,
        Err(resp) => return resp,
    };
    let project_id = extract_json_string(&body, "project_id");
    if let Some(resp) = ensure_permission(
        api,
        auth,
        Permission::ManagePrincipals,
        project_id.as_deref(),
    ) {
        return resp;
    }
    let name = match extract_json_string(&body, "name") {
        Some(name) => name,
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"name is required"}"#.to_string(),
            );
        }
    };
    match api.create_principal(&name).await {
        Some(Ok(principal)) => json_response(
            StatusCode::CREATED,
            format!(
                r#"{{"principal_id":"{}","name":"{}"}}"#,
                principal.principal_id, principal.name
            ),
        ),
        Some(Err(error)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, error),
        ),
        None => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":"auth service not enabled"}"#.to_string(),
        ),
    }
}

fn handle_list_roles(api: &LlmGatewayApi, auth: Option<&AuthContext>) -> Response<Full<Bytes>> {
    if let Some(resp) = ensure_permission(api, auth, Permission::ViewRoleBindings, None) {
        return resp;
    }

    let roles = api.role_catalog().unwrap_or_default();
    json_response(
        StatusCode::OK,
        serde_json::json!({
            "roles": roles.into_iter().map(|entry| {
                serde_json::json!({
                    "role": entry.role.as_str(),
                    "description": entry.description,
                    "permissions": entry.permissions.into_iter().map(|permission| permission.as_str()).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>(),
        })
        .to_string(),
    )
}

fn handle_list_role_bindings(
    api: &LlmGatewayApi,
    req: &Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let query = req.uri().query().unwrap_or("");
    let project_id = extract_query_param(query, "project_id");
    let principal_id = extract_query_param(query, "principal_id");
    let scoped_project = match project_id {
        Some(project_id) => Some(project_id),
        None => match scoped_project_query(api, auth) {
            Ok(project_id) => project_id,
            Err(resp) => return resp,
        },
    };
    if let Some(resp) = ensure_permission(
        api,
        auth,
        Permission::ViewRoleBindings,
        scoped_project.as_deref(),
    ) {
        return resp;
    }
    let bindings = api
        .list_role_bindings(principal_id.as_deref(), scoped_project.as_deref())
        .unwrap_or_default();
    json_response(
        StatusCode::OK,
        format!(
            r#"{{"bindings":[{}]}}"#,
            bindings
                .iter()
                .map(|binding| {
                    format!(
                        r#"{{"binding_id":"{}","principal_id":"{}","role":"{}","project_id":{},"created_at":"{}"}}"#,
                        binding.binding_id,
                        binding.principal_id,
                        binding.role,
                        binding.project_id.as_ref().map(|value| format!("\"{}\"", value)).unwrap_or_else(|| "null".to_string()),
                        binding.created_at,
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        ),
    )
}

async fn handle_create_role_binding(
    api: &LlmGatewayApi,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let body = match read_body_string(req).await {
        Ok(body) => body,
        Err(resp) => return resp,
    };
    let principal_id = match extract_json_string(&body, "principal_id") {
        Some(principal_id) => principal_id,
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"principal_id is required"}"#.to_string(),
            )
        }
    };
    let role = match extract_json_string(&body, "role").and_then(|role| Role::parse(&role)) {
        Some(role) => role,
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                r#"{"error":"valid role is required"}"#.to_string(),
            )
        }
    };
    let project_id = extract_json_string(&body, "project_id");
    if let Some(resp) = ensure_permission(
        api,
        auth,
        Permission::ManageRoleBindings,
        project_id.as_deref(),
    ) {
        return resp;
    }
    match api
        .create_role_binding(&principal_id, role, project_id)
        .await
    {
        Some(Ok(binding)) => json_response(
            StatusCode::CREATED,
            format!(r#"{{"binding_id":"{}"}}"#, binding.binding_id),
        ),
        Some(Err(error)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, error),
        ),
        None => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":"auth service not enabled"}"#.to_string(),
        ),
    }
}

fn handle_list_tokens(
    api: &LlmGatewayApi,
    req: &Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    if let Some(resp) = ensure_permission(api, auth, Permission::ViewPrincipals, None) {
        return resp;
    }
    let principal_id = req
        .uri()
        .query()
        .and_then(|query| extract_query_param(query, "principal_id"));
    let tokens = api.list_tokens(principal_id.as_deref()).unwrap_or_default();
    json_response(
        StatusCode::OK,
        format!(
            r#"{{"tokens":[{}]}}"#,
            tokens
                .iter()
                .map(|token| {
                    format!(
                        r#"{{"token_hash":"{}","principal_id":"{}","name":"{}","active":{},"created_at":"{}"}}"#,
                        token.token_hash, token.principal_id, token.name, token.active, token.created_at
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        ),
    )
}

async fn handle_principal_subroutes(
    api: &LlmGatewayApi,
    tail: &str,
    req: Request<Incoming>,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    if let Some(principal_id) = tail.strip_suffix("/access") {
        if req.method() != Method::GET {
            return method_not_allowed();
        }
        let query = req.uri().query().unwrap_or("");
        let project_id = extract_query_param(query, "project_id");
        if let Some(resp) = ensure_permission(
            api,
            auth,
            Permission::ViewRoleBindings,
            project_id.as_deref(),
        ) {
            return resp;
        }
        return match api.principal_access(principal_id, project_id.as_deref()) {
            Some(snapshot) => json_response(
                StatusCode::OK,
                serde_json::json!({
                    "principal": {
                        "principal_id": snapshot.principal.principal_id,
                        "name": snapshot.principal.name,
                        "active": snapshot.principal.active,
                        "created_at": snapshot.principal.created_at,
                    },
                    "scope": {
                        "project_id": project_id,
                    },
                    "role_bindings": snapshot.role_bindings.into_iter().map(|binding| {
                        serde_json::json!({
                            "binding_id": binding.binding_id,
                            "principal_id": binding.principal_id,
                            "role": binding.role,
                            "project_id": binding.project_id,
                            "created_at": binding.created_at,
                        })
                    }).collect::<Vec<_>>(),
                    "instance_permissions": snapshot
                        .instance_permissions
                        .into_iter()
                        .map(|permission| permission.as_str())
                        .collect::<Vec<_>>(),
                    "project_access": snapshot.project_access.into_iter().map(|project| {
                        serde_json::json!({
                            "project_id": project.project_id,
                            "permissions": project.permissions.into_iter().map(|permission| permission.as_str()).collect::<Vec<_>>(),
                            "role_bindings": project.role_bindings.into_iter().map(|binding| {
                                serde_json::json!({
                                    "binding_id": binding.binding_id,
                                    "role": binding.role,
                                    "project_id": binding.project_id,
                                })
                            }).collect::<Vec<_>>(),
                        })
                    }).collect::<Vec<_>>(),
                })
                .to_string(),
            ),
            None => json_response(
                StatusCode::NOT_FOUND,
                r#"{"error":"principal not found"}"#.to_string(),
            ),
        };
    }

    if let Some(principal_id) = tail.strip_suffix("/tokens") {
        if req.method() != Method::POST {
            return method_not_allowed();
        }
        if let Some(resp) = ensure_permission(api, auth, Permission::ManagePrincipals, None) {
            return resp;
        }
        let body = match read_body_string(req).await {
            Ok(body) => body,
            Err(resp) => return resp,
        };
        let name =
            extract_json_string(&body, "name").unwrap_or_else(|| "runtime token".to_string());
        return match api.create_token(principal_id, &name).await {
            Some(Ok((plaintext, token))) => json_response(
                StatusCode::CREATED,
                format!(
                    r#"{{"token":"{}","token_hash":"{}","principal_id":"{}","name":"{}"}}"#,
                    plaintext, token.token_hash, token.principal_id, token.name
                ),
            ),
            Some(Err(error)) => json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(r#"{{"error":"{}"}}"#, error),
            ),
            None => json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                r#"{"error":"auth service not enabled"}"#.to_string(),
            ),
        };
    }

    if req.method() != Method::DELETE {
        return method_not_allowed();
    }
    if let Some(resp) = ensure_permission(api, auth, Permission::ManagePrincipals, None) {
        return resp;
    }
    match api.delete_principal(tail).await {
        Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
        Some(Ok(false)) => json_response(
            StatusCode::NOT_FOUND,
            r#"{"error":"principal not found"}"#.to_string(),
        ),
        Some(Err(error)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, error),
        ),
        None => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":"auth service not enabled"}"#.to_string(),
        ),
    }
}

async fn handle_delete_token(
    api: &LlmGatewayApi,
    token_hash: &str,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    if let Some(resp) = ensure_permission(api, auth, Permission::ManagePrincipals, None) {
        return resp;
    }
    match api.delete_token(token_hash).await {
        Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
        Some(Ok(false)) => json_response(
            StatusCode::NOT_FOUND,
            r#"{"error":"token not found"}"#.to_string(),
        ),
        Some(Err(error)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, error),
        ),
        None => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":"auth service not enabled"}"#.to_string(),
        ),
    }
}

async fn handle_delete_role_binding(
    api: &LlmGatewayApi,
    binding_id: &str,
    auth: Option<&AuthContext>,
) -> Response<Full<Bytes>> {
    let binding = api
        .list_role_bindings(None, None)
        .unwrap_or_default()
        .into_iter()
        .find(|binding| binding.binding_id == binding_id);
    if let Some(binding) = &binding {
        if let Some(resp) = ensure_permission(
            api,
            auth,
            Permission::ManageRoleBindings,
            binding.project_id.as_deref(),
        ) {
            return resp;
        }
    } else {
        return json_response(
            StatusCode::NOT_FOUND,
            r#"{"error":"binding not found"}"#.to_string(),
        );
    }

    match api.delete_role_binding(binding_id).await {
        Some(Ok(true)) => json_response(StatusCode::OK, r#"{"ok":true}"#.to_string()),
        Some(Ok(false)) => json_response(
            StatusCode::NOT_FOUND,
            r#"{"error":"binding not found"}"#.to_string(),
        ),
        Some(Err(error)) => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(r#"{{"error":"{}"}}"#, error),
        ),
        None => json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":"auth service not enabled"}"#.to_string(),
        ),
    }
}

/// Extract a float value for a given key from a JSON string.
fn extract_json_float(text: &str, key: &str) -> Option<f64> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    value.get(key)?.as_f64()
}

fn extract_json_optional_float(text: &str, key: &str) -> Option<f64> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    if !value.as_object()?.contains_key(key) {
        return None;
    }
    value.get(key).and_then(|entry| entry.as_f64())
}

/// Extract a string value for a given key from a JSON string.
fn extract_json_string(text: &str, key: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    value.get(key)?.as_str().map(ToString::to_string)
}

fn extract_json_optional_string(text: &str, key: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    if !value.as_object()?.contains_key(key) {
        return None;
    }
    value
        .get(key)
        .and_then(|entry| entry.as_str().map(ToString::to_string))
}

/// Extract a boolean value for a given key from a JSON string.
fn extract_json_bool(text: &str, key: &str) -> Option<bool> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    value.get(key)?.as_bool()
}

/// Extract a JSON array of strings for a given key (e.g., "allowed_models": ["a", "b"]).
fn extract_json_string_array(text: &str, key: &str) -> Option<Vec<String>> {
    let result = extract_json_string_array_allow_empty(text, key)?;
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

fn extract_json_string_array_allow_empty(text: &str, key: &str) -> Option<Vec<String>> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let arr = value.get(key)?.as_array()?;
    let mut result = Vec::with_capacity(arr.len());
    for item in arr {
        result.push(item.as_str()?.to_string());
    }
    Some(result)
}

fn extract_json_raw(text: &str, key: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let entry = value.get(key)?;
    if entry.is_null() {
        None
    } else {
        Some(entry.to_string())
    }
}

fn opt_json_string(value: &Option<String>) -> String {
    value
        .as_ref()
        .map(|value| format!("\"{}\"", value))
        .unwrap_or_else(|| "null".to_string())
}

fn opt_json_bool(value: Option<bool>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn opt_json_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn serialize_string_array(values: BTreeSet<String>) -> Option<String> {
    if values.is_empty() {
        return None;
    }
    serde_json::to_string(&values.into_iter().collect::<Vec<_>>()).ok()
}

fn parse_string_array_value(value: Option<&str>) -> serde_json::Value {
    value
        .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
        .unwrap_or_else(|| serde_json::Value::Array(Vec::new()))
}

fn parse_json_value(value: Option<&str>) -> serde_json::Value {
    value
        .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
        .unwrap_or(serde_json::Value::Null)
}

fn normalize_session_status(status: String) -> Option<String> {
    let trimmed = status.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

fn normalize_nonempty_str(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn normalize_nonempty_string(value: Option<String>) -> Option<String> {
    value.and_then(|value| normalize_nonempty_str(&value).map(ToString::to_string))
}

fn is_supported_session_status(status: &str) -> bool {
    matches!(
        status,
        "active" | "paused" | "completed" | "cancelled" | "failed"
    )
}

fn session_status_is_terminal(status: &str) -> bool {
    matches!(status, "completed" | "cancelled" | "failed")
}

fn session_owner_is_active(record: &SessionRecord, now: i64) -> bool {
    record.owner_id.is_some()
        && record
            .lease_expires_at_unix
            .map(|expires_at| expires_at > now)
            .unwrap_or(false)
}

fn clear_session_handoff(record: &mut SessionRecord) {
    record.handoff_target_owner_id = None;
    record.handoff_requested_at_unix = None;
    record.handoff_reason = None;
}

fn validate_session_transition(current: Option<&str>, next: &str) -> Result<(), String> {
    if !is_supported_session_status(next) {
        return Err(format!("unsupported session status '{next}'"));
    }
    match current {
        None => Ok(()),
        Some(current) if current == next => Ok(()),
        Some("active") if matches!(next, "paused" | "completed" | "cancelled" | "failed") => Ok(()),
        Some("paused") if matches!(next, "active" | "cancelled" | "failed") => Ok(()),
        Some(current) if session_status_is_terminal(current) => Err(format!(
            "cannot transition terminal session from '{current}' to '{next}'"
        )),
        Some(_) => Ok(()),
    }
}

fn current_unix_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_float() {
        assert_eq!(
            extract_json_float(
                r#"{"input_cost_per_1k":0.01,"output_cost_per_1k":0.02}"#,
                "input_cost_per_1k"
            ),
            Some(0.01)
        );
        assert_eq!(
            extract_json_float(
                r#"{"input_cost_per_1k": 0.01, "output_cost_per_1k": 0.02}"#,
                "output_cost_per_1k"
            ),
            Some(0.02)
        );
        assert_eq!(
            extract_json_float(r#"{"foo":"bar"}"#, "input_cost_per_1k"),
            None
        );
    }

    #[test]
    fn test_mask_key() {
        assert_eq!(mask_key("sk-test-1234567890"), "sk-test-...");
        assert_eq!(mask_key("short"), "short");
    }

    #[test]
    fn test_is_authorized_with_valid_bearer() {
        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            "authorization",
            hyper::header::HeaderValue::from_static("Bearer top-secret"),
        );
        assert!(is_authorized(&headers, "top-secret"));
    }

    #[test]
    fn test_is_authorized_rejects_missing_or_wrong_token() {
        let headers = hyper::HeaderMap::new();
        assert!(!is_authorized(&headers, "top-secret"));

        let mut headers = hyper::HeaderMap::new();
        headers.insert(
            "authorization",
            hyper::header::HeaderValue::from_static("Bearer wrong"),
        );
        assert!(!is_authorized(&headers, "top-secret"));
    }

    // --- Gap tests ---

    #[test]
    fn test_extract_json_string() {
        assert_eq!(
            extract_json_string(r#"{"name":"test-key","provider_name":"openai"}"#, "name"),
            Some("test-key".to_string())
        );
        assert_eq!(
            extract_json_string(
                r#"{"name":"test-key","provider_name":"openai"}"#,
                "provider_name"
            ),
            Some("openai".to_string())
        );
        assert_eq!(extract_json_string(r#"{"foo":42}"#, "name"), None);
    }

    #[test]
    fn test_extract_json_bool() {
        assert_eq!(
            extract_json_bool(r#"{"active":true,"name":"x"}"#, "active"),
            Some(true)
        );
        assert_eq!(
            extract_json_bool(r#"{"active":false}"#, "active"),
            Some(false)
        );
        assert_eq!(extract_json_bool(r#"{"name":"x"}"#, "active"), None);
    }

    #[test]
    fn test_extract_json_string_array() {
        assert_eq!(
            extract_json_string_array(
                r#"{"allowed_models":["gpt-4o","gpt-4o-mini"]}"#,
                "allowed_models"
            ),
            Some(vec!["gpt-4o".to_string(), "gpt-4o-mini".to_string()])
        );
        // Single element
        assert_eq!(
            extract_json_string_array(r#"{"allowed_models":["gpt-4o"]}"#, "allowed_models"),
            Some(vec!["gpt-4o".to_string()])
        );
        // Missing field
        assert_eq!(
            extract_json_string_array(r#"{"name":"x"}"#, "allowed_models"),
            None
        );
        // Empty array returns None (current behavior)
        assert_eq!(
            extract_json_string_array(r#"{"allowed_models":[]}"#, "allowed_models"),
            None
        );
    }

    #[test]
    fn test_extract_json_string_with_spaces() {
        // Verify extraction works with spaces around colons
        assert_eq!(
            extract_json_string(r#"{"name" : "test-key"}"#, "name"),
            Some("test-key".to_string())
        );
    }

    #[test]
    fn test_extract_json_string_handles_escaped_quotes() {
        assert_eq!(
            extract_json_string(r#"{"name":"test-\"key\""}"#, "name"),
            Some("test-\"key\"".to_string())
        );
    }

    #[test]
    fn test_extract_json_bool_not_confused_by_embedded_json_text() {
        let body = r#"{"note":"the string says \"active\":true","active":false}"#;
        assert_eq!(extract_json_bool(body, "active"), Some(false));
    }

    #[test]
    fn test_extract_json_float_integer_values() {
        // Integer value (no decimal)
        assert_eq!(
            extract_json_float(r#"{"budget_limit":100}"#, "budget_limit"),
            Some(100.0)
        );
        // Float value
        assert_eq!(
            extract_json_float(r#"{"budget_limit":10.5}"#, "budget_limit"),
            Some(10.5)
        );
    }

    #[test]
    fn test_handle_update_key_parses_allowed_models_and_expires_at() {
        // Verify the extractors correctly parse allowed_models and expires_at
        // from a PATCH body — these are now forwarded to update_virtual_key.
        let body = r#"{"allowed_models":["gpt-4o"],"timeout_secs":15,"expires_at":"1999999999"}"#;
        let models = extract_json_string_array(body, "allowed_models");
        assert_eq!(models, Some(vec!["gpt-4o".to_string()]));
        assert_eq!(extract_json_float(body, "timeout_secs"), Some(15.0));
        let expires = extract_json_string(body, "expires_at");
        assert_eq!(expires, Some("1999999999".to_string()));

        // Combined with other fields
        let body2 = r#"{"active":false,"allowed_models":["gpt-4o","gpt-4o-mini"],"expires_at":"9999999999","budget_limit":50.0,"timeout_secs":90}"#;
        assert_eq!(extract_json_bool(body2, "active"), Some(false));
        assert_eq!(
            extract_json_string_array(body2, "allowed_models"),
            Some(vec!["gpt-4o".to_string(), "gpt-4o-mini".to_string()])
        );
        assert_eq!(
            extract_json_string(body2, "expires_at"),
            Some("9999999999".to_string())
        );
        assert_eq!(extract_json_float(body2, "budget_limit"), Some(50.0));
        assert_eq!(extract_json_float(body2, "timeout_secs"), Some(90.0));
    }
}
