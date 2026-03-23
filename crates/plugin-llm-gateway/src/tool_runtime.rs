use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::{HeaderValue, CONTENT_LENGTH, CONTENT_TYPE, HOST};
use hyper::{Method, Request, Response, StatusCode};
use proxy_core::config::{
    ManagedToolRequestShape, ProviderKeyConfig, ProviderRuntimeSemantics, ProviderSurfaceCatalog,
};
use proxy_core::handlers::proxy::build_client;
use proxy_core::plugin::{Action, Plugin, ProviderCandidates, RequestContext};
use rand::Rng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tracing::Instrument;

use crate::content_filter::{ContentFilterReplayHandle, SafetyAudit};
use crate::estimate_prompt_tokens;
use crate::governance::{current_timestamp_string, GovernanceState};
use crate::semantic_safety::{SemanticSafetyAudit, SemanticSafetyReplayHandle};
use crate::store::ProjectToolRecord;
use crate::virtual_keys::{ToolApprovalMode, VirtualKeyMeta};

#[derive(Clone, Debug)]
pub struct ToolUsageOverride {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Debug)]
pub struct ForwardedRequestBody(pub Bytes);

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolTraceEntry {
    pub tool_name: String,
    pub executor_kind: String,
    pub status: String,
    pub error_code: Option<String>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolRuntimeAudit {
    pub calls: Vec<ToolTraceEntry>,
}

#[derive(Clone)]
struct ToolRuntimeRequest {
    provider_name: String,
    request_shape: ManagedToolRequestShape,
    project_id: String,
    request_json: Value,
    selected_tools: HashMap<String, ProjectToolRecord>,
    allowed_tool_names: Option<Vec<String>>,
    stream: bool,
}

#[derive(Clone)]
struct WebSearchBackendConfig {
    url: String,
    method: String,
    headers: HashMap<String, String>,
}

#[derive(Clone)]
enum McpAuthConfig {
    None,
    StaticBearer {
        access_token: String,
    },
    OAuthClientCredentials {
        token_url: Option<String>,
        protected_resource_metadata_url: Option<String>,
        authorization_server_metadata_url: Option<String>,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
        resource: Option<String>,
    },
    OAuthAuthorizationCode {
        token_url: Option<String>,
        authorization_url: Option<String>,
        protected_resource_metadata_url: Option<String>,
        authorization_server_metadata_url: Option<String>,
        client_id: String,
        client_secret: Option<String>,
        redirect_uri: String,
        scope: Option<String>,
        resource: Option<String>,
    },
}

#[derive(Clone)]
enum McpTransportConfig {
    Http {
        url: String,
        method: String,
        headers: HashMap<String, String>,
    },
    Sse {
        url: String,
        headers: HashMap<String, String>,
    },
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
        cwd: Option<String>,
    },
}

#[derive(Clone)]
struct McpServerConfig {
    transport: McpTransportConfig,
    auth: McpAuthConfig,
    timeout_ms: Option<u64>,
    max_retries: u32,
    max_calls_per_request: Option<u32>,
    max_total_time_ms: Option<u64>,
    max_output_tokens: Option<u64>,
}

#[derive(Clone)]
struct ArxivRuntimeConfig {
    base_url: String,
    default_max_results: u64,
}

impl McpServerConfig {
    fn auth_mode(&self) -> Option<&'static str> {
        match &self.auth {
            McpAuthConfig::None => None,
            McpAuthConfig::StaticBearer { .. } => Some("bearer"),
            McpAuthConfig::OAuthClientCredentials { .. } => Some("oauth_client_credentials"),
            McpAuthConfig::OAuthAuthorizationCode { .. } => Some("oauth_authorization_code"),
        }
    }

    fn transport_name(&self) -> &'static str {
        match &self.transport {
            McpTransportConfig::Http { .. } => "http",
            McpTransportConfig::Sse { .. } => "sse",
            McpTransportConfig::Stdio { .. } => "stdio",
        }
    }

    fn display_url(&self) -> String {
        match &self.transport {
            McpTransportConfig::Http { url, .. } => url.clone(),
            McpTransportConfig::Sse { url, .. } => url.clone(),
            McpTransportConfig::Stdio { .. } => String::new(),
        }
    }

    fn display_method(&self) -> String {
        match &self.transport {
            McpTransportConfig::Http { method, .. } => method.clone(),
            McpTransportConfig::Sse { .. } => "sse".to_string(),
            McpTransportConfig::Stdio { .. } => "stdio".to_string(),
        }
    }

    fn display_command(&self) -> Option<String> {
        match &self.transport {
            McpTransportConfig::Http { .. } | McpTransportConfig::Sse { .. } => None,
            McpTransportConfig::Stdio { command, .. } => Some(command.clone()),
        }
    }

    fn display_args(&self) -> Vec<String> {
        match &self.transport {
            McpTransportConfig::Http { .. } | McpTransportConfig::Sse { .. } => Vec::new(),
            McpTransportConfig::Stdio { args, .. } => args.clone(),
        }
    }

    fn display_cwd(&self) -> Option<String> {
        match &self.transport {
            McpTransportConfig::Http { .. } | McpTransportConfig::Sse { .. } => None,
            McpTransportConfig::Stdio { cwd, .. } => cwd.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ToolRuntimeBackendSnapshot {
    pub name: String,
    pub url: String,
    pub method: String,
}

#[derive(Clone, Debug)]
pub struct ToolRuntimeMcpServerSnapshot {
    pub name: String,
    pub transport: String,
    pub auth_mode: Option<String>,
    pub auth_status: String,
    pub auth_last_error: Option<String>,
    pub auth_last_discovery_error: Option<String>,
    pub auth_refreshes: u64,
    pub auth_last_refreshed_at: Option<String>,
    pub auth_token_expires_at_unix_ms: Option<u64>,
    pub auth_resource: Option<String>,
    pub auth_authorization_url: Option<String>,
    pub auth_token_url: Option<String>,
    pub auth_authorization_server_url: Option<String>,
    pub auth_pending_authorization: bool,
    pub auth_pending_authorization_expires_at_unix_ms: Option<u64>,
    pub url: String,
    pub method: String,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub timeout_ms: u64,
    pub max_retries: u32,
    pub max_calls_per_request: Option<u32>,
    pub max_total_time_ms: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub operator_state: String,
    pub operator_state_at: Option<String>,
    pub operator_state_actor: Option<String>,
    pub operator_state_reason: Option<String>,
    pub health_state: String,
    pub health_reason: Option<String>,
    pub recommended_action: Option<String>,
    pub reachable: bool,
    pub protocol_version: Option<String>,
    pub session_id_present: bool,
    pub discovered_tools: Vec<String>,
    pub last_error: Option<String>,
    pub discovery_refreshes: u64,
    pub last_discovery_at: Option<String>,
    pub last_discovery_status: Option<String>,
    pub last_discovery_error: Option<String>,
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub retried_calls: u64,
    pub session_reinitializations: u64,
    pub last_session_reinitialized_at: Option<String>,
    pub last_recovery_error: Option<String>,
    pub budget_exceeded_calls: u64,
    pub last_budget_exceeded_at: Option<String>,
    pub last_budget_exceeded_error: Option<String>,
    pub last_session_reset_at: Option<String>,
    pub last_session_reset_status: Option<String>,
    pub last_session_reset_error: Option<String>,
    pub last_session_reset_http_status: Option<u16>,
    pub last_call_at: Option<String>,
    pub last_call_tool: Option<String>,
    pub last_call_status: Option<String>,
    pub last_call_error: Option<String>,
    pub last_call_http_status: Option<u16>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolRuntimeProviderSnapshot {
    pub name: String,
    pub family: String,
    pub surfaces: ProviderSurfaceCatalog,
    #[serde(flatten)]
    pub semantics: ProviderRuntimeSemantics,
    pub data_collection: Option<String>,
    pub zdr: bool,
    pub distillable_text: bool,
    pub quantizations: Vec<String>,
    pub supported_parameter_families: Vec<String>,
    pub models: Vec<String>,
    pub timeout_secs: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct ToolRuntimeExecutorSummary {
    pub executor_kind: String,
    pub total: u64,
    pub enabled: u64,
}

#[derive(Clone, Debug)]
pub struct ToolRuntimeStatusSnapshot {
    pub default_timeout_ms: u64,
    pub max_round_trips: usize,
    pub responses_stream_mode: String,
    pub supported_executors: Vec<String>,
    pub web_search_backends: Vec<ToolRuntimeBackendSnapshot>,
    pub mcp_servers: Vec<ToolRuntimeMcpServerSnapshot>,
    pub arxiv_base_url: String,
    pub arxiv_default_max_results: u64,
    pub providers: Vec<ToolRuntimeProviderSnapshot>,
    pub registered_tools_total: u64,
    pub enabled_tools_total: u64,
    pub executors: Vec<ToolRuntimeExecutorSummary>,
}

#[derive(Clone)]
pub struct ToolRuntime {
    governance: Arc<GovernanceState>,
    providers: Arc<RwLock<HashMap<String, ProviderKeyConfig>>>,
    web_search_backends: Arc<HashMap<String, WebSearchBackendConfig>>,
    mcp_servers: Arc<HashMap<String, McpServerConfig>>,
    mcp_inventory: Arc<RwLock<HashMap<String, McpServerInventory>>>,
    arxiv: ArxivRuntimeConfig,
    timeout: Duration,
    max_round_trips: usize,
    responses_stream_mode: ResponsesStreamMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResponsesStreamMode {
    Strict,
    Composed,
}

impl ResponsesStreamMode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "strict" => Some(Self::Strict),
            "composed" => Some(Self::Composed),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Composed => "composed",
        }
    }
}

static MCP_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
const MCP_CLIENT_PROTOCOL_VERSION: &str = "2025-11-25";
const MCP_SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26"];
const MCP_SESSION_HEADER: &str = "mcp-session-id";
const MCP_PROTOCOL_HEADER: &str = "mcp-protocol-version";
const MCP_ACCEPT_HEADER: &str = "application/json, text/event-stream";
const MCP_AUTH_REFRESH_SKEW_MS: u64 = 30_000;
const MCP_OAUTH_PENDING_TTL_MS: u64 = 10 * 60 * 1000;

#[derive(Clone)]
struct McpServerInventory {
    operator_enabled: bool,
    operator_state_at: Option<String>,
    operator_state_actor: Option<String>,
    operator_state_reason: Option<String>,
    reachable: bool,
    protocol_version: Option<String>,
    session_id: Option<String>,
    discovered_tools: Vec<String>,
    last_error: Option<String>,
    discovery_refreshes: u64,
    last_discovery_at: Option<String>,
    last_discovery_status: Option<String>,
    last_discovery_error: Option<String>,
    total_calls: u64,
    successful_calls: u64,
    failed_calls: u64,
    retried_calls: u64,
    session_reinitializations: u64,
    last_session_reinitialized_at: Option<String>,
    last_recovery_error: Option<String>,
    budget_exceeded_calls: u64,
    last_budget_exceeded_at: Option<String>,
    last_budget_exceeded_error: Option<String>,
    last_session_reset_at: Option<String>,
    last_session_reset_status: Option<String>,
    last_session_reset_error: Option<String>,
    last_session_reset_http_status: Option<u16>,
    last_call_at: Option<String>,
    last_call_tool: Option<String>,
    last_call_status: Option<String>,
    last_call_error: Option<String>,
    last_call_http_status: Option<u16>,
    auth_cached_access_token: Option<String>,
    auth_cached_refresh_token: Option<String>,
    auth_token_expires_at_unix_ms: Option<u64>,
    auth_refreshes: u64,
    auth_last_refreshed_at: Option<String>,
    auth_last_error: Option<String>,
    auth_last_discovery_error: Option<String>,
    auth_resource: Option<String>,
    auth_authorization_url: Option<String>,
    auth_token_url: Option<String>,
    auth_authorization_server_url: Option<String>,
    auth_pending_state: Option<String>,
    auth_pending_code_verifier: Option<String>,
    auth_pending_authorization_expires_at_unix_ms: Option<u64>,
}

impl Default for McpServerInventory {
    fn default() -> Self {
        Self {
            operator_enabled: true,
            operator_state_at: None,
            operator_state_actor: None,
            operator_state_reason: None,
            reachable: false,
            protocol_version: None,
            session_id: None,
            discovered_tools: Vec::new(),
            last_error: None,
            discovery_refreshes: 0,
            last_discovery_at: None,
            last_discovery_status: None,
            last_discovery_error: None,
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            retried_calls: 0,
            session_reinitializations: 0,
            last_session_reinitialized_at: None,
            last_recovery_error: None,
            budget_exceeded_calls: 0,
            last_budget_exceeded_at: None,
            last_budget_exceeded_error: None,
            last_session_reset_at: None,
            last_session_reset_status: None,
            last_session_reset_error: None,
            last_session_reset_http_status: None,
            last_call_at: None,
            last_call_tool: None,
            last_call_status: None,
            last_call_error: None,
            last_call_http_status: None,
            auth_cached_access_token: None,
            auth_cached_refresh_token: None,
            auth_token_expires_at_unix_ms: None,
            auth_refreshes: 0,
            auth_last_refreshed_at: None,
            auth_last_error: None,
            auth_last_discovery_error: None,
            auth_resource: None,
            auth_authorization_url: None,
            auth_token_url: None,
            auth_authorization_server_url: None,
            auth_pending_state: None,
            auth_pending_code_verifier: None,
            auth_pending_authorization_expires_at_unix_ms: None,
        }
    }
}

#[derive(Default)]
struct ToolExecutionBudgetState {
    mcp_server_calls: HashMap<String, u32>,
    mcp_tool_calls: HashMap<String, u32>,
    mcp_server_time_ms: HashMap<String, u64>,
    mcp_tool_time_ms: HashMap<String, u64>,
    mcp_server_output_tokens: HashMap<String, u64>,
    mcp_tool_output_tokens: HashMap<String, u64>,
}

fn merge_mcp_server_inventory(
    previous: Option<&McpServerInventory>,
    mut next: McpServerInventory,
) -> McpServerInventory {
    if let Some(previous) = previous {
        next.operator_enabled = previous.operator_enabled;
        if next.operator_state_at.is_none() {
            next.operator_state_at = previous.operator_state_at.clone();
        }
        if next.operator_state_actor.is_none() {
            next.operator_state_actor = previous.operator_state_actor.clone();
        }
        if next.operator_state_reason.is_none() {
            next.operator_state_reason = previous.operator_state_reason.clone();
        }
        next.discovery_refreshes = next.discovery_refreshes.max(previous.discovery_refreshes);
        if next.last_discovery_at.is_none() {
            next.last_discovery_at = previous.last_discovery_at.clone();
        }
        if next.last_discovery_status.is_none() {
            next.last_discovery_status = previous.last_discovery_status.clone();
        }
        if next.last_discovery_error.is_none() {
            next.last_discovery_error = previous.last_discovery_error.clone();
        }
        next.total_calls = previous.total_calls;
        next.successful_calls = previous.successful_calls;
        next.failed_calls = previous.failed_calls;
        next.retried_calls = previous.retried_calls;
        next.session_reinitializations = previous.session_reinitializations;
        if next.last_session_reinitialized_at.is_none() {
            next.last_session_reinitialized_at = previous.last_session_reinitialized_at.clone();
        }
        if next.last_recovery_error.is_none() {
            next.last_recovery_error = previous.last_recovery_error.clone();
        }
        next.budget_exceeded_calls = previous.budget_exceeded_calls;
        if next.last_budget_exceeded_at.is_none() {
            next.last_budget_exceeded_at = previous.last_budget_exceeded_at.clone();
        }
        if next.last_budget_exceeded_error.is_none() {
            next.last_budget_exceeded_error = previous.last_budget_exceeded_error.clone();
        }
        if next.last_session_reset_at.is_none() {
            next.last_session_reset_at = previous.last_session_reset_at.clone();
        }
        if next.last_session_reset_status.is_none() {
            next.last_session_reset_status = previous.last_session_reset_status.clone();
        }
        if next.last_session_reset_error.is_none() {
            next.last_session_reset_error = previous.last_session_reset_error.clone();
        }
        if next.last_session_reset_http_status.is_none() {
            next.last_session_reset_http_status = previous.last_session_reset_http_status;
        }
        next.last_call_at = previous.last_call_at.clone();
        next.last_call_tool = previous.last_call_tool.clone();
        next.last_call_status = previous.last_call_status.clone();
        next.last_call_error = previous.last_call_error.clone();
        next.last_call_http_status = previous.last_call_http_status;
        if next.auth_cached_access_token.is_none() {
            next.auth_cached_access_token = previous.auth_cached_access_token.clone();
        }
        if next.auth_token_expires_at_unix_ms.is_none() {
            next.auth_token_expires_at_unix_ms = previous.auth_token_expires_at_unix_ms;
        }
        if next.auth_cached_refresh_token.is_none() {
            next.auth_cached_refresh_token = previous.auth_cached_refresh_token.clone();
        }
        next.auth_refreshes = previous.auth_refreshes.max(next.auth_refreshes);
        if next.auth_last_refreshed_at.is_none() {
            next.auth_last_refreshed_at = previous.auth_last_refreshed_at.clone();
        }
        if next.auth_last_error.is_none() {
            next.auth_last_error = previous.auth_last_error.clone();
        }
        if next.auth_last_discovery_error.is_none() {
            next.auth_last_discovery_error = previous.auth_last_discovery_error.clone();
        }
        if next.auth_resource.is_none() {
            next.auth_resource = previous.auth_resource.clone();
        }
        if next.auth_authorization_url.is_none() {
            next.auth_authorization_url = previous.auth_authorization_url.clone();
        }
        if next.auth_token_url.is_none() {
            next.auth_token_url = previous.auth_token_url.clone();
        }
        if next.auth_authorization_server_url.is_none() {
            next.auth_authorization_server_url = previous.auth_authorization_server_url.clone();
        }
        if next.auth_pending_state.is_none() {
            next.auth_pending_state = previous.auth_pending_state.clone();
        }
        if next.auth_pending_code_verifier.is_none() {
            next.auth_pending_code_verifier = previous.auth_pending_code_verifier.clone();
        }
        if next.auth_pending_authorization_expires_at_unix_ms.is_none() {
            next.auth_pending_authorization_expires_at_unix_ms =
                previous.auth_pending_authorization_expires_at_unix_ms;
        }
    }
    next
}

fn current_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn auth_token_is_fresh(expires_at_unix_ms: Option<u64>) -> bool {
    match expires_at_unix_ms {
        Some(expires_at) => expires_at > current_unix_ms().saturating_add(MCP_AUTH_REFRESH_SKEW_MS),
        None => true,
    }
}

pub async fn create_plugin(
    config: &toml::Value,
    providers: &[ProviderKeyConfig],
    governance: Arc<GovernanceState>,
) -> Result<ToolRuntime, Box<dyn std::error::Error>> {
    let timeout = config
        .get("tool_timeout_ms")
        .and_then(|value| value.as_integer())
        .map(|value| value as u64)
        .unwrap_or(5_000);
    let max_round_trips = config
        .get("max_round_trips")
        .and_then(|value| value.as_integer())
        .map(|value| value as usize)
        .unwrap_or(8);
    let responses_stream_mode = match config
        .get("responses_stream_mode")
        .and_then(|value| value.as_str())
    {
        Some(value) => ResponsesStreamMode::parse(value)
            .ok_or_else(|| format!("invalid responses_stream_mode '{value}'"))?,
        None => ResponsesStreamMode::Strict,
    };
    let web_search_backends = parse_web_search_backends(config)?;
    let mcp_servers = parse_mcp_servers(config)?;
    let arxiv = ArxivRuntimeConfig {
        base_url: config
            .get("arxiv_base_url")
            .and_then(|value| value.as_str())
            .unwrap_or("https://export.arxiv.org/api/query")
            .to_string(),
        default_max_results: config
            .get("arxiv_default_max_results")
            .and_then(|value| value.as_integer())
            .map(|value| value as u64)
            .unwrap_or(5),
    };

    let runtime = ToolRuntime {
        governance,
        providers: Arc::new(RwLock::new(
            providers
                .iter()
                .map(|provider| (provider.name.clone(), provider.clone()))
                .collect(),
        )),
        web_search_backends: Arc::new(web_search_backends),
        mcp_servers: Arc::new(mcp_servers),
        mcp_inventory: Arc::new(RwLock::new(HashMap::new())),
        arxiv,
        timeout: Duration::from_millis(timeout),
        max_round_trips,
        responses_stream_mode,
    };
    runtime.refresh_mcp_inventory().await;
    runtime
        .validate_registered_tools()
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
    Ok(runtime)
}

#[async_trait]
impl Plugin for ToolRuntime {
    fn name(&self) -> &str {
        "tool_runtime"
    }

    async fn on_request(&self, ctx: &mut RequestContext) -> Action {
        if ctx.method != Method::POST {
            return Action::Continue;
        }

        let Some(body) = ctx.body.as_ref() else {
            return Action::Continue;
        };
        let Ok(mut request_json) = serde_json::from_slice::<Value>(body) else {
            return Action::Continue;
        };

        let opt_in = match extract_opt_in(&mut request_json) {
            Ok(opt_in) => opt_in,
            Err(error) => return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error)),
        };
        let Some(opt_in) = opt_in else {
            return Action::Continue;
        };
        if !opt_in.enabled {
            return Action::Continue;
        }

        let virtual_key = match ctx.extensions.get::<VirtualKeyMeta>() {
            Some(meta) => meta.clone(),
            None => {
                return Action::Respond(json_error(
                    StatusCode::BAD_REQUEST,
                    "trp_tools requires a managed runtime key",
                ))
            }
        };

        let providers = self
            .providers
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        let provider = match providers.get(&virtual_key.provider_name) {
            Some(provider) => provider,
            None => {
                return Action::Respond(json_error(
                    StatusCode::BAD_GATEWAY,
                    "provider configuration not found for tool runtime",
                ))
            }
        };

        if !provider.supports_managed_tools() {
            return Action::Respond(json_error(
                StatusCode::BAD_REQUEST,
                "provider does not support gateway-managed tools",
            ));
        }

        let stream = request_json
            .get("stream")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        let available = self
            .governance
            .list_project_tools(Some(&virtual_key.project_id));
        let selected = match select_tools(
            &available,
            opt_in.names.as_deref(),
            &virtual_key.tool_approval_mode,
            virtual_key.allowed_tools.as_deref(),
        ) {
            Ok(selected) => selected,
            Err(error) => return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error)),
        };
        let selected_map: HashMap<String, ProjectToolRecord> = selected
            .iter()
            .map(|tool| (tool.tool_name.clone(), tool.clone()))
            .collect();

        let request_shape = request_shape_for(provider, ctx.uri.path());
        if stream
            && request_shape == ManagedToolRequestShape::OpenAiResponses
            && self.responses_stream_mode == ResponsesStreamMode::Strict
        {
            return Action::Respond(json_error(
                StatusCode::BAD_REQUEST,
                "managed tools for /v1/responses streaming require tool_runtime.responses_stream_mode = \"composed\"",
            ));
        }
        let allowed_tool_names = match extract_allowed_tool_names(&request_shape, &request_json) {
            Ok(names) => names,
            Err(error) => return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error)),
        };

        let merge_result = match request_shape {
            ManagedToolRequestShape::OpenAiChatCompletions => {
                merge_openai_chat_tools(&mut request_json, &selected)
            }
            ManagedToolRequestShape::OpenAiResponses => {
                merge_openai_responses_tools(&mut request_json, &selected)
            }
            ManagedToolRequestShape::AnthropicMessages => {
                merge_anthropic_tools(&mut request_json, &selected)
            }
        };
        if let Err(error) = merge_result {
            return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error));
        }

        let updated_body = match serde_json::to_vec(&request_json) {
            Ok(bytes) => bytes,
            Err(error) => {
                return Action::Respond(json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("failed to encode tool request: {error}"),
                ))
            }
        };
        if let Ok(value) = HeaderValue::from_str(&updated_body.len().to_string()) {
            ctx.headers.insert(CONTENT_LENGTH, value);
        }
        if let Some(candidates) = ctx.extensions.get_mut::<ProviderCandidates>() {
            for candidate in &mut candidates.0 {
                if let Ok(value) = HeaderValue::from_str(&updated_body.len().to_string()) {
                    candidate.headers.insert(CONTENT_LENGTH, value);
                }
            }
        }
        let forwarded_body = Bytes::from(updated_body);
        ctx.extensions
            .insert(ForwardedRequestBody(forwarded_body.clone()));
        ctx.body = Some(forwarded_body);
        ctx.extensions.insert(ToolRuntimeRequest {
            provider_name: virtual_key.provider_name.clone(),
            request_shape,
            project_id: virtual_key.project_id.clone(),
            request_json,
            selected_tools: selected_map,
            allowed_tool_names,
            stream,
        });

        Action::Continue
    }

    async fn transform_response(
        &self,
        ctx: &mut RequestContext,
        resp: Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>,
    ) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
        let Some(request_meta) = ctx.extensions.get::<ToolRuntimeRequest>().cloned() else {
            return resp;
        };
        let Some(upstream) = ctx.selected_upstream.clone() else {
            return resp;
        };

        let result = if request_meta.stream {
            self.run_streaming_loop(ctx, request_meta, upstream, resp)
                .await
        } else {
            self.run_non_streaming_loop(ctx, request_meta, upstream, resp)
                .await
        };

        match result {
            Ok(response) => response,
            Err(error) => json_error(StatusCode::BAD_GATEWAY, &error),
        }
    }
}

impl ToolRuntime {
    fn mcp_server_health_summary(
        state: &McpServerInventory,
    ) -> (String, Option<String>, Option<String>) {
        if !state.operator_enabled {
            return (
                "disabled".to_string(),
                state
                    .operator_state_reason
                    .clone()
                    .or_else(|| Some("MCP server is disabled by operator".to_string())),
                Some("enable".to_string()),
            );
        }

        if matches!(state.last_discovery_status.as_deref(), Some("error")) {
            return (
                "discovery_failed".to_string(),
                state
                    .last_discovery_error
                    .clone()
                    .or_else(|| state.last_error.clone()),
                Some("refresh".to_string()),
            );
        }

        if state.last_recovery_error.is_some() {
            return (
                "recovery_failed".to_string(),
                state.last_recovery_error.clone(),
                Some("refresh".to_string()),
            );
        }

        if matches!(state.last_call_status.as_deref(), Some("budget_exceeded")) {
            return (
                "budget_exhausted".to_string(),
                state
                    .last_budget_exceeded_error
                    .clone()
                    .or_else(|| state.last_call_error.clone()),
                Some("adjust_budget".to_string()),
            );
        }

        if matches!(state.last_session_reset_status.as_deref(), Some("error")) {
            return (
                "session_reset_failed".to_string(),
                state.last_session_reset_error.clone(),
                Some("refresh".to_string()),
            );
        }

        if matches!(state.last_call_status.as_deref(), Some("error")) {
            return (
                "degraded".to_string(),
                state
                    .last_call_error
                    .clone()
                    .or_else(|| state.last_error.clone()),
                Some(if state.session_id.is_some() {
                    "reset_session".to_string()
                } else {
                    "refresh".to_string()
                }),
            );
        }

        if state.reachable && state.protocol_version.is_some() {
            return ("ready".to_string(), None, None);
        }

        if !state.reachable {
            return (
                "unreachable".to_string(),
                state.last_error.clone(),
                Some("refresh".to_string()),
            );
        }

        (
            "unknown".to_string(),
            state.last_error.clone(),
            Some("refresh".to_string()),
        )
    }

    fn mcp_server_auth_status(
        server: &McpServerConfig,
        state: &McpServerInventory,
    ) -> (String, Option<String>) {
        match &server.auth {
            McpAuthConfig::None => ("none".to_string(), None),
            McpAuthConfig::StaticBearer { .. } => (
                if state.auth_last_error.is_some() {
                    "error".to_string()
                } else {
                    "ready".to_string()
                },
                state.auth_last_error.clone(),
            ),
            McpAuthConfig::OAuthClientCredentials { .. } => {
                if state.auth_last_error.is_some() {
                    ("error".to_string(), state.auth_last_error.clone())
                } else if state.auth_cached_access_token.is_some()
                    && auth_token_is_fresh(state.auth_token_expires_at_unix_ms)
                {
                    ("ready".to_string(), None)
                } else if state.auth_refreshes > 0 {
                    ("refreshing".to_string(), None)
                } else {
                    ("configured".to_string(), None)
                }
            }
            McpAuthConfig::OAuthAuthorizationCode { .. } => {
                if state.auth_last_error.is_some() {
                    ("error".to_string(), state.auth_last_error.clone())
                } else if state.auth_cached_access_token.is_some()
                    && auth_token_is_fresh(state.auth_token_expires_at_unix_ms)
                {
                    ("ready".to_string(), None)
                } else if state.auth_pending_state.is_some()
                    && state
                        .auth_pending_authorization_expires_at_unix_ms
                        .map(|expires_at| expires_at > current_unix_ms())
                        .unwrap_or(true)
                {
                    ("pending_authorization".to_string(), None)
                } else {
                    ("authorization_required".to_string(), None)
                }
            }
        }
    }

    fn snapshot_mcp_server(
        &self,
        name: &str,
        server: &McpServerConfig,
        state: McpServerInventory,
    ) -> ToolRuntimeMcpServerSnapshot {
        let (health_state, health_reason, recommended_action) =
            Self::mcp_server_health_summary(&state);
        let (auth_status, auth_last_error) = Self::mcp_server_auth_status(server, &state);
        ToolRuntimeMcpServerSnapshot {
            name: name.to_string(),
            transport: server.transport_name().to_string(),
            auth_mode: server.auth_mode().map(ToString::to_string),
            auth_status,
            auth_last_error,
            auth_last_discovery_error: state.auth_last_discovery_error,
            auth_refreshes: state.auth_refreshes,
            auth_last_refreshed_at: state.auth_last_refreshed_at,
            auth_token_expires_at_unix_ms: state.auth_token_expires_at_unix_ms,
            auth_resource: state.auth_resource,
            auth_authorization_url: state.auth_authorization_url,
            auth_token_url: state.auth_token_url,
            auth_authorization_server_url: state.auth_authorization_server_url,
            auth_pending_authorization: state.auth_pending_state.is_some(),
            auth_pending_authorization_expires_at_unix_ms: state
                .auth_pending_authorization_expires_at_unix_ms,
            url: server.display_url(),
            method: server.display_method(),
            command: server.display_command(),
            args: server.display_args(),
            cwd: server.display_cwd(),
            timeout_ms: server.timeout_ms.unwrap_or(self.timeout.as_millis() as u64),
            max_retries: server.max_retries,
            max_calls_per_request: server.max_calls_per_request,
            max_total_time_ms: server.max_total_time_ms,
            max_output_tokens: server.max_output_tokens,
            operator_state: if state.operator_enabled {
                "enabled".to_string()
            } else {
                "disabled".to_string()
            },
            operator_state_at: state.operator_state_at,
            operator_state_actor: state.operator_state_actor,
            operator_state_reason: state.operator_state_reason,
            health_state,
            health_reason,
            recommended_action,
            reachable: state.reachable,
            protocol_version: state.protocol_version,
            session_id_present: state.session_id.is_some(),
            discovered_tools: state.discovered_tools,
            last_error: state.last_error,
            discovery_refreshes: state.discovery_refreshes,
            last_discovery_at: state.last_discovery_at,
            last_discovery_status: state.last_discovery_status,
            last_discovery_error: state.last_discovery_error,
            total_calls: state.total_calls,
            successful_calls: state.successful_calls,
            failed_calls: state.failed_calls,
            retried_calls: state.retried_calls,
            session_reinitializations: state.session_reinitializations,
            last_session_reinitialized_at: state.last_session_reinitialized_at,
            last_recovery_error: state.last_recovery_error,
            budget_exceeded_calls: state.budget_exceeded_calls,
            last_budget_exceeded_at: state.last_budget_exceeded_at,
            last_budget_exceeded_error: state.last_budget_exceeded_error,
            last_session_reset_at: state.last_session_reset_at,
            last_session_reset_status: state.last_session_reset_status,
            last_session_reset_error: state.last_session_reset_error,
            last_session_reset_http_status: state.last_session_reset_http_status,
            last_call_at: state.last_call_at,
            last_call_tool: state.last_call_tool,
            last_call_status: state.last_call_status,
            last_call_error: state.last_call_error,
            last_call_http_status: state.last_call_http_status,
        }
    }

    pub fn status(&self) -> ToolRuntimeStatusSnapshot {
        let provider_configs = self
            .providers
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut providers = provider_configs
            .values()
            .map(|provider| ToolRuntimeProviderSnapshot {
                name: provider.name.clone(),
                family: provider.family_kind().as_str().to_string(),
                surfaces: provider.surfaces().clone(),
                semantics: provider.runtime_semantics(),
                data_collection: provider
                    .routing_metadata
                    .data_collection
                    .as_ref()
                    .map(|value| value.as_str().to_string()),
                zdr: provider.routing_metadata.zdr,
                distillable_text: provider.routing_metadata.distillable_text,
                quantizations: provider.routing_metadata.quantizations.clone(),
                supported_parameter_families: provider
                    .routing_metadata
                    .supported_parameter_families
                    .clone(),
                models: provider.models.clone(),
                timeout_secs: provider.timeout_secs,
            })
            .collect::<Vec<_>>();
        providers.sort_by(|left, right| left.name.cmp(&right.name));

        let mut web_search_backends = self
            .web_search_backends
            .iter()
            .map(|(name, backend)| ToolRuntimeBackendSnapshot {
                name: name.clone(),
                url: backend.url.clone(),
                method: backend.method.clone(),
            })
            .collect::<Vec<_>>();
        web_search_backends.sort_by(|left, right| left.name.cmp(&right.name));
        let inventory = self
            .mcp_inventory
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut mcp_servers = self
            .mcp_servers
            .iter()
            .map(|(name, server)| {
                let state = inventory.get(name).cloned().unwrap_or_default();
                self.snapshot_mcp_server(name, server, state)
            })
            .collect::<Vec<_>>();
        mcp_servers.sort_by(|left, right| left.name.cmp(&right.name));

        let tools = self.governance.list_project_tools(None);
        let registered_tools_total = tools.len() as u64;
        let enabled_tools_total = tools.iter().filter(|tool| tool.enabled).count() as u64;
        let mut executor_counts = HashMap::<String, (u64, u64)>::new();
        for tool in tools {
            let entry = executor_counts
                .entry(tool.executor_kind.clone())
                .or_insert((0, 0));
            entry.0 += 1;
            if tool.enabled {
                entry.1 += 1;
            }
        }
        let mut executors = executor_counts
            .into_iter()
            .map(
                |(executor_kind, (total, enabled))| ToolRuntimeExecutorSummary {
                    executor_kind,
                    total,
                    enabled,
                },
            )
            .collect::<Vec<_>>();
        executors.sort_by(|left, right| left.executor_kind.cmp(&right.executor_kind));

        ToolRuntimeStatusSnapshot {
            default_timeout_ms: self.timeout.as_millis() as u64,
            max_round_trips: self.max_round_trips,
            responses_stream_mode: self.responses_stream_mode.as_str().to_string(),
            supported_executors: vec![
                "webhook".to_string(),
                "web_search".to_string(),
                "mcp".to_string(),
                "arxiv_search".to_string(),
            ],
            web_search_backends,
            mcp_servers,
            arxiv_base_url: self.arxiv.base_url.clone(),
            arxiv_default_max_results: self.arxiv.default_max_results,
            providers,
            registered_tools_total,
            enabled_tools_total,
            executors,
        }
    }

    pub fn set_provider_configs(&self, providers: &[ProviderKeyConfig]) {
        *self
            .providers
            .write()
            .unwrap_or_else(|poison| poison.into_inner()) = providers
            .iter()
            .map(|provider| (provider.name.clone(), provider.clone()))
            .collect();
    }

    fn update_mcp_inventory<F>(&self, server_name: &str, update: F)
    where
        F: FnOnce(&mut McpServerInventory),
    {
        let mut inventory = self
            .mcp_inventory
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        let state = inventory.entry(server_name.to_string()).or_default();
        update(state);
    }

    fn record_mcp_auth_ready(&self, server_name: &str, expires_at_unix_ms: Option<u64>) {
        let now = current_timestamp_string();
        self.update_mcp_inventory(server_name, |state| {
            state.auth_last_error = None;
            state.auth_last_refreshed_at = Some(now);
            state.auth_token_expires_at_unix_ms = expires_at_unix_ms;
        });
    }

    fn record_mcp_auth_resolution(
        &self,
        server_name: &str,
        resolved: &ResolvedMcpOAuthClientCredentials,
    ) {
        self.update_mcp_inventory(server_name, |state| {
            state.auth_last_discovery_error = None;
            state.auth_resource = resolved.resource.clone();
            state.auth_token_url = Some(resolved.token_url.clone());
            state.auth_authorization_server_url = resolved.authorization_server_url.clone();
        });
    }

    fn record_mcp_auth_code_resolution(
        &self,
        server_name: &str,
        resolved: &ResolvedMcpOAuthAuthorizationCode,
    ) {
        self.update_mcp_inventory(server_name, |state| {
            state.auth_last_discovery_error = None;
            state.auth_resource = resolved.resource.clone();
            state.auth_authorization_url = Some(resolved.authorization_url.clone());
            state.auth_token_url = Some(resolved.token_url.clone());
            state.auth_authorization_server_url = resolved.authorization_server_url.clone();
        });
    }

    fn record_mcp_auth_refresh(&self, server_name: &str, token: &McpOAuthToken) {
        let now = current_timestamp_string();
        self.update_mcp_inventory(server_name, |state| {
            state.auth_cached_access_token = Some(token.access_token.clone());
            if token.refresh_token.is_some() {
                state.auth_cached_refresh_token = token.refresh_token.clone();
            }
            state.auth_token_expires_at_unix_ms = token.expires_at_unix_ms;
            state.auth_refreshes += 1;
            state.auth_last_refreshed_at = Some(now);
            state.auth_last_error = None;
            state.auth_pending_state = None;
            state.auth_pending_code_verifier = None;
            state.auth_pending_authorization_expires_at_unix_ms = None;
        });
    }

    fn record_mcp_auth_error(&self, server_name: &str, error: &str) {
        self.update_mcp_inventory(server_name, |state| {
            state.auth_last_error = Some(error.to_string());
        });
    }

    fn record_mcp_auth_discovery_error(&self, server_name: &str, error: &str) {
        self.update_mcp_inventory(server_name, |state| {
            state.auth_last_error = Some(error.to_string());
            state.auth_last_discovery_error = Some(error.to_string());
        });
    }

    async fn ensure_mcp_auth_bearer_token(
        &self,
        server_name: &str,
        server: &McpServerConfig,
    ) -> Result<Option<String>, String> {
        match &server.auth {
            McpAuthConfig::None => Ok(None),
            McpAuthConfig::StaticBearer { access_token } => {
                self.record_mcp_auth_ready(server_name, None);
                Ok(Some(access_token.clone()))
            }
            McpAuthConfig::OAuthClientCredentials { .. } => {
                let current = self
                    .mcp_inventory
                    .read()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .get(server_name)
                    .cloned()
                    .unwrap_or_default();
                if let Some(token) = current.auth_cached_access_token.clone() {
                    if auth_token_is_fresh(current.auth_token_expires_at_unix_ms) {
                        return Ok(Some(token));
                    }
                }
                let resolved = resolve_mcp_oauth_client_credentials(
                    server_name,
                    &server.auth,
                    &server.transport,
                    Some(&current),
                    mcp_server_timeout(server, self.timeout),
                )
                .await
                .inspect_err(|error| self.record_mcp_auth_discovery_error(server_name, error))?;
                self.record_mcp_auth_resolution(server_name, &resolved);
                let token = fetch_mcp_oauth_access_token(
                    server_name,
                    &resolved,
                    mcp_server_timeout(server, self.timeout),
                )
                .await
                .inspect_err(|error| self.record_mcp_auth_error(server_name, error))?;
                self.record_mcp_auth_refresh(server_name, &token);
                Ok(Some(token.access_token))
            }
            McpAuthConfig::OAuthAuthorizationCode { .. } => {
                let current = self
                    .mcp_inventory
                    .read()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .get(server_name)
                    .cloned()
                    .unwrap_or_default();
                if let Some(token) = current.auth_cached_access_token.clone() {
                    if auth_token_is_fresh(current.auth_token_expires_at_unix_ms) {
                        return Ok(Some(token));
                    }
                }
                let resolved = resolve_mcp_oauth_authorization_code(
                    server_name,
                    &server.auth,
                    &server.transport,
                    Some(&current),
                    mcp_server_timeout(server, self.timeout),
                )
                .await
                .inspect_err(|error| self.record_mcp_auth_discovery_error(server_name, error))?;
                self.record_mcp_auth_code_resolution(server_name, &resolved);
                if let Some(refresh_token) = current.auth_cached_refresh_token.clone() {
                    let token = refresh_mcp_oauth_access_token(
                        server_name,
                        &resolved,
                        &refresh_token,
                        mcp_server_timeout(server, self.timeout),
                    )
                    .await
                    .inspect_err(|error| self.record_mcp_auth_error(server_name, error))?;
                    self.record_mcp_auth_refresh(server_name, &token);
                    return Ok(Some(token.access_token));
                }
                let message = format!(
                    "MCP server '{}' requires OAuth authorization; start the flow via the management API",
                    server_name
                );
                self.record_mcp_auth_error(server_name, &message);
                Err(message)
            }
        }
    }

    fn record_mcp_tool_call_started(&self, server_name: &str, tool_name: &str) {
        let now = current_timestamp_string();
        self.update_mcp_inventory(server_name, |state| {
            state.total_calls += 1;
            state.last_call_at = Some(now);
            state.last_call_tool = Some(tool_name.to_string());
            state.last_call_status = Some("running".to_string());
            state.last_call_error = None;
            state.last_call_http_status = None;
        });
    }

    fn record_mcp_tool_call_succeeded(&self, server_name: &str, tool_name: &str, retry_count: u32) {
        let now = current_timestamp_string();
        self.update_mcp_inventory(server_name, |state| {
            state.reachable = true;
            state.successful_calls += 1;
            state.retried_calls += u64::from(retry_count);
            state.last_error = None;
            state.last_recovery_error = None;
            state.last_call_at = Some(now);
            state.last_call_tool = Some(tool_name.to_string());
            state.last_call_status = Some("ok".to_string());
            state.last_call_error = None;
            state.last_call_http_status = None;
        });
    }

    fn record_mcp_tool_call_failed(
        &self,
        server_name: &str,
        tool_name: &str,
        error: &str,
        status_code: Option<StatusCode>,
        retry_count: u32,
    ) {
        let now = current_timestamp_string();
        self.update_mcp_inventory(server_name, |state| {
            state.reachable = status_code.is_some();
            state.failed_calls += 1;
            state.retried_calls += u64::from(retry_count);
            state.last_error = Some(error.to_string());
            state.last_call_at = Some(now);
            state.last_call_tool = Some(tool_name.to_string());
            state.last_call_status = Some("error".to_string());
            state.last_call_error = Some(error.to_string());
            state.last_call_http_status = status_code.map(|status| status.as_u16());
        });
    }

    fn record_mcp_tool_call_budget_exceeded(
        &self,
        server_name: &str,
        tool_name: &str,
        error: &str,
        count_as_new_call: bool,
    ) {
        let now = current_timestamp_string();
        self.update_mcp_inventory(server_name, |state| {
            if count_as_new_call {
                state.total_calls += 1;
            }
            state.failed_calls += 1;
            state.budget_exceeded_calls += 1;
            state.last_error = Some(error.to_string());
            state.last_budget_exceeded_at = Some(now.clone());
            state.last_budget_exceeded_error = Some(error.to_string());
            state.last_call_at = Some(now);
            state.last_call_tool = Some(tool_name.to_string());
            state.last_call_status = Some("budget_exceeded".to_string());
            state.last_call_error = Some(error.to_string());
            state.last_call_http_status = None;
        });
    }

    fn record_mcp_tool_call_disabled(&self, server_name: &str, tool_name: &str, error: &str) {
        let now = current_timestamp_string();
        self.update_mcp_inventory(server_name, |state| {
            state.total_calls += 1;
            state.failed_calls += 1;
            state.last_error = Some(error.to_string());
            state.last_call_at = Some(now);
            state.last_call_tool = Some(tool_name.to_string());
            state.last_call_status = Some("disabled".to_string());
            state.last_call_error = Some(error.to_string());
            state.last_call_http_status = None;
        });
    }

    fn record_mcp_session_reinitialized(&self, server_name: &str) {
        let now = current_timestamp_string();
        self.update_mcp_inventory(server_name, |state| {
            state.session_reinitializations += 1;
            state.last_session_reinitialized_at = Some(now);
            state.last_recovery_error = None;
        });
    }

    fn record_mcp_session_recovery_failed(&self, server_name: &str, error: &str) {
        self.update_mcp_inventory(server_name, |state| {
            state.last_recovery_error = Some(error.to_string());
        });
    }

    fn mcp_server_disabled_message(&self, server_name: &str) -> Option<String> {
        let inventory = self
            .mcp_inventory
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        let state = inventory.get(server_name)?;
        if state.operator_enabled {
            return None;
        }
        Some(match state.operator_state_reason.as_deref() {
            Some(reason) if !reason.is_empty() => {
                format!("MCP server '{server_name}' is disabled by operator: {reason}")
            }
            _ => format!("MCP server '{server_name}' is disabled by operator"),
        })
    }

    fn validate_registered_tools(&self) -> Result<(), String> {
        let enabled_tools = self
            .governance
            .list_project_tools(None)
            .into_iter()
            .filter(|tool| tool.enabled)
            .collect::<Vec<_>>();
        if enabled_tools.is_empty() {
            return Ok(());
        }
        let providers = self
            .providers
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        if !providers
            .values()
            .any(|provider| provider.supports_managed_tools())
        {
            return Err(
                "tool_runtime has enabled project tools but no providers expose a managed tools surface"
                    .to_string(),
            );
        }
        for tool in &enabled_tools {
            self.validate_tool_record(tool)?;
        }
        Ok(())
    }

    fn validate_tool_record(&self, tool: &ProjectToolRecord) -> Result<(), String> {
        match tool.executor_kind.as_str() {
            "webhook" => {
                let _ = parse_webhook_executor_config(tool)?;
                Ok(())
            }
            "web_search" => {
                let config = parse_web_search_executor_config(tool)?;
                if !self.web_search_backends.contains_key(&config.backend) {
                    return Err(format!(
                        "tool '{}' references unknown web_search backend '{}'",
                        tool.tool_name, config.backend
                    ));
                }
                Ok(())
            }
            "mcp" => {
                let config = parse_mcp_executor_config(tool)?;
                if !self.mcp_servers.contains_key(&config.server) {
                    return Err(format!(
                        "tool '{}' references unknown mcp server '{}'",
                        tool.tool_name, config.server
                    ));
                }
                let inventory = self
                    .mcp_inventory
                    .read()
                    .unwrap_or_else(|poison| poison.into_inner());
                if let Some(state) = inventory.get(&config.server) {
                    if state.reachable
                        && !state
                            .discovered_tools
                            .iter()
                            .any(|name| name == &config.remote_tool)
                    {
                        return Err(format!(
                            "tool '{}' references remote MCP tool '{}' that was not found on server '{}'",
                            tool.tool_name, config.remote_tool, config.server
                        ));
                    }
                }
                Ok(())
            }
            "arxiv_search" => {
                let _ = parse_arxiv_executor_config(tool, self.arxiv.default_max_results)?;
                Ok(())
            }
            other => Err(format!(
                "tool '{}' uses unsupported executor_kind '{}'",
                tool.tool_name, other
            )),
        }
    }

    pub async fn refresh_mcp_server(
        &self,
        server_name: &str,
    ) -> Result<ToolRuntimeMcpServerSnapshot, String> {
        let server = self
            .mcp_servers
            .get(server_name)
            .ok_or_else(|| format!("unknown mcp server '{}'", server_name))?;
        let current = self
            .mcp_inventory
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(server_name)
            .cloned();
        let state = self
            .discover_mcp_server(server_name, server, current.as_ref())
            .await;
        let snapshot = self.snapshot_mcp_server(server_name, server, state.clone());
        self.mcp_inventory
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(server_name.to_string(), state);
        Ok(snapshot)
    }

    pub async fn reset_mcp_server_session(
        &self,
        server_name: &str,
    ) -> Result<ToolRuntimeMcpServerSnapshot, String> {
        let server = self
            .mcp_servers
            .get(server_name)
            .ok_or_else(|| format!("unknown mcp server '{}'", server_name))?;
        let timeout = mcp_server_timeout(server, self.timeout);
        let current = self
            .mcp_inventory
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(server_name)
            .cloned()
            .unwrap_or_default();
        let now = current_timestamp_string();
        let mut next = current.clone();
        next.protocol_version = None;
        next.session_id = None;

        if let Some(session_id) = current.session_id.as_deref() {
            if let McpTransportConfig::Http {
                url,
                headers: default_headers,
                ..
            } = &server.transport
            {
                let bearer_token = self
                    .ensure_mcp_auth_bearer_token(server_name, server)
                    .await?;
                let mut headers = build_mcp_auth_headers(default_headers, bearer_token.as_deref());
                headers.insert("accept".to_string(), MCP_ACCEPT_HEADER.to_string());
                if let Some(protocol_version) = current.protocol_version.as_deref() {
                    headers.insert(
                        MCP_PROTOCOL_HEADER.to_string(),
                        protocol_version.to_string(),
                    );
                }
                headers.insert(MCP_SESSION_HEADER.to_string(), session_id.to_string());
                match execute_runtime_http_request(
                    url,
                    "DELETE",
                    &headers,
                    Bytes::new(),
                    timeout,
                    &format!("MCP server '{server_name}' session delete"),
                )
                .await
                {
                    Ok(_) => {
                        next.reachable = true;
                        next.last_error = None;
                        next.last_session_reset_at = Some(now);
                        next.last_session_reset_status = Some("ok".to_string());
                        next.last_session_reset_error = None;
                        next.last_session_reset_http_status = None;
                    }
                    Err(RuntimeHttpError::Status(StatusCode::NOT_FOUND, message)) => {
                        next.reachable = true;
                        next.last_error = None;
                        next.last_session_reset_at = Some(now);
                        next.last_session_reset_status = Some("gone".to_string());
                        next.last_session_reset_error = Some(message);
                        next.last_session_reset_http_status = Some(StatusCode::NOT_FOUND.as_u16());
                    }
                    Err(error) => {
                        let status_code = error.status_code();
                        let message = error.into_message();
                        self.update_mcp_inventory(server_name, |state| {
                            state.last_error = Some(message.clone());
                            state.last_session_reset_at = Some(now.clone());
                            state.last_session_reset_status = Some("error".to_string());
                            state.last_session_reset_error = Some(message.clone());
                            state.last_session_reset_http_status =
                                status_code.map(|status| status.as_u16());
                        });
                        return Err(message);
                    }
                }
            }
        } else {
            next.last_session_reset_at = Some(now);
            next.last_session_reset_status = Some("no_session".to_string());
            next.last_session_reset_error = None;
            next.last_session_reset_http_status = None;
        }

        let snapshot = self.snapshot_mcp_server(server_name, server, next.clone());
        self.mcp_inventory
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(server_name.to_string(), next);
        Ok(snapshot)
    }

    pub fn disable_mcp_server(
        &self,
        server_name: &str,
        actor: Option<String>,
        reason: Option<String>,
    ) -> Result<ToolRuntimeMcpServerSnapshot, String> {
        let server = self
            .mcp_servers
            .get(server_name)
            .ok_or_else(|| format!("unknown mcp server '{}'", server_name))?;
        let now = current_timestamp_string();
        let next = {
            let mut inventory = self
                .mcp_inventory
                .write()
                .unwrap_or_else(|poison| poison.into_inner());
            let state = inventory.entry(server_name.to_string()).or_default();
            state.operator_enabled = false;
            state.operator_state_at = Some(now);
            state.operator_state_actor = actor;
            state.operator_state_reason = reason;
            state.clone()
        };
        Ok(self.snapshot_mcp_server(server_name, server, next))
    }

    pub fn enable_mcp_server(
        &self,
        server_name: &str,
        actor: Option<String>,
        reason: Option<String>,
    ) -> Result<ToolRuntimeMcpServerSnapshot, String> {
        let server = self
            .mcp_servers
            .get(server_name)
            .ok_or_else(|| format!("unknown mcp server '{}'", server_name))?;
        let now = current_timestamp_string();
        let next = {
            let mut inventory = self
                .mcp_inventory
                .write()
                .unwrap_or_else(|poison| poison.into_inner());
            let state = inventory.entry(server_name.to_string()).or_default();
            state.operator_enabled = true;
            state.operator_state_at = Some(now);
            state.operator_state_actor = actor;
            state.operator_state_reason = reason;
            state.clone()
        };
        Ok(self.snapshot_mcp_server(server_name, server, next))
    }

    pub async fn begin_mcp_oauth_authorization(
        &self,
        server_name: &str,
    ) -> Result<ToolRuntimeMcpOAuthAuthorizationRequest, String> {
        let server = self
            .mcp_servers
            .get(server_name)
            .ok_or_else(|| format!("unknown mcp server '{}'", server_name))?;
        let current = self
            .mcp_inventory
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(server_name)
            .cloned()
            .unwrap_or_default();
        if !matches!(server.auth, McpAuthConfig::OAuthAuthorizationCode { .. }) {
            return Err(format!(
                "MCP server '{}' is not configured for OAuth authorization code",
                server_name
            ));
        }
        let resolved = resolve_mcp_oauth_authorization_code(
            server_name,
            &server.auth,
            &server.transport,
            Some(&current),
            mcp_server_timeout(server, self.timeout),
        )
        .await
        .inspect_err(|error| self.record_mcp_auth_discovery_error(server_name, error))?;
        self.record_mcp_auth_code_resolution(server_name, &resolved);
        let state = generate_mcp_oauth_secret(24);
        let code_verifier = generate_mcp_oauth_secret(48);
        let code_challenge = build_pkce_code_challenge(&code_verifier);
        let authorization_url =
            build_mcp_oauth_authorization_url(&resolved, &state, &code_challenge);
        let expires_at_unix_ms = current_unix_ms().saturating_add(MCP_OAUTH_PENDING_TTL_MS);
        self.update_mcp_inventory(server_name, |inventory| {
            inventory.auth_pending_state = Some(state.clone());
            inventory.auth_pending_code_verifier = Some(code_verifier.clone());
            inventory.auth_pending_authorization_expires_at_unix_ms = Some(expires_at_unix_ms);
            inventory.auth_authorization_url = Some(resolved.authorization_url.clone());
            inventory.auth_last_error = None;
        });
        Ok(ToolRuntimeMcpOAuthAuthorizationRequest {
            server_name: server_name.to_string(),
            authorization_url,
            redirect_uri: resolved.redirect_uri,
            state,
            expires_at_unix_ms,
        })
    }

    pub async fn complete_mcp_oauth_authorization(
        &self,
        server_name: &str,
        state: &str,
        code: &str,
    ) -> Result<ToolRuntimeMcpServerSnapshot, String> {
        let server = self
            .mcp_servers
            .get(server_name)
            .ok_or_else(|| format!("unknown mcp server '{}'", server_name))?;
        if !matches!(server.auth, McpAuthConfig::OAuthAuthorizationCode { .. }) {
            return Err(format!(
                "MCP server '{}' is not configured for OAuth authorization code",
                server_name
            ));
        }
        let current = self
            .mcp_inventory
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(server_name)
            .cloned()
            .unwrap_or_default();
        let pending_state = current.auth_pending_state.clone().ok_or_else(|| {
            format!(
                "MCP server '{}' has no pending OAuth authorization",
                server_name
            )
        })?;
        if current
            .auth_pending_authorization_expires_at_unix_ms
            .map(|expires_at| expires_at <= current_unix_ms())
            .unwrap_or(false)
        {
            return Err(format!(
                "MCP server '{}' OAuth authorization request has expired",
                server_name
            ));
        }
        if pending_state != state {
            return Err(format!(
                "MCP server '{}' OAuth authorization state did not match",
                server_name
            ));
        }
        let code_verifier = current.auth_pending_code_verifier.clone().ok_or_else(|| {
            format!(
                "MCP server '{}' OAuth authorization is missing a PKCE verifier",
                server_name
            )
        })?;
        let resolved = resolve_mcp_oauth_authorization_code(
            server_name,
            &server.auth,
            &server.transport,
            Some(&current),
            mcp_server_timeout(server, self.timeout),
        )
        .await
        .inspect_err(|error| self.record_mcp_auth_discovery_error(server_name, error))?;
        self.record_mcp_auth_code_resolution(server_name, &resolved);
        let token = exchange_mcp_oauth_authorization_code(
            server_name,
            &resolved,
            code,
            &code_verifier,
            mcp_server_timeout(server, self.timeout),
        )
        .await
        .inspect_err(|error| self.record_mcp_auth_error(server_name, error))?;
        self.record_mcp_auth_refresh(server_name, &token);
        self.refresh_mcp_server(server_name).await
    }

    async fn refresh_mcp_inventory(&self) {
        let current = self
            .mcp_inventory
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        let mut discovered = HashMap::new();
        for (name, server) in self.mcp_servers.iter() {
            let state = self
                .discover_mcp_server(name, server, current.get(name))
                .await;
            discovered.insert(name.clone(), state);
        }
        *self
            .mcp_inventory
            .write()
            .unwrap_or_else(|poison| poison.into_inner()) = discovered;
    }

    async fn ensure_mcp_server_ready(
        &self,
        server_name: &str,
    ) -> Result<McpServerInventory, String> {
        let current = self
            .mcp_inventory
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(server_name)
            .cloned()
            .unwrap_or_default();
        if current.protocol_version.is_some() {
            return Ok(current);
        }

        let server = self
            .mcp_servers
            .get(server_name)
            .ok_or_else(|| format!("unknown mcp server '{}'", server_name))?;
        let state = match self
            .discover_mcp_server(server_name, server, Some(&current))
            .await
        {
            state => state,
        };
        let mut inventory = self
            .mcp_inventory
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        inventory.insert(server_name.to_string(), state.clone());
        drop(inventory);
        if state.protocol_version.is_some() {
            Ok(state)
        } else {
            Err(state
                .last_error
                .unwrap_or_else(|| format!("failed to initialize mcp server '{}'", server_name)))
        }
    }

    async fn discover_mcp_server(
        &self,
        server_name: &str,
        server: &McpServerConfig,
        previous: Option<&McpServerInventory>,
    ) -> McpServerInventory {
        match self.establish_mcp_server_session(server_name, server).await {
            Ok(mut state) => {
                let latest = self
                    .mcp_inventory
                    .read()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .get(server_name)
                    .cloned();
                state.discovery_refreshes = previous
                    .map(|state| state.discovery_refreshes)
                    .unwrap_or(0)
                    .saturating_add(1);
                let now = current_timestamp_string();
                state.last_discovery_at = Some(now);
                state.last_discovery_status = Some("ok".to_string());
                state.last_discovery_error = None;
                merge_mcp_server_inventory(latest.as_ref().or(previous), state)
            }
            Err(error) => {
                let latest = self
                    .mcp_inventory
                    .read()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .get(server_name)
                    .cloned();
                tracing::warn!(
                    server = %server_name,
                    error = %error,
                    "tool_runtime failed to discover MCP tools"
                );
                let now = current_timestamp_string();
                let mut state = McpServerInventory {
                    reachable: false,
                    protocol_version: None,
                    session_id: None,
                    discovered_tools: Vec::new(),
                    last_error: Some(error.clone()),
                    discovery_refreshes: previous
                        .map(|state| state.discovery_refreshes)
                        .unwrap_or(0)
                        .saturating_add(1),
                    last_discovery_at: Some(now),
                    last_discovery_status: Some("error".to_string()),
                    last_discovery_error: Some(error),
                    ..Default::default()
                };
                state = merge_mcp_server_inventory(latest.as_ref().or(previous), state);
                state.protocol_version = None;
                state.session_id = None;
                state.discovered_tools = Vec::new();
                state.reachable = false;
                state
            }
        }
    }

    async fn establish_mcp_server_session(
        &self,
        server_name: &str,
        server: &McpServerConfig,
    ) -> Result<McpServerInventory, String> {
        let timeout = mcp_server_timeout(server, self.timeout);
        let bearer_token = self
            .ensure_mcp_auth_bearer_token(server_name, server)
            .await?;
        let (protocol_version, session_id, tools) = match &server.transport {
            McpTransportConfig::Http { .. } => {
                let initialized =
                    initialize_mcp_server(server_name, server, bearer_token.as_deref(), timeout)
                        .await?;
                send_mcp_initialized_notification(
                    server_name,
                    server,
                    &initialized,
                    bearer_token.as_deref(),
                    timeout,
                )
                .await?;
                let tools = fetch_mcp_server_tools(
                    server_name,
                    server,
                    &initialized,
                    bearer_token.as_deref(),
                    timeout,
                )
                .await?;
                (initialized.protocol_version, initialized.session_id, tools)
            }
            McpTransportConfig::Sse { .. } => {
                let mut session =
                    start_mcp_sse_session(server_name, server, bearer_token.as_deref(), timeout)
                        .await?;
                let initialized =
                    initialize_mcp_server_sse(server_name, server, &mut session, timeout).await?;
                send_mcp_initialized_notification_sse(server_name, server, &session, timeout)
                    .await?;
                let tools =
                    fetch_mcp_server_tools_sse(server_name, server, &mut session, timeout).await?;
                (initialized.protocol_version, initialized.session_id, tools)
            }
            McpTransportConfig::Stdio { .. } => {
                let (mut process, initialized) =
                    establish_mcp_stdio_process(server_name, server, timeout).await?;
                let tools =
                    fetch_mcp_server_tools_stdio(server_name, &mut process, timeout).await?;
                (initialized.protocol_version, initialized.session_id, tools)
            }
        };
        Ok(McpServerInventory {
            reachable: true,
            protocol_version: Some(protocol_version),
            session_id,
            discovered_tools: tools,
            last_error: None,
            ..Default::default()
        })
    }

    async fn run_non_streaming_loop(
        &self,
        ctx: &mut RequestContext,
        request_meta: ToolRuntimeRequest,
        upstream: String,
        initial_resp: Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>,
    ) -> Result<Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>, String> {
        let client = build_client();
        let original_headers = initial_resp.headers().clone();
        let (mut status, mut headers, mut body_bytes) = collect_response(initial_resp).await?;
        let mut current_request = current_forwarded_request(ctx, &request_meta)?;
        let mut total_input_tokens = 0u64;
        let mut total_output_tokens = 0u64;
        let mut round_trips = 0usize;
        let mut budget_state = ToolExecutionBudgetState::default();
        let mut safety_audit = ctx.extensions.get::<SafetyAudit>().cloned();
        let mut semantic_audit = ctx.extensions.get::<SemanticSafetyAudit>().cloned();
        let mut tool_audit = ctx
            .extensions
            .get::<ToolRuntimeAudit>()
            .cloned()
            .unwrap_or_default();

        loop {
            let body_json = match serde_json::from_slice::<Value>(&body_bytes) {
                Ok(body) => body,
                Err(_) => {
                    apply_aggregated_audits(ctx, safety_audit, semantic_audit, tool_audit);
                    return Ok(response_from_bytes(
                        status,
                        headers,
                        body_bytes,
                        &original_headers,
                    ));
                }
            };

            let usage = match request_meta.request_shape {
                ManagedToolRequestShape::OpenAiChatCompletions => {
                    parse_openai_chat_usage(&body_json)
                }
                ManagedToolRequestShape::OpenAiResponses => {
                    parse_openai_responses_usage(&body_json)
                }
                ManagedToolRequestShape::AnthropicMessages => parse_anthropic_usage(&body_json),
            };
            if let Some((input_tokens, output_tokens)) = usage {
                total_input_tokens = total_input_tokens.saturating_add(input_tokens);
                total_output_tokens = total_output_tokens.saturating_add(output_tokens);
            }

            let tool_calls = match request_meta.request_shape {
                ManagedToolRequestShape::OpenAiChatCompletions => {
                    extract_openai_chat_tool_calls(&body_json)?
                }
                ManagedToolRequestShape::OpenAiResponses => {
                    extract_openai_responses_tool_calls(&body_json)?
                }
                ManagedToolRequestShape::AnthropicMessages => {
                    extract_anthropic_tool_calls(&body_json)?
                }
            };
            if tool_calls.is_empty() {
                if round_trips > 0 {
                    ctx.extensions.insert(ToolUsageOverride {
                        input_tokens: total_input_tokens,
                        output_tokens: total_output_tokens,
                    });
                }
                apply_aggregated_audits(ctx, safety_audit, semantic_audit, tool_audit);
                return Ok(response_from_bytes(
                    status,
                    headers,
                    body_bytes,
                    &original_headers,
                ));
            }

            round_trips += 1;
            if round_trips > self.max_round_trips {
                return Err("tool execution exceeded max_round_trips".to_string());
            }

            let mut tool_outputs = Vec::with_capacity(tool_calls.len());
            for tool_call in &tool_calls {
                ensure_allowed_tool_call(&request_meta, &tool_call.name)?;
                let Some(tool) = request_meta.selected_tools.get(&tool_call.name) else {
                    return Err(format!(
                        "tool '{}' is not registered for this project",
                        tool_call.name
                    ));
                };
                let output = self
                    .execute_tool(
                        &request_meta.project_id,
                        &request_meta.provider_name,
                        tool,
                        &tool_call.arguments,
                        &mut budget_state,
                    )
                    .await
                    .unwrap_or_else(|error| tool_error_payload("tool_execution_failed", &error));
                let replayed = self
                    .replay_tool_output(
                        ctx,
                        &request_meta,
                        &current_request,
                        ToolOutput {
                            call_id: tool_call.call_id.clone(),
                            tool_name: tool_call.name.clone(),
                            content: output,
                            assistant_message: tool_call.assistant_message.clone(),
                        },
                    )
                    .await?;
                merge_safety_audit(&mut safety_audit, replayed.safety_audit);
                merge_semantic_audit(&mut semantic_audit, replayed.semantic_audit);
                record_tool_trace(&mut tool_audit, tool, &replayed.output);
                tool_outputs.push(replayed.output);
            }

            match request_meta.request_shape {
                ManagedToolRequestShape::OpenAiChatCompletions => {
                    append_openai_chat_tool_results(&mut current_request, &tool_outputs)?
                }
                ManagedToolRequestShape::OpenAiResponses => {
                    append_openai_responses_tool_results(&mut current_request, &tool_outputs)?
                }
                ManagedToolRequestShape::AnthropicMessages => {
                    append_anthropic_tool_results(&mut current_request, &tool_outputs)?
                }
            }

            let follow_up = send_provider_request(
                &client,
                ctx,
                &request_meta.provider_name,
                round_trips,
                &upstream,
                &current_request,
                self.timeout,
            )
            .await?;
            status = follow_up.status;
            headers = follow_up.headers;
            body_bytes = follow_up.body;
        }
    }

    fn reserve_mcp_budget(
        &self,
        tool: &ProjectToolRecord,
        config: &McpExecutorConfig,
        budgets: &mut ToolExecutionBudgetState,
    ) -> Result<Duration, String> {
        let server = self.mcp_servers.get(&config.server);
        let current_server_calls = budgets
            .mcp_server_calls
            .get(&config.server)
            .copied()
            .unwrap_or(0);
        let current_tool_calls = budgets
            .mcp_tool_calls
            .get(&tool.tool_name)
            .copied()
            .unwrap_or(0);
        let current_server_time_ms = budgets
            .mcp_server_time_ms
            .get(&config.server)
            .copied()
            .unwrap_or(0);
        let current_tool_time_ms = budgets
            .mcp_tool_time_ms
            .get(&tool.tool_name)
            .copied()
            .unwrap_or(0);

        if let Some(limit) = server.and_then(|server| server.max_calls_per_request) {
            if current_server_calls >= limit {
                let error = format!(
                    "MCP server '{}' exceeded max_calls_per_request {}",
                    config.server, limit
                );
                self.record_mcp_tool_call_budget_exceeded(
                    &config.server,
                    &tool.tool_name,
                    &error,
                    true,
                );
                return Err(error);
            }
        }

        if let Some(limit) = server.and_then(|server| server.max_total_time_ms) {
            if current_server_time_ms >= limit {
                let error = format!(
                    "MCP server '{}' exceeded max_total_time_ms {}",
                    config.server, limit
                );
                self.record_mcp_tool_call_budget_exceeded(
                    &config.server,
                    &tool.tool_name,
                    &error,
                    true,
                );
                return Err(error);
            }
        }

        if let Some(limit) = config.max_calls_per_request {
            if current_tool_calls >= limit {
                let error = format!(
                    "tool '{}' exceeded max_calls_per_request {}",
                    tool.tool_name, limit
                );
                self.record_mcp_tool_call_budget_exceeded(
                    &config.server,
                    &tool.tool_name,
                    &error,
                    true,
                );
                return Err(error);
            }
        }

        if let Some(limit) = config.max_total_time_ms {
            if current_tool_time_ms >= limit {
                let error = format!(
                    "tool '{}' exceeded max_total_time_ms {}",
                    tool.tool_name, limit
                );
                self.record_mcp_tool_call_budget_exceeded(
                    &config.server,
                    &tool.tool_name,
                    &error,
                    true,
                );
                return Err(error);
            }
        }

        budgets
            .mcp_server_calls
            .insert(config.server.clone(), current_server_calls + 1);
        budgets
            .mcp_tool_calls
            .insert(tool.tool_name.clone(), current_tool_calls + 1);

        let mut timeout = server
            .map(|server| mcp_server_timeout(server, tool_timeout(tool, self.timeout)))
            .unwrap_or_else(|| tool_timeout(tool, self.timeout));
        if let Some(limit) = server.and_then(|server| server.max_total_time_ms) {
            let remaining = limit.saturating_sub(current_server_time_ms);
            timeout = timeout.min(Duration::from_millis(remaining.max(1)));
        }
        if let Some(limit) = config.max_total_time_ms {
            let remaining = limit.saturating_sub(current_tool_time_ms);
            timeout = timeout.min(Duration::from_millis(remaining.max(1)));
        }
        Ok(timeout)
    }

    fn record_mcp_elapsed_budget(
        &self,
        budgets: &mut ToolExecutionBudgetState,
        server_name: &str,
        tool_name: &str,
        elapsed: Duration,
    ) {
        let elapsed_ms = elapsed.as_millis() as u64;
        budgets
            .mcp_server_time_ms
            .entry(server_name.to_string())
            .and_modify(|value| *value += elapsed_ms)
            .or_insert(elapsed_ms);
        budgets
            .mcp_tool_time_ms
            .entry(tool_name.to_string())
            .and_modify(|value| *value += elapsed_ms)
            .or_insert(elapsed_ms);
    }

    fn enforce_mcp_output_budget(
        &self,
        tool: &ProjectToolRecord,
        config: &McpExecutorConfig,
        budgets: &mut ToolExecutionBudgetState,
        output: &str,
    ) -> Result<(), String> {
        let output_tokens = estimate_prompt_tokens(output.as_bytes());
        let server = self.mcp_servers.get(&config.server);
        let current_server_output_tokens = budgets
            .mcp_server_output_tokens
            .get(&config.server)
            .copied()
            .unwrap_or(0);
        let current_tool_output_tokens = budgets
            .mcp_tool_output_tokens
            .get(&tool.tool_name)
            .copied()
            .unwrap_or(0);

        if let Some(limit) = server.and_then(|server| server.max_output_tokens) {
            if current_server_output_tokens.saturating_add(output_tokens) > limit {
                let error = format!(
                    "MCP server '{}' exceeded max_output_tokens {}",
                    config.server, limit
                );
                self.record_mcp_tool_call_budget_exceeded(
                    &config.server,
                    &tool.tool_name,
                    &error,
                    false,
                );
                return Err(error);
            }
        }

        if let Some(limit) = config.max_output_tokens {
            if current_tool_output_tokens.saturating_add(output_tokens) > limit {
                let error = format!(
                    "tool '{}' exceeded max_output_tokens {}",
                    tool.tool_name, limit
                );
                self.record_mcp_tool_call_budget_exceeded(
                    &config.server,
                    &tool.tool_name,
                    &error,
                    false,
                );
                return Err(error);
            }
        }

        budgets
            .mcp_server_output_tokens
            .entry(config.server.clone())
            .and_modify(|value| *value = value.saturating_add(output_tokens))
            .or_insert(output_tokens);
        budgets
            .mcp_tool_output_tokens
            .entry(tool.tool_name.clone())
            .and_modify(|value| *value = value.saturating_add(output_tokens))
            .or_insert(output_tokens);
        Ok(())
    }

    async fn execute_tool(
        &self,
        project_id: &str,
        provider_name: &str,
        tool: &ProjectToolRecord,
        arguments: &Value,
        budgets: &mut ToolExecutionBudgetState,
    ) -> Result<String, String> {
        let span = tracing::info_span!(
            "tool_execution",
            llm.project_id = %project_id,
            llm.provider = %provider_name,
            llm.tool_name = %tool.tool_name,
            llm.tool_executor = %tool.executor_kind,
        );
        async move {
            match tool.executor_kind.as_str() {
                "webhook" => {
                    self.execute_webhook_tool(project_id, provider_name, tool, arguments)
                        .await
                }
                "web_search" => {
                    self.execute_web_search_tool(project_id, provider_name, tool, arguments)
                        .await
                }
                "mcp" => {
                    let config = parse_mcp_executor_config(tool)?;
                    let timeout = self.reserve_mcp_budget(tool, &config, budgets)?;
                    self.execute_mcp_tool(tool, &config, arguments, budgets, timeout)
                        .await
                }
                "arxiv_search" => self.execute_arxiv_search_tool(tool, arguments).await,
                other => Err(format!(
                    "unsupported executor_kind '{}' for tool '{}'",
                    other, tool.tool_name
                )),
            }
        }
        .instrument(span)
        .await
    }

    async fn execute_webhook_tool(
        &self,
        project_id: &str,
        provider_name: &str,
        tool: &ProjectToolRecord,
        arguments: &Value,
    ) -> Result<String, String> {
        let config = parse_webhook_executor_config(tool)?;
        let payload = json!({
            "project_id": project_id,
            "provider_name": provider_name,
            "tool_name": tool.tool_name,
            "arguments": arguments,
        })
        .to_string();

        self.execute_http_tool_request(
            tool,
            &config.url,
            &config.method,
            &config.headers,
            Bytes::from(payload),
            CONTENT_TYPE,
            "application/json",
        )
        .await
    }

    async fn execute_web_search_tool(
        &self,
        project_id: &str,
        provider_name: &str,
        tool: &ProjectToolRecord,
        arguments: &Value,
    ) -> Result<String, String> {
        let config = parse_web_search_executor_config(tool)?;
        let backend = self
            .web_search_backends
            .get(&config.backend)
            .ok_or_else(|| {
                format!(
                    "tool '{}' references unknown web_search backend '{}'",
                    tool.tool_name, config.backend
                )
            })?;
        let query = extract_tool_query(arguments)?;
        let payload = json!({
            "project_id": project_id,
            "provider_name": provider_name,
            "tool_name": tool.tool_name,
            "query": query,
            "arguments": arguments,
        })
        .to_string();

        self.execute_http_tool_request(
            tool,
            &backend.url,
            &backend.method,
            &backend.headers,
            Bytes::from(payload),
            CONTENT_TYPE,
            "application/json",
        )
        .await
    }

    async fn execute_mcp_tool(
        &self,
        tool: &ProjectToolRecord,
        config: &McpExecutorConfig,
        arguments: &Value,
        budgets: &mut ToolExecutionBudgetState,
        timeout: Duration,
    ) -> Result<String, String> {
        let started = Instant::now();
        let result = async {
            let server = self.mcp_servers.get(&config.server).ok_or_else(|| {
                format!(
                    "tool '{}' references unknown mcp server '{}'",
                    tool.tool_name, config.server
                )
            })?;
            if let Some(error) = self.mcp_server_disabled_message(&config.server) {
                self.record_mcp_tool_call_disabled(&config.server, &tool.tool_name, &error);
                return Err(error);
            }
            self.record_mcp_tool_call_started(&config.server, &tool.tool_name);
            let state = match self.ensure_mcp_server_ready(&config.server).await {
                Ok(state) => state,
                Err(error) => {
                    self.record_mcp_tool_call_failed(
                        &config.server,
                        &tool.tool_name,
                        &error,
                        None,
                        0,
                    );
                    return Err(error);
                }
            };
            let bearer_token = self
                .ensure_mcp_auth_bearer_token(&config.server, server)
                .await?;
            if state.reachable
                && !state
                    .discovered_tools
                    .iter()
                    .any(|name| name == &config.remote_tool)
            {
                let error = format!(
                    "tool '{}' references remote MCP tool '{}' that was not found on server '{}'",
                    tool.tool_name, config.remote_tool, config.server
                );
                self.record_mcp_tool_call_failed(
                    &config.server,
                    &tool.tool_name,
                    &error,
                    None,
                    0,
                );
                return Err(error);
            }

            let context = format!("tool '{}' tools/call", tool.tool_name);
            let request_payload = json!({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": config.remote_tool,
                    "arguments": arguments,
                }
            });
            let mut retry_count = 0u32;
            let response = match &server.transport {
                McpTransportConfig::Http { .. } => match send_mcp_jsonrpc_request(
                    &context,
                    server,
                    bearer_token.as_deref(),
                    state.protocol_version.as_deref(),
                    state.session_id.as_deref(),
                    request_payload.clone(),
                    timeout,
                )
                .await
                {
                    Ok(response) => {
                        retry_count += response.retry_count;
                        response
                    }
                    Err(error)
                        if error.status_code() == Some(StatusCode::NOT_FOUND)
                            && state.session_id.is_some() =>
                    {
                        tracing::info!(
                            server = %config.server,
                            tool = %tool.tool_name,
                            "MCP session expired; reinitializing and retrying tool call"
                        );
                        self.record_mcp_session_reinitialized(&config.server);
                        let refreshed = match self
                            .establish_mcp_server_session(&config.server, server)
                            .await
                        {
                            Ok(refreshed) => refreshed,
                            Err(error) => {
                                self.record_mcp_session_recovery_failed(&config.server, &error);
                                self.record_mcp_tool_call_failed(
                                    &config.server,
                                    &tool.tool_name,
                                    &error,
                                    Some(StatusCode::NOT_FOUND),
                                    retry_count,
                                );
                                return Err(error);
                            }
                        };
                        let merged = {
                            let mut inventory = self
                                .mcp_inventory
                                .write()
                                .unwrap_or_else(|poison| poison.into_inner());
                            let merged =
                                merge_mcp_server_inventory(inventory.get(&config.server), refreshed);
                            inventory.insert(config.server.clone(), merged.clone());
                            merged
                        };
                        if !merged
                            .discovered_tools
                            .iter()
                            .any(|name| name == &config.remote_tool)
                        {
                            let error = format!(
                                "tool '{}' references remote MCP tool '{}' that was not found on server '{}'",
                                tool.tool_name, config.remote_tool, config.server
                            );
                            self.record_mcp_tool_call_failed(
                                &config.server,
                                &tool.tool_name,
                                &error,
                                None,
                                retry_count,
                            );
                            return Err(error);
                        }
                        let response = send_mcp_jsonrpc_request(
                            &context,
                            server,
                            bearer_token.as_deref(),
                            merged.protocol_version.as_deref(),
                            merged.session_id.as_deref(),
                            request_payload,
                            timeout,
                        )
                        .await
                        .map_err(|error| {
                            let status_code = error.status_code();
                            let message = error.into_message();
                            self.record_mcp_session_recovery_failed(&config.server, &message);
                            self.record_mcp_tool_call_failed(
                                &config.server,
                                &tool.tool_name,
                                &message,
                                status_code,
                                retry_count,
                            );
                            message
                        })?;
                        retry_count += response.retry_count;
                        response
                    }
                    Err(error) => {
                        let status_code = error.status_code();
                        let message = error.into_message();
                        self.record_mcp_tool_call_failed(
                            &config.server,
                            &tool.tool_name,
                            &message,
                            status_code,
                            retry_count,
                        );
                        return Err(message);
                    }
                },
                McpTransportConfig::Sse { .. } => {
                    let mut session = match start_mcp_sse_session(
                        &config.server,
                        server,
                        bearer_token.as_deref(),
                        timeout,
                    )
                    .await
                    {
                        Ok(session) => session,
                        Err(error) => {
                            self.record_mcp_tool_call_failed(
                                &config.server,
                                &tool.tool_name,
                                &error,
                                None,
                                retry_count,
                            );
                            return Err(error);
                        }
                    };
                    let _initialized = initialize_mcp_server_sse(
                        &config.server,
                        server,
                        &mut session,
                        timeout,
                    )
                    .await
                    .map_err(|error| {
                        self.record_mcp_tool_call_failed(
                            &config.server,
                            &tool.tool_name,
                            &error,
                            None,
                            retry_count,
                        );
                        error
                    })?;
                    send_mcp_initialized_notification_sse(
                        &config.server,
                        server,
                        &session,
                        timeout,
                    )
                    .await
                    .map_err(|error| {
                        self.record_mcp_tool_call_failed(
                            &config.server,
                            &tool.tool_name,
                            &error,
                            None,
                            retry_count,
                        );
                        error
                    })?;
                    let response = send_mcp_sse_jsonrpc_request(
                        server,
                        &mut session,
                        &context,
                        json!({
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": config.remote_tool,
                                "arguments": arguments,
                            }
                        }),
                        timeout,
                    )
                    .await
                    .map_err(|error| {
                        self.record_mcp_tool_call_failed(
                            &config.server,
                            &tool.tool_name,
                            &error,
                            None,
                            retry_count,
                        );
                        error
                    })?;
                    retry_count += response.retry_count;
                    response
                }
                McpTransportConfig::Stdio { .. } => {
                    let (mut process, _) =
                        match establish_mcp_stdio_process(&config.server, server, timeout).await {
                            Ok(process) => process,
                            Err(error) => {
                                self.record_mcp_tool_call_failed(
                                    &config.server,
                                    &tool.tool_name,
                                    &error,
                                    None,
                                    retry_count,
                                );
                                return Err(error);
                            }
                        };
                    let response = send_stdio_jsonrpc_request(
                        &mut process,
                        &context,
                        json!({
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": config.remote_tool,
                                "arguments": arguments,
                            }
                        }),
                        timeout,
                    )
                    .await
                    .map_err(|error| {
                        self.record_mcp_tool_call_failed(
                            &config.server,
                            &tool.tool_name,
                            &error,
                            None,
                            retry_count,
                        );
                        error
                    })?;
                    response
                }
            };

            let result = parse_mcp_tool_response(
                tool,
                &config.server,
                &config.remote_tool,
                &response.response,
            );
            match result {
                Ok(output) => {
                    self.enforce_mcp_output_budget(tool, config, budgets, &output)?;
                    self.record_mcp_tool_call_succeeded(
                        &config.server,
                        &tool.tool_name,
                        retry_count,
                    );
                    Ok(output)
                }
                Err(error) => {
                    self.record_mcp_tool_call_failed(
                        &config.server,
                        &tool.tool_name,
                        &error,
                        None,
                        retry_count,
                    );
                    Err(error)
                }
            }
        }
        .await;
        self.record_mcp_elapsed_budget(budgets, &config.server, &tool.tool_name, started.elapsed());
        result
    }

    async fn execute_arxiv_search_tool(
        &self,
        tool: &ProjectToolRecord,
        arguments: &Value,
    ) -> Result<String, String> {
        let config = parse_arxiv_executor_config(tool, self.arxiv.default_max_results)?;
        let query = extract_tool_query(arguments)?;
        let query_value = encode_query_component(&format!("all:{query}"));
        let uri = format!(
            "{}?search_query={query_value}&start=0&max_results={}",
            self.arxiv.base_url, config.max_results
        );

        let req = Request::builder()
            .method(Method::GET)
            .uri(&uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .map_err(|error| format!("failed to build arxiv request: {error}"))?;

        let response = tokio::time::timeout(
            tool_timeout(tool, self.timeout),
            build_client().request(req),
        )
        .await
        .map_err(|_| format!("tool '{}' timed out", tool.tool_name))?
        .map_err(|error| format!("tool '{}' request failed: {error}", tool.tool_name))?;
        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .map_err(|error| format!("failed to read tool response: {error}"))?
            .to_bytes();
        if !status.is_success() {
            return Err(format!(
                "tool '{}' returned status {}",
                tool.tool_name,
                status.as_u16()
            ));
        }

        let xml = String::from_utf8_lossy(&bytes);
        Ok(json!({
            "results": parse_arxiv_feed(&xml),
        })
        .to_string())
    }

    async fn execute_http_tool_request(
        &self,
        tool: &ProjectToolRecord,
        url: &str,
        method: &str,
        extra_headers: &HashMap<String, String>,
        body: Bytes,
        content_type_name: hyper::header::HeaderName,
        content_type_value: &'static str,
    ) -> Result<String, String> {
        let mut headers = extra_headers.clone();
        headers.insert(
            content_type_name.as_str().to_string(),
            content_type_value.to_string(),
        );
        execute_runtime_http_request(
            url,
            method,
            &headers,
            body,
            tool_timeout(tool, self.timeout),
            &format!("tool '{}'", tool.tool_name),
        )
        .await
        .map(|response| String::from_utf8_lossy(&response.body).to_string())
        .map_err(RuntimeHttpError::into_message)
    }

    async fn replay_tool_output(
        &self,
        ctx: &RequestContext,
        request_meta: &ToolRuntimeRequest,
        current_request: &Value,
        mut output: ToolOutput,
    ) -> Result<ReplayResult, String> {
        let model = current_request.get("model").cloned();
        let candidate = build_replay_candidate(&request_meta.request_shape, model, &output);
        let body = serde_json::to_vec(&candidate)
            .map_err(|error| format!("failed to encode replay candidate: {error}"))?;
        let mut synthetic_ctx = synthetic_request_context(ctx, Bytes::from(body));

        let mut safety_audit = None;
        if let Some(filter) = ctx.extensions.get::<ContentFilterReplayHandle>().cloned() {
            let action = filter.on_synthetic_request(&mut synthetic_ctx).await;
            safety_audit = synthetic_ctx.extensions.get::<SafetyAudit>().cloned();
            if let Action::Respond(_) = action {
                let message = blocked_tool_message(&safety_audit, &output.call_id);
                output.content = tool_error_payload("tool_output_blocked", &message);
                return Ok(ReplayResult {
                    output,
                    safety_audit,
                    semantic_audit: None,
                });
            }
        }

        let mut semantic_audit = None;
        if let Some(semantic) = ctx.extensions.get::<SemanticSafetyReplayHandle>().cloned() {
            let _ = semantic.on_synthetic_request(&mut synthetic_ctx).await;
            semantic_audit = synthetic_ctx
                .extensions
                .get::<SemanticSafetyAudit>()
                .cloned();
        }

        if let Some(body) = synthetic_ctx.body.as_ref() {
            let candidate_json: Value = serde_json::from_slice(body)
                .map_err(|error| format!("failed to decode replay candidate: {error}"))?;
            output.content =
                extract_replayed_content(&request_meta.request_shape, &candidate_json)?;
        }

        Ok(ReplayResult {
            output,
            safety_audit,
            semantic_audit,
        })
    }
}

#[derive(Clone)]
struct ToolOutput {
    call_id: String,
    tool_name: String,
    content: String,
    assistant_message: Value,
}

struct ReplayResult {
    output: ToolOutput,
    safety_audit: Option<SafetyAudit>,
    semantic_audit: Option<SemanticSafetyAudit>,
}

#[derive(Clone)]
struct ParsedToolCall {
    call_id: String,
    name: String,
    arguments: Value,
    assistant_message: Value,
}

struct ProviderResponse {
    status: StatusCode,
    headers: hyper::HeaderMap,
    body: Bytes,
}

struct OptInTools {
    enabled: bool,
    names: Option<Vec<String>>,
}

#[derive(Default)]
struct ParsedStreamingResponse {
    tool_calls: Vec<ParsedToolCall>,
    usage: Option<(u64, u64)>,
}

#[derive(Default)]
struct OpenAiToolCallAccumulator {
    call_id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct AnthropicToolCallAccumulator {
    call_id: String,
    name: String,
    raw_input: String,
    input: Option<Value>,
}

#[derive(Default)]
struct SseEvent {
    event: Option<String>,
    data: String,
}

struct WebhookExecutorConfig {
    url: String,
    method: String,
    headers: HashMap<String, String>,
}

struct WebSearchExecutorConfig {
    backend: String,
}

struct McpExecutorConfig {
    server: String,
    remote_tool: String,
    max_calls_per_request: Option<u32>,
    max_total_time_ms: Option<u64>,
    max_output_tokens: Option<u64>,
}

struct ArxivExecutorConfig {
    max_results: u64,
}

fn tool_timeout(tool: &ProjectToolRecord, default_timeout: Duration) -> Duration {
    tool.timeout_ms
        .map(Duration::from_millis)
        .unwrap_or(default_timeout)
}

fn mcp_server_timeout(server: &McpServerConfig, default_timeout: Duration) -> Duration {
    Duration::from_millis(
        server
            .timeout_ms
            .unwrap_or(default_timeout.as_millis() as u64),
    )
}

fn mcp_request_is_retryable(error: &RuntimeHttpError) -> bool {
    match error {
        RuntimeHttpError::Message(_) => true,
        RuntimeHttpError::Status(status, _) => status.is_server_error(),
    }
}

fn parse_web_search_backends(
    config: &toml::Value,
) -> Result<HashMap<String, WebSearchBackendConfig>, String> {
    let Some(backends) = config.get("web_search_backends") else {
        return Ok(HashMap::new());
    };
    let table = backends
        .as_table()
        .ok_or_else(|| "tool_runtime.web_search_backends must be a table".to_string())?;

    let mut parsed = HashMap::new();
    for (name, value) in table {
        let backend = value
            .as_table()
            .ok_or_else(|| format!("tool_runtime.web_search_backends.{name} must be a table"))?;
        let url = backend
            .get("url")
            .and_then(|value| value.as_str())
            .ok_or_else(|| format!("tool_runtime.web_search_backends.{name}.url is required"))?;
        let method = backend
            .get("method")
            .and_then(|value| value.as_str())
            .unwrap_or("POST")
            .to_string();
        let headers = backend
            .get("headers")
            .map(parse_toml_string_map)
            .transpose()?
            .unwrap_or_default();
        parsed.insert(
            name.clone(),
            WebSearchBackendConfig {
                url: url.to_string(),
                method,
                headers,
            },
        );
    }

    Ok(parsed)
}

fn parse_mcp_servers(config: &toml::Value) -> Result<HashMap<String, McpServerConfig>, String> {
    let Some(servers) = config.get("mcp_servers") else {
        return Ok(HashMap::new());
    };
    let table = servers
        .as_table()
        .ok_or_else(|| "tool_runtime.mcp_servers must be a table".to_string())?;

    let mut parsed = HashMap::new();
    for (name, value) in table {
        let server = value
            .as_table()
            .ok_or_else(|| format!("tool_runtime.mcp_servers.{name} must be a table"))?;
        let transport_name = server
            .get("transport")
            .and_then(|value| value.as_str())
            .unwrap_or("http");
        let transport = match transport_name {
            "http" => {
                let url = server
                    .get("url")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| format!("tool_runtime.mcp_servers.{name}.url is required"))?;
                let method = server
                    .get("method")
                    .and_then(|value| value.as_str())
                    .unwrap_or("POST")
                    .to_string();
                let headers = server
                    .get("headers")
                    .map(parse_toml_string_map)
                    .transpose()?
                    .unwrap_or_default();
                McpTransportConfig::Http {
                    url: url.to_string(),
                    method,
                    headers,
                }
            }
            "sse" => {
                let url = server
                    .get("url")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| format!("tool_runtime.mcp_servers.{name}.url is required"))?;
                let headers = server
                    .get("headers")
                    .map(parse_toml_string_map)
                    .transpose()?
                    .unwrap_or_default();
                McpTransportConfig::Sse {
                    url: url.to_string(),
                    headers,
                }
            }
            "stdio" => {
                let command = server
                    .get("command")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| {
                        format!("tool_runtime.mcp_servers.{name}.command is required")
                    })?;
                let args = server
                    .get("args")
                    .map(parse_toml_string_array)
                    .transpose()?
                    .unwrap_or_default();
                let env = server
                    .get("env")
                    .map(parse_toml_string_map)
                    .transpose()?
                    .unwrap_or_default();
                let cwd = server
                    .get("cwd")
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string);
                McpTransportConfig::Stdio {
                    command: command.to_string(),
                    args,
                    env,
                    cwd,
                }
            }
            other => {
                return Err(format!(
                    "tool_runtime.mcp_servers.{name}.transport must be 'http', 'sse', or 'stdio', got '{other}'"
                ))
            }
        };
        let auth = parse_mcp_auth_config(name, server)?;
        let timeout_ms = server
            .get("timeout_ms")
            .and_then(|value| value.as_integer())
            .map(|value| value as u64);
        let max_retries = server
            .get("max_retries")
            .and_then(|value| value.as_integer())
            .map(|value| value as u32)
            .unwrap_or(0);
        let max_calls_per_request = server
            .get("max_calls_per_request")
            .and_then(|value| value.as_integer())
            .map(|value| value as u32);
        let max_total_time_ms = server
            .get("max_total_time_ms")
            .and_then(|value| value.as_integer())
            .map(|value| value as u64);
        let max_output_tokens = server
            .get("max_output_tokens")
            .and_then(|value| value.as_integer())
            .map(|value| value as u64);
        if matches!(max_output_tokens, Some(0)) {
            return Err(format!(
                "tool_runtime.mcp_servers.{name}.max_output_tokens must be greater than 0"
            ));
        }
        parsed.insert(
            name.clone(),
            McpServerConfig {
                transport,
                auth,
                timeout_ms,
                max_retries,
                max_calls_per_request,
                max_total_time_ms,
                max_output_tokens,
            },
        );
    }

    Ok(parsed)
}

fn parse_mcp_auth_config(
    server_name: &str,
    server: &toml::value::Map<String, toml::Value>,
) -> Result<McpAuthConfig, String> {
    let Some(auth) = server.get("auth") else {
        return Ok(McpAuthConfig::None);
    };
    let auth = auth
        .as_table()
        .ok_or_else(|| format!("tool_runtime.mcp_servers.{server_name}.auth must be a table"))?;
    let kind = auth
        .get("type")
        .and_then(|value| value.as_str())
        .ok_or_else(|| format!("tool_runtime.mcp_servers.{server_name}.auth.type is required"))?;
    match kind {
        "bearer" => {
            let access_token =
                if let Some(value) = auth.get("access_token").and_then(|value| value.as_str()) {
                    value.to_string()
                } else if let Some(env_var) = auth
                    .get("access_token_env")
                    .and_then(|value| value.as_str())
                {
                    std::env::var(env_var).map_err(|_| {
                        format!(
                            "tool_runtime.mcp_servers.{server_name}.auth.access_token_env '{env_var}' is not set"
                        )
                    })?
                } else {
                    return Err(format!(
                        "tool_runtime.mcp_servers.{server_name}.auth must set access_token or access_token_env"
                    ));
                };
            Ok(McpAuthConfig::StaticBearer { access_token })
        }
        "oauth_client_credentials" => {
            let token_url = auth
                .get("token_url")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty());
            let protected_resource_metadata_url = auth
                .get("protected_resource_metadata_url")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty());
            let authorization_server_metadata_url = auth
                .get("authorization_server_metadata_url")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty());
            let client_id = auth
                .get("client_id")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!("tool_runtime.mcp_servers.{server_name}.auth.client_id is required")
                })?
                .to_string();
            let client_secret =
                if let Some(value) = auth.get("client_secret").and_then(|value| value.as_str()) {
                    value.to_string()
                } else if let Some(env_var) = auth
                    .get("client_secret_env")
                    .and_then(|value| value.as_str())
                {
                    std::env::var(env_var).map_err(|_| {
                        format!(
                            "tool_runtime.mcp_servers.{server_name}.auth.client_secret_env '{env_var}' is not set"
                        )
                    })?
                } else {
                    return Err(format!(
                        "tool_runtime.mcp_servers.{server_name}.auth must set client_secret or client_secret_env"
                    ));
                };
            let scope = auth
                .get("scope")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .or_else(|| {
                    auth.get("scopes")
                        .and_then(|value| value.as_array())
                        .map(|values| {
                            values
                                .iter()
                                .filter_map(|value| value.as_str())
                                .collect::<Vec<_>>()
                                .join(" ")
                        })
                })
                .filter(|value| !value.trim().is_empty());
            let resource = auth
                .get("resource")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty());
            Ok(McpAuthConfig::OAuthClientCredentials {
                token_url,
                protected_resource_metadata_url,
                authorization_server_metadata_url,
                client_id,
                client_secret,
                scope,
                resource,
            })
        }
        "oauth_authorization_code" => {
            let token_url = auth
                .get("token_url")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty());
            let authorization_url = auth
                .get("authorization_url")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty());
            let protected_resource_metadata_url = auth
                .get("protected_resource_metadata_url")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty());
            let authorization_server_metadata_url = auth
                .get("authorization_server_metadata_url")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty());
            let client_id = auth
                .get("client_id")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!("tool_runtime.mcp_servers.{server_name}.auth.client_id is required")
                })?
                .to_string();
            let client_secret = if let Some(value) =
                auth.get("client_secret").and_then(|value| value.as_str())
            {
                Some(value.to_string())
            } else if let Some(env_var) = auth
                .get("client_secret_env")
                .and_then(|value| value.as_str())
            {
                Some(std::env::var(env_var).map_err(|_| {
                    format!(
                        "tool_runtime.mcp_servers.{server_name}.auth.client_secret_env '{env_var}' is not set"
                    )
                })?)
            } else {
                None
            };
            let redirect_uri = auth
                .get("redirect_uri")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!("tool_runtime.mcp_servers.{server_name}.auth.redirect_uri is required")
                })?
                .to_string();
            let scope = auth
                .get("scope")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .or_else(|| {
                    auth.get("scopes")
                        .and_then(|value| value.as_array())
                        .map(|values| {
                            values
                                .iter()
                                .filter_map(|value| value.as_str())
                                .collect::<Vec<_>>()
                                .join(" ")
                        })
                })
                .filter(|value| !value.trim().is_empty());
            let resource = auth
                .get("resource")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty());
            Ok(McpAuthConfig::OAuthAuthorizationCode {
                token_url,
                authorization_url,
                protected_resource_metadata_url,
                authorization_server_metadata_url,
                client_id,
                client_secret,
                redirect_uri,
                scope,
                resource,
            })
        }
        other => Err(format!(
            "tool_runtime.mcp_servers.{server_name}.auth.type must be 'bearer', 'oauth_client_credentials', or 'oauth_authorization_code', got '{other}'"
        )),
    }
}

fn parse_toml_string_map(value: &toml::Value) -> Result<HashMap<String, String>, String> {
    let table = value
        .as_table()
        .ok_or_else(|| "expected table of string values".to_string())?;
    let mut parsed = HashMap::new();
    for (key, value) in table {
        let string_value = value
            .as_str()
            .ok_or_else(|| format!("expected string value for '{key}'"))?;
        parsed.insert(key.clone(), string_value.to_string());
    }
    Ok(parsed)
}

fn parse_toml_string_array(value: &toml::Value) -> Result<Vec<String>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "expected array of string values".to_string())?;
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| "expected string value".to_string())
        })
        .collect()
}

fn parse_executor_config_json(tool: &ProjectToolRecord) -> Result<Value, String> {
    match tool.executor_config_json.as_deref() {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str::<Value>(raw)
            .map_err(|error| format!("invalid executor_config for '{}': {error}", tool.tool_name)),
        _ => Ok(Value::Object(serde_json::Map::new())),
    }
}

fn parse_webhook_executor_config(
    tool: &ProjectToolRecord,
) -> Result<WebhookExecutorConfig, String> {
    let config_json = parse_executor_config_json(tool)?;
    let url = config_json
        .get("url")
        .and_then(|value| value.as_str())
        .ok_or_else(|| format!("tool '{}' is missing executor_config.url", tool.tool_name))?;
    let method = config_json
        .get("method")
        .and_then(|value| value.as_str())
        .unwrap_or("POST")
        .to_string();
    let headers = config_json
        .get("headers")
        .and_then(|value| value.as_object())
        .map(|headers| {
            headers
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();

    Ok(WebhookExecutorConfig {
        url: url.to_string(),
        method,
        headers,
    })
}

fn parse_web_search_executor_config(
    tool: &ProjectToolRecord,
) -> Result<WebSearchExecutorConfig, String> {
    let config_json = parse_executor_config_json(tool)?;
    let backend = config_json
        .get("backend")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            format!(
                "tool '{}' is missing executor_config.backend",
                tool.tool_name
            )
        })?;
    Ok(WebSearchExecutorConfig {
        backend: backend.to_string(),
    })
}

fn parse_mcp_executor_config(tool: &ProjectToolRecord) -> Result<McpExecutorConfig, String> {
    let config_json = parse_executor_config_json(tool)?;
    let server = config_json
        .get("server")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            format!(
                "tool '{}' is missing executor_config.server",
                tool.tool_name
            )
        })?;
    let remote_tool = config_json
        .get("remote_tool")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            format!(
                "tool '{}' is missing executor_config.remote_tool",
                tool.tool_name
            )
        })?;
    let max_calls_per_request = config_json
        .get("max_calls_per_request")
        .and_then(|value| value.as_u64())
        .map(|value| value as u32);
    let max_total_time_ms = config_json
        .get("max_total_time_ms")
        .and_then(|value| value.as_u64());
    let max_output_tokens = config_json
        .get("max_output_tokens")
        .and_then(|value| value.as_u64());
    if matches!(max_output_tokens, Some(0)) {
        return Err(format!(
            "tool '{}' must use max_output_tokens greater than 0",
            tool.tool_name
        ));
    }
    Ok(McpExecutorConfig {
        server: server.to_string(),
        remote_tool: remote_tool.to_string(),
        max_calls_per_request,
        max_total_time_ms,
        max_output_tokens,
    })
}

fn parse_arxiv_executor_config(
    tool: &ProjectToolRecord,
    default_max_results: u64,
) -> Result<ArxivExecutorConfig, String> {
    let config_json = parse_executor_config_json(tool)?;
    let max_results = config_json
        .get("max_results")
        .and_then(|value| value.as_u64())
        .unwrap_or(default_max_results);
    if max_results == 0 {
        return Err(format!(
            "tool '{}' must use max_results greater than 0",
            tool.tool_name
        ));
    }
    Ok(ArxivExecutorConfig { max_results })
}

fn extract_tool_query(arguments: &Value) -> Result<String, String> {
    arguments
        .get("query")
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "tool arguments must include a non-empty string query".to_string())
}

fn encode_query_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            b' ' => encoded.push_str("%20"),
            _ => encoded.push_str(&format!("%{:02X}", byte)),
        }
    }
    encoded
}

fn parse_arxiv_feed(xml: &str) -> Vec<Value> {
    extract_xml_blocks(xml, "entry")
        .into_iter()
        .map(|entry| {
            let authors = extract_xml_values(&entry, "name");
            json!({
                "id": extract_xml_value(&entry, "id"),
                "title": extract_xml_value(&entry, "title"),
                "summary": extract_xml_value(&entry, "summary"),
                "published": extract_xml_value(&entry, "published"),
                "authors": authors,
            })
        })
        .collect()
}

fn extract_xml_blocks(text: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(&open) {
        let after_open = &rest[start + open.len()..];
        let Some(end) = after_open.find(&close) else {
            break;
        };
        blocks.push(after_open[..end].to_string());
        rest = &after_open[end + close.len()..];
    }
    blocks
}

fn extract_xml_value(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)?;
    let after_open = &text[start + open.len()..];
    let end = after_open.find(&close)?;
    Some(decode_xml_text(&after_open[..end]))
}

fn extract_xml_values(text: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut values = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(&open) {
        let after_open = &rest[start + open.len()..];
        let Some(end) = after_open.find(&close) else {
            break;
        };
        values.push(decode_xml_text(&after_open[..end]));
        rest = &after_open[end + close.len()..];
    }
    values
}

fn decode_xml_text(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
        .trim()
        .to_string()
}

fn parse_mcp_tool_response(
    tool: &ProjectToolRecord,
    server_name: &str,
    remote_tool: &str,
    value: &Value,
) -> Result<String, String> {
    if let Some(error) = value.get("error") {
        let code = error.get("code").and_then(|value| value.as_i64());
        let message = error
            .get("message")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown MCP error");
        return Err(match code {
            Some(code) => format!(
                "tool '{}' MCP server '{}' returned error {}: {}",
                tool.tool_name, server_name, code, message
            ),
            None => format!(
                "tool '{}' MCP server '{}' returned error: {}",
                tool.tool_name, server_name, message
            ),
        });
    }

    let result = value.get("result").ok_or_else(|| {
        format!(
            "tool '{}' MCP server '{}' response is missing result",
            tool.tool_name, server_name
        )
    })?;

    if result.get("isError").and_then(|value| value.as_bool()) == Some(true) {
        let message = extract_mcp_result_text(result)
            .unwrap_or_else(|| format!("remote MCP tool '{remote_tool}' returned isError"));
        return Err(format!(
            "tool '{}' remote MCP tool '{}' failed: {}",
            tool.tool_name, remote_tool, message
        ));
    }

    Ok(extract_mcp_result_text(result).unwrap_or_else(|| {
        result
            .get("structuredContent")
            .cloned()
            .unwrap_or_else(|| result.clone())
            .to_string()
    }))
}

fn extract_mcp_result_text(result: &Value) -> Option<String> {
    if let Some(content) = result.get("content") {
        return extract_mcp_content_text(content);
    }
    if let Some(text) = result.get("text").and_then(|value| value.as_str()) {
        return Some(text.to_string());
    }
    if let Some(value) = result.get("structuredContent") {
        return Some(match value {
            Value::String(text) => text.to_string(),
            other => other.to_string(),
        });
    }
    result.as_str().map(ToString::to_string)
}

fn extract_mcp_content_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.to_string()),
        Value::Array(blocks) => {
            let texts = blocks
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|value| value.as_str()) == Some("text") {
                        block
                            .get("text")
                            .and_then(|value| value.as_str())
                            .map(ToString::to_string)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if texts.is_empty() {
                Some(content.to_string())
            } else {
                Some(texts.join("\n"))
            }
        }
        other => Some(other.to_string()),
    }
}

struct RuntimeHttpResponse {
    headers: hyper::HeaderMap,
    body: Bytes,
}

#[derive(Debug)]
enum RuntimeHttpError {
    Message(String),
    Status(StatusCode, String),
}

impl RuntimeHttpError {
    fn status_code(&self) -> Option<StatusCode> {
        match self {
            Self::Status(status, _) => Some(*status),
            Self::Message(_) => None,
        }
    }

    fn into_message(self) -> String {
        match self {
            Self::Message(message) | Self::Status(_, message) => message,
        }
    }
}

struct McpRpcResponse {
    headers: hyper::HeaderMap,
    response: Value,
    retry_count: u32,
}

struct McpOAuthToken {
    access_token: String,
    refresh_token: Option<String>,
    expires_at_unix_ms: Option<u64>,
}

#[derive(Clone, Debug)]
struct ResolvedMcpOAuthClientCredentials {
    token_url: String,
    client_id: String,
    client_secret: String,
    scope: Option<String>,
    resource: Option<String>,
    authorization_server_url: Option<String>,
}

#[derive(Clone, Debug)]
struct ResolvedMcpOAuthAuthorizationCode {
    authorization_url: String,
    token_url: String,
    client_id: String,
    client_secret: Option<String>,
    redirect_uri: String,
    scope: Option<String>,
    resource: Option<String>,
    authorization_server_url: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ToolRuntimeMcpOAuthAuthorizationRequest {
    pub server_name: String,
    pub authorization_url: String,
    pub redirect_uri: String,
    pub state: String,
    pub expires_at_unix_ms: u64,
}

struct StdioMcpProcess {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct SseMcpSession {
    endpoint_url: String,
    body: Incoming,
    buffer: String,
    bearer_token: Option<String>,
}

struct McpInitializedSession {
    protocol_version: String,
    session_id: Option<String>,
}

fn parse_mcp_tools_list_response(server_name: &str, value: &Value) -> Result<Vec<String>, String> {
    if let Some(error) = value.get("error") {
        let code = error.get("code").and_then(|value| value.as_i64());
        let message = error
            .get("message")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown MCP error");
        return Err(match code {
            Some(code) => format!(
                "MCP server '{}' returned tools/list error {}: {}",
                server_name, code, message
            ),
            None => format!(
                "MCP server '{}' returned tools/list error: {}",
                server_name, message
            ),
        });
    }

    let tools = value
        .get("result")
        .and_then(|value| value.get("tools"))
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            format!(
                "MCP server '{}' tools/list response is missing result.tools",
                server_name
            )
        })?;
    let mut names = tools
        .iter()
        .filter_map(|tool| {
            tool.get("name")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
        })
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    Ok(names)
}

fn build_mcp_auth_headers(
    base_headers: &HashMap<String, String>,
    bearer_token: Option<&str>,
) -> HashMap<String, String> {
    let mut headers = base_headers.clone();
    if let Some(token) = bearer_token {
        headers.insert("authorization".to_string(), format!("Bearer {token}"));
    }
    headers
}

async fn fetch_mcp_oauth_access_token(
    server_name: &str,
    auth: &ResolvedMcpOAuthClientCredentials,
    timeout: Duration,
) -> Result<McpOAuthToken, String> {
    let mut form_fields = vec![
        ("grant_type", "client_credentials".to_string()),
        ("client_id", auth.client_id.clone()),
        ("client_secret", auth.client_secret.clone()),
    ];
    if let Some(scope) = &auth.scope {
        form_fields.push(("scope", scope.clone()));
    }
    if let Some(resource) = &auth.resource {
        form_fields.push(("resource", resource.clone()));
    }
    let body = form_fields
        .into_iter()
        .map(|(key, value)| format!("{key}={}", encode_query_component(&value)))
        .collect::<Vec<_>>()
        .join("&");

    let mut headers = HashMap::new();
    headers.insert(
        "content-type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    );
    headers.insert("accept".to_string(), "application/json".to_string());
    let response = execute_runtime_http_request(
        &auth.token_url,
        "POST",
        &headers,
        Bytes::from(body),
        timeout,
        &format!("MCP server '{server_name}' OAuth token request"),
    )
    .await
    .map_err(RuntimeHttpError::into_message)?;
    let body_json: Value = serde_json::from_slice(&response.body).map_err(|error| {
        format!(
            "MCP server '{}' OAuth token response was not valid JSON: {error}",
            server_name
        )
    })?;
    let access_token = body_json
        .get("access_token")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            format!(
                "MCP server '{}' OAuth token response is missing access_token",
                server_name
            )
        })?
        .to_string();
    let refresh_token = body_json
        .get("refresh_token")
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
        .filter(|value| !value.trim().is_empty());
    let expires_at_unix_ms = body_json
        .get("expires_in")
        .and_then(|value| value.as_u64())
        .map(|expires_in| current_unix_ms().saturating_add(expires_in.saturating_mul(1000)));
    Ok(McpOAuthToken {
        access_token,
        refresh_token,
        expires_at_unix_ms,
    })
}

fn default_mcp_oauth_resource(transport: &McpTransportConfig) -> Option<String> {
    match transport {
        McpTransportConfig::Http { url, .. } | McpTransportConfig::Sse { url, .. } => {
            Some(url.clone())
        }
        McpTransportConfig::Stdio { .. } => None,
    }
}

fn derive_oauth_well_known_url(base_url: &str, kind: &str) -> Result<String, String> {
    let base: hyper::Uri = base_url
        .parse()
        .map_err(|error| format!("invalid OAuth metadata base URL '{}': {error}", base_url))?;
    let scheme = base
        .scheme_str()
        .ok_or_else(|| format!("OAuth metadata base URL '{}' is missing a scheme", base_url))?;
    let authority = base.authority().ok_or_else(|| {
        format!(
            "OAuth metadata base URL '{}' is missing an authority",
            base_url
        )
    })?;
    let path = base.path().trim_start_matches('/');
    if path.is_empty() {
        Ok(format!("{scheme}://{authority}/.well-known/{kind}"))
    } else {
        Ok(format!("{scheme}://{authority}/.well-known/{kind}/{path}"))
    }
}

async fn fetch_runtime_json(url: &str, timeout: Duration, context: &str) -> Result<Value, String> {
    let mut headers = HashMap::new();
    headers.insert("accept".to_string(), "application/json".to_string());
    let response =
        execute_runtime_http_request(url, "GET", &headers, Bytes::new(), timeout, context)
            .await
            .map_err(RuntimeHttpError::into_message)?;
    serde_json::from_slice(&response.body)
        .map_err(|error| format!("{context} returned invalid JSON: {error}"))
}

fn extract_first_string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(value)) if !value.trim().is_empty() => Some(value.to_string()),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_str())
            .find(|value| !value.trim().is_empty())
            .map(ToString::to_string),
        _ => None,
    }
}

fn generate_mcp_oauth_secret(bytes_len: usize) -> String {
    let mut rng = rand::thread_rng();
    let bytes = (0..bytes_len).map(|_| rng.gen::<u8>()).collect::<Vec<_>>();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn build_pkce_code_challenge(verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hasher.finalize())
}

async fn resolve_mcp_oauth_client_credentials(
    server_name: &str,
    auth: &McpAuthConfig,
    transport: &McpTransportConfig,
    current: Option<&McpServerInventory>,
    timeout: Duration,
) -> Result<ResolvedMcpOAuthClientCredentials, String> {
    let McpAuthConfig::OAuthClientCredentials {
        token_url,
        protected_resource_metadata_url,
        authorization_server_metadata_url,
        client_id,
        client_secret,
        scope,
        resource,
    } = auth
    else {
        return Err(format!(
            "MCP server '{}' is not configured for OAuth client credentials",
            server_name
        ));
    };

    let mut resolved_resource = resource
        .clone()
        .or_else(|| current.and_then(|state| state.auth_resource.clone()))
        .or_else(|| default_mcp_oauth_resource(transport));
    let mut resolved_token_url = token_url
        .clone()
        .or_else(|| current.and_then(|state| state.auth_token_url.clone()));
    let mut resolved_authorization_server_url =
        current.and_then(|state| state.auth_authorization_server_url.clone());

    if resolved_token_url.is_none() {
        let protected_resource_metadata_url = protected_resource_metadata_url
            .clone()
            .or_else(|| {
                resolved_resource
                    .as_deref()
                    .map(|resource| derive_oauth_well_known_url(resource, "oauth-protected-resource"))
                    .transpose()
                    .ok()
                    .flatten()
            })
            .ok_or_else(|| {
                format!(
                    "MCP server '{}' OAuth auth discovery requires token_url, protected_resource_metadata_url, or an HTTP/SSE MCP URL",
                    server_name
                )
            })?;
        let protected_resource_metadata = fetch_runtime_json(
            &protected_resource_metadata_url,
            timeout,
            &format!(
                "MCP server '{}' protected resource metadata request",
                server_name
            ),
        )
        .await?;
        if resolved_resource.is_none() {
            resolved_resource = protected_resource_metadata
                .get("resource")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .or_else(|| default_mcp_oauth_resource(transport));
        }
        resolved_authorization_server_url =
            extract_first_string(protected_resource_metadata.get("authorization_servers"))
                .or_else(|| {
                    protected_resource_metadata
                        .get("authorization_server")
                        .and_then(|value| value.as_str())
                        .map(ToString::to_string)
                })
                .or(resolved_authorization_server_url);
        let authorization_server_metadata_url = authorization_server_metadata_url
            .clone()
            .or_else(|| {
                resolved_authorization_server_url
                    .as_deref()
                    .map(|url| derive_oauth_well_known_url(url, "oauth-authorization-server"))
                    .transpose()
                    .ok()
                    .flatten()
            })
            .ok_or_else(|| {
                format!(
                    "MCP server '{}' OAuth protected resource metadata did not provide an authorization server",
                    server_name
                )
            })?;
        let authorization_server_metadata = fetch_runtime_json(
            &authorization_server_metadata_url,
            timeout,
            &format!(
                "MCP server '{}' authorization server metadata request",
                server_name
            ),
        )
        .await?;
        resolved_token_url = authorization_server_metadata
            .get("token_endpoint")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
    }

    let resolved_token_url = resolved_token_url.ok_or_else(|| {
        format!(
            "MCP server '{}' OAuth client credentials did not resolve a token endpoint",
            server_name
        )
    })?;

    Ok(ResolvedMcpOAuthClientCredentials {
        token_url: resolved_token_url,
        client_id: client_id.clone(),
        client_secret: client_secret.clone(),
        scope: scope.clone(),
        resource: resolved_resource,
        authorization_server_url: resolved_authorization_server_url,
    })
}

async fn resolve_mcp_oauth_authorization_code(
    server_name: &str,
    auth: &McpAuthConfig,
    transport: &McpTransportConfig,
    current: Option<&McpServerInventory>,
    timeout: Duration,
) -> Result<ResolvedMcpOAuthAuthorizationCode, String> {
    let McpAuthConfig::OAuthAuthorizationCode {
        token_url,
        authorization_url,
        protected_resource_metadata_url,
        authorization_server_metadata_url,
        client_id,
        client_secret,
        redirect_uri,
        scope,
        resource,
    } = auth
    else {
        return Err(format!(
            "MCP server '{}' is not configured for OAuth authorization code",
            server_name
        ));
    };

    let mut resolved_resource = resource
        .clone()
        .or_else(|| current.and_then(|state| state.auth_resource.clone()))
        .or_else(|| default_mcp_oauth_resource(transport));
    let mut resolved_token_url = token_url
        .clone()
        .or_else(|| current.and_then(|state| state.auth_token_url.clone()));
    let mut resolved_authorization_url = authorization_url
        .clone()
        .or_else(|| current.and_then(|state| state.auth_authorization_url.clone()));
    let mut resolved_authorization_server_url =
        current.and_then(|state| state.auth_authorization_server_url.clone());

    if resolved_token_url.is_none() || resolved_authorization_url.is_none() {
        let protected_resource_metadata_url = protected_resource_metadata_url
            .clone()
            .or_else(|| {
                resolved_resource
                    .as_deref()
                    .map(|resource| {
                        derive_oauth_well_known_url(resource, "oauth-protected-resource")
                    })
                    .transpose()
                    .ok()
                    .flatten()
            })
            .ok_or_else(|| {
                format!(
                    "MCP server '{}' OAuth auth discovery requires token_url/authorization_url, protected_resource_metadata_url, or an HTTP/SSE MCP URL",
                    server_name
                )
            })?;
        let protected_resource_metadata = fetch_runtime_json(
            &protected_resource_metadata_url,
            timeout,
            &format!(
                "MCP server '{}' protected resource metadata request",
                server_name
            ),
        )
        .await?;
        if resolved_resource.is_none() {
            resolved_resource = protected_resource_metadata
                .get("resource")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .or_else(|| default_mcp_oauth_resource(transport));
        }
        resolved_authorization_server_url =
            extract_first_string(protected_resource_metadata.get("authorization_servers"))
                .or_else(|| {
                    protected_resource_metadata
                        .get("authorization_server")
                        .and_then(|value| value.as_str())
                        .map(ToString::to_string)
                })
                .or(resolved_authorization_server_url);
        let authorization_server_metadata_url = authorization_server_metadata_url
            .clone()
            .or_else(|| {
                resolved_authorization_server_url
                    .as_deref()
                    .map(|url| derive_oauth_well_known_url(url, "oauth-authorization-server"))
                    .transpose()
                    .ok()
                    .flatten()
            })
            .ok_or_else(|| {
                format!(
                    "MCP server '{}' OAuth protected resource metadata did not provide an authorization server",
                    server_name
                )
            })?;
        let authorization_server_metadata = fetch_runtime_json(
            &authorization_server_metadata_url,
            timeout,
            &format!(
                "MCP server '{}' authorization server metadata request",
                server_name
            ),
        )
        .await?;
        if resolved_token_url.is_none() {
            resolved_token_url = authorization_server_metadata
                .get("token_endpoint")
                .and_then(|value| value.as_str())
                .map(ToString::to_string);
        }
        if resolved_authorization_url.is_none() {
            resolved_authorization_url = authorization_server_metadata
                .get("authorization_endpoint")
                .and_then(|value| value.as_str())
                .map(ToString::to_string);
        }
    }

    let resolved_token_url = resolved_token_url.ok_or_else(|| {
        format!(
            "MCP server '{}' OAuth authorization code did not resolve a token endpoint",
            server_name
        )
    })?;
    let resolved_authorization_url = resolved_authorization_url.ok_or_else(|| {
        format!(
            "MCP server '{}' OAuth authorization code did not resolve an authorization endpoint",
            server_name
        )
    })?;

    Ok(ResolvedMcpOAuthAuthorizationCode {
        authorization_url: resolved_authorization_url,
        token_url: resolved_token_url,
        client_id: client_id.clone(),
        client_secret: client_secret.clone(),
        redirect_uri: redirect_uri.clone(),
        scope: scope.clone(),
        resource: resolved_resource,
        authorization_server_url: resolved_authorization_server_url,
    })
}

fn build_mcp_oauth_authorization_url(
    resolved: &ResolvedMcpOAuthAuthorizationCode,
    state: &str,
    code_challenge: &str,
) -> String {
    let mut query = vec![
        ("response_type", "code".to_string()),
        ("client_id", resolved.client_id.clone()),
        ("redirect_uri", resolved.redirect_uri.clone()),
        ("code_challenge", code_challenge.to_string()),
        ("code_challenge_method", "S256".to_string()),
        ("state", state.to_string()),
    ];
    if let Some(scope) = &resolved.scope {
        query.push(("scope", scope.clone()));
    }
    if let Some(resource) = &resolved.resource {
        query.push(("resource", resource.clone()));
    }
    let separator = if resolved.authorization_url.contains('?') {
        "&"
    } else {
        "?"
    };
    format!(
        "{}{}{}",
        resolved.authorization_url,
        separator,
        query
            .into_iter()
            .map(|(key, value)| format!("{key}={}", encode_query_component(&value)))
            .collect::<Vec<_>>()
            .join("&")
    )
}

async fn exchange_mcp_oauth_authorization_code(
    server_name: &str,
    auth: &ResolvedMcpOAuthAuthorizationCode,
    code: &str,
    code_verifier: &str,
    timeout: Duration,
) -> Result<McpOAuthToken, String> {
    let mut form_fields = vec![
        ("grant_type", "authorization_code".to_string()),
        ("code", code.to_string()),
        ("client_id", auth.client_id.clone()),
        ("redirect_uri", auth.redirect_uri.clone()),
        ("code_verifier", code_verifier.to_string()),
    ];
    if let Some(client_secret) = &auth.client_secret {
        form_fields.push(("client_secret", client_secret.clone()));
    }
    if let Some(resource) = &auth.resource {
        form_fields.push(("resource", resource.clone()));
    }
    let body = form_fields
        .into_iter()
        .map(|(key, value)| format!("{key}={}", encode_query_component(&value)))
        .collect::<Vec<_>>()
        .join("&");
    let mut headers = HashMap::new();
    headers.insert(
        "content-type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    );
    headers.insert("accept".to_string(), "application/json".to_string());
    let response = execute_runtime_http_request(
        &auth.token_url,
        "POST",
        &headers,
        Bytes::from(body),
        timeout,
        &format!("MCP server '{server_name}' OAuth authorization code exchange"),
    )
    .await
    .map_err(RuntimeHttpError::into_message)?;
    let body_json: Value = serde_json::from_slice(&response.body).map_err(|error| {
        format!(
            "MCP server '{}' OAuth authorization code response was not valid JSON: {error}",
            server_name
        )
    })?;
    let access_token = body_json
        .get("access_token")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            format!(
                "MCP server '{}' OAuth authorization code response is missing access_token",
                server_name
            )
        })?
        .to_string();
    let refresh_token = body_json
        .get("refresh_token")
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
        .filter(|value| !value.trim().is_empty());
    let expires_at_unix_ms = body_json
        .get("expires_in")
        .and_then(|value| value.as_u64())
        .map(|expires_in| current_unix_ms().saturating_add(expires_in.saturating_mul(1000)));
    Ok(McpOAuthToken {
        access_token,
        refresh_token,
        expires_at_unix_ms,
    })
}

async fn refresh_mcp_oauth_access_token(
    server_name: &str,
    auth: &ResolvedMcpOAuthAuthorizationCode,
    refresh_token: &str,
    timeout: Duration,
) -> Result<McpOAuthToken, String> {
    let mut form_fields = vec![
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", refresh_token.to_string()),
        ("client_id", auth.client_id.clone()),
    ];
    if let Some(client_secret) = &auth.client_secret {
        form_fields.push(("client_secret", client_secret.clone()));
    }
    if let Some(resource) = &auth.resource {
        form_fields.push(("resource", resource.clone()));
    }
    let body = form_fields
        .into_iter()
        .map(|(key, value)| format!("{key}={}", encode_query_component(&value)))
        .collect::<Vec<_>>()
        .join("&");
    let mut headers = HashMap::new();
    headers.insert(
        "content-type".to_string(),
        "application/x-www-form-urlencoded".to_string(),
    );
    headers.insert("accept".to_string(), "application/json".to_string());
    let response = execute_runtime_http_request(
        &auth.token_url,
        "POST",
        &headers,
        Bytes::from(body),
        timeout,
        &format!("MCP server '{server_name}' OAuth refresh token request"),
    )
    .await
    .map_err(RuntimeHttpError::into_message)?;
    let body_json: Value = serde_json::from_slice(&response.body).map_err(|error| {
        format!(
            "MCP server '{}' OAuth refresh response was not valid JSON: {error}",
            server_name
        )
    })?;
    let access_token = body_json
        .get("access_token")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            format!(
                "MCP server '{}' OAuth refresh response is missing access_token",
                server_name
            )
        })?
        .to_string();
    let next_refresh_token = body_json
        .get("refresh_token")
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
        .filter(|value| !value.trim().is_empty())
        .or_else(|| Some(refresh_token.to_string()));
    let expires_at_unix_ms = body_json
        .get("expires_in")
        .and_then(|value| value.as_u64())
        .map(|expires_in| current_unix_ms().saturating_add(expires_in.saturating_mul(1000)));
    Ok(McpOAuthToken {
        access_token,
        refresh_token: next_refresh_token,
        expires_at_unix_ms,
    })
}

fn resolve_mcp_sse_endpoint(base_url: &str, endpoint: &str) -> Result<String, String> {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        return Ok(endpoint.to_string());
    }

    let base: hyper::Uri = base_url
        .parse()
        .map_err(|error| format!("invalid MCP SSE base URL '{}': {error}", base_url))?;
    let scheme = base
        .scheme_str()
        .ok_or_else(|| format!("MCP SSE base URL '{}' is missing a scheme", base_url))?;
    let authority = base
        .authority()
        .ok_or_else(|| format!("MCP SSE base URL '{}' is missing an authority", base_url))?;

    if endpoint.starts_with('/') {
        return Ok(format!("{scheme}://{authority}{endpoint}"));
    }

    let path = base.path();
    let prefix = path
        .rsplit_once('/')
        .map(|(head, _)| {
            if head.is_empty() {
                "/".to_string()
            } else {
                format!("{head}/")
            }
        })
        .unwrap_or_else(|| "/".to_string());
    Ok(format!("{scheme}://{authority}{prefix}{endpoint}"))
}

fn parse_single_sse_event(block: &str) -> Option<SseEvent> {
    let block = block.trim();
    if block.is_empty() {
        return None;
    }
    let mut event = None;
    let mut data_lines = Vec::new();
    for line in block.lines() {
        if let Some(value) = line.strip_prefix("event:") {
            event = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            data_lines.push(value.trim_start().to_string());
        }
    }
    if data_lines.is_empty() {
        return None;
    }
    Some(SseEvent {
        event,
        data: data_lines.join("\n"),
    })
}

fn take_next_sse_event(buffer: &mut String) -> Option<SseEvent> {
    let delimiter_index = buffer.find("\n\n")?;
    let block = buffer[..delimiter_index].to_string();
    buffer.drain(..delimiter_index + 2);
    parse_single_sse_event(&block)
}

async fn next_sse_session_event(
    session: &mut SseMcpSession,
    context: &str,
    timeout: Duration,
) -> Result<SseEvent, String> {
    tokio::time::timeout(timeout, async {
        loop {
            if let Some(event) = take_next_sse_event(&mut session.buffer) {
                return Ok(event);
            }
            let frame = session
                .body
                .frame()
                .await
                .transpose()
                .map_err(|error| format!("{context} SSE stream failed: {error}"))?;
            let Some(frame) = frame else {
                return Err(format!("{context} SSE stream closed unexpectedly"));
            };
            if let Some(data) = frame.data_ref() {
                session
                    .buffer
                    .push_str(&String::from_utf8_lossy(data).replace("\r\n", "\n"));
            }
        }
    })
    .await
    .map_err(|_| format!("{context} timed out"))?
}

async fn start_mcp_sse_session(
    server_name: &str,
    server: &McpServerConfig,
    bearer_token: Option<&str>,
    timeout: Duration,
) -> Result<SseMcpSession, String> {
    let McpTransportConfig::Sse { url, headers } = &server.transport else {
        return Err(format!(
            "MCP server '{}' is not configured for SSE transport",
            server_name
        ));
    };

    let mut builder = Request::builder().method(Method::GET).uri(url);
    if let Some(request_headers) = builder.headers_mut() {
        for (key, value) in build_mcp_auth_headers(headers, bearer_token) {
            if let (Ok(name), Ok(value)) = (
                hyper::header::HeaderName::from_bytes(key.as_bytes()),
                HeaderValue::from_str(&value),
            ) {
                request_headers.insert(name, value);
            }
        }
        request_headers.insert(
            hyper::header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );
    }
    let request = builder
        .body(
            Full::new(Bytes::new())
                .map_err(|never| match never {})
                .boxed(),
        )
        .map_err(|error| format!("failed to build MCP SSE request: {error}"))?;
    let response = tokio::time::timeout(timeout, build_client().request(request))
        .await
        .map_err(|_| format!("MCP server '{server_name}' SSE connect timed out"))?
        .map_err(|error| format!("MCP server '{server_name}' SSE connect failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!(
            "MCP server '{}' SSE connect returned status {}",
            server_name,
            status.as_u16()
        ));
    }
    let (_, body) = response.into_parts();
    let mut session = SseMcpSession {
        endpoint_url: String::new(),
        body,
        buffer: String::new(),
        bearer_token: bearer_token.map(ToString::to_string),
    };
    loop {
        let event = next_sse_session_event(
            &mut session,
            &format!("MCP server '{server_name}' SSE endpoint"),
            timeout,
        )
        .await?;
        if event.event.as_deref() == Some("endpoint") {
            session.endpoint_url = resolve_mcp_sse_endpoint(url, event.data.trim())?;
            return Ok(session);
        }
    }
}

async fn post_mcp_sse_payload(
    server: &McpServerConfig,
    session: &SseMcpSession,
    payload: Bytes,
    timeout: Duration,
    context: &str,
) -> Result<u32, RuntimeHttpError> {
    let McpTransportConfig::Sse { headers, .. } = &server.transport else {
        return Err(RuntimeHttpError::Message(format!(
            "{context} requires an SSE MCP server transport"
        )));
    };
    let mut post_headers = build_mcp_auth_headers(headers, session.bearer_token.as_deref());
    post_headers.insert("content-type".to_string(), "application/json".to_string());
    post_headers.insert("accept".to_string(), MCP_ACCEPT_HEADER.to_string());

    let mut attempt = 0u32;
    loop {
        match execute_runtime_http_request(
            &session.endpoint_url,
            "POST",
            &post_headers,
            payload.clone(),
            timeout,
            context,
        )
        .await
        {
            Ok(_) => return Ok(attempt),
            Err(error) if attempt < server.max_retries && mcp_request_is_retryable(&error) => {
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

async fn send_mcp_sse_notification(
    server: &McpServerConfig,
    session: &SseMcpSession,
    context: &str,
    payload: Value,
    timeout: Duration,
) -> Result<(), String> {
    let payload = Bytes::from(
        serde_json::to_vec(&payload)
            .map_err(|error| format!("failed to encode {context} payload: {error}"))?,
    );
    post_mcp_sse_payload(server, session, payload, timeout, context)
        .await
        .map(|_| ())
        .map_err(RuntimeHttpError::into_message)
}

async fn send_mcp_sse_jsonrpc_request(
    server: &McpServerConfig,
    session: &mut SseMcpSession,
    context: &str,
    mut payload: Value,
    timeout: Duration,
) -> Result<McpRpcResponse, String> {
    let request_id = MCP_REQUEST_ID.fetch_add(1, Ordering::Relaxed).to_string();
    payload
        .as_object_mut()
        .ok_or_else(|| format!("{context} payload must be a JSON object"))?
        .insert("id".to_string(), Value::String(request_id.clone()));
    let payload_bytes = Bytes::from(
        serde_json::to_vec(&payload)
            .map_err(|error| format!("failed to encode {context} payload: {error}"))?,
    );
    let retry_count = post_mcp_sse_payload(server, session, payload_bytes, timeout, context)
        .await
        .map_err(RuntimeHttpError::into_message)?;
    loop {
        let event = next_sse_session_event(session, context, timeout).await?;
        if matches!(event.event.as_deref(), Some("endpoint")) {
            continue;
        }
        if !matches!(event.event.as_deref(), Some("message") | None) {
            continue;
        }
        let value: Value = serde_json::from_str(&event.data)
            .map_err(|error| format!("{context} returned invalid SSE JSON: {error}"))?;
        if let Some(matched) = find_jsonrpc_response_by_id(&value, &request_id) {
            return Ok(McpRpcResponse {
                headers: hyper::HeaderMap::new(),
                response: matched.clone(),
                retry_count,
            });
        }
    }
}

fn spawn_mcp_stderr_logger(server_name: String, stderr: ChildStderr) {
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {
                    let message = line.trim();
                    if !message.is_empty() {
                        tracing::debug!(server = %server_name, stderr = %message, "MCP stdio server stderr");
                    }
                }
                Err(error) => {
                    tracing::debug!(
                        server = %server_name,
                        error = %error,
                        "failed to read MCP stdio server stderr"
                    );
                    break;
                }
            }
        }
    });
}

async fn start_mcp_stdio_process(
    server_name: &str,
    server: &McpServerConfig,
) -> Result<StdioMcpProcess, String> {
    let McpTransportConfig::Stdio {
        command,
        args,
        env,
        cwd,
    } = &server.transport
    else {
        return Err(format!(
            "MCP server '{}' is not configured for stdio transport",
            server_name
        ));
    };

    let mut process = Command::new(command);
    process.args(args);
    process.stdin(std::process::Stdio::piped());
    process.stdout(std::process::Stdio::piped());
    process.stderr(std::process::Stdio::piped());
    process.kill_on_drop(true);
    if let Some(cwd) = cwd {
        process.current_dir(cwd);
    }
    for (key, value) in env {
        process.env(key, value);
    }

    let mut child = process.spawn().map_err(|error| {
        format!(
            "failed to spawn MCP stdio server '{}' with command '{}': {}",
            server_name, command, error
        )
    })?;
    let stdin = child.stdin.take().ok_or_else(|| {
        format!(
            "failed to capture stdin for MCP stdio server '{}'",
            server_name
        )
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        format!(
            "failed to capture stdout for MCP stdio server '{}'",
            server_name
        )
    })?;
    if let Some(stderr) = child.stderr.take() {
        spawn_mcp_stderr_logger(server_name.to_string(), stderr);
    }

    Ok(StdioMcpProcess {
        _child: child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

fn serialize_stdio_jsonrpc_message(context: &str, payload: &Value) -> Result<String, String> {
    let serialized = serde_json::to_string(payload)
        .map_err(|error| format!("failed to encode {context} JSON-RPC payload: {error}"))?;
    if serialized.contains('\n') || serialized.contains('\r') {
        return Err(format!(
            "{context} JSON-RPC payload contains unexpected line breaks"
        ));
    }
    Ok(serialized)
}

async fn send_stdio_notification(
    process: &mut StdioMcpProcess,
    context: &str,
    payload: Value,
    timeout: Duration,
) -> Result<(), String> {
    let serialized = serialize_stdio_jsonrpc_message(context, &payload)?;
    tokio::time::timeout(timeout, async {
        process
            .stdin
            .write_all(serialized.as_bytes())
            .await
            .map_err(|error| format!("{context} write failed: {error}"))?;
        process
            .stdin
            .write_all(b"\n")
            .await
            .map_err(|error| format!("{context} write failed: {error}"))?;
        process
            .stdin
            .flush()
            .await
            .map_err(|error| format!("{context} flush failed: {error}"))?;
        Ok(())
    })
    .await
    .map_err(|_| format!("{context} timed out"))?
}

async fn send_stdio_jsonrpc_request(
    process: &mut StdioMcpProcess,
    context: &str,
    mut payload: Value,
    timeout: Duration,
) -> Result<McpRpcResponse, String> {
    let request_id = MCP_REQUEST_ID.fetch_add(1, Ordering::Relaxed).to_string();
    payload
        .as_object_mut()
        .ok_or_else(|| format!("{context} payload must be a JSON object"))?
        .insert("id".to_string(), Value::String(request_id.clone()));
    let serialized = serialize_stdio_jsonrpc_message(context, &payload)?;

    tokio::time::timeout(timeout, async {
        process
            .stdin
            .write_all(serialized.as_bytes())
            .await
            .map_err(|error| format!("{context} write failed: {error}"))?;
        process
            .stdin
            .write_all(b"\n")
            .await
            .map_err(|error| format!("{context} write failed: {error}"))?;
        process
            .stdin
            .flush()
            .await
            .map_err(|error| format!("{context} flush failed: {error}"))?;

        let mut line = String::new();
        loop {
            line.clear();
            let read = process
                .stdout
                .read_line(&mut line)
                .await
                .map_err(|error| format!("{context} read failed: {error}"))?;
            if read == 0 {
                return Err(format!(
                    "{context} MCP stdio server exited before sending a response"
                ));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(trimmed)
                .map_err(|error| format!("{context} returned invalid JSON: {error}"))?;
            if let Some(matched) = find_jsonrpc_response_by_id(&value, &request_id) {
                return Ok(McpRpcResponse {
                    headers: hyper::HeaderMap::new(),
                    response: matched.clone(),
                    retry_count: 0,
                });
            }
        }
    })
    .await
    .map_err(|_| format!("{context} timed out"))?
}

async fn execute_runtime_http_request(
    url: &str,
    method: &str,
    extra_headers: &HashMap<String, String>,
    body: Bytes,
    timeout: Duration,
    context: &str,
) -> Result<RuntimeHttpResponse, RuntimeHttpError> {
    let mut builder = Request::builder().method(method).uri(url);
    if let Some(headers) = builder.headers_mut() {
        for (key, value) in extra_headers {
            if let (Ok(name), Ok(value)) = (
                hyper::header::HeaderName::from_bytes(key.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                headers.insert(name, value);
            }
        }
    }

    let req = builder
        .body(Full::new(body).map_err(|never| match never {}).boxed())
        .map_err(|error| {
            RuntimeHttpError::Message(format!("failed to build {context} request: {error}"))
        })?;

    let response = tokio::time::timeout(timeout, build_client().request(req))
        .await
        .map_err(|_| RuntimeHttpError::Message(format!("{context} timed out")))?
        .map_err(|error| RuntimeHttpError::Message(format!("{context} request failed: {error}")))?;
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .into_body()
        .collect()
        .await
        .map_err(|error| {
            RuntimeHttpError::Message(format!("failed to read {context} response: {error}"))
        })?
        .to_bytes();
    if !status.is_success() {
        return Err(RuntimeHttpError::Status(
            status,
            format!("{context} returned status {}", status.as_u16()),
        ));
    }
    Ok(RuntimeHttpResponse {
        headers,
        body: bytes,
    })
}

fn parse_mcp_response_envelope(
    context: &str,
    request_id: &str,
    headers: &hyper::HeaderMap,
    body: &Bytes,
) -> Result<Value, RuntimeHttpError> {
    let content_type = headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");

    if content_type.starts_with("text/event-stream") {
        for event in parse_sse_events(body.as_ref()) {
            let value: Value = serde_json::from_str(&event.data).map_err(|error| {
                RuntimeHttpError::Message(format!("{context} returned invalid SSE JSON: {error}"))
            })?;
            if let Some(matched) = find_jsonrpc_response_by_id(&value, request_id) {
                return Ok(matched.clone());
            }
        }
        return Err(RuntimeHttpError::Message(format!(
            "{context} did not include a JSON-RPC response for request id '{}'",
            request_id
        )));
    }

    let value: Value = serde_json::from_slice(body).map_err(|error| {
        RuntimeHttpError::Message(format!("{context} returned invalid JSON: {error}"))
    })?;
    find_jsonrpc_response_by_id(&value, request_id)
        .cloned()
        .ok_or_else(|| {
            RuntimeHttpError::Message({
                format!(
                    "{context} did not include a JSON-RPC response for request id '{}'",
                    request_id
                )
            })
        })
}

fn find_jsonrpc_response_by_id<'a>(value: &'a Value, request_id: &str) -> Option<&'a Value> {
    match value {
        Value::Array(values) => values
            .iter()
            .find(|entry| jsonrpc_id_matches(entry.get("id"), request_id)),
        Value::Object(_) if jsonrpc_id_matches(value.get("id"), request_id) => Some(value),
        _ => None,
    }
}

fn jsonrpc_id_matches(value: Option<&Value>, request_id: &str) -> bool {
    match value {
        Some(Value::String(id)) => id == request_id,
        Some(Value::Number(id)) => id.to_string() == request_id,
        _ => false,
    }
}

async fn send_mcp_jsonrpc_request(
    context: &str,
    server: &McpServerConfig,
    bearer_token: Option<&str>,
    protocol_version: Option<&str>,
    session_id: Option<&str>,
    mut payload: Value,
    timeout: Duration,
) -> Result<McpRpcResponse, RuntimeHttpError> {
    let McpTransportConfig::Http {
        url,
        method,
        headers: default_headers,
    } = &server.transport
    else {
        return Err(RuntimeHttpError::Message(format!(
            "{context} requires an HTTP MCP server transport"
        )));
    };
    let request_id = MCP_REQUEST_ID.fetch_add(1, Ordering::Relaxed).to_string();
    payload
        .as_object_mut()
        .ok_or_else(|| {
            RuntimeHttpError::Message(format!("{context} payload must be a JSON object"))
        })?
        .insert("id".to_string(), Value::String(request_id.clone()));

    let mut headers = build_mcp_auth_headers(default_headers, bearer_token);
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("accept".to_string(), MCP_ACCEPT_HEADER.to_string());
    if let Some(protocol_version) = protocol_version {
        headers.insert(
            MCP_PROTOCOL_HEADER.to_string(),
            protocol_version.to_string(),
        );
    }
    if let Some(session_id) = session_id {
        headers.insert(MCP_SESSION_HEADER.to_string(), session_id.to_string());
    }

    let payload_bytes = Bytes::from(payload.to_string());
    let mut attempt = 0u32;
    let response = loop {
        match execute_runtime_http_request(
            url,
            method,
            &headers,
            payload_bytes.clone(),
            timeout,
            context,
        )
        .await
        {
            Ok(response) => break response,
            Err(error) if attempt < server.max_retries && mcp_request_is_retryable(&error) => {
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    };
    let envelope =
        parse_mcp_response_envelope(context, &request_id, &response.headers, &response.body)?;
    Ok(McpRpcResponse {
        headers: response.headers,
        response: envelope,
        retry_count: attempt,
    })
}

async fn initialize_mcp_server(
    server_name: &str,
    server: &McpServerConfig,
    bearer_token: Option<&str>,
    timeout: Duration,
) -> Result<McpInitializedSession, String> {
    let response = send_mcp_jsonrpc_request(
        &format!("MCP server '{server_name}' initialize"),
        server,
        bearer_token,
        None,
        None,
        json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_CLIENT_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "tiny-reverse-proxy",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }
        }),
        timeout,
    )
    .await
    .map_err(RuntimeHttpError::into_message)?;
    let protocol_version = response
        .response
        .get("result")
        .and_then(|value| value.get("protocolVersion"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            format!(
                "MCP server '{}' initialize response is missing result.protocolVersion",
                server_name
            )
        })?
        .to_string();
    if !MCP_SUPPORTED_PROTOCOL_VERSIONS
        .iter()
        .any(|supported| *supported == protocol_version)
    {
        return Err(format!(
            "MCP server '{}' negotiated unsupported protocol version '{}'",
            server_name, protocol_version
        ));
    }
    let session_id = response
        .headers
        .get(MCP_SESSION_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string);
    Ok(McpInitializedSession {
        protocol_version,
        session_id,
    })
}

async fn send_mcp_initialized_notification(
    server_name: &str,
    server: &McpServerConfig,
    session: &McpInitializedSession,
    bearer_token: Option<&str>,
    timeout: Duration,
) -> Result<(), String> {
    let McpTransportConfig::Http {
        url,
        method,
        headers: default_headers,
    } = &server.transport
    else {
        return Err(format!(
            "MCP server '{}' is not configured for HTTP transport",
            server_name
        ));
    };
    let mut headers = build_mcp_auth_headers(default_headers, bearer_token);
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("accept".to_string(), MCP_ACCEPT_HEADER.to_string());
    headers.insert(
        MCP_PROTOCOL_HEADER.to_string(),
        session.protocol_version.clone(),
    );
    if let Some(session_id) = session.session_id.as_deref() {
        headers.insert(MCP_SESSION_HEADER.to_string(), session_id.to_string());
    }
    execute_runtime_http_request(
        url,
        method,
        &headers,
        Bytes::from(
            json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            })
            .to_string(),
        ),
        timeout,
        &format!("MCP server '{server_name}' notifications/initialized"),
    )
    .await
    .map_err(RuntimeHttpError::into_message)?;
    Ok(())
}

async fn fetch_mcp_server_tools(
    server_name: &str,
    server: &McpServerConfig,
    session: &McpInitializedSession,
    bearer_token: Option<&str>,
    timeout: Duration,
) -> Result<Vec<String>, String> {
    let response = send_mcp_jsonrpc_request(
        &format!("MCP server '{server_name}' tools/list"),
        server,
        bearer_token,
        Some(&session.protocol_version),
        session.session_id.as_deref(),
        json!({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {}
        }),
        timeout,
    )
    .await
    .map_err(RuntimeHttpError::into_message)?;
    parse_mcp_tools_list_response(server_name, &response.response)
}

async fn initialize_mcp_server_sse(
    server_name: &str,
    server: &McpServerConfig,
    session: &mut SseMcpSession,
    timeout: Duration,
) -> Result<McpInitializedSession, String> {
    let response = send_mcp_sse_jsonrpc_request(
        server,
        session,
        &format!("MCP server '{server_name}' initialize"),
        json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_CLIENT_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "tiny-reverse-proxy",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }
        }),
        timeout,
    )
    .await?;
    let protocol_version = response
        .response
        .get("result")
        .and_then(|value| value.get("protocolVersion"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            format!(
                "MCP server '{}' initialize response is missing result.protocolVersion",
                server_name
            )
        })?
        .to_string();
    if !MCP_SUPPORTED_PROTOCOL_VERSIONS
        .iter()
        .any(|supported| *supported == protocol_version)
    {
        return Err(format!(
            "MCP server '{}' negotiated unsupported protocol version '{}'",
            server_name, protocol_version
        ));
    }
    Ok(McpInitializedSession {
        protocol_version,
        session_id: None,
    })
}

async fn send_mcp_initialized_notification_sse(
    server_name: &str,
    server: &McpServerConfig,
    session: &SseMcpSession,
    timeout: Duration,
) -> Result<(), String> {
    send_mcp_sse_notification(
        server,
        session,
        &format!("MCP server '{server_name}' notifications/initialized"),
        json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }),
        timeout,
    )
    .await
}

async fn fetch_mcp_server_tools_sse(
    server_name: &str,
    server: &McpServerConfig,
    session: &mut SseMcpSession,
    timeout: Duration,
) -> Result<Vec<String>, String> {
    let response = send_mcp_sse_jsonrpc_request(
        server,
        session,
        &format!("MCP server '{server_name}' tools/list"),
        json!({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {}
        }),
        timeout,
    )
    .await?;
    parse_mcp_tools_list_response(server_name, &response.response)
}

async fn initialize_mcp_server_stdio(
    server_name: &str,
    process: &mut StdioMcpProcess,
    timeout: Duration,
) -> Result<McpInitializedSession, String> {
    let response = send_stdio_jsonrpc_request(
        process,
        &format!("MCP server '{server_name}' initialize"),
        json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_CLIENT_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "tiny-reverse-proxy",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }
        }),
        timeout,
    )
    .await?;
    let protocol_version = response
        .response
        .get("result")
        .and_then(|value| value.get("protocolVersion"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            format!(
                "MCP stdio server '{}' initialize response is missing result.protocolVersion",
                server_name
            )
        })?
        .to_string();
    if !MCP_SUPPORTED_PROTOCOL_VERSIONS
        .iter()
        .any(|supported| *supported == protocol_version)
    {
        return Err(format!(
            "MCP stdio server '{}' negotiated unsupported protocol version '{}'",
            server_name, protocol_version
        ));
    }
    Ok(McpInitializedSession {
        protocol_version,
        session_id: None,
    })
}

async fn send_mcp_initialized_notification_stdio(
    server_name: &str,
    process: &mut StdioMcpProcess,
    timeout: Duration,
) -> Result<(), String> {
    send_stdio_notification(
        process,
        &format!(
            "MCP stdio server '{}' notifications/initialized",
            server_name
        ),
        json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }),
        timeout,
    )
    .await
}

async fn fetch_mcp_server_tools_stdio(
    server_name: &str,
    process: &mut StdioMcpProcess,
    timeout: Duration,
) -> Result<Vec<String>, String> {
    let response = send_stdio_jsonrpc_request(
        process,
        &format!("MCP stdio server '{}' tools/list", server_name),
        json!({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {}
        }),
        timeout,
    )
    .await?;
    parse_mcp_tools_list_response(server_name, &response.response)
}

async fn establish_mcp_stdio_process(
    server_name: &str,
    server: &McpServerConfig,
    timeout: Duration,
) -> Result<(StdioMcpProcess, McpInitializedSession), String> {
    let mut process = start_mcp_stdio_process(server_name, server).await?;
    let session = initialize_mcp_server_stdio(server_name, &mut process, timeout).await?;
    send_mcp_initialized_notification_stdio(server_name, &mut process, timeout).await?;
    Ok((process, session))
}

fn request_shape_for(provider: &ProviderKeyConfig, path: &str) -> ManagedToolRequestShape {
    provider
        .surfaces()
        .managed_tool_request_shape_for_path(path)
        .unwrap_or(ManagedToolRequestShape::OpenAiChatCompletions)
}

fn extract_allowed_tool_names(
    request_shape: &ManagedToolRequestShape,
    request: &Value,
) -> Result<Option<Vec<String>>, String> {
    if *request_shape != ManagedToolRequestShape::OpenAiResponses {
        return Ok(None);
    }

    if let Some(tool_choice) = request.get("tool_choice").and_then(Value::as_object) {
        if tool_choice.get("type").and_then(Value::as_str) == Some("allowed_tools") {
            let tools = tool_choice
                .get("tools")
                .ok_or_else(|| "tool_choice.allowed_tools is missing tools".to_string())?;
            return parse_allowed_tools_array(tools).map(Some);
        }
    }

    request
        .get("allowed_tools")
        .map(parse_allowed_tools_array)
        .transpose()
}

fn parse_allowed_tools_array(value: &Value) -> Result<Vec<String>, String> {
    let tools = value
        .as_array()
        .ok_or_else(|| "allowed_tools must be an array".to_string())?;
    let mut names = Vec::with_capacity(tools.len());
    for entry in tools {
        let name = if let Some(name) = entry.as_str() {
            Some(name.to_string())
        } else if let Some(name) = entry.get("name").and_then(Value::as_str) {
            Some(name.to_string())
        } else {
            entry
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        }
        .ok_or_else(|| {
            "allowed_tools entries must be strings or objects with a name".to_string()
        })?;

        if !names.iter().any(|existing| existing == &name) {
            names.push(name);
        }
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proxy_core::config::{
        ProviderCommonConfig, ProviderFamilyConfig, ProviderSurfaceCatalog, ResponsesSurface,
        ToolSurface,
    };

    fn provider_with_surfaces(surfaces: ProviderSurfaceCatalog) -> ProviderKeyConfig {
        ProviderKeyConfig::new(
            ProviderCommonConfig {
                name: "provider".to_string(),
                api_key: "sk-test".to_string(),
                base_url: "https://example.test".to_string(),
                models: vec!["gpt-4o".to_string()],
                api_key_header: "authorization".to_string(),
                timeout_secs: None,
                routing_metadata: Default::default(),
            },
            ProviderFamilyConfig::OpenAi { surfaces },
        )
    }

    #[test]
    fn request_shape_uses_responses_surface_for_openai_tools() {
        let provider = provider_with_surfaces(ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            responses: Some(ResponsesSurface::OpenAiCompatible),
            ..ProviderSurfaceCatalog::default()
        });

        assert_eq!(
            request_shape_for(&provider, "/v1/responses"),
            ManagedToolRequestShape::OpenAiResponses
        );
        assert_eq!(
            request_shape_for(&provider, "/v1/chat/completions"),
            ManagedToolRequestShape::OpenAiChatCompletions
        );
    }

    #[test]
    fn request_shape_does_not_treat_responses_path_as_responses_without_surface() {
        let provider = provider_with_surfaces(ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            ..ProviderSurfaceCatalog::default()
        });

        assert_eq!(
            request_shape_for(&provider, "/v1/responses"),
            ManagedToolRequestShape::OpenAiChatCompletions
        );
    }
}

fn ensure_allowed_tool_call(
    request_meta: &ToolRuntimeRequest,
    tool_name: &str,
) -> Result<(), String> {
    let Some(allowed) = request_meta.allowed_tool_names.as_ref() else {
        return Ok(());
    };
    if allowed.iter().any(|name| name == tool_name) {
        Ok(())
    } else {
        Err(format!(
            "provider attempted disallowed tool '{}' for this request",
            tool_name
        ))
    }
}

fn current_forwarded_request(
    ctx: &RequestContext,
    request_meta: &ToolRuntimeRequest,
) -> Result<Value, String> {
    let body = ctx
        .extensions
        .get::<ForwardedRequestBody>()
        .map(|body| body.0.as_ref())
        .or_else(|| ctx.body.as_ref().map(Bytes::as_ref));
    match body {
        Some(bytes) => serde_json::from_slice(bytes)
            .map_err(|error| format!("failed to decode forwarded request body: {error}")),
        None => Ok(request_meta.request_json.clone()),
    }
}

fn apply_aggregated_audits(
    ctx: &mut RequestContext,
    safety_audit: Option<SafetyAudit>,
    semantic_audit: Option<SemanticSafetyAudit>,
    tool_audit: ToolRuntimeAudit,
) {
    if let Some(audit) = safety_audit {
        ctx.extensions.insert(audit);
    }
    if let Some(audit) = semantic_audit {
        ctx.extensions.insert(audit);
    }
    if !tool_audit.calls.is_empty() {
        ctx.extensions.insert(tool_audit);
    }
}

fn merge_safety_audit(target: &mut Option<SafetyAudit>, incoming: Option<SafetyAudit>) {
    let Some(incoming) = incoming else {
        return;
    };
    match target {
        Some(current) => {
            if safety_mode_rank(&incoming.mode) > safety_mode_rank(&current.mode) {
                current.mode = incoming.mode.clone();
            }
            for entry in incoming.matches {
                if !current.matches.contains(&entry) {
                    current.matches.push(entry);
                }
            }
        }
        None => *target = Some(incoming),
    }
}

fn merge_semantic_audit(
    target: &mut Option<SemanticSafetyAudit>,
    incoming: Option<SemanticSafetyAudit>,
) {
    let Some(incoming) = incoming else {
        return;
    };
    match target {
        Some(current) => {
            current.service_latency_ms = current
                .service_latency_ms
                .saturating_add(incoming.service_latency_ms);
            current.findings.extend(incoming.findings);

            if current.policy_version.is_empty() {
                current.policy_version = incoming.policy_version.clone();
            }

            if semantic_state_rank(&incoming.index_state)
                > semantic_state_rank(&current.index_state)
            {
                current.index_state = incoming.index_state.clone();
                current.degraded_reason = incoming.degraded_reason.clone();
            } else {
                current.degraded_reason = merge_degraded_reason(
                    current.degraded_reason.take(),
                    incoming.degraded_reason.clone(),
                );
            }
        }
        None => *target = Some(incoming),
    }
}

fn safety_mode_rank(mode: &str) -> u8 {
    match mode {
        "block" => 3,
        "redact_and_forward" => 2,
        "observe_only" => 1,
        _ => 0,
    }
}

fn semantic_state_rank(state: &str) -> u8 {
    match state {
        "degraded" => 4,
        "stale" => 3,
        "missing" => 2,
        "disabled" => 1,
        "ready" => 0,
        _ => 5,
    }
}

fn merge_degraded_reason(current: Option<String>, incoming: Option<String>) -> Option<String> {
    match (current, incoming) {
        (Some(current), Some(incoming)) if current == incoming => Some(current),
        (Some(current), Some(incoming)) => Some(format!("{current}; {incoming}")),
        (Some(current), None) => Some(current),
        (None, Some(incoming)) => Some(incoming),
        (None, None) => None,
    }
}

fn record_tool_trace(audit: &mut ToolRuntimeAudit, tool: &ProjectToolRecord, output: &ToolOutput) {
    let (status, error_code) = classify_tool_output(&output.content);
    audit.calls.push(ToolTraceEntry {
        tool_name: tool.tool_name.clone(),
        executor_kind: tool.executor_kind.clone(),
        status,
        error_code,
    });
}

fn classify_tool_output(content: &str) -> (String, Option<String>) {
    let parsed = match serde_json::from_str::<Value>(content) {
        Ok(parsed) => parsed,
        Err(_) => return ("success".to_string(), None),
    };
    let Some(error) = parsed.get("error") else {
        return ("success".to_string(), None);
    };
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    ("error".to_string(), code)
}

fn build_replay_candidate(
    request_shape: &ManagedToolRequestShape,
    model: Option<Value>,
    output: &ToolOutput,
) -> Value {
    let model = model.unwrap_or(Value::Null);
    match request_shape {
        ManagedToolRequestShape::OpenAiChatCompletions => json!({
            "model": model,
            "messages": [{
                "role": "tool",
                "tool_call_id": output.call_id,
                "content": output.content,
            }]
        }),
        ManagedToolRequestShape::OpenAiResponses => json!({
            "model": model,
            "input": [{
                "type": "function_call_output",
                "call_id": output.call_id,
                "output": output.content,
            }]
        }),
        ManagedToolRequestShape::AnthropicMessages => json!({
            "model": model,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": output.call_id,
                    "content": output.content,
                }]
            }]
        }),
    }
}

fn extract_replayed_content(
    request_shape: &ManagedToolRequestShape,
    candidate: &Value,
) -> Result<String, String> {
    match request_shape {
        ManagedToolRequestShape::OpenAiChatCompletions => candidate
            .get("messages")
            .and_then(|value| value.as_array())
            .and_then(|messages| messages.first())
            .and_then(|message| message.get("content"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
            .ok_or_else(|| "OpenAI replay candidate is missing tool content".to_string()),
        ManagedToolRequestShape::OpenAiResponses => candidate
            .get("input")
            .and_then(|value| value.as_array())
            .and_then(|input| input.first())
            .and_then(|message| message.get("output"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
            .ok_or_else(|| "OpenAI responses replay candidate is missing tool content".to_string()),
        ManagedToolRequestShape::AnthropicMessages => candidate
            .get("messages")
            .and_then(|value| value.as_array())
            .and_then(|messages| messages.first())
            .and_then(|message| message.get("content"))
            .and_then(|value| value.as_array())
            .and_then(|blocks| blocks.first())
            .and_then(|block| block.get("content"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
            .ok_or_else(|| "Anthropic replay candidate is missing tool content".to_string()),
    }
}

fn synthetic_request_context(ctx: &RequestContext, body: Bytes) -> RequestContext {
    let mut headers = hyper::HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Ok(value) = HeaderValue::from_str(&body.len().to_string()) {
        headers.insert(CONTENT_LENGTH, value);
    }

    RequestContext {
        peer_addr: ctx.peer_addr,
        method: ctx.method.clone(),
        uri: ctx.uri.clone(),
        version: ctx.version,
        headers,
        body: Some(body),
        route: ctx.route.clone(),
        selected_upstream: ctx.selected_upstream.clone(),
        auth: ctx.auth.clone(),
        connection: Arc::clone(&ctx.connection),
        extensions: hyper::http::Extensions::new(),
    }
}

fn blocked_tool_message(audit: &Option<SafetyAudit>, call_id: &str) -> String {
    audit
        .as_ref()
        .and_then(|audit| {
            audit
                .matches
                .iter()
                .find(|entry| entry.action == "block")
                .or_else(|| audit.matches.first())
                .map(|entry| entry.description.clone())
        })
        .unwrap_or_else(|| format!("tool output for call '{call_id}' was blocked"))
}

fn tool_error_payload(code: &str, message: &str) -> String {
    json!({
        "error": {
            "message": message,
            "code": code,
        }
    })
    .to_string()
}

fn json_error(
    status: StatusCode,
    message: &str,
) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(
            Full::new(Bytes::from(
                json!({
                    "error": {
                        "message": message,
                    }
                })
                .to_string(),
            ))
            .map_err(|never| match never {})
            .boxed(),
        )
        .unwrap()
}

fn extract_opt_in(body: &mut Value) -> Result<Option<OptInTools>, String> {
    let Some(object) = body.as_object_mut() else {
        return Ok(None);
    };
    let Some(opt_in_value) = object.remove("trp_tools") else {
        return Ok(None);
    };
    let Some(opt_in_object) = opt_in_value.as_object() else {
        return Err("trp_tools must be an object".to_string());
    };
    let enabled = opt_in_object
        .get("enabled")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let names = match opt_in_object.get("names") {
        Some(Value::Array(values)) => Some(
            values
                .iter()
                .filter_map(|value| value.as_str().map(ToString::to_string))
                .collect(),
        ),
        Some(Value::Null) | None => None,
        _ => return Err("trp_tools.names must be an array of strings".to_string()),
    };
    Ok(Some(OptInTools { enabled, names }))
}

fn select_tools(
    tools: &[ProjectToolRecord],
    requested_names: Option<&[String]>,
    approval_mode: &ToolApprovalMode,
    approved_names: Option<&[String]>,
) -> Result<Vec<ProjectToolRecord>, String> {
    if matches!(approval_mode, ToolApprovalMode::DenyAll) {
        return Err("managed tools are disabled by runtime policy".to_string());
    }

    let enabled_tools: HashMap<&str, &ProjectToolRecord> = tools
        .iter()
        .filter(|tool| tool.enabled)
        .map(|tool| (tool.tool_name.as_str(), tool))
        .collect();

    let is_approved = |name: &str| match approval_mode {
        ToolApprovalMode::AllowAll => true,
        ToolApprovalMode::DenyAll => false,
        ToolApprovalMode::AllowList => approved_names
            .map(|approved| approved.iter().any(|candidate| candidate == name))
            .unwrap_or(false),
    };

    let selected = match requested_names {
        Some(names) if !names.is_empty() => {
            let mut selected = Vec::with_capacity(names.len());
            for name in names {
                let Some(tool) = enabled_tools.get(name.as_str()) else {
                    return Err(format!("tool '{}' is not enabled for this project", name));
                };
                if !is_approved(name) {
                    return Err(format!("tool '{}' is not approved by runtime policy", name));
                }
                selected.push((*tool).clone());
            }
            selected
        }
        _ => {
            let selected = enabled_tools
                .values()
                .filter(|tool| is_approved(&tool.tool_name))
                .map(|tool| (*tool).clone())
                .collect::<Vec<_>>();
            if selected.is_empty() {
                return Err(match approval_mode {
                    ToolApprovalMode::AllowAll => {
                        "no enabled project tools are registered".to_string()
                    }
                    ToolApprovalMode::DenyAll => {
                        "managed tools are disabled by runtime policy".to_string()
                    }
                    ToolApprovalMode::AllowList => {
                        "no approved project tools are available for this runtime policy"
                            .to_string()
                    }
                });
            }
            selected
        }
    };

    Ok(selected)
}

fn merge_openai_chat_tools(
    request: &mut Value,
    project_tools: &[ProjectToolRecord],
) -> Result<(), String> {
    let object = request
        .as_object_mut()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;
    let tools_value = object
        .entry("tools")
        .or_insert_with(|| Value::Array(Vec::new()));
    let tools = tools_value
        .as_array_mut()
        .ok_or_else(|| "tools must be an array".to_string())?;
    let existing_names = extract_openai_tool_names(tools)?;

    for tool in project_tools {
        if existing_names.iter().any(|name| name == &tool.tool_name) {
            return Err(format!(
                "tool '{}' conflicts with a client-supplied tool",
                tool.tool_name
            ));
        }
        let input_schema: Value =
            serde_json::from_str(&tool.input_schema_json).map_err(|error| {
                format!(
                    "invalid input_schema_json for '{}': {error}",
                    tool.tool_name
                )
            })?;
        tools.push(json!({
            "type": "function",
            "function": {
                "name": tool.tool_name,
                "description": tool.description,
                "parameters": input_schema,
            }
        }));
    }
    Ok(())
}

fn merge_openai_responses_tools(
    request: &mut Value,
    project_tools: &[ProjectToolRecord],
) -> Result<(), String> {
    let object = request
        .as_object_mut()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;
    let tools_value = object
        .entry("tools")
        .or_insert_with(|| Value::Array(Vec::new()));
    let tools = tools_value
        .as_array_mut()
        .ok_or_else(|| "tools must be an array".to_string())?;
    let existing_names = extract_openai_responses_tool_names(tools)?;

    for tool in project_tools {
        if existing_names.iter().any(|name| name == &tool.tool_name) {
            return Err(format!(
                "tool '{}' conflicts with a client-supplied tool",
                tool.tool_name
            ));
        }
        let input_schema: Value =
            serde_json::from_str(&tool.input_schema_json).map_err(|error| {
                format!(
                    "invalid input_schema_json for '{}': {error}",
                    tool.tool_name
                )
            })?;
        tools.push(json!({
            "type": "function",
            "name": tool.tool_name,
            "description": tool.description,
            "parameters": input_schema,
        }));
    }
    Ok(())
}

fn merge_anthropic_tools(
    request: &mut Value,
    project_tools: &[ProjectToolRecord],
) -> Result<(), String> {
    let object = request
        .as_object_mut()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;
    let tools_value = object
        .entry("tools")
        .or_insert_with(|| Value::Array(Vec::new()));
    let tools = tools_value
        .as_array_mut()
        .ok_or_else(|| "tools must be an array".to_string())?;
    let existing_names = extract_anthropic_tool_names(tools)?;

    for tool in project_tools {
        if existing_names.iter().any(|name| name == &tool.tool_name) {
            return Err(format!(
                "tool '{}' conflicts with a client-supplied tool",
                tool.tool_name
            ));
        }
        let input_schema: Value =
            serde_json::from_str(&tool.input_schema_json).map_err(|error| {
                format!(
                    "invalid input_schema_json for '{}': {error}",
                    tool.tool_name
                )
            })?;
        tools.push(json!({
            "name": tool.tool_name,
            "description": tool.description,
            "input_schema": input_schema,
        }));
    }
    Ok(())
}

fn extract_openai_tool_names(tools: &[Value]) -> Result<Vec<String>, String> {
    let mut names = Vec::with_capacity(tools.len());
    for tool in tools {
        if let Some(name) = tool
            .get("function")
            .and_then(|value| value.get("name"))
            .and_then(|value| value.as_str())
        {
            names.push(name.to_string());
        } else {
            return Err("OpenAI tools must use function.name".to_string());
        }
    }
    Ok(names)
}

fn extract_openai_responses_tool_names(tools: &[Value]) -> Result<Vec<String>, String> {
    let mut names = Vec::with_capacity(tools.len());
    for tool in tools {
        if let Some(name) = tool.get("name").and_then(|value| value.as_str()) {
            names.push(name.to_string());
        } else {
            return Err("OpenAI responses tools must use name".to_string());
        }
    }
    Ok(names)
}

fn extract_anthropic_tool_names(tools: &[Value]) -> Result<Vec<String>, String> {
    let mut names = Vec::with_capacity(tools.len());
    for tool in tools {
        if let Some(name) = tool.get("name").and_then(|value| value.as_str()) {
            names.push(name.to_string());
        } else {
            return Err("Anthropic tools must use name".to_string());
        }
    }
    Ok(names)
}

fn parse_sse_events(body: &[u8]) -> Vec<SseEvent> {
    let text = String::from_utf8_lossy(body).replace("\r\n", "\n");
    text.split("\n\n")
        .filter_map(|block| {
            let block = block.trim();
            if block.is_empty() {
                return None;
            }
            let mut event = None;
            let mut data_lines = Vec::new();
            for line in block.lines() {
                if let Some(value) = line.strip_prefix("event:") {
                    event = Some(value.trim().to_string());
                } else if let Some(value) = line.strip_prefix("data:") {
                    data_lines.push(value.trim_start().to_string());
                }
            }
            if data_lines.is_empty() {
                return None;
            }
            Some(SseEvent {
                event,
                data: data_lines.join("\n"),
            })
        })
        .collect()
}

async fn collect_response(
    resp: Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>,
) -> Result<(StatusCode, hyper::HeaderMap, Bytes), String> {
    let (parts, body) = resp.into_parts();
    let bytes = body
        .collect()
        .await
        .map_err(|error| format!("failed to read provider response: {error}"))?
        .to_bytes();
    Ok((parts.status, parts.headers, bytes))
}

fn parse_openai_chat_usage(body: &Value) -> Option<(u64, u64)> {
    let usage = body.get("usage")?;
    Some((
        usage.get("prompt_tokens")?.as_u64()?,
        usage.get("completion_tokens")?.as_u64()?,
    ))
}

fn parse_openai_responses_usage(body: &Value) -> Option<(u64, u64)> {
    let usage = body.get("usage")?;
    Some((
        usage.get("input_tokens")?.as_u64()?,
        usage.get("output_tokens")?.as_u64()?,
    ))
}

fn parse_anthropic_usage(body: &Value) -> Option<(u64, u64)> {
    let usage = body.get("usage")?;
    Some((
        usage.get("input_tokens")?.as_u64()?,
        usage.get("output_tokens")?.as_u64()?,
    ))
}

fn parse_streaming_response(
    request_shape: &ManagedToolRequestShape,
    body: &[u8],
) -> Result<ParsedStreamingResponse, String> {
    match request_shape {
        ManagedToolRequestShape::OpenAiChatCompletions => {
            parse_openai_chat_streaming_response(body)
        }
        ManagedToolRequestShape::OpenAiResponses => parse_openai_responses_streaming_response(body),
        ManagedToolRequestShape::AnthropicMessages => parse_anthropic_streaming_response(body),
    }
}

fn parse_openai_chat_streaming_response(body: &[u8]) -> Result<ParsedStreamingResponse, String> {
    let mut usage = None;
    let mut accumulators: HashMap<usize, OpenAiToolCallAccumulator> = HashMap::new();

    for event in parse_sse_events(body) {
        if event.data == "[DONE]" {
            continue;
        }
        let value: Value = serde_json::from_str(&event.data)
            .map_err(|error| format!("invalid OpenAI SSE chunk: {error}"))?;
        if let Some(parsed_usage) = parse_openai_chat_usage(&value) {
            usage = Some(parsed_usage);
        }
        let Some(choices) = value.get("choices").and_then(|entry| entry.as_array()) else {
            continue;
        };
        for choice in choices {
            let Some(tool_calls) = choice
                .get("delta")
                .and_then(|entry| entry.get("tool_calls"))
                .and_then(|entry| entry.as_array())
            else {
                continue;
            };
            for tool_call in tool_calls {
                let index = tool_call
                    .get("index")
                    .and_then(|entry| entry.as_u64())
                    .unwrap_or(accumulators.len() as u64) as usize;
                let entry = accumulators.entry(index).or_default();
                if let Some(call_id) = tool_call.get("id").and_then(|entry| entry.as_str()) {
                    entry.call_id = call_id.to_string();
                }
                if let Some(name) = tool_call
                    .get("function")
                    .and_then(|entry| entry.get("name"))
                    .and_then(|entry| entry.as_str())
                {
                    entry.name.push_str(name);
                }
                if let Some(arguments) = tool_call
                    .get("function")
                    .and_then(|entry| entry.get("arguments"))
                    .and_then(|entry| entry.as_str())
                {
                    entry.arguments.push_str(arguments);
                }
            }
        }
    }

    if accumulators.is_empty() {
        return Ok(ParsedStreamingResponse {
            tool_calls: Vec::new(),
            usage,
        });
    }

    let mut ordered = accumulators.into_iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(index, _)| *index);

    let tool_calls_json = ordered
        .iter()
        .map(|(_, tool_call)| {
            json!({
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
            })
        })
        .collect::<Vec<_>>();
    let assistant_message = json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": tool_calls_json,
    });

    let mut parsed = Vec::with_capacity(ordered.len());
    for (_, tool_call) in ordered {
        if tool_call.call_id.is_empty() {
            return Err("OpenAI streaming tool call is missing id".to_string());
        }
        if tool_call.name.is_empty() {
            return Err("OpenAI streaming tool call is missing function.name".to_string());
        }
        let arguments = serde_json::from_str::<Value>(&tool_call.arguments)
            .unwrap_or_else(|_| Value::String(tool_call.arguments.clone()));
        parsed.push(ParsedToolCall {
            call_id: tool_call.call_id,
            name: tool_call.name,
            arguments,
            assistant_message: assistant_message.clone(),
        });
    }

    Ok(ParsedStreamingResponse {
        tool_calls: parsed,
        usage,
    })
}

fn parse_openai_responses_streaming_response(
    body: &[u8],
) -> Result<ParsedStreamingResponse, String> {
    let mut usage = None;
    let mut response_id = None;
    let mut parsed = Vec::new();

    for event in parse_sse_events(body) {
        if event.data == "[DONE]" {
            continue;
        }
        let value: Value = serde_json::from_str(&event.data)
            .map_err(|error| format!("invalid OpenAI responses SSE chunk: {error}"))?;

        if let Some(id) = value
            .get("response")
            .and_then(|response| response.get("id"))
            .and_then(|id| id.as_str())
        {
            response_id = Some(id.to_string());
        }
        if let Some(parsed_usage) = value.get("response").and_then(parse_openai_responses_usage) {
            usage = Some(parsed_usage);
        }

        if value.get("type").and_then(|entry| entry.as_str()) != Some("response.output_item.done") {
            continue;
        }
        let Some(item) = value.get("item") else {
            continue;
        };
        if item.get("type").and_then(|entry| entry.as_str()) != Some("function_call") {
            continue;
        }

        let call_id = item
            .get("call_id")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "OpenAI responses tool call is missing call_id".to_string())?;
        let name = item
            .get("name")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "OpenAI responses tool call is missing name".to_string())?;
        let raw_arguments = item
            .get("arguments")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "OpenAI responses tool call is missing arguments".to_string())?;
        let arguments = serde_json::from_str::<Value>(raw_arguments)
            .unwrap_or_else(|_| Value::String(raw_arguments.to_string()));
        parsed.push(ParsedToolCall {
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments,
            assistant_message: Value::Null,
        });
    }

    if parsed.is_empty() {
        return Ok(ParsedStreamingResponse {
            tool_calls: Vec::new(),
            usage,
        });
    }

    let response_id = response_id
        .ok_or_else(|| "OpenAI responses tool stream is missing response id".to_string())?;
    for tool_call in &mut parsed {
        tool_call.assistant_message = json!({
            "id": response_id,
        });
    }

    Ok(ParsedStreamingResponse {
        tool_calls: parsed,
        usage,
    })
}

fn parse_anthropic_streaming_response(body: &[u8]) -> Result<ParsedStreamingResponse, String> {
    let mut usage = None;
    let mut accumulators: HashMap<usize, AnthropicToolCallAccumulator> = HashMap::new();

    for event in parse_sse_events(body) {
        let value: Value = serde_json::from_str(&event.data)
            .map_err(|error| format!("invalid Anthropic SSE chunk: {error}"))?;
        if let Some(parsed_usage) = parse_anthropic_usage(&value) {
            usage = Some(parsed_usage);
        }
        match event.event.as_deref() {
            Some("content_block_start") => {
                if value.get("type").and_then(|entry| entry.as_str()) != Some("content_block_start")
                {
                    continue;
                }
                let Some(block) = value.get("content_block") else {
                    continue;
                };
                if block.get("type").and_then(|entry| entry.as_str()) != Some("tool_use") {
                    continue;
                }
                let index = value
                    .get("index")
                    .and_then(|entry| entry.as_u64())
                    .unwrap_or(accumulators.len() as u64) as usize;
                let entry = accumulators.entry(index).or_default();
                entry.call_id = block
                    .get("id")
                    .and_then(|entry| entry.as_str())
                    .unwrap_or_default()
                    .to_string();
                entry.name = block
                    .get("name")
                    .and_then(|entry| entry.as_str())
                    .unwrap_or_default()
                    .to_string();
                if let Some(input) = block.get("input") {
                    entry.input = Some(input.clone());
                    entry.raw_input = serde_json::to_string(input).unwrap_or_default();
                }
            }
            Some("content_block_delta") => {
                if value.get("type").and_then(|entry| entry.as_str()) != Some("content_block_delta")
                {
                    continue;
                }
                let Some(delta) = value.get("delta") else {
                    continue;
                };
                if delta.get("type").and_then(|entry| entry.as_str()) != Some("input_json_delta") {
                    continue;
                }
                let index = value
                    .get("index")
                    .and_then(|entry| entry.as_u64())
                    .unwrap_or(accumulators.len() as u64) as usize;
                let entry = accumulators.entry(index).or_default();
                if let Some(partial) = delta.get("partial_json").and_then(|entry| entry.as_str()) {
                    entry.raw_input.push_str(partial);
                }
            }
            _ => {}
        }
    }

    if accumulators.is_empty() {
        return Ok(ParsedStreamingResponse {
            tool_calls: Vec::new(),
            usage,
        });
    }

    let mut ordered = accumulators.into_iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(index, _)| *index);

    let tool_blocks = ordered
        .iter()
        .map(|(_, tool_call)| {
            json!({
                "type": "tool_use",
                "id": tool_call.call_id,
                "name": tool_call.name,
                "input": tool_call.input.clone().unwrap_or(Value::Null),
            })
        })
        .collect::<Vec<_>>();
    let assistant_message = json!({
        "role": "assistant",
        "content": tool_blocks,
    });

    let mut parsed = Vec::with_capacity(ordered.len());
    for (_, mut tool_call) in ordered {
        if tool_call.call_id.is_empty() {
            return Err("Anthropic streaming tool call is missing id".to_string());
        }
        if tool_call.name.is_empty() {
            return Err("Anthropic streaming tool call is missing name".to_string());
        }
        if tool_call.input.is_none() && !tool_call.raw_input.is_empty() {
            tool_call.input = serde_json::from_str::<Value>(&tool_call.raw_input).ok();
        }
        parsed.push(ParsedToolCall {
            call_id: tool_call.call_id,
            name: tool_call.name,
            arguments: tool_call.input.unwrap_or(Value::Null),
            assistant_message: assistant_message.clone(),
        });
    }

    Ok(ParsedStreamingResponse {
        tool_calls: parsed,
        usage,
    })
}

fn extract_openai_chat_tool_calls(body: &Value) -> Result<Vec<ParsedToolCall>, String> {
    let Some(choice) = body
        .get("choices")
        .and_then(|value| value.as_array())
        .and_then(|choices| choices.first())
    else {
        return Ok(Vec::new());
    };
    let Some(message) = choice.get("message") else {
        return Ok(Vec::new());
    };
    let Some(tool_calls) = message.get("tool_calls").and_then(|value| value.as_array()) else {
        return Ok(Vec::new());
    };

    let mut parsed = Vec::with_capacity(tool_calls.len());
    for tool_call in tool_calls {
        let call_id = tool_call
            .get("id")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "OpenAI tool call is missing id".to_string())?;
        let name = tool_call
            .get("function")
            .and_then(|value| value.get("name"))
            .and_then(|value| value.as_str())
            .ok_or_else(|| "OpenAI tool call is missing function.name".to_string())?;
        let raw_arguments = tool_call
            .get("function")
            .and_then(|value| value.get("arguments"))
            .and_then(|value| value.as_str())
            .ok_or_else(|| "OpenAI tool call is missing function.arguments".to_string())?;
        let arguments = serde_json::from_str::<Value>(raw_arguments)
            .unwrap_or_else(|_| Value::String(raw_arguments.to_string()));
        parsed.push(ParsedToolCall {
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments,
            assistant_message: message.clone(),
        });
    }
    Ok(parsed)
}

fn extract_openai_responses_tool_calls(body: &Value) -> Result<Vec<ParsedToolCall>, String> {
    let response_id = body
        .get("id")
        .and_then(|value| value.as_str())
        .ok_or_else(|| "OpenAI responses body is missing id".to_string())?;
    let Some(output) = body.get("output").and_then(|value| value.as_array()) else {
        return Ok(Vec::new());
    };

    let mut parsed = Vec::new();
    for item in output {
        if item.get("type").and_then(|value| value.as_str()) != Some("function_call") {
            continue;
        }
        let call_id = item
            .get("call_id")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "OpenAI responses tool call is missing call_id".to_string())?;
        let name = item
            .get("name")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "OpenAI responses tool call is missing name".to_string())?;
        let raw_arguments = item
            .get("arguments")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "OpenAI responses tool call is missing arguments".to_string())?;
        let arguments = serde_json::from_str::<Value>(raw_arguments)
            .unwrap_or_else(|_| Value::String(raw_arguments.to_string()));
        parsed.push(ParsedToolCall {
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments,
            assistant_message: json!({
                "id": response_id,
            }),
        });
    }
    Ok(parsed)
}

fn extract_anthropic_tool_calls(body: &Value) -> Result<Vec<ParsedToolCall>, String> {
    let Some(content) = body.get("content").and_then(|value| value.as_array()) else {
        return Ok(Vec::new());
    };
    let assistant_message = json!({
        "role": "assistant",
        "content": content.clone(),
    });
    let mut parsed = Vec::new();
    for block in content {
        if block.get("type").and_then(|value| value.as_str()) != Some("tool_use") {
            continue;
        }
        let call_id = block
            .get("id")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "Anthropic tool_use block is missing id".to_string())?;
        let name = block
            .get("name")
            .and_then(|value| value.as_str())
            .ok_or_else(|| "Anthropic tool_use block is missing name".to_string())?;
        let arguments = block.get("input").cloned().unwrap_or(Value::Null);
        parsed.push(ParsedToolCall {
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments,
            assistant_message: assistant_message.clone(),
        });
    }
    Ok(parsed)
}

fn append_openai_chat_tool_results(
    request: &mut Value,
    outputs: &[ToolOutput],
) -> Result<(), String> {
    let messages = request
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "OpenAI request is missing messages".to_string())?;
    messages.push(outputs[0].assistant_message.clone());
    for output in outputs {
        messages.push(json!({
            "role": "tool",
            "tool_call_id": output.call_id,
            "content": output.content,
        }));
    }
    Ok(())
}

fn append_openai_responses_tool_results(
    request: &mut Value,
    outputs: &[ToolOutput],
) -> Result<(), String> {
    let request_object = request
        .as_object_mut()
        .ok_or_else(|| "OpenAI responses request must be a JSON object".to_string())?;
    let response_id = outputs
        .first()
        .and_then(|output| output.assistant_message.get("id"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| "OpenAI responses tool output is missing response id".to_string())?;
    request_object.insert(
        "previous_response_id".to_string(),
        Value::String(response_id.to_string()),
    );
    request_object.insert(
        "input".to_string(),
        Value::Array(
            outputs
                .iter()
                .map(|output| {
                    json!({
                        "type": "function_call_output",
                        "call_id": output.call_id,
                        "output": output.content,
                    })
                })
                .collect(),
        ),
    );
    Ok(())
}

fn append_anthropic_tool_results(
    request: &mut Value,
    outputs: &[ToolOutput],
) -> Result<(), String> {
    let messages = request
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "Anthropic request is missing messages".to_string())?;
    messages.push(outputs[0].assistant_message.clone());
    let content = outputs
        .iter()
        .map(|output| {
            json!({
                "type": "tool_result",
                "tool_use_id": output.call_id,
                "content": output.content,
            })
        })
        .collect::<Vec<_>>();
    messages.push(json!({
        "role": "user",
        "content": content,
    }));
    Ok(())
}

async fn send_provider_request(
    client: &proxy_core::handlers::proxy::HttpClient,
    ctx: &RequestContext,
    provider_name: &str,
    round_trip: usize,
    upstream: &str,
    body_json: &Value,
    timeout: Duration,
) -> Result<ProviderResponse, String> {
    let body = serde_json::to_vec(body_json)
        .map_err(|error| format!("failed to encode follow-up provider request: {error}"))?;
    let uri = join_upstream_uri(
        upstream,
        ctx.uri
            .path_and_query()
            .map(|value| value.as_str())
            .unwrap_or(ctx.uri.path()),
    )?;
    let mut builder = Request::builder()
        .method(ctx.method.clone())
        .uri(uri.clone());
    if let Some(headers) = builder.headers_mut() {
        *headers = ctx.headers.clone();
        headers.remove(CONTENT_LENGTH);
        headers.insert(
            CONTENT_LENGTH,
            HeaderValue::from_str(&body.len().to_string()).map_err(|error| error.to_string())?,
        );
        if let Some(authority) = uri.authority() {
            headers.insert(
                HOST,
                HeaderValue::from_str(authority.as_str()).map_err(|error| error.to_string())?,
            );
        }
    }
    let req = builder
        .body(
            Full::new(Bytes::from(body))
                .map_err(|never| match never {})
                .boxed(),
        )
        .map_err(|error| format!("failed to build follow-up provider request: {error}"))?;
    let provider_span = tracing::info_span!(
        "provider_follow_up_turn",
        llm.provider = %provider_name,
        net.upstream = %upstream,
        llm.tool_round_trip = round_trip as u64,
        http.method = %ctx.method,
        http.target = %ctx.uri.path(),
        proxy.timeout_ms = timeout.as_millis() as u64,
    );
    let resp = tokio::time::timeout(timeout, client.request(req).instrument(provider_span))
        .await
        .map_err(|_| "follow-up provider request timed out".to_string())?
        .map_err(|error| format!("follow-up provider request failed: {error}"))?;
    let (parts, body) = resp.into_parts();
    let bytes = body
        .collect()
        .await
        .map_err(|error| format!("failed to read follow-up provider response: {error}"))?
        .to_bytes();
    Ok(ProviderResponse {
        status: parts.status,
        headers: parts.headers,
        body: bytes,
    })
}

fn sse_event(event_name: &str, payload: Value) -> String {
    format!("event: {event_name}\ndata: {}\n\n", payload)
}

fn sse_event_with_optional_name(event_name: Option<&str>, payload: Value) -> String {
    match event_name {
        Some(event_name) => sse_event(event_name, payload),
        None => format!("data: {}\n\n", payload),
    }
}

fn sse_raw_data(data: &str) -> String {
    format!("data: {data}\n\n")
}

fn tool_call_event(tool_call: &ParsedToolCall) -> String {
    sse_event(
        "trp_tool_call",
        json!({
            "call_id": tool_call.call_id,
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
        }),
    )
}

fn tool_output_event(output: &ToolOutput) -> String {
    if let Ok(value) = serde_json::from_str::<Value>(&output.content) {
        if let Some(error) = value.get("error") {
            return sse_event(
                "trp_tool_error",
                json!({
                    "call_id": output.call_id,
                    "tool_name": output.tool_name,
                    "error": error,
                }),
            );
        }
    }

    sse_event(
        "trp_tool_result",
        json!({
            "call_id": output.call_id,
            "tool_name": output.tool_name,
            "content": output.content,
        }),
    )
}

fn prepend_sse_events(prefix_events: &[String], body: Bytes) -> Bytes {
    if prefix_events.is_empty() {
        return body;
    }
    let mut combined = prefix_events.join("").into_bytes();
    combined.extend_from_slice(body.as_ref());
    Bytes::from(combined)
}

fn prepend_openai_responses_events(prefix_events: &[String], body: Bytes) -> Result<Bytes, String> {
    if prefix_events.is_empty() {
        return Ok(body);
    }
    let offset = prefix_events.len() as u64;
    let adjusted_body = offset_openresponses_sequence_numbers(&body, offset)?;
    let mut combined = prefix_events.join("").into_bytes();
    combined.extend_from_slice(adjusted_body.as_ref());
    Ok(Bytes::from(combined))
}

fn offset_openresponses_sequence_numbers(body: &[u8], offset: u64) -> Result<Bytes, String> {
    if offset == 0 {
        return Ok(Bytes::copy_from_slice(body));
    }

    let mut rewritten = String::new();
    for event in parse_sse_events(body) {
        if event.data == "[DONE]" {
            rewritten.push_str(&sse_raw_data("[DONE]"));
            continue;
        }
        let mut payload: Value = serde_json::from_str(&event.data)
            .map_err(|error| format!("invalid OpenResponses SSE chunk: {error}"))?;
        if let Some(sequence) = payload.get("sequence_number").and_then(Value::as_u64) {
            payload["sequence_number"] = Value::from(sequence.saturating_add(offset));
        }
        rewritten.push_str(&sse_event_with_optional_name(
            event.event.as_deref(),
            payload,
        ));
    }

    Ok(Bytes::from(rewritten))
}

fn build_openai_responses_tool_round_trip_events(
    request: &Value,
    tool_calls: &[ParsedToolCall],
    outputs: &[ToolOutput],
    usage: Option<(u64, u64)>,
    starting_sequence: u64,
) -> Result<Vec<String>, String> {
    if tool_calls.is_empty() {
        return Ok(Vec::new());
    }

    let response_id = tool_calls
        .first()
        .and_then(|tool_call| tool_call.assistant_message.get("id"))
        .and_then(Value::as_str)
        .ok_or_else(|| "OpenAI responses tool stream is missing response id".to_string())?;
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let mut sequence_number = starting_sequence;
    let mut events = Vec::new();
    events.push(sse_event(
        "response.created",
        json!({
            "type": "response.created",
            "sequence_number": sequence_number,
            "response": {
                "id": response_id,
                "object": "response",
                "model": model,
                "status": "in_progress",
                "output": [],
            }
        }),
    ));
    sequence_number = sequence_number.saturating_add(1);

    for (index, tool_call) in tool_calls.iter().enumerate() {
        let function_call_id = format!("fc_{}", tool_call.call_id);
        events.push(sse_event(
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": sequence_number,
                "output_index": index,
                "item": {
                    "type": "function_call",
                    "id": function_call_id,
                    "status": "in_progress",
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": "",
                }
            }),
        ));
        sequence_number = sequence_number.saturating_add(1);
        events.push(sse_event(
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "sequence_number": sequence_number,
                "output_index": index,
                "item": {
                    "type": "function_call",
                    "id": function_call_id,
                    "status": "completed",
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": serde_json::to_string(&tool_call.arguments)
                        .unwrap_or_else(|_| "null".to_string()),
                }
            }),
        ));
        sequence_number = sequence_number.saturating_add(1);
    }

    for (index, output) in outputs.iter().enumerate() {
        let output_index = tool_calls.len() + index;
        let output_item_id = format!("fco_{}", output.call_id);
        let item = json!({
            "type": "function_call_output",
            "id": output_item_id,
            "call_id": output.call_id,
            "output": output.content,
        });
        events.push(sse_event(
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": sequence_number,
                "output_index": output_index,
                "item": item,
            }),
        ));
        sequence_number = sequence_number.saturating_add(1);
        events.push(sse_event(
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "sequence_number": sequence_number,
                "output_index": output_index,
                "item": item,
            }),
        ));
        sequence_number = sequence_number.saturating_add(1);
    }

    let mut completed_response = json!({
        "id": response_id,
        "object": "response",
        "model": model,
        "status": "completed",
        "output": [],
    });
    if let Some((input_tokens, output_tokens)) = usage {
        completed_response["usage"] = json!({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens.saturating_add(output_tokens),
        });
    }
    events.push(sse_event(
        "response.completed",
        json!({
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": completed_response,
        }),
    ));

    Ok(events)
}

fn join_upstream_uri(upstream: &str, path_and_query: &str) -> Result<hyper::Uri, String> {
    let base = upstream.trim_end_matches('/');
    let path = if path_and_query.starts_with('/') {
        path_and_query.to_string()
    } else {
        format!("/{path_and_query}")
    };
    format!("{base}{path}")
        .parse::<hyper::Uri>()
        .map_err(|error| format!("invalid upstream URI: {error}"))
}

fn response_from_bytes(
    status: StatusCode,
    mut headers: hyper::HeaderMap,
    body: Bytes,
    original_headers: &hyper::HeaderMap,
) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
    headers.remove(CONTENT_LENGTH);
    if let Ok(length) = HeaderValue::from_str(&body.len().to_string()) {
        headers.insert(CONTENT_LENGTH, length);
    }
    for key in ["via", "x-cache", "alt-svc"] {
        if let Some(value) = original_headers.get(key) {
            headers.insert(
                hyper::header::HeaderName::from_bytes(key.as_bytes()).unwrap(),
                value.clone(),
            );
        }
    }

    let mut builder = Response::builder().status(status);
    if let Some(response_headers) = builder.headers_mut() {
        *response_headers = headers;
    }
    builder
        .body(Full::new(body).map_err(|never| match never {}).boxed())
        .unwrap()
}

impl ToolRuntime {
    async fn run_streaming_loop(
        &self,
        ctx: &mut RequestContext,
        request_meta: ToolRuntimeRequest,
        upstream: String,
        initial_resp: Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>,
    ) -> Result<Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>, String> {
        let client = build_client();
        let original_headers = initial_resp.headers().clone();
        let (mut status, mut headers, mut body_bytes) = collect_response(initial_resp).await?;
        let mut current_request = current_forwarded_request(ctx, &request_meta)?;
        let mut total_input_tokens = 0u64;
        let mut total_output_tokens = 0u64;
        let mut round_trips = 0usize;
        let mut budget_state = ToolExecutionBudgetState::default();
        let mut synthetic_events = Vec::new();
        let mut safety_audit = ctx.extensions.get::<SafetyAudit>().cloned();
        let mut semantic_audit = ctx.extensions.get::<SemanticSafetyAudit>().cloned();
        let mut tool_audit = ctx
            .extensions
            .get::<ToolRuntimeAudit>()
            .cloned()
            .unwrap_or_default();

        loop {
            let parsed = parse_streaming_response(&request_meta.request_shape, &body_bytes)?;
            if let Some((input_tokens, output_tokens)) = parsed.usage {
                total_input_tokens = total_input_tokens.saturating_add(input_tokens);
                total_output_tokens = total_output_tokens.saturating_add(output_tokens);
            }

            if parsed.tool_calls.is_empty() {
                if round_trips > 0 {
                    ctx.extensions.insert(ToolUsageOverride {
                        input_tokens: total_input_tokens,
                        output_tokens: total_output_tokens,
                    });
                }
                apply_aggregated_audits(ctx, safety_audit, semantic_audit, tool_audit);
                let body = match request_meta.request_shape {
                    ManagedToolRequestShape::OpenAiResponses => {
                        prepend_openai_responses_events(&synthetic_events, body_bytes)?
                    }
                    ManagedToolRequestShape::OpenAiChatCompletions
                    | ManagedToolRequestShape::AnthropicMessages => {
                        prepend_sse_events(&synthetic_events, body_bytes)
                    }
                };
                return Ok(response_from_bytes(
                    status,
                    headers,
                    body,
                    &original_headers,
                ));
            }

            round_trips += 1;
            if round_trips > self.max_round_trips {
                return Err("tool execution exceeded max_round_trips".to_string());
            }

            let mut tool_outputs = Vec::with_capacity(parsed.tool_calls.len());
            for tool_call in &parsed.tool_calls {
                if request_meta.request_shape != ManagedToolRequestShape::OpenAiResponses {
                    synthetic_events.push(tool_call_event(tool_call));
                }
                ensure_allowed_tool_call(&request_meta, &tool_call.name)?;
                let Some(tool) = request_meta.selected_tools.get(&tool_call.name) else {
                    return Err(format!(
                        "tool '{}' is not registered for this project",
                        tool_call.name
                    ));
                };
                let output = self
                    .execute_tool(
                        &request_meta.project_id,
                        &request_meta.provider_name,
                        tool,
                        &tool_call.arguments,
                        &mut budget_state,
                    )
                    .await
                    .unwrap_or_else(|error| tool_error_payload("tool_execution_failed", &error));
                let replayed = self
                    .replay_tool_output(
                        ctx,
                        &request_meta,
                        &current_request,
                        ToolOutput {
                            call_id: tool_call.call_id.clone(),
                            tool_name: tool_call.name.clone(),
                            content: output,
                            assistant_message: tool_call.assistant_message.clone(),
                        },
                    )
                    .await?;
                merge_safety_audit(&mut safety_audit, replayed.safety_audit);
                merge_semantic_audit(&mut semantic_audit, replayed.semantic_audit);
                record_tool_trace(&mut tool_audit, tool, &replayed.output);
                if request_meta.request_shape != ManagedToolRequestShape::OpenAiResponses {
                    synthetic_events.push(tool_output_event(&replayed.output));
                }
                tool_outputs.push(replayed.output);
            }

            if request_meta.request_shape == ManagedToolRequestShape::OpenAiResponses {
                synthetic_events.extend(build_openai_responses_tool_round_trip_events(
                    &current_request,
                    &parsed.tool_calls,
                    &tool_outputs,
                    parsed.usage,
                    synthetic_events.len() as u64 + 1,
                )?);
            }

            match request_meta.request_shape {
                ManagedToolRequestShape::OpenAiChatCompletions => {
                    append_openai_chat_tool_results(&mut current_request, &tool_outputs)?
                }
                ManagedToolRequestShape::OpenAiResponses => {
                    append_openai_responses_tool_results(&mut current_request, &tool_outputs)?
                }
                ManagedToolRequestShape::AnthropicMessages => {
                    append_anthropic_tool_results(&mut current_request, &tool_outputs)?
                }
            }

            let follow_up = send_provider_request(
                &client,
                ctx,
                &request_meta.provider_name,
                round_trips,
                &upstream,
                &current_request,
                self.timeout,
            )
            .await?;
            status = follow_up.status;
            headers = follow_up.headers;
            body_bytes = follow_up.body;
        }
    }
}
