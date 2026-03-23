use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use proxy_core::plugin::{Action, Plugin, RequestContext};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, Endpoint};
use tonic::Request as GrpcRequest;
use tracing::Instrument;

use semantic_safety_protocol::{
    index_state_name, Chunk, DeleteProjectPolicyRequest, EvaluateRequest,
    GetProjectSyncStatusRequest, HealthRequest, IndexState, ListProjectSyncStatesRequest,
    ProjectSemanticPolicy, SemanticEntity, SemanticSafetyServiceClient, SemanticTopic,
    UpsertProjectPolicyRequest,
};

use crate::extract_model;
use crate::governance::{current_timestamp_string, GovernanceState};
use crate::metrics::LlmMetrics;
use crate::store::ProjectSemanticPolicyRecord;

const DEFAULT_TIMEOUT_MS: u64 = 750;
const DEFAULT_MAX_CHUNKS: usize = 16;
const DEFAULT_MAX_CHARS_PER_CHUNK: usize = 1024;
const DEFAULT_TOP_K: u32 = 8;
const DEFAULT_RECONCILE_INTERVAL_SECS: u64 = 300;
static SEMANTIC_POLICY_VERSION_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticFindingAudit {
    pub chunk_path: String,
    pub entity_id: String,
    pub entity_text: String,
    pub topic_id: String,
    pub matched_exemplar: String,
    pub embedding_score: f32,
    pub rerank_score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticSafetyAudit {
    pub mode: String,
    pub project_id: String,
    pub policy_version: String,
    pub index_state: String,
    pub service_latency_ms: u64,
    pub degraded_reason: Option<String>,
    pub findings: Vec<SemanticFindingAudit>,
}

#[derive(Clone)]
pub struct SemanticSafetyReplayHandle(SemanticSafety);

impl SemanticSafetyReplayHandle {
    pub async fn on_synthetic_request(&self, ctx: &mut RequestContext) -> Action {
        self.0.on_request(ctx).await
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticPolicySyncStatus {
    pub project_id: String,
    pub policy_version: String,
    pub index_state: String,
    pub updated_at: String,
    pub stored_exemplar_count: u64,
    pub available: bool,
    pub ready: bool,
    pub backend: Option<String>,
    pub message: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticPolicyMutationResult {
    pub policy_version: String,
    pub synced: bool,
    pub sync_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticPolicyDeleteResult {
    pub existed: bool,
    pub synced: bool,
    pub sync_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SemanticSafetyConfig {
    endpoint: String,
    auth_token: Option<String>,
    timeout: Duration,
    max_chunks: usize,
    max_chars_per_chunk: usize,
    top_k: u32,
    reconcile_interval: Duration,
}

impl SemanticSafetyConfig {
    pub fn from_toml(config: &toml::Value) -> Result<Self, Box<dyn std::error::Error>> {
        let endpoint = config
            .get("endpoint")
            .and_then(|value| value.as_str())
            .ok_or("semantic_safety.endpoint is required")?
            .to_string();
        let auth_token = config
            .get("auth_token")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string());
        let timeout = Duration::from_millis(
            config
                .get("timeout_ms")
                .and_then(|value| value.as_integer())
                .unwrap_or(DEFAULT_TIMEOUT_MS as i64)
                .max(1) as u64,
        );
        let max_chunks = config
            .get("max_chunks")
            .and_then(|value| value.as_integer())
            .unwrap_or(DEFAULT_MAX_CHUNKS as i64)
            .max(1) as usize;
        let max_chars_per_chunk = config
            .get("max_chars_per_chunk")
            .and_then(|value| value.as_integer())
            .unwrap_or(DEFAULT_MAX_CHARS_PER_CHUNK as i64)
            .max(64) as usize;
        let top_k = config
            .get("top_k")
            .and_then(|value| value.as_integer())
            .unwrap_or(DEFAULT_TOP_K as i64)
            .max(1) as u32;
        let reconcile_interval = Duration::from_secs(
            config
                .get("reconcile_interval_secs")
                .and_then(|value| value.as_integer())
                .unwrap_or(DEFAULT_RECONCILE_INTERVAL_SECS as i64)
                .max(30) as u64,
        );
        Ok(Self {
            endpoint,
            auth_token,
            timeout,
            max_chunks,
            max_chars_per_chunk,
            top_k,
            reconcile_interval,
        })
    }
}

#[derive(Clone)]
struct SemanticSafetyGrpcClient {
    channel: Channel,
    auth_header: Option<String>,
}

impl SemanticSafetyGrpcClient {
    fn new(endpoint: &str, auth_token: Option<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let channel = Endpoint::from_shared(endpoint.to_string())?.connect_lazy();
        Ok(Self {
            channel,
            auth_header: auth_token.map(|token| format!("Bearer {token}")),
        })
    }

    fn with_auth<T>(
        &self,
        request: T,
    ) -> Result<GrpcRequest<T>, Box<dyn std::error::Error + Send + Sync>> {
        let mut request = GrpcRequest::new(request);
        if let Some(header) = &self.auth_header {
            request
                .metadata_mut()
                .insert("authorization", MetadataValue::try_from(header.as_str())?);
        }
        Ok(request)
    }

    async fn evaluate(
        &self,
        request: EvaluateRequest,
    ) -> Result<
        tonic::Response<semantic_safety_protocol::EvaluateResponse>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let mut client = SemanticSafetyServiceClient::new(self.channel.clone());
        Ok(client.evaluate(self.with_auth(request)?).await?)
    }

    async fn upsert_policy(
        &self,
        policy: ProjectSemanticPolicy,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut client = SemanticSafetyServiceClient::new(self.channel.clone());
        client
            .upsert_project_policy(self.with_auth(UpsertProjectPolicyRequest {
                policy: Some(policy),
            })?)
            .await?;
        Ok(())
    }

    async fn delete_policy(
        &self,
        project_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut client = SemanticSafetyServiceClient::new(self.channel.clone());
        client
            .delete_project_policy(self.with_auth(DeleteProjectPolicyRequest {
                project_id: project_id.to_string(),
            })?)
            .await?;
        Ok(())
    }

    async fn get_sync_status(
        &self,
        project_id: &str,
    ) -> Result<SemanticPolicySyncStatus, Box<dyn std::error::Error + Send + Sync>> {
        let mut client = SemanticSafetyServiceClient::new(self.channel.clone());
        let response = client
            .get_project_sync_status(self.with_auth(GetProjectSyncStatusRequest {
                project_id: project_id.to_string(),
            })?)
            .await?
            .into_inner();
        Ok(SemanticPolicySyncStatus {
            project_id: response.project_id,
            policy_version: response.policy_version,
            index_state: index_state_name(response.index_state).to_string(),
            updated_at: response.updated_at,
            stored_exemplar_count: response.stored_exemplar_count,
            available: true,
            ready: true,
            backend: None,
            message: None,
            error: None,
        })
    }

    async fn list_sync_states(
        &self,
    ) -> Result<Vec<SemanticPolicySyncStatus>, Box<dyn std::error::Error + Send + Sync>> {
        let mut client = SemanticSafetyServiceClient::new(self.channel.clone());
        let response = client
            .list_project_sync_states(self.with_auth(ListProjectSyncStatesRequest {})?)
            .await?
            .into_inner();
        Ok(response
            .projects
            .into_iter()
            .map(|state| SemanticPolicySyncStatus {
                project_id: state.project_id,
                policy_version: state.policy_version,
                index_state: index_state_name(state.index_state).to_string(),
                updated_at: state.updated_at,
                stored_exemplar_count: state.stored_exemplar_count,
                available: true,
                ready: true,
                backend: None,
                message: None,
                error: None,
            })
            .collect())
    }

    async fn health(
        &self,
    ) -> Result<
        tonic::Response<semantic_safety_protocol::HealthResponse>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let mut client = SemanticSafetyServiceClient::new(self.channel.clone());
        Ok(client.health(self.with_auth(HealthRequest {})?).await?)
    }
}

#[derive(Clone)]
pub struct SemanticSafety {
    config: SemanticSafetyConfig,
    client: SemanticSafetyGrpcClient,
    governance: Arc<GovernanceState>,
    metrics: Option<LlmMetrics>,
}

impl SemanticSafety {
    pub fn new(
        config: SemanticSafetyConfig,
        governance: Arc<GovernanceState>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let client = SemanticSafetyGrpcClient::new(&config.endpoint, config.auth_token.clone())?;
        Ok(Self {
            config,
            client,
            governance,
            metrics: None,
        })
    }

    pub fn with_metrics(mut self, metrics: LlmMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn spawn_reconciliation_task(&self) {
        let this = self.clone();
        tokio::spawn(async move {
            this.reconcile_once().await;
            loop {
                tokio::time::sleep(this.config.reconcile_interval).await;
                this.reconcile_once().await;
            }
        });
    }

    pub async fn reconcile_once(&self) {
        let records = self.governance.list_semantic_policies();
        let remote_statuses = match self.client.list_sync_states().await {
            Ok(statuses) => Some(
                statuses
                    .into_iter()
                    .map(|status| (status.project_id.clone(), status))
                    .collect::<HashMap<_, _>>(),
            ),
            Err(error) => {
                tracing::warn!(error = %error, "semantic_safety: failed to list remote sync states");
                None
            }
        };

        let mut local_projects = HashSet::new();
        for record in records {
            local_projects.insert(record.project_id.clone());
            let policy = match record_to_proto(&record) {
                Ok(policy) => policy,
                Err(error) => {
                    tracing::warn!(project_id = %record.project_id, error = %error, "semantic_safety: failed to parse policy for reconciliation");
                    continue;
                }
            };
            let expected_state = if record.enabled { "ready" } else { "disabled" };
            let should_push = match remote_statuses
                .as_ref()
                .and_then(|statuses| statuses.get(&record.project_id))
            {
                Some(status) => {
                    status.policy_version != record.version || status.index_state != expected_state
                }
                None => true,
            };
            if should_push {
                if let Err(error) = self.client.upsert_policy(policy).await {
                    tracing::warn!(project_id = %record.project_id, error = %error, "semantic_safety: reconcile push failed");
                }
            }
        }

        if let Some(remote_statuses) = remote_statuses {
            for (project_id, _) in remote_statuses {
                if local_projects.contains(&project_id) {
                    continue;
                }
                if let Err(error) = self.client.delete_policy(&project_id).await {
                    tracing::warn!(project_id = %project_id, error = %error, "semantic_safety: reconcile delete failed");
                }
            }
        }
    }

    pub async fn sync_policy_record(
        &self,
        record: &ProjectSemanticPolicyRecord,
    ) -> SemanticPolicyMutationResult {
        match record_to_proto(record) {
            Ok(policy) => match self.client.upsert_policy(policy).await {
                Ok(()) => SemanticPolicyMutationResult {
                    policy_version: record.version.clone(),
                    synced: true,
                    sync_error: None,
                },
                Err(error) => SemanticPolicyMutationResult {
                    policy_version: record.version.clone(),
                    synced: false,
                    sync_error: Some(error.to_string()),
                },
            },
            Err(error) => SemanticPolicyMutationResult {
                policy_version: record.version.clone(),
                synced: false,
                sync_error: Some(error.to_string()),
            },
        }
    }

    pub async fn delete_policy(
        &self,
        project_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.client.delete_policy(project_id).await
    }

    pub async fn get_sync_status(&self, project_id: &str) -> SemanticPolicySyncStatus {
        let (sync_result, health_result) = tokio::join!(
            self.client.get_sync_status(project_id),
            self.client.health(),
        );

        match (sync_result, health_result) {
            (Ok(mut status), Ok(health)) => {
                let health = health.into_inner();
                status.ready = health.ready;
                status.backend = Some(health.backend);
                status.message = Some(health.message);
                status
            }
            (Ok(mut status), Err(error)) => {
                status.ready = false;
                status.error = Some(error.to_string());
                status
            }
            (Err(sync_error), Ok(health)) => {
                let health = health.into_inner();
                SemanticPolicySyncStatus {
                    project_id: project_id.to_string(),
                    policy_version: String::new(),
                    index_state: "degraded".to_string(),
                    updated_at: String::new(),
                    stored_exemplar_count: 0,
                    available: true,
                    ready: health.ready,
                    backend: Some(health.backend),
                    message: Some(health.message),
                    error: Some(sync_error.to_string()),
                }
            }
            (Err(sync_error), Err(health_error)) => SemanticPolicySyncStatus {
                project_id: project_id.to_string(),
                policy_version: String::new(),
                index_state: "degraded".to_string(),
                updated_at: String::new(),
                stored_exemplar_count: 0,
                available: false,
                ready: false,
                backend: None,
                message: None,
                error: Some(format!(
                    "sync status failed: {}; health failed: {}",
                    sync_error, health_error
                )),
            },
        }
    }
}

#[async_trait]
impl Plugin for SemanticSafety {
    fn name(&self) -> &str {
        "semantic_safety"
    }

    async fn on_request(&self, ctx: &mut RequestContext) -> Action {
        ctx.extensions
            .insert(SemanticSafetyReplayHandle(self.clone()));

        let Some(project_id) = ctx
            .auth
            .as_ref()
            .and_then(|auth| auth.resolved_project())
            .map(|project| project.0.clone())
        else {
            if let Some(metrics) = &self.metrics {
                metrics
                    .semantic_requests_total
                    .with_label_values(&["skipped_no_project"])
                    .inc();
            }
            return Action::Continue;
        };

        let Some(record) = self.governance.semantic_policy(&project_id) else {
            if let Some(metrics) = &self.metrics {
                metrics
                    .semantic_requests_total
                    .with_label_values(&["skipped_no_policy"])
                    .inc();
            }
            return Action::Continue;
        };

        if !record.enabled {
            if let Some(metrics) = &self.metrics {
                metrics
                    .semantic_requests_total
                    .with_label_values(&["skipped_disabled"])
                    .inc();
            }
            return Action::Continue;
        }

        let Some(body) = ctx.body.as_ref() else {
            if let Some(metrics) = &self.metrics {
                metrics
                    .semantic_requests_total
                    .with_label_values(&["skipped_no_body"])
                    .inc();
            }
            return Action::Continue;
        };

        let chunks = match extract_semantic_chunks(
            body,
            self.config.max_chunks,
            self.config.max_chars_per_chunk,
        ) {
            Some(chunks) if !chunks.is_empty() => chunks,
            _ => {
                if let Some(metrics) = &self.metrics {
                    metrics
                        .semantic_requests_total
                        .with_label_values(&["skipped_no_chunks"])
                        .inc();
                }
                return Action::Continue;
            }
        };

        let model = ctx
            .body
            .as_ref()
            .and_then(|body| extract_model(body))
            .unwrap_or_default();
        let streaming = extract_stream_flag(body);
        let chunk_count = chunks.len() as u64;

        let request = EvaluateRequest {
            project_id: project_id.clone(),
            policy_version: record.version.clone(),
            request_id: generate_request_id(),
            path: ctx.uri.path().to_string(),
            model: model.clone(),
            streaming,
            chunks,
            top_k: self.config.top_k,
        };

        let semantic_span = tracing::info_span!(
            "semantic_safety_check",
            llm.project_id = %project_id,
            llm.semantic_policy_version = %record.version,
            llm.model = %model,
            llm.streaming = streaming,
            llm.semantic_chunk_count = chunk_count,
        );
        let result = tokio::time::timeout(
            self.config.timeout,
            self.client.evaluate(request).instrument(semantic_span),
        )
        .await;
        match result {
            Ok(Ok(response)) => {
                let payload = response.into_inner();
                if let Some(metrics) = &self.metrics {
                    metrics
                        .semantic_requests_total
                        .with_label_values(&["evaluated"])
                        .inc();
                    metrics
                        .semantic_findings_total
                        .inc_by(payload.findings.len() as u64);
                    metrics
                        .semantic_service_latency_ms
                        .with_label_values(&["ok"])
                        .observe(payload.service_latency_ms as f64);
                    if payload.index_state != IndexState::Ready as i32 {
                        metrics.semantic_degraded_total.inc();
                    }
                }
                ctx.extensions.insert(SemanticSafetyAudit {
                    mode: "observe_only".to_string(),
                    project_id,
                    policy_version: payload.policy_version,
                    index_state: index_state_name(payload.index_state).to_string(),
                    service_latency_ms: payload.service_latency_ms,
                    degraded_reason: if payload.degraded_reason.is_empty() {
                        None
                    } else {
                        Some(payload.degraded_reason)
                    },
                    findings: payload
                        .findings
                        .into_iter()
                        .map(|finding| SemanticFindingAudit {
                            chunk_path: finding.chunk_path,
                            entity_id: finding.entity_id,
                            entity_text: finding.entity_text,
                            topic_id: finding.topic_id,
                            matched_exemplar: finding.matched_exemplar,
                            embedding_score: finding.embedding_score,
                            rerank_score: finding.rerank_score,
                        })
                        .collect(),
                });
            }
            Ok(Err(error)) => {
                if let Some(metrics) = &self.metrics {
                    metrics
                        .semantic_requests_total
                        .with_label_values(&["error"])
                        .inc();
                    metrics.semantic_degraded_total.inc();
                }
                ctx.extensions.insert(SemanticSafetyAudit {
                    mode: "observe_only".to_string(),
                    project_id,
                    policy_version: record.version,
                    index_state: "degraded".to_string(),
                    service_latency_ms: 0,
                    degraded_reason: Some(error.to_string()),
                    findings: Vec::new(),
                });
            }
            Err(_) => {
                if let Some(metrics) = &self.metrics {
                    metrics
                        .semantic_requests_total
                        .with_label_values(&["timeout"])
                        .inc();
                    metrics.semantic_degraded_total.inc();
                }
                ctx.extensions.insert(SemanticSafetyAudit {
                    mode: "observe_only".to_string(),
                    project_id,
                    policy_version: record.version,
                    index_state: "degraded".to_string(),
                    service_latency_ms: self.config.timeout.as_millis() as u64,
                    degraded_reason: Some("semantic safety timeout".to_string()),
                    findings: Vec::new(),
                });
            }
        }

        Action::Continue
    }
}

pub fn create_plugin(
    config: &toml::Value,
    governance: Arc<GovernanceState>,
) -> Result<SemanticSafety, Box<dyn std::error::Error>> {
    let config = SemanticSafetyConfig::from_toml(config)?;
    SemanticSafety::new(config, governance)
}

pub fn create(config: &toml::Value) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>> {
    Ok(Box::new(create_plugin(
        config,
        Arc::new(GovernanceState::new(None)),
    )?))
}

pub fn record_to_proto(
    record: &ProjectSemanticPolicyRecord,
) -> Result<ProjectSemanticPolicy, Box<dyn std::error::Error + Send + Sync>> {
    let entities = serde_json::from_str::<Vec<SemanticEntity>>(
        record.entities_json.as_deref().unwrap_or("[]"),
    )?;
    let topics =
        serde_json::from_str::<Vec<SemanticTopic>>(record.topics_json.as_deref().unwrap_or("[]"))?;
    Ok(ProjectSemanticPolicy {
        project_id: record.project_id.clone(),
        version: record.version.clone(),
        enabled: record.enabled,
        entities,
        topics,
        updated_at: record.updated_at.clone(),
    })
}

pub fn proto_to_record(
    policy: &ProjectSemanticPolicy,
) -> Result<ProjectSemanticPolicyRecord, Box<dyn std::error::Error + Send + Sync>> {
    Ok(ProjectSemanticPolicyRecord {
        project_id: policy.project_id.clone(),
        version: if policy.version.trim().is_empty() {
            generate_semantic_policy_version()
        } else {
            policy.version.clone()
        },
        enabled: policy.enabled,
        entities_json: Some(serde_json::to_string(&policy.entities)?),
        topics_json: Some(serde_json::to_string(&policy.topics)?),
        updated_at: if policy.updated_at.trim().is_empty() {
            current_timestamp_string()
        } else {
            policy.updated_at.clone()
        },
    })
}

pub(crate) fn generate_semantic_policy_version() -> String {
    let now_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = SEMANTIC_POLICY_VERSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("sem-{now_nanos:032x}-{sequence:016x}")
}

fn generate_request_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let suffix = rand::thread_rng().gen::<u64>();
    format!("sem-{now}-{suffix:016x}")
}

fn extract_stream_flag(body: &[u8]) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| value.get("stream").and_then(|entry| entry.as_bool()))
        .unwrap_or(false)
}

fn extract_semantic_chunks(
    body: &[u8],
    max_chunks: usize,
    max_chars_per_chunk: usize,
) -> Option<Vec<Chunk>> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    let mut chunks = Vec::new();
    collect_chunks(
        &value,
        "$",
        None,
        &mut chunks,
        max_chunks,
        max_chars_per_chunk,
    );
    Some(chunks)
}

fn collect_chunks(
    value: &Value,
    path: &str,
    field_name: Option<&str>,
    chunks: &mut Vec<Chunk>,
    max_chunks: usize,
    max_chars_per_chunk: usize,
) {
    if chunks.len() >= max_chunks {
        return;
    }
    match value {
        Value::String(text) => {
            if should_capture_field(field_name) {
                let normalized = text.trim();
                if !normalized.is_empty() {
                    let limited = normalized
                        .chars()
                        .take(max_chars_per_chunk)
                        .collect::<String>();
                    chunks.push(Chunk {
                        path: path.to_string(),
                        text: limited,
                    });
                }
            }
        }
        Value::Array(values) => {
            for (idx, entry) in values.iter().enumerate() {
                collect_chunks(
                    entry,
                    &format!("{path}[{idx}]"),
                    field_name,
                    chunks,
                    max_chunks,
                    max_chars_per_chunk,
                );
                if chunks.len() >= max_chunks {
                    break;
                }
            }
        }
        Value::Object(map) => {
            for (key, entry) in map {
                let next_path = format!("{path}.{key}");
                if should_capture_field(Some(key)) {
                    match entry {
                        Value::String(_) => collect_chunks(
                            entry,
                            &next_path,
                            Some(key),
                            chunks,
                            max_chunks,
                            max_chars_per_chunk,
                        ),
                        Value::Array(_) | Value::Object(_) => {
                            let text = serde_json::to_string(entry).unwrap_or_default();
                            if !text.is_empty() && chunks.len() < max_chunks {
                                let limited =
                                    text.chars().take(max_chars_per_chunk).collect::<String>();
                                chunks.push(Chunk {
                                    path: next_path.clone(),
                                    text: limited,
                                });
                            }
                            collect_chunks(
                                entry,
                                &next_path,
                                Some(key),
                                chunks,
                                max_chunks,
                                max_chars_per_chunk,
                            );
                        }
                        _ => {}
                    }
                } else {
                    collect_chunks(
                        entry,
                        &next_path,
                        Some(key),
                        chunks,
                        max_chunks,
                        max_chars_per_chunk,
                    );
                }
                if chunks.len() >= max_chunks {
                    break;
                }
            }
        }
        _ => {}
    }
}

fn should_capture_field(field_name: Option<&str>) -> bool {
    matches!(
        field_name.unwrap_or_default(),
        "content" | "text" | "prompt" | "input" | "instructions" | "arguments" | "query"
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use hyper::header::HeaderMap;
    use hyper::http::Extensions;
    use hyper::{Method, Uri, Version};
    use semantic_safety_protocol::{SemanticSafetyService, SemanticSafetyServiceServer};
    use semantic_safety_service::backend::TensorRtBackend;
    use semantic_safety_service::persistence::FileProjectIndexStore;
    use semantic_safety_service::service::{
        SemanticSafetyConfig as ServiceConfig, SemanticSafetyGrpcService,
    };
    use tempfile::tempdir;
    use tokio::net::TcpListener;
    use tonic::transport::Server;

    use super::*;

    fn make_ctx(body: Option<&[u8]>) -> RequestContext {
        RequestContext {
            peer_addr: None,
            method: Method::POST,
            uri: Uri::from_static("http://localhost/v1/chat/completions"),
            version: Version::HTTP_11,
            headers: HeaderMap::new(),
            body: body.map(Bytes::copy_from_slice),
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        }
    }

    async fn start_service() -> (String, ProjectSemanticPolicyRecord) {
        let dir = tempdir().unwrap();
        let service = SemanticSafetyGrpcService::new(
            ServiceConfig { auth_token: None },
            Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap()),
            Arc::new(TensorRtBackend::new_dev_stub()),
        );
        let policy = ProjectSemanticPolicyRecord {
            project_id: "project-a".to_string(),
            version: "v1".to_string(),
            enabled: true,
            entities_json: Some(
                serde_json::to_string(&vec![SemanticEntity {
                    entity_id: "company-x".to_string(),
                    name: "Company X".to_string(),
                    aliases: vec!["companyx".to_string()],
                }])
                .unwrap(),
            ),
            topics_json: Some(
                serde_json::to_string(&vec![SemanticTopic {
                    topic_id: "layoffs".to_string(),
                    name: "Layoffs".to_string(),
                    exemplars: vec!["company x layoffs next week".to_string()],
                    rerank_threshold: 0.1,
                    require_entity_match: true,
                }])
                .unwrap(),
            ),
            updated_at: "1".to_string(),
        };
        service
            .upsert_project_policy(GrpcRequest::new(UpsertProjectPolicyRequest {
                policy: Some(record_to_proto(&policy).unwrap()),
            }))
            .await
            .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            Server::builder()
                .add_service(SemanticSafetyServiceServer::new(service))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        (format!("http://{}", addr), policy)
    }

    #[test]
    fn extracts_supported_chunks() {
        let body = br#"{"messages":[{"role":"user","content":"Company X layoffs next week"}],"input":"hello","tools":[{"function":{"arguments":"{\"secret\":\"value\"}"}}]}"#;
        let chunks = extract_semantic_chunks(body, 16, 128).unwrap();
        assert!(chunks
            .iter()
            .any(|chunk| chunk.path == "$.messages[0].content"));
        assert!(chunks.iter().any(|chunk| chunk.path == "$.input"));
        assert!(chunks
            .iter()
            .any(|chunk| chunk.path.ends_with(".arguments")));
    }

    #[test]
    fn generated_policy_versions_are_unique() {
        let first = generate_semantic_policy_version();
        let second = generate_semantic_policy_version();
        assert_ne!(first, second);
        assert!(first.starts_with("sem-"));
        assert!(second.starts_with("sem-"));
    }

    #[tokio::test]
    async fn skips_without_project_auth() {
        let plugin = SemanticSafety::new(
            SemanticSafetyConfig {
                endpoint: "http://127.0.0.1:50061".to_string(),
                auth_token: None,
                timeout: Duration::from_millis(50),
                max_chunks: DEFAULT_MAX_CHUNKS,
                max_chars_per_chunk: DEFAULT_MAX_CHARS_PER_CHUNK,
                top_k: DEFAULT_TOP_K,
                reconcile_interval: Duration::from_secs(DEFAULT_RECONCILE_INTERVAL_SECS),
            },
            Arc::new(GovernanceState::new(None)),
        )
        .unwrap();
        let mut ctx = make_ctx(Some(br#"{"input":"hello"}"#));
        matches!(plugin.on_request(&mut ctx).await, Action::Continue);
        assert!(ctx.extensions.get::<SemanticSafetyAudit>().is_none());
    }

    #[tokio::test]
    async fn timeout_is_fail_open_and_records_audit() {
        let governance = Arc::new(GovernanceState::new(None));
        governance
            .upsert_semantic_policy(ProjectSemanticPolicyRecord {
                project_id: "project-a".to_string(),
                version: "v1".to_string(),
                enabled: true,
                entities_json: Some("[]".to_string()),
                topics_json: Some("[]".to_string()),
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();
        let plugin = SemanticSafety::new(
            SemanticSafetyConfig {
                endpoint: "http://127.0.0.1:9".to_string(),
                auth_token: None,
                timeout: Duration::from_millis(10),
                max_chunks: DEFAULT_MAX_CHUNKS,
                max_chars_per_chunk: DEFAULT_MAX_CHARS_PER_CHUNK,
                top_k: DEFAULT_TOP_K,
                reconcile_interval: Duration::from_secs(DEFAULT_RECONCILE_INTERVAL_SECS),
            },
            governance,
        )
        .unwrap();
        let mut ctx = make_ctx(Some(br#"{"input":"hello"}"#));
        ctx.auth = Some(proxy_auth::AuthContext::runtime("project-a", "runtime"));
        let result = plugin.on_request(&mut ctx).await;
        assert!(matches!(result, Action::Continue));
        let audit = ctx.extensions.get::<SemanticSafetyAudit>().unwrap();
        assert_eq!(audit.index_state, "degraded");
    }

    #[tokio::test]
    async fn attaches_semantic_audit_from_service() {
        let (endpoint, policy) = start_service().await;
        let governance = Arc::new(GovernanceState::new(None));
        governance.upsert_semantic_policy(policy).await.unwrap();
        let plugin = SemanticSafety::new(
            SemanticSafetyConfig {
                endpoint,
                auth_token: None,
                timeout: Duration::from_millis(200),
                max_chunks: DEFAULT_MAX_CHUNKS,
                max_chars_per_chunk: DEFAULT_MAX_CHARS_PER_CHUNK,
                top_k: DEFAULT_TOP_K,
                reconcile_interval: Duration::from_secs(DEFAULT_RECONCILE_INTERVAL_SECS),
            },
            governance,
        )
        .unwrap();
        let mut ctx = make_ctx(Some(br#"{"messages":[{"role":"user","content":"Something is happening at Company X, layoffs next week"}]}"#));
        ctx.auth = Some(proxy_auth::AuthContext::runtime("project-a", "runtime"));
        let result = plugin.on_request(&mut ctx).await;
        assert!(matches!(result, Action::Continue));
        let audit = ctx.extensions.get::<SemanticSafetyAudit>().unwrap();
        assert_eq!(audit.index_state, "degraded");
        assert_eq!(audit.findings.len(), 1);
        assert!(audit
            .degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("dev stub"));
    }

    #[tokio::test]
    async fn reconciliation_prunes_remote_projects_missing_locally() {
        let (endpoint, _) = start_service().await;
        let plugin = SemanticSafety::new(
            SemanticSafetyConfig {
                endpoint,
                auth_token: None,
                timeout: Duration::from_millis(200),
                max_chunks: DEFAULT_MAX_CHUNKS,
                max_chars_per_chunk: DEFAULT_MAX_CHARS_PER_CHUNK,
                top_k: DEFAULT_TOP_K,
                reconcile_interval: Duration::from_secs(DEFAULT_RECONCILE_INTERVAL_SECS),
            },
            Arc::new(GovernanceState::new(None)),
        )
        .unwrap();

        let initial = plugin.get_sync_status("project-a").await;
        assert_eq!(initial.index_state, "ready");
        assert!(!initial.ready);
        assert_eq!(initial.backend.as_deref(), Some("tensorrt-dev-stub"));
        plugin.reconcile_once().await;
        let reconciled = plugin.get_sync_status("project-a").await;
        assert_eq!(reconciled.index_state, "missing");
        assert!(!reconciled.ready);
        assert_eq!(reconciled.backend.as_deref(), Some("tensorrt-dev-stub"));
    }
}
