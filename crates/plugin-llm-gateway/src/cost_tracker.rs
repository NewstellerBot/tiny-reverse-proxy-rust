use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use http_body_util::{BodyExt, Full};
use hyper::header::{HeaderValue, CONTENT_LENGTH, CONTENT_TYPE};
use hyper::{Response, StatusCode};

use http_body_util::combinators::BoxBody;
use proxy_core::plugin::ProviderCandidates;
use proxy_core::plugin::{Action, Plugin, RequestContext, ResponseContext};

use crate::content_filter::SafetyAudit;
use crate::metrics::{mask_key, LlmMetrics};
use crate::prompt_registry::PromptResolutionAudit;
use crate::semantic_safety::SemanticSafetyAudit;
use crate::store::{
    GatewayStore, KeyModelUsageRecord, KeyUsageRecord, ModelCostRecord, RequestLogEntry,
    SessionEventRecord, SessionRecord, Store, StoreError,
};
use crate::tool_runtime::{ToolRuntimeAudit, ToolUsageOverride};
use crate::virtual_keys::VirtualKeyMeta;
use crate::{
    estimate_prompt_tokens, extract_api_key, extract_model, take_request_custom_cost,
    take_request_metadata, RequestCustomCost, RequestMetadata,
};

use tokio::sync::mpsc;

const OPENROUTER_API: &str = "https://openrouter.ai/api/v1/models";

/// Per-1k-token costs (input, output).
#[derive(Debug, Clone, Copy)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
}

/// Metadata stashed in `ctx.extensions` by `on_request` for use in `on_response`.
#[derive(Clone)]
struct LlmRequestMeta {
    api_key: String,
    model: Option<String>,
    estimated_input_tokens: u64,
}

#[derive(Clone)]
struct SessionTraceMeta {
    session_id: String,
}

#[derive(Clone, Default)]
struct RequestAuditFields {
    project_id: Option<String>,
    session_id: Option<String>,
    metadata_json: Option<String>,
    custom_cost_json: Option<String>,
    provider_name: Option<String>,
    prompt_name: Option<String>,
    prompt_version: Option<String>,
    prompt_environment: Option<String>,
    safety_mode: Option<String>,
    safety_matches: Option<String>,
    semantic_policy_version: Option<String>,
    semantic_index_state: Option<String>,
    semantic_degraded_reason: Option<String>,
    semantic_findings: Option<String>,
    tool_trace: Option<String>,
}

#[derive(Clone, Debug)]
struct ObservedResponseUsage {
    input_tokens: u64,
    output_tokens: u64,
}

const SESSION_ID_HEADER: &str = "x-trp-session-id";

#[derive(Debug, Clone)]
pub struct KeyUsage {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost: f64,
}

#[derive(Clone)]
pub struct CostTracker {
    usage: Arc<DashMap<String, KeyUsage>>,
    budget_limit: f64,
    default_cost_per_1k_input: f64,
    default_cost_per_1k_output: f64,
    model_costs: Arc<DashMap<String, ModelCost>>,
    log_interval_secs: u64,
    store: Option<Arc<Store>>,
    metrics: Option<LlmMetrics>,
    webhook_url: Option<String>,
    budget_alert_thresholds: Vec<u64>,
    budget_alert_ttl: u64,
    /// (key_hash, threshold_pct) → last fired instant. Used to deduplicate alerts.
    webhook_fired: Arc<DashMap<(String, u64), std::time::Instant>>,
    /// Latest known budget window start per virtual key.
    /// This prevents repeated resets if upstream metadata is stale.
    budget_window_starts: Arc<DashMap<String, i64>>,
    /// Per-key per-model usage breakdown.
    model_usage: Arc<DashMap<(String, String), KeyUsage>>,
    /// Channel for off-path audit log entries.
    audit_tx: Option<Arc<mpsc::UnboundedSender<RequestLogEntry>>>,
    /// Receiver held until spawn_audit_drain_task takes it.
    audit_rx: Arc<std::sync::Mutex<Option<mpsc::UnboundedReceiver<RequestLogEntry>>>>,
}

impl CostTracker {
    pub fn new(
        budget_limit: f64,
        default_cost_per_1k_input: f64,
        default_cost_per_1k_output: f64,
        log_interval_secs: u64,
    ) -> Self {
        let (audit_tx, audit_rx) = mpsc::unbounded_channel();
        Self {
            usage: Arc::new(DashMap::new()),
            budget_limit,
            default_cost_per_1k_input,
            default_cost_per_1k_output,
            model_costs: Arc::new(DashMap::new()),
            log_interval_secs,
            store: None,
            metrics: None,
            webhook_url: None,
            budget_alert_thresholds: vec![50, 80, 100],
            budget_alert_ttl: 86400,
            webhook_fired: Arc::new(DashMap::new()),
            budget_window_starts: Arc::new(DashMap::new()),
            model_usage: Arc::new(DashMap::new()),
            audit_tx: Some(Arc::new(audit_tx)),
            audit_rx: Arc::new(std::sync::Mutex::new(Some(audit_rx))),
        }
    }

    /// Attach a persistent store. Call `load_from_store()` after this.
    pub fn with_store(mut self, store: Arc<Store>) -> Self {
        self.store = Some(store);
        self
    }

    /// Attach LLM Prometheus metrics.
    pub fn with_metrics(mut self, metrics: LlmMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Configure webhook budget alerts.
    pub fn with_webhook(mut self, url: String, thresholds: Vec<u64>, ttl: u64) -> Self {
        self.webhook_url = Some(url);
        self.budget_alert_thresholds = thresholds;
        self.budget_alert_ttl = ttl;
        self
    }

    /// Check budget thresholds and fire webhook alerts as needed.
    fn check_and_fire_webhook(&self, tracking_key: &str, total_cost: f64, effective_budget: f64) {
        let webhook_url = match &self.webhook_url {
            Some(url) => url.clone(),
            None => return,
        };
        if effective_budget <= 0.0 {
            return;
        }

        let pct = (total_cost / effective_budget) * 100.0;
        let ttl = std::time::Duration::from_secs(self.budget_alert_ttl);

        for &threshold in &self.budget_alert_thresholds {
            if pct < threshold as f64 {
                continue;
            }
            let dedup_key = (tracking_key.to_string(), threshold);
            // Check dedup window
            if let Some(last) = self.webhook_fired.get(&dedup_key) {
                if last.elapsed() < ttl {
                    continue;
                }
            }
            self.webhook_fired
                .insert(dedup_key, std::time::Instant::now());

            let token = if tracking_key.len() > 12 {
                &tracking_key[..12]
            } else {
                tracking_key
            };
            let body = format!(
                r#"{{"event":"threshold_crossed","threshold_percent":{},"spend":{:.2},"max_budget":{:.2},"token":"{}","event_message":"Budget alert: {}% of ${:.2} budget used (${:.2} spent)"}}"#,
                threshold,
                total_cost,
                effective_budget,
                token,
                threshold,
                effective_budget,
                total_cost
            );

            let url = webhook_url.clone();
            tokio::spawn(async move {
                let client = proxy_core::handlers::proxy::build_client();
                let req = match hyper::Request::builder()
                    .method("POST")
                    .uri(&url)
                    .header("content-type", "application/json")
                    .body(
                        Full::new(Bytes::from(body))
                            .map_err(|never| match never {})
                            .boxed(),
                    ) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("cost_tracker: failed to build webhook request: {e}");
                        return;
                    }
                };
                match tokio::time::timeout(std::time::Duration::from_secs(5), client.request(req))
                    .await
                {
                    Ok(Ok(_)) => tracing::debug!("cost_tracker: webhook alert sent to {url}"),
                    Ok(Err(e)) => tracing::warn!("cost_tracker: webhook request failed: {e}"),
                    Err(_) => tracing::warn!("cost_tracker: webhook request timed out"),
                }
            });
        }
    }

    /// Load existing usage and model pricing from the store into DashMaps.
    pub async fn load_from_store(&self) -> Result<(), Box<dyn std::error::Error>> {
        let store = match &self.store {
            Some(s) => s,
            None => return Ok(()),
        };

        let usage_rows = store.get_all_usage().await?;
        for (key, record) in usage_rows {
            self.usage.insert(
                key,
                KeyUsage {
                    total_input_tokens: record.total_input_tokens,
                    total_output_tokens: record.total_output_tokens,
                    total_cost: record.total_cost,
                },
            );
        }

        let cost_rows = store.get_all_model_costs().await?;
        for (model, record) in cost_rows {
            self.model_costs.insert(
                model,
                ModelCost {
                    input: record.input_cost_per_1k,
                    output: record.output_cost_per_1k,
                },
            );
        }

        // Load per-model usage.
        let model_usage_rows = store.get_all_per_model_usage().await?;
        for record in model_usage_rows {
            self.model_usage.insert(
                (record.api_key.clone(), record.model.clone()),
                KeyUsage {
                    total_input_tokens: record.total_input_tokens,
                    total_output_tokens: record.total_output_tokens,
                    total_cost: record.total_cost,
                },
            );
        }

        Ok(())
    }

    /// Record per-model usage in the DashMap.
    fn record_model_usage(
        &self,
        api_key: &str,
        model: Option<&str>,
        input_tokens: u64,
        output_tokens: u64,
        cost: f64,
    ) {
        let model_name = model.unwrap_or("unknown").to_string();
        let mut entry = self
            .model_usage
            .entry((api_key.to_string(), model_name))
            .or_insert(KeyUsage {
                total_input_tokens: 0,
                total_output_tokens: 0,
                total_cost: 0.0,
            });
        entry.total_input_tokens += input_tokens;
        entry.total_output_tokens += output_tokens;
        entry.total_cost += cost;
    }

    /// Send an audit log entry to the drain task.
    fn send_audit_log(
        &self,
        api_key: &str,
        model: Option<&str>,
        input_tokens: u64,
        output_tokens: u64,
        cost: f64,
        is_streaming: bool,
        custom_cost_applied: bool,
        audit: RequestAuditFields,
    ) {
        if let Some(ref tx) = self.audit_tx {
            let entry = RequestLogEntry {
                timestamp_unix: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64,
                api_key: api_key.to_string(),
                project_id: audit.project_id,
                session_id: audit.session_id,
                metadata_json: audit.metadata_json,
                custom_cost_json: audit.custom_cost_json,
                custom_cost_applied,
                provider_name: audit.provider_name,
                prompt_name: audit.prompt_name,
                prompt_version: audit.prompt_version,
                prompt_environment: audit.prompt_environment,
                model: model.map(|m| m.to_string()),
                input_tokens,
                output_tokens,
                cost,
                is_streaming,
                safety_mode: audit.safety_mode,
                safety_matches: audit.safety_matches,
                semantic_policy_version: audit.semantic_policy_version,
                semantic_index_state: audit.semantic_index_state,
                semantic_degraded_reason: audit.semantic_degraded_reason,
                semantic_findings: audit.semantic_findings,
                tool_trace: audit.tool_trace,
            };
            let _ = tx.send(entry);
        }
    }

    /// Spawn a background task that periodically flushes DashMap state to the store.
    pub fn spawn_flush_task(&self, interval_secs: u64) -> Option<tokio::task::JoinHandle<()>> {
        let store = self.store.clone()?;
        let usage = Arc::clone(&self.usage);
        let model_costs = Arc::clone(&self.model_costs);
        let model_usage = Arc::clone(&self.model_usage);

        Some(tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
                // Flush usage.
                for entry in usage.iter() {
                    let record = KeyUsageRecord {
                        total_input_tokens: entry.value().total_input_tokens,
                        total_output_tokens: entry.value().total_output_tokens,
                        total_cost: entry.value().total_cost,
                    };
                    if let Err(e) = store.upsert_usage(entry.key(), &record).await {
                        tracing::warn!("cost_tracker flush usage error: {}", e);
                    }
                }
                // Flush model costs.
                for entry in model_costs.iter() {
                    let record = ModelCostRecord {
                        input_cost_per_1k: entry.value().input,
                        output_cost_per_1k: entry.value().output,
                    };
                    if let Err(e) = store.upsert_model_cost(entry.key(), &record).await {
                        tracing::warn!("cost_tracker flush model cost error: {}", e);
                    }
                }
                // Flush per-model usage.
                for entry in model_usage.iter() {
                    let (ref api_key, ref model) = *entry.key();
                    let record = KeyModelUsageRecord {
                        api_key: api_key.clone(),
                        model: model.clone(),
                        total_input_tokens: entry.value().total_input_tokens,
                        total_output_tokens: entry.value().total_output_tokens,
                        total_cost: entry.value().total_cost,
                    };
                    if let Err(e) = store.upsert_per_model_usage(&record).await {
                        tracing::warn!("cost_tracker flush per-model usage error: {}", e);
                    }
                }
                tracing::debug!("cost_tracker: flushed state to store");
            }
        }))
    }

    /// Spawn the audit log drain task that batch-inserts entries to the store.
    pub fn spawn_audit_drain_task(&self) -> Option<tokio::task::JoinHandle<()>> {
        let store = self.store.clone()?;
        let rx = self.audit_rx.lock().unwrap().take()?;

        Some(tokio::spawn(async move {
            let mut rx = rx;
            let mut buf = Vec::with_capacity(64);
            loop {
                let count = rx.recv_many(&mut buf, 64).await;
                if count == 0 {
                    break; // Channel closed
                }
                if let Err(e) = store.append_request_logs(&buf).await {
                    tracing::warn!("cost_tracker audit drain error: {}", e);
                    buf.clear();
                    continue;
                }
                if let Err(e) = upsert_session_rollups(store.as_ref(), &buf).await {
                    tracing::warn!("cost_tracker session rollup error: {}", e);
                }
                buf.clear();
            }
        }))
    }

    /// Perform a one-time flush of all state to the store.
    pub async fn flush_to_store(&self) -> Result<(), Box<dyn std::error::Error>> {
        let store = match &self.store {
            Some(s) => s,
            None => return Ok(()),
        };

        for entry in self.usage.iter() {
            let record = KeyUsageRecord {
                total_input_tokens: entry.value().total_input_tokens,
                total_output_tokens: entry.value().total_output_tokens,
                total_cost: entry.value().total_cost,
            };
            store.upsert_usage(entry.key(), &record).await?;
        }
        for entry in self.model_costs.iter() {
            let record = ModelCostRecord {
                input_cost_per_1k: entry.value().input,
                output_cost_per_1k: entry.value().output,
            };
            store.upsert_model_cost(entry.key(), &record).await?;
        }
        for entry in self.model_usage.iter() {
            let (ref api_key, ref model) = *entry.key();
            let record = KeyModelUsageRecord {
                api_key: api_key.clone(),
                model: model.clone(),
                total_input_tokens: entry.value().total_input_tokens,
                total_output_tokens: entry.value().total_output_tokens,
                total_cost: entry.value().total_cost,
            };
            store.upsert_per_model_usage(&record).await?;
        }
        Ok(())
    }

    /// Look up per-1k-token costs for a model, falling back to defaults.
    fn costs_for_model(&self, model: Option<&str>) -> (f64, f64) {
        if let Some(name) = model {
            if let Some(mc) = self.model_costs.get(name) {
                return (mc.input, mc.output);
            }
        }
        (
            self.default_cost_per_1k_input,
            self.default_cost_per_1k_output,
        )
    }

    pub fn spawn_logger_task(&self) -> tokio::task::JoinHandle<()> {
        let usage = Arc::clone(&self.usage);
        let interval_secs = self.log_interval_secs;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
                for entry in usage.iter() {
                    let key = entry.key();
                    let u = entry.value();
                    // Mask API key for logging (show first 8 chars + "...").
                    let masked = if key.len() > 8 {
                        format!("{}...", &key[..8])
                    } else {
                        key.clone()
                    };
                    tracing::info!(
                        api_key = %masked,
                        input_tokens = u.total_input_tokens,
                        output_tokens = u.total_output_tokens,
                        cost = format!("{:.6}", u.total_cost),
                        "cost_tracker usage"
                    );
                }
            }
        })
    }

    // --- Accessor methods for the API layer ---

    pub fn get_all_usage(&self) -> Vec<(String, KeyUsage)> {
        self.usage
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    pub fn get_usage(&self, api_key: &str) -> Option<KeyUsage> {
        self.usage.get(api_key).map(|entry| entry.value().clone())
    }

    pub fn budget_limit(&self) -> f64 {
        self.budget_limit
    }

    pub fn get_model_costs(&self) -> Vec<(String, ModelCost)> {
        self.model_costs
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    pub fn reset_usage(&self, api_key: &str) -> bool {
        self.usage.remove(api_key).is_some()
    }

    pub fn reset_all_usage(&self) {
        self.usage.clear();
    }

    pub fn set_model_cost(&self, model: &str, input: f64, output: f64) {
        self.model_costs
            .insert(model.to_string(), ModelCost { input, output });
    }

    pub fn delete_model_cost(&self, model: &str) -> bool {
        self.model_costs.remove(model).is_some()
    }

    pub fn store(&self) -> Option<&Arc<Store>> {
        self.store.as_ref()
    }

    pub fn get_all_model_usage(&self) -> Vec<((String, String), KeyUsage)> {
        self.model_usage
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
}

fn request_audit_fields(ctx: &RequestContext) -> RequestAuditFields {
    let project_id = ctx
        .auth
        .as_ref()
        .and_then(|auth| auth.resolved_project())
        .map(|project| project.0.clone())
        .or_else(|| {
            ctx.extensions
                .get::<VirtualKeyMeta>()
                .map(|meta| meta.project_id.clone())
        });
    let session_id = ctx
        .extensions
        .get::<SessionTraceMeta>()
        .map(|meta| meta.session_id.clone());
    let metadata_json = ctx
        .extensions
        .get::<RequestMetadata>()
        .and_then(|metadata| serde_json::to_string(&metadata.value).ok());
    let custom_cost_json = ctx
        .extensions
        .get::<RequestCustomCost>()
        .and_then(|custom_cost| serde_json::to_string(&custom_cost.to_json_value()).ok());
    let provider_name = ctx
        .extensions
        .get::<VirtualKeyMeta>()
        .map(|meta| meta.provider_name.clone());
    let prompt = ctx.extensions.get::<PromptResolutionAudit>();
    let safety = ctx.extensions.get::<SafetyAudit>();
    let safety_mode = safety.as_ref().map(|audit| audit.mode.clone());
    let safety_matches = safety.as_ref().map(|audit| {
        let values = audit
            .matches
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "detector_class": entry.detector_class,
                    "description": entry.description,
                    "path": entry.path,
                    "action": entry.action,
                })
            })
            .collect::<Vec<_>>();
        serde_json::Value::Array(values).to_string()
    });
    let semantic = ctx.extensions.get::<SemanticSafetyAudit>();
    let tool_trace = ctx
        .extensions
        .get::<ToolRuntimeAudit>()
        .and_then(|audit| serde_json::to_string(audit).ok());
    RequestAuditFields {
        project_id,
        session_id,
        metadata_json,
        custom_cost_json,
        provider_name,
        prompt_name: prompt.as_ref().map(|audit| audit.prompt_name.clone()),
        prompt_version: prompt.as_ref().map(|audit| audit.prompt_version.clone()),
        prompt_environment: prompt
            .as_ref()
            .map(|audit| audit.prompt_environment.clone()),
        safety_mode,
        safety_matches,
        semantic_policy_version: semantic.as_ref().map(|audit| audit.policy_version.clone()),
        semantic_index_state: semantic.as_ref().map(|audit| audit.index_state.clone()),
        semantic_degraded_reason: semantic
            .as_ref()
            .and_then(|audit| audit.degraded_reason.clone()),
        semantic_findings: semantic
            .as_ref()
            .and_then(|audit| serde_json::to_string(&audit.findings).ok()),
        tool_trace,
    }
}

async fn upsert_session_rollups(
    store: &Store,
    entries: &[RequestLogEntry],
) -> Result<(), StoreError> {
    let mut grouped: HashMap<String, Vec<&RequestLogEntry>> = HashMap::new();
    for entry in entries {
        let Some(session_id) = entry.session_id.as_ref() else {
            continue;
        };
        grouped.entry(session_id.clone()).or_default().push(entry);
    }

    for (session_id, session_entries) in grouped {
        let mut record = store
            .get_session(&session_id)
            .await?
            .unwrap_or_else(|| empty_session_record(&session_id));
        for entry in &session_entries {
            merge_session_entry(&mut record, entry);
        }
        store.upsert_session(&record).await?;
        for entry in &session_entries {
            store
                .append_session_event(&build_request_session_event(&session_id, entry))
                .await?;
        }
    }

    Ok(())
}

fn empty_session_record(session_id: &str) -> SessionRecord {
    SessionRecord {
        session_id: session_id.to_string(),
        project_id: None,
        project_ids_json: None,
        first_request_unix: None,
        last_request_unix: None,
        updated_at_unix: 0,
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

fn merge_session_entry(record: &mut SessionRecord, entry: &RequestLogEntry) {
    let mut project_ids = parse_json_string_set(record.project_ids_json.as_deref());
    if let Some(project_id) = entry.project_id.as_ref() {
        project_ids.insert(project_id.clone());
    }
    record.project_id = if project_ids.len() == 1 {
        project_ids.iter().next().cloned()
    } else {
        None
    };
    record.project_ids_json = serialize_json_string_set(project_ids);

    let mut providers = parse_json_string_set(record.providers_json.as_deref());
    if let Some(provider_name) = entry.provider_name.as_ref() {
        providers.insert(provider_name.clone());
    }
    record.providers_json = serialize_json_string_set(providers);

    let mut models = parse_json_string_set(record.models_json.as_deref());
    if let Some(model) = entry.model.as_ref() {
        models.insert(model.clone());
    }
    record.models_json = serialize_json_string_set(models);

    let mut prompt_names = parse_json_string_set(record.prompt_names_json.as_deref());
    if let Some(prompt_name) = entry.prompt_name.as_ref() {
        prompt_names.insert(prompt_name.clone());
    }
    record.prompt_names_json = serialize_json_string_set(prompt_names);

    let mut prompt_versions = parse_json_string_set(record.prompt_versions_json.as_deref());
    if let (Some(prompt_name), Some(prompt_version)) =
        (entry.prompt_name.as_ref(), entry.prompt_version.as_ref())
    {
        prompt_versions.insert(format!("{prompt_name}@{prompt_version}"));
    }
    record.prompt_versions_json = serialize_json_string_set(prompt_versions);

    let mut tool_names = parse_json_string_set(record.tool_names_json.as_deref());
    if let Some(audit) = parse_tool_runtime_audit(entry.tool_trace.as_deref()) {
        record.tool_call_count = record
            .tool_call_count
            .saturating_add(audit.calls.len() as u64);
        record.tool_error_count = record.tool_error_count.saturating_add(
            audit
                .calls
                .iter()
                .filter(|call| call.status == "error")
                .count() as u64,
        );
        for call in audit.calls {
            tool_names.insert(call.tool_name);
        }
    }
    record.tool_names_json = serialize_json_string_set(tool_names);

    record.first_request_unix = Some(
        record
            .first_request_unix
            .map(|current| current.min(entry.timestamp_unix))
            .unwrap_or(entry.timestamp_unix),
    );

    if record
        .last_request_unix
        .map(|current| entry.timestamp_unix >= current)
        .unwrap_or(true)
    {
        record.last_request_unix = Some(entry.timestamp_unix);
        record.latest_request_json = Some(build_latest_request_json(entry).to_string());
    }

    record.updated_at_unix = record.updated_at_unix.max(entry.timestamp_unix);
    record.request_count = record.request_count.saturating_add(1);
    if entry.is_streaming {
        record.streaming_request_count = record.streaming_request_count.saturating_add(1);
    }
    record.total_input_tokens = record.total_input_tokens.saturating_add(entry.input_tokens);
    record.total_output_tokens = record
        .total_output_tokens
        .saturating_add(entry.output_tokens);
    record.total_cost += entry.cost;
    if request_has_safety_event(entry) {
        record.safety_event_count = record.safety_event_count.saturating_add(1);
    }
    if request_has_semantic_event(entry) {
        record.semantic_event_count = record.semantic_event_count.saturating_add(1);
    }
    if request_has_semantic_degraded_event(entry) {
        record.semantic_degraded_count = record.semantic_degraded_count.saturating_add(1);
    }
}

fn build_request_session_event(session_id: &str, entry: &RequestLogEntry) -> SessionEventRecord {
    let tool_summary = parse_tool_runtime_audit(entry.tool_trace.as_deref()).map(|audit| {
        serde_json::json!({
            "tool_names": audit
                .calls
                .iter()
                .map(|call| call.tool_name.clone())
                .collect::<Vec<_>>(),
            "tool_call_count": audit.calls.len(),
            "tool_error_count": audit.calls.iter().filter(|call| call.status == "error").count(),
        })
    });
    SessionEventRecord {
        event_seq: 0,
        session_id: session_id.to_string(),
        project_id: entry.project_id.clone(),
        event_kind: "request_observed".to_string(),
        actor_id: None,
        reason: None,
        payload_json: Some(
            serde_json::json!({
                "timestamp_unix": entry.timestamp_unix,
                "metadata": entry
                    .metadata_json
                    .as_deref()
                    .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
                    .unwrap_or(serde_json::Value::Null),
                "custom_cost": entry
                    .custom_cost_json
                    .as_deref()
                    .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
                    .unwrap_or(serde_json::Value::Null),
                "custom_cost_applied": entry.custom_cost_applied,
                "provider_name": entry.provider_name.clone(),
                "model": entry.model.clone(),
                "prompt_name": entry.prompt_name.clone(),
                "prompt_version": entry.prompt_version.clone(),
                "prompt_environment": entry.prompt_environment.clone(),
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "cost": entry.cost,
                "is_streaming": entry.is_streaming,
                "safety_mode": entry.safety_mode.clone(),
                "semantic_index_state": entry.semantic_index_state.clone(),
                "semantic_degraded_reason": entry.semantic_degraded_reason.clone(),
                "tool_summary": tool_summary,
            })
            .to_string(),
        ),
        created_at_unix: entry.timestamp_unix,
    }
}

fn parse_json_string_set(raw: Option<&str>) -> BTreeSet<String> {
    raw.and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
        .unwrap_or_default()
        .into_iter()
        .collect()
}

fn serialize_json_string_set(values: BTreeSet<String>) -> Option<String> {
    if values.is_empty() {
        return None;
    }
    serde_json::to_string(&values.into_iter().collect::<Vec<_>>()).ok()
}

fn build_latest_request_json(entry: &RequestLogEntry) -> serde_json::Value {
    serde_json::json!({
        "timestamp_unix": entry.timestamp_unix,
        "metadata": entry
            .metadata_json
            .as_deref()
            .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
            .unwrap_or(serde_json::Value::Null),
        "custom_cost": entry
            .custom_cost_json
            .as_deref()
            .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
            .unwrap_or(serde_json::Value::Null),
        "custom_cost_applied": entry.custom_cost_applied,
        "provider_name": entry.provider_name.clone(),
        "model": entry.model.clone(),
        "prompt_name": entry.prompt_name.clone(),
        "prompt_version": entry.prompt_version.clone(),
        "prompt_environment": entry.prompt_environment.clone(),
        "safety_mode": entry.safety_mode.clone(),
        "semantic_index_state": entry.semantic_index_state.clone(),
        "semantic_degraded_reason": entry.semantic_degraded_reason.clone(),
    })
}

fn request_has_safety_event(entry: &RequestLogEntry) -> bool {
    entry.safety_mode.is_some()
        || entry
            .safety_matches
            .as_deref()
            .map(|value| value != "[]" && !value.is_empty())
            .unwrap_or(false)
}

fn request_has_semantic_event(entry: &RequestLogEntry) -> bool {
    entry.semantic_policy_version.is_some()
        || entry.semantic_index_state.is_some()
        || entry.semantic_degraded_reason.is_some()
        || entry
            .semantic_findings
            .as_deref()
            .map(|value| value != "[]" && !value.is_empty())
            .unwrap_or(false)
}

fn request_has_semantic_degraded_event(entry: &RequestLogEntry) -> bool {
    entry.semantic_degraded_reason.is_some()
        || entry.semantic_index_state.as_deref() == Some("degraded")
}

fn parse_tool_runtime_audit(value: Option<&str>) -> Option<ToolRuntimeAudit> {
    value.and_then(|value| serde_json::from_str(value).ok())
}

const BUDGET_EXCEEDED_BODY: &str = r#"{"error":{"message":"Budget limit exceeded","type":"budget_exceeded_error","code":"budget_exceeded"}}"#;
const INVALID_REQUEST_METADATA_CODE: &str = "invalid_request_metadata";
const INVALID_REQUEST_CUSTOM_COST_CODE: &str = "invalid_request_custom_cost";

fn extract_usage_u64(value: Option<&serde_json::Value>) -> Option<u64> {
    value.and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_i64().and_then(|number| u64::try_from(number).ok()))
    })
}

fn extract_observed_response_usage(body_bytes: &[u8]) -> Option<ObservedResponseUsage> {
    let response_json: serde_json::Value = serde_json::from_slice(body_bytes).ok()?;
    let usage = response_json.get("usage")?;
    let input_tokens = extract_usage_u64(usage.get("prompt_tokens"))
        .or_else(|| extract_usage_u64(usage.get("input_tokens")));
    let output_tokens = extract_usage_u64(usage.get("completion_tokens"))
        .or_else(|| extract_usage_u64(usage.get("output_tokens")));
    if input_tokens.is_none() && output_tokens.is_none() {
        return None;
    }
    Some(ObservedResponseUsage {
        input_tokens: input_tokens.unwrap_or(0),
        output_tokens: output_tokens.unwrap_or(0),
    })
}

fn request_cost_from_usage(
    model_costs: (f64, f64),
    custom_cost: Option<&RequestCustomCost>,
    input_tokens: u64,
    output_tokens: u64,
) -> (f64, bool) {
    match custom_cost {
        Some(custom_cost) => (
            (input_tokens as f64 * custom_cost.per_token_in)
                + (output_tokens as f64 * custom_cost.per_token_out),
            true,
        ),
        None => {
            let input_cost = (input_tokens as f64 / 1000.0) * model_costs.0;
            let output_cost = (output_tokens as f64 / 1000.0) * model_costs.1;
            (input_cost + output_cost, false)
        }
    }
}

fn budget_exceeded_response() -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>
{
    let mut resp = Response::builder()
        .status(StatusCode::PAYMENT_REQUIRED)
        .body(
            Full::new(Bytes::from(BUDGET_EXCEEDED_BODY))
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap();
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp
}

fn invalid_request_metadata_response(
    message: &str,
) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": INVALID_REQUEST_METADATA_CODE,
        }
    })
    .to_string();
    let mut resp = Response::builder()
        .status(StatusCode::BAD_REQUEST)
        .body(
            Full::new(Bytes::from(body))
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap();
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp
}

fn invalid_request_custom_cost_response(
    message: &str,
) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": INVALID_REQUEST_CUSTOM_COST_CODE,
        }
    })
    .to_string();
    let mut resp = Response::builder()
        .status(StatusCode::BAD_REQUEST)
        .body(
            Full::new(Bytes::from(body))
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap();
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    resp
}

#[async_trait]
impl Plugin for CostTracker {
    fn name(&self) -> &str {
        "cost_tracker"
    }

    async fn on_request(&self, ctx: &mut RequestContext) -> Action {
        if let Some(session_id) = extract_session_id(&mut ctx.headers) {
            if let Some(candidates) = ctx.extensions.get_mut::<ProviderCandidates>() {
                for candidate in &mut candidates.0 {
                    candidate.headers.remove(SESSION_ID_HEADER);
                }
            }
            ctx.extensions.insert(SessionTraceMeta { session_id });
        }

        if ctx.extensions.get::<RequestMetadata>().is_none() {
            match take_request_metadata(&mut ctx.headers) {
                Ok(Some(metadata)) => {
                    ctx.extensions.insert(metadata);
                }
                Ok(None) => {}
                Err(error) => {
                    return Action::Respond(invalid_request_metadata_response(&error));
                }
            }
        }
        if ctx.extensions.get::<RequestCustomCost>().is_none() {
            match take_request_custom_cost(&mut ctx.headers) {
                Ok(Some(custom_cost)) => {
                    ctx.extensions.insert(custom_cost);
                }
                Ok(None) => {}
                Err(error) => {
                    return Action::Respond(invalid_request_custom_cost_response(&error));
                }
            }
        }

        // If a virtual key was validated upstream, use its hash as the tracking key
        // and its per-key budget instead of the global one.
        let (tracking_key, per_key_budget) =
            if let Some(vk_meta) = ctx.extensions.get::<VirtualKeyMeta>() {
                (vk_meta.key_hash.clone(), vk_meta.budget_limit)
            } else {
                match extract_api_key(&ctx.headers) {
                    Some(key) => (key, None),
                    None => return Action::Continue,
                }
            };

        let estimated_input_tokens = ctx
            .body
            .as_ref()
            .map(|b| estimate_prompt_tokens(b))
            .unwrap_or(0);

        let model = ctx.body.as_ref().and_then(|b| extract_model(b));

        // Budget enforcement: per-key budget takes precedence over global.
        let effective_budget = per_key_budget.unwrap_or(self.budget_limit);
        if effective_budget > 0.0 {
            // Check time-windowed budget reset for virtual keys.
            if let Some(vk_meta) = ctx.extensions.get::<VirtualKeyMeta>() {
                if let Some(ref duration) = vk_meta.budget_duration {
                    let window_start = self
                        .budget_window_starts
                        .get(&tracking_key)
                        .map(|v| *v)
                        .or(vk_meta.budget_window_start);
                    if let Some(window_start) = window_start {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs() as i64;
                        if is_window_expired(duration, window_start, now) {
                            // Reset usage for this key and update window start.
                            self.usage.remove(&tracking_key);
                            // Clear webhook dedup entries for this key.
                            self.webhook_fired.retain(|k, _| k.0 != tracking_key);
                            self.budget_window_starts.insert(tracking_key.clone(), now);
                            if let Some(ref store) = self.store {
                                if let Err(e) = store
                                    .update_virtual_key_budget_window(&tracking_key, now)
                                    .await
                                {
                                    tracing::warn!(
                                        error = %e,
                                        key = %if tracking_key.len() > 12 { format!("{}...", &tracking_key[..12]) } else { tracking_key.clone() },
                                        "cost_tracker: failed to persist budget window start"
                                    );
                                }
                            }
                            tracing::info!(
                                key = %if tracking_key.len() > 12 { format!("{}...", &tracking_key[..12]) } else { tracking_key.clone() },
                                duration = %duration,
                                "cost_tracker: budget window expired, resetting usage"
                            );
                        }
                    }
                }
            }

            if let Some(entry) = self.usage.get(&tracking_key) {
                if entry.total_cost >= effective_budget {
                    if let Some(ref m) = self.metrics {
                        m.budget_rejections_total.inc();
                    }
                    return Action::Respond(budget_exceeded_response());
                }
            }
        }

        // Stash metadata for on_response.
        ctx.extensions.insert(LlmRequestMeta {
            api_key: tracking_key,
            model,
            estimated_input_tokens,
        });

        Action::Continue
    }

    async fn transform_response(
        &self,
        ctx: &mut RequestContext,
        resp: Response<BoxBody<Bytes, hyper::Error>>,
    ) -> Response<BoxBody<Bytes, hyper::Error>> {
        if ctx.extensions.get::<RequestCustomCost>().is_none() {
            return resp;
        }

        let is_sse = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|content_type| content_type.contains("text/event-stream"))
            .unwrap_or(false);
        if is_sse || !resp.status().is_success() {
            return resp;
        }

        let (mut parts, body) = resp.into_parts();
        let body_bytes = match body.collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(_) => {
                return Response::from_parts(
                    parts,
                    Full::new(Bytes::new())
                        .map_err(|never| match never {})
                        .boxed(),
                )
            }
        };

        if let Some(observed_usage) = extract_observed_response_usage(&body_bytes) {
            ctx.extensions.insert(observed_usage);
        }
        parts.headers.remove(CONTENT_LENGTH);
        if let Ok(value) = HeaderValue::from_str(&body_bytes.len().to_string()) {
            parts.headers.insert(CONTENT_LENGTH, value);
        }
        Response::from_parts(
            parts,
            Full::new(body_bytes)
                .map_err(|never| match never {})
                .boxed(),
        )
    }

    async fn on_response(&self, ctx: &mut RequestContext, resp: &mut ResponseContext) -> Action {
        // For SSE responses, don't remove the meta — wrap_response_body will handle cost tracking.
        let is_sse = resp
            .headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.contains("text/event-stream"))
            .unwrap_or(false);

        if is_sse {
            // Don't count cost for error SSE responses either.
            if resp.status.is_client_error() || resp.status.is_server_error() {
                ctx.extensions.remove::<LlmRequestMeta>();
            }
            // Leave meta in place for wrap_response_body.
            return Action::Continue;
        }

        let meta = match ctx.extensions.remove::<LlmRequestMeta>() {
            Some(m) => m,
            None => return Action::Continue,
        };

        // Don't count cost for failed upstream responses (4xx/5xx).
        if resp.status.is_client_error() || resp.status.is_server_error() {
            tracing::debug!(
                api_key = %if meta.api_key.len() > 8 { format!("{}...", &meta.api_key[..8]) } else { meta.api_key.clone() },
                status = resp.status.as_u16(),
                "cost_tracker: skipping cost for error response"
            );
            return Action::Continue;
        }

        let usage_override = ctx.extensions.remove::<ToolUsageOverride>();
        let observed_usage = ctx.extensions.remove::<ObservedResponseUsage>();
        let custom_cost = ctx.extensions.get::<RequestCustomCost>().cloned();
        let actual_usage = usage_override
            .as_ref()
            .map(|usage| ObservedResponseUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
            })
            .or(observed_usage);
        let usage = if let Some(actual_usage) = actual_usage {
            Some((actual_usage.input_tokens, actual_usage.output_tokens))
        } else if custom_cost.is_some() {
            None
        } else {
            Some((
                meta.estimated_input_tokens,
                resp.headers
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(|len| len / 4)
                    .unwrap_or(0),
            ))
        };
        let (cost_per_1k_in, cost_per_1k_out) = self.costs_for_model(meta.model.as_deref());
        if let Some((input_tokens, output_tokens)) = usage {
            let (request_cost, custom_cost_applied) = request_cost_from_usage(
                (cost_per_1k_in, cost_per_1k_out),
                custom_cost.as_ref(),
                input_tokens,
                output_tokens,
            );

            let mut entry = self.usage.entry(meta.api_key.clone()).or_insert(KeyUsage {
                total_input_tokens: 0,
                total_output_tokens: 0,
                total_cost: 0.0,
            });
            entry.total_input_tokens += input_tokens;
            entry.total_output_tokens += output_tokens;
            entry.total_cost += request_cost;

            // Check webhook budget alerts.
            let total_cost = entry.total_cost;
            drop(entry);

            // Record per-model usage.
            self.record_model_usage(
                &meta.api_key,
                meta.model.as_deref(),
                input_tokens,
                output_tokens,
                request_cost,
            );

            // Send audit log entry.
            let audit = request_audit_fields(ctx);
            self.send_audit_log(
                &meta.api_key,
                meta.model.as_deref(),
                input_tokens,
                output_tokens,
                request_cost,
                false,
                custom_cost_applied,
                audit,
            );

            let effective_budget = ctx
                .extensions
                .get::<VirtualKeyMeta>()
                .and_then(|vk| vk.budget_limit)
                .unwrap_or(self.budget_limit);
            self.check_and_fire_webhook(&meta.api_key, total_cost, effective_budget);

            if let Some(ref m) = self.metrics {
                let masked = mask_key(&meta.api_key);
                let model_label = meta.model.as_deref().unwrap_or("unknown");
                m.tokens_total
                    .with_label_values(&[masked.as_str(), model_label, "input"])
                    .inc_by(input_tokens);
                m.tokens_total
                    .with_label_values(&[masked.as_str(), model_label, "output"])
                    .inc_by(output_tokens);
                m.cost_dollars_total
                    .with_label_values(&[masked.as_str(), model_label])
                    .inc_by(request_cost);
                m.request_tokens
                    .with_label_values(&["input"])
                    .observe(input_tokens as f64);
                m.request_tokens
                    .with_label_values(&["output"])
                    .observe(output_tokens as f64);
            }

            // Enrich the OTEL span with LLM attributes.
            #[cfg(feature = "opentelemetry")]
            if let Some(otel) = ctx.extensions.get::<proxy_core::otel::OtelSpan>() {
                otel.0
                    .record("llm.model", meta.model.as_deref().unwrap_or("unknown"));
                otel.0.record("llm.input_tokens", input_tokens);
                otel.0.record("llm.output_tokens", output_tokens);
                otel.0
                    .record("llm.cost_usd", format!("{:.6}", request_cost).as_str());
            }

            tracing::debug!(
                api_key = %if meta.api_key.len() > 8 { format!("{}...", &meta.api_key[..8]) } else { meta.api_key },
                model = meta.model.as_deref().unwrap_or("unknown"),
                input_tokens = input_tokens,
                output_tokens = output_tokens,
                request_cost = format!("{:.6}", request_cost),
                total_cost = format!("{:.6}", total_cost),
                custom_cost_applied = custom_cost_applied,
                "cost_tracker recorded"
            );
        } else {
            let audit = request_audit_fields(ctx);
            self.send_audit_log(
                &meta.api_key,
                meta.model.as_deref(),
                0,
                0,
                0.0,
                false,
                false,
                audit,
            );
            tracing::debug!(
                api_key = %if meta.api_key.len() > 8 { format!("{}...", &meta.api_key[..8]) } else { meta.api_key },
                model = meta.model.as_deref().unwrap_or("unknown"),
                "cost_tracker: custom cost skipped because actual token usage was unavailable"
            );
        }

        Action::Continue
    }

    fn wrap_response_body(
        &self,
        ctx: &RequestContext,
        resp: &ResponseContext,
        body: BoxBody<Bytes, hyper::Error>,
    ) -> BoxBody<Bytes, hyper::Error> {
        // Only wrap SSE (streaming) responses
        let is_sse = resp
            .headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.contains("text/event-stream"))
            .unwrap_or(false);

        if !is_sse {
            return body;
        }

        // Extract the stashed request metadata
        let meta = match ctx.extensions.get::<LlmRequestMeta>() {
            Some(m) => m.clone(),
            None => return body,
        };

        let effective_budget = ctx
            .extensions
            .get::<VirtualKeyMeta>()
            .and_then(|vk| vk.budget_limit)
            .unwrap_or(self.budget_limit);
        let audit = request_audit_fields(ctx);
        let usage_override = ctx.extensions.get::<ToolUsageOverride>().cloned();
        let custom_cost = ctx.extensions.get::<RequestCustomCost>().cloned();

        let (wrapper, rx) = crate::streaming::UsageExtractorBody::new(body);
        let usage_map = Arc::clone(&self.usage);
        let model_costs = Arc::clone(&self.model_costs);
        let model_usage_map = Arc::clone(&self.model_usage);
        let audit_tx = self.audit_tx.clone();
        let default_in = self.default_cost_per_1k_input;
        let default_out = self.default_cost_per_1k_output;
        let metrics = self.metrics.clone();
        let webhook_self = self.clone();

        if let Some(ref m) = metrics {
            m.streaming_requests_total.inc();
        }

        // Spawn a task to wait for stream completion and update usage
        tokio::spawn(async move {
            let stream_usage = rx.await.ok().flatten();
            if usage_override.is_none() && stream_usage.is_none() {
                if custom_cost.is_some() {
                    if let Some(ref tx) = audit_tx {
                        let entry = RequestLogEntry {
                            timestamp_unix: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs() as i64,
                            api_key: meta.api_key.clone(),
                            project_id: audit.project_id.clone(),
                            session_id: audit.session_id.clone(),
                            metadata_json: audit.metadata_json.clone(),
                            custom_cost_json: audit.custom_cost_json.clone(),
                            custom_cost_applied: false,
                            provider_name: audit.provider_name.clone(),
                            prompt_name: audit.prompt_name.clone(),
                            prompt_version: audit.prompt_version.clone(),
                            prompt_environment: audit.prompt_environment.clone(),
                            model: meta.model.clone(),
                            input_tokens: 0,
                            output_tokens: 0,
                            cost: 0.0,
                            is_streaming: true,
                            safety_mode: audit.safety_mode.clone(),
                            safety_matches: audit.safety_matches.clone(),
                            semantic_policy_version: audit.semantic_policy_version.clone(),
                            semantic_index_state: audit.semantic_index_state.clone(),
                            semantic_degraded_reason: audit.semantic_degraded_reason.clone(),
                            semantic_findings: audit.semantic_findings.clone(),
                            tool_trace: audit.tool_trace.clone(),
                        };
                        let _ = tx.send(entry);
                    }
                }
                return;
            }

            let output_tokens = usage_override
                .as_ref()
                .map(|usage| usage.output_tokens)
                .or_else(|| {
                    stream_usage
                        .as_ref()
                        .and_then(|usage| usage.completion_tokens)
                })
                .unwrap_or(0);
            let input_tokens = usage_override
                .as_ref()
                .map(|usage| usage.input_tokens)
                .or_else(|| stream_usage.as_ref().and_then(|usage| usage.prompt_tokens))
                .unwrap_or(meta.estimated_input_tokens);

            let (cost_per_1k_in, cost_per_1k_out) = if let Some(ref model) = meta.model {
                if let Some(mc) = model_costs.get(model.as_str()) {
                    (mc.input, mc.output)
                } else {
                    (default_in, default_out)
                }
            } else {
                (default_in, default_out)
            };

            let (request_cost, custom_cost_applied) = request_cost_from_usage(
                (cost_per_1k_in, cost_per_1k_out),
                custom_cost.as_ref(),
                input_tokens,
                output_tokens,
            );

            let mut entry = usage_map.entry(meta.api_key.clone()).or_insert(KeyUsage {
                total_input_tokens: 0,
                total_output_tokens: 0,
                total_cost: 0.0,
            });
            entry.total_input_tokens += input_tokens;
            entry.total_output_tokens += output_tokens;
            entry.total_cost += request_cost;
            let total_cost = entry.total_cost;
            drop(entry);
            webhook_self.check_and_fire_webhook(&meta.api_key, total_cost, effective_budget);

            // Record per-model usage.
            {
                let model_name = meta.model.as_deref().unwrap_or("unknown").to_string();
                let mut mu = model_usage_map
                    .entry((meta.api_key.clone(), model_name))
                    .or_insert(KeyUsage {
                        total_input_tokens: 0,
                        total_output_tokens: 0,
                        total_cost: 0.0,
                    });
                mu.total_input_tokens += input_tokens;
                mu.total_output_tokens += output_tokens;
                mu.total_cost += request_cost;
            }

            // Send audit log entry.
            if let Some(ref tx) = audit_tx {
                let entry = RequestLogEntry {
                    timestamp_unix: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as i64,
                    api_key: meta.api_key.clone(),
                    project_id: audit.project_id.clone(),
                    session_id: audit.session_id.clone(),
                    metadata_json: audit.metadata_json.clone(),
                    custom_cost_json: audit.custom_cost_json.clone(),
                    custom_cost_applied,
                    provider_name: audit.provider_name.clone(),
                    prompt_name: audit.prompt_name.clone(),
                    prompt_version: audit.prompt_version.clone(),
                    prompt_environment: audit.prompt_environment.clone(),
                    model: meta.model.clone(),
                    input_tokens,
                    output_tokens,
                    cost: request_cost,
                    is_streaming: true,
                    safety_mode: audit.safety_mode.clone(),
                    safety_matches: audit.safety_matches.clone(),
                    semantic_policy_version: audit.semantic_policy_version.clone(),
                    semantic_index_state: audit.semantic_index_state.clone(),
                    semantic_degraded_reason: audit.semantic_degraded_reason.clone(),
                    semantic_findings: audit.semantic_findings.clone(),
                    tool_trace: audit.tool_trace.clone(),
                };
                let _ = tx.send(entry);
            }

            if let Some(ref m) = metrics {
                let masked = mask_key(&meta.api_key);
                let model_label = meta.model.as_deref().unwrap_or("unknown");
                m.tokens_total
                    .with_label_values(&[masked.as_str(), model_label, "input"])
                    .inc_by(input_tokens);
                m.tokens_total
                    .with_label_values(&[masked.as_str(), model_label, "output"])
                    .inc_by(output_tokens);
                m.cost_dollars_total
                    .with_label_values(&[masked.as_str(), model_label])
                    .inc_by(request_cost);
                m.request_tokens
                    .with_label_values(&["input"])
                    .observe(input_tokens as f64);
                m.request_tokens
                    .with_label_values(&["output"])
                    .observe(output_tokens as f64);
            }

            tracing::debug!(
                api_key = %if meta.api_key.len() > 8 { format!("{}...", &meta.api_key[..8]) } else { meta.api_key },
                model = meta.model.as_deref().unwrap_or("unknown"),
                input_tokens = input_tokens,
                output_tokens = output_tokens,
                request_cost = format!("{:.6}", request_cost),
                custom_cost_applied = custom_cost_applied,
                "cost_tracker: streaming usage recorded"
            );
        });

        BoxBody::new(wrapper)
    }
}

fn extract_session_id(headers: &mut hyper::HeaderMap) -> Option<String> {
    let value = headers.remove(SESSION_ID_HEADER)?;
    value
        .to_str()
        .ok()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

/// Fetch model pricing from OpenRouter in the background and populate the map.
fn spawn_openrouter_fetch(model_costs: Arc<DashMap<String, ModelCost>>) {
    tokio::spawn(async move {
        tracing::info!("cost_tracker: fetching model pricing from OpenRouter...");

        let client = proxy_core::handlers::proxy::build_client();
        let req = match hyper::Request::builder()
            .method("GET")
            .uri(OPENROUTER_API)
            .header("user-agent", "tiny-reverse-proxy/cost-tracker")
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            ) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("cost_tracker: failed to build OpenRouter request: {e}");
                return;
            }
        };

        let resp =
            match tokio::time::timeout(std::time::Duration::from_secs(15), client.request(req))
                .await
            {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    tracing::warn!("cost_tracker: OpenRouter request failed: {e}");
                    return;
                }
                Err(_) => {
                    tracing::warn!("cost_tracker: OpenRouter request timed out");
                    return;
                }
            };

        if !resp.status().is_success() {
            tracing::warn!("cost_tracker: OpenRouter returned status {}", resp.status());
            return;
        }

        let body = match resp.into_body().collect().await {
            Ok(b) => b.to_bytes(),
            Err(e) => {
                tracing::warn!("cost_tracker: failed to read OpenRouter response: {e}");
                return;
            }
        };

        let count = parse_openrouter_models(&body, &model_costs);
        if count > 0 {
            tracing::info!(
                "cost_tracker: loaded pricing for {count} models from OpenRouter. \
                 To customize, add [plugins.config.model_costs] to your config file."
            );
        } else {
            tracing::warn!("cost_tracker: no models with pricing found in OpenRouter response");
        }
    });
}

/// Parse OpenRouter JSON response and populate model_costs. Returns count of models added.
///
/// Uses simple byte scanning to avoid a serde_json dependency — the response format is:
/// `{"data": [{"id": "provider/model", "pricing": {"prompt": "0.00001", "completion": "0.00002"}}, ...]}`
fn parse_openrouter_models(body: &[u8], model_costs: &DashMap<String, ModelCost>) -> usize {
    // Minimal JSON extraction: find each "id" + "pricing.prompt" + "pricing.completion" object.
    // We scan for `"id":"<value>"` and `"prompt":"<value>"` / `"completion":"<value>"` pairs.
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(_) => return 0,
    };

    let mut count = 0;

    // Split by `{"id":` or `{ "id":` to get rough per-model chunks.
    // Each chunk should contain one model's data.
    let chunks: Vec<&str> = text.split("\"id\"").collect();

    for chunk in chunks.iter().skip(1) {
        // Extract model id value.
        let model_id = match extract_json_string_after(chunk, "") {
            Some(id) => id,
            None => continue,
        };

        // We only want "provider/model" format; skip the provider prefix for the key.
        let short_name = match model_id.split_once('/') {
            Some((_, name)) => name.to_string(),
            None => continue,
        };

        // Extract prompt price.
        let prompt_price = match extract_json_string_value(chunk, "\"prompt\"") {
            Some(p) => match p.parse::<f64>() {
                Ok(v) => v,
                Err(_) => continue,
            },
            None => continue,
        };

        // Extract completion price.
        let completion_price = match extract_json_string_value(chunk, "\"completion\"") {
            Some(p) => match p.parse::<f64>() {
                Ok(v) => v,
                Err(_) => continue,
            },
            None => continue,
        };

        // OpenRouter prices are per-token, convert to per-1k.
        let input_per_1k = prompt_price * 1000.0;
        let output_per_1k = completion_price * 1000.0;

        if input_per_1k > 0.0 || output_per_1k > 0.0 {
            // Don't overwrite config-provided prices.
            model_costs.entry(short_name).or_insert(ModelCost {
                input: input_per_1k,
                output: output_per_1k,
            });
            count += 1;
        }
    }

    count
}

/// Extract a JSON string value immediately after a position (expects `:"value"` or `: "value"`).
fn extract_json_string_after(text: &str, _prefix: &str) -> Option<String> {
    // Expect text to start like: `:"some-value",...` or `: "some-value",...`
    let text = text.trim_start();
    let text = text.strip_prefix(':')?;
    let text = text.trim_start();
    let text = text.strip_prefix('"')?;
    let end = text.find('"')?;
    Some(text[..end].to_string())
}

/// Find `"key":"value"` or `"key": "value"` in text and return the value.
fn extract_json_string_value(text: &str, key: &str) -> Option<String> {
    let idx = text.find(key)?;
    let after = &text[idx + key.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();
    let after = after.strip_prefix('"')?;
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

/// Check if a budget window has expired.
///
/// Durations: "daily" (24h), "weekly" (7d), "monthly" (30d).
fn is_window_expired(duration: &str, window_start: i64, now: i64) -> bool {
    let window_secs = match duration {
        "daily" => 24 * 3600,
        "weekly" => 7 * 24 * 3600,
        "monthly" => 30 * 24 * 3600,
        _ => return false,
    };
    now >= window_start + window_secs
}

/// Create a CostTracker directly (not boxed) for use with the API layer.
pub fn create_tracker(config: &toml::Value) -> Result<CostTracker, Box<dyn std::error::Error>> {
    let budget_limit = config
        .get("budget_limit")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .unwrap_or(0.0);
    let log_interval_secs = config
        .get("log_interval_secs")
        .and_then(|v| v.as_integer())
        .unwrap_or(60) as u64;
    let default_cost_per_1k_input = config
        .get("default_cost_per_1k_input")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .unwrap_or(0.01);
    let default_cost_per_1k_output = config
        .get("default_cost_per_1k_output")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .unwrap_or(0.02);

    let mut plugin = CostTracker::new(
        budget_limit,
        default_cost_per_1k_input,
        default_cost_per_1k_output,
        log_interval_secs,
    );

    // Parse webhook configuration.
    if let Some(raw_url) = config.get("webhook_url").and_then(|v| v.as_str()) {
        let url = if let Some(var_name) = raw_url.strip_prefix('$') {
            std::env::var(var_name).unwrap_or_else(|_| {
                tracing::warn!("cost_tracker: env var {var_name} not set for webhook_url");
                String::new()
            })
        } else {
            raw_url.to_string()
        };
        if !url.is_empty() {
            let thresholds = config
                .get("budget_alert_thresholds")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_integer().map(|i| i as u64))
                        .collect()
                })
                .unwrap_or_else(|| vec![50, 80, 100]);
            let ttl = config
                .get("budget_alert_ttl")
                .and_then(|v| v.as_integer())
                .unwrap_or(86400) as u64;
            plugin = plugin.with_webhook(url, thresholds, ttl);
        }
    }

    // Parse per-model cost overrides from config.
    if let Some(table) = config.get("model_costs").and_then(|v| v.as_table()) {
        for (model, entry) in table {
            let input = entry
                .get("input")
                .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
                .unwrap_or(default_cost_per_1k_input);
            let output = entry
                .get("output")
                .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
                .unwrap_or(default_cost_per_1k_output);
            plugin
                .model_costs
                .insert(model.clone(), ModelCost { input, output });
        }
    }

    Ok(plugin)
}

/// Factory function for creating a CostTracker from TOML config.
pub fn create(config: &toml::Value) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>> {
    let plugin = create_tracker(config)?;

    let model_count = plugin.model_costs.len();
    if model_count == 0 {
        tracing::info!(
            "cost_tracker: no per-model pricing in config, fetching from OpenRouter \
             (using ${}/1k input, ${}/1k output as defaults until loaded)",
            plugin.default_cost_per_1k_input,
            plugin.default_cost_per_1k_output,
        );
        spawn_openrouter_fetch(Arc::clone(&plugin.model_costs));
    } else {
        tracing::info!(
            models = model_count,
            "cost_tracker loaded pricing for {} models from config",
            model_count
        );
    }

    plugin.spawn_logger_task();
    Ok(Box::new(plugin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{self, GatewayStore, VirtualKeyRecord};
    use hyper::header::{HeaderMap, HeaderValue};
    use hyper::http::Extensions;
    use hyper::{Method, Uri, Version};
    use std::sync::Arc;
    use std::time::Duration;

    fn make_ctx(api_key: &str, body: &[u8]) -> RequestContext {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap(),
        );
        RequestContext {
            peer_addr: None,
            method: Method::POST,
            uri: Uri::from_static("http://localhost/v1/chat/completions"),
            version: Version::HTTP_11,
            headers,
            body: Some(Bytes::from(body.to_vec())),
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        }
    }

    fn make_resp_ctx(content_length: u64) -> ResponseContext {
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-length",
            HeaderValue::from_str(&content_length.to_string()).unwrap(),
        );
        ResponseContext {
            status: StatusCode::OK,
            headers,
            upstream: "https://api.openai.com".into(),
            duration: Duration::from_millis(100),
        }
    }

    fn make_json_response(body: &'static str) -> Response<BoxBody<Bytes, hyper::Error>> {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .header("content-length", body.len().to_string())
            .body(
                Full::new(Bytes::from_static(body.as_bytes()))
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap()
    }

    #[tokio::test]
    async fn test_cost_accumulation() {
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        // Request: 4000 bytes body => 1000 input tokens
        // No model_costs configured, so falls back to defaults (0.01/0.02 per 1k)
        let body = br#"{"model":"gpt-4","messages":[{"role":"user","content":"x"]}"#;
        let padded_body = {
            let mut v = body.to_vec();
            v.resize(4000, b' ');
            v
        };
        let mut ctx = make_ctx("sk-test", &padded_body);

        // on_request should stash metadata.
        match tracker.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("should continue"),
        }

        // on_response: 2000 bytes content-length => 500 output tokens
        let mut resp_ctx = make_resp_ctx(2000);
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        let entry = tracker.usage.get("sk-test").unwrap();
        assert_eq!(entry.total_input_tokens, 1000);
        assert_eq!(entry.total_output_tokens, 500);
        // Cost: (1000/1000)*0.01 + (500/1000)*0.02 = 0.01 + 0.01 = 0.02
        assert!(
            (entry.total_cost - 0.02).abs() < 1e-9,
            "expected 0.02, got {}",
            entry.total_cost
        );
    }

    #[tokio::test]
    async fn test_budget_enforcement() {
        let tracker = CostTracker::new(0.05, 0.01, 0.02, 60);

        // Pre-fill usage to exceed budget.
        tracker.usage.insert(
            "sk-expensive".into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 0.10, // Over 0.05 limit
            },
        );

        let mut ctx = make_ctx("sk-expensive", b"hello");
        match tracker.on_request(&mut ctx).await {
            Action::Continue => panic!("should reject over budget"),
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
            }
        }
    }

    #[tokio::test]
    async fn test_no_budget_limit() {
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60); // 0 = no limit

        // Pre-fill huge cost.
        tracker.usage.insert(
            "sk-unlimited".into(),
            KeyUsage {
                total_input_tokens: 1_000_000,
                total_output_tokens: 500_000,
                total_cost: 999.99,
            },
        );

        let mut ctx = make_ctx("sk-unlimited", b"hello");
        match tracker.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("no budget limit should allow"),
        }
    }

    #[tokio::test]
    async fn test_no_api_key_passes_through() {
        let tracker = CostTracker::new(0.01, 0.01, 0.02, 60);
        let mut ctx = RequestContext {
            peer_addr: None,
            method: Method::POST,
            uri: Uri::from_static("http://localhost/v1/chat/completions"),
            version: Version::HTTP_11,
            headers: HeaderMap::new(),
            body: Some(Bytes::from("hello")),
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        };
        match tracker.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("no API key should pass through"),
        }
    }

    #[tokio::test]
    async fn test_model_specific_pricing() {
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);
        // Configure gpt-4o pricing via model_costs (as the script would generate)
        tracker.model_costs.insert(
            "gpt-4o".into(),
            ModelCost {
                input: 0.0025,
                output: 0.01,
            },
        );

        // Body with "model":"gpt-4o", padded to 4000 bytes => 1000 input tokens
        let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"x"}]}"#;
        let padded = {
            let mut v = body.to_vec();
            v.resize(4000, b' ');
            v
        };
        let mut ctx = make_ctx("sk-model", &padded);
        tracker.on_request(&mut ctx).await;

        // 2000 bytes response => 500 output tokens
        let mut resp_ctx = make_resp_ctx(2000);
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        let entry = tracker.usage.get("sk-model").unwrap();
        // Cost: (1000/1000)*0.0025 + (500/1000)*0.01 = 0.0025 + 0.005 = 0.0075
        assert!(
            (entry.total_cost - 0.0075).abs() < 1e-9,
            "expected 0.0075, got {}",
            entry.total_cost
        );
    }

    #[tokio::test]
    async fn test_unknown_model_uses_defaults() {
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        let body = br#"{"model":"some-custom-finetune","messages":[]}"#;
        let padded = {
            let mut v = body.to_vec();
            v.resize(4000, b' ');
            v
        };
        let mut ctx = make_ctx("sk-custom", &padded);
        tracker.on_request(&mut ctx).await;

        let mut resp_ctx = make_resp_ctx(2000);
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        let entry = tracker.usage.get("sk-custom").unwrap();
        // Falls back to defaults: (1000/1000)*0.01 + (500/1000)*0.02 = 0.01 + 0.01 = 0.02
        assert!(
            (entry.total_cost - 0.02).abs() < 1e-9,
            "expected 0.02, got {}",
            entry.total_cost
        );
    }

    #[tokio::test]
    async fn test_custom_cost_uses_observed_non_streaming_usage() {
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        let mut ctx = make_ctx("sk-custom-cost", br#"{"model":"gpt-4o","messages":[]}"#);
        ctx.headers.insert(
            crate::REQUEST_CUSTOM_COST_HEADER,
            HeaderValue::from_static(r#"{"per_token_in":0.001,"per_token_out":0.002}"#),
        );
        tracker.on_request(&mut ctx).await;

        let resp = make_json_response(
            r#"{"id":"chatcmpl-abc","usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}"#,
        );
        let resp = <CostTracker as Plugin>::transform_response(&tracker, &mut ctx, resp).await;
        let mut resp_ctx = ResponseContext {
            status: resp.status(),
            headers: resp.headers().clone(),
            upstream: "https://api.openai.com".into(),
            duration: Duration::from_millis(50),
        };
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        let entry = tracker.usage.get("sk-custom-cost").unwrap();
        assert_eq!(entry.total_input_tokens, 10);
        assert_eq!(entry.total_output_tokens, 8);
        assert!(
            (entry.total_cost - 0.026).abs() < 1e-9,
            "expected 0.026, got {}",
            entry.total_cost
        );
    }

    #[tokio::test]
    async fn test_custom_cost_skips_when_non_streaming_usage_is_unavailable() {
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        let mut ctx = make_ctx(
            "sk-custom-cost-missing",
            br#"{"model":"gpt-4o","messages":[]}"#,
        );
        ctx.headers.insert(
            crate::REQUEST_CUSTOM_COST_HEADER,
            HeaderValue::from_static(r#"{"per_token_in":0.001,"per_token_out":0.002}"#),
        );
        tracker.on_request(&mut ctx).await;

        let resp = make_json_response(r#"{"id":"chatcmpl-abc","choices":[]}"#);
        let resp = <CostTracker as Plugin>::transform_response(&tracker, &mut ctx, resp).await;
        let mut resp_ctx = ResponseContext {
            status: resp.status(),
            headers: resp.headers().clone(),
            upstream: "https://api.openai.com".into(),
            duration: Duration::from_millis(50),
        };
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        assert!(tracker.usage.get("sk-custom-cost-missing").is_none());
    }

    #[tokio::test]
    async fn test_multiple_requests_accumulate() {
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        for _ in 0..3 {
            let mut ctx = make_ctx("sk-multi", &vec![b'a'; 400]); // 100 input tokens each
            tracker.on_request(&mut ctx).await;
            let mut resp_ctx = make_resp_ctx(800); // 200 output tokens each
            tracker.on_response(&mut ctx, &mut resp_ctx).await;
        }

        let entry = tracker.usage.get("sk-multi").unwrap();
        assert_eq!(entry.total_input_tokens, 300);
        assert_eq!(entry.total_output_tokens, 600);
    }

    #[test]
    fn test_parse_openrouter_models() {
        let json = br#"{"data":[
            {"id":"openai/gpt-4o","pricing":{"prompt":"0.0000025","completion":"0.00001"}},
            {"id":"anthropic/claude-3-opus","pricing":{"prompt":"0.000015","completion":"0.000075"}},
            {"id":"free/model","pricing":{"prompt":"0","completion":"0"}},
            {"id":"bad-format","pricing":{"prompt":"0.001","completion":"0.002"}}
        ]}"#;
        let costs = DashMap::new();
        let count = parse_openrouter_models(json, &costs);

        // free model (0/0) should be skipped, bad-format (no slash) should be skipped
        assert_eq!(count, 2);
        let gpt4o = costs.get("gpt-4o").unwrap();
        assert!((gpt4o.input - 0.0025).abs() < 1e-9); // 0.0000025 * 1000
        assert!((gpt4o.output - 0.01).abs() < 1e-9); // 0.00001 * 1000

        let opus = costs.get("claude-3-opus").unwrap();
        assert!((opus.input - 0.015).abs() < 1e-9);
        assert!((opus.output - 0.075).abs() < 1e-9);
    }

    #[test]
    fn test_parse_openrouter_config_takes_precedence() {
        let json = br#"{"data":[
            {"id":"openai/gpt-4o","pricing":{"prompt":"0.0000025","completion":"0.00001"}}
        ]}"#;
        let costs = DashMap::new();
        // Pre-insert a config-provided price.
        costs.insert(
            "gpt-4o".into(),
            ModelCost {
                input: 0.999,
                output: 0.888,
            },
        );
        parse_openrouter_models(json, &costs);

        // Config value should NOT be overwritten.
        let gpt4o = costs.get("gpt-4o").unwrap();
        assert!((gpt4o.input - 0.999).abs() < 1e-9);
        assert!((gpt4o.output - 0.888).abs() < 1e-9);
    }

    #[test]
    fn test_clone_shares_state() {
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);
        tracker.usage.insert(
            "key1".into(),
            KeyUsage {
                total_input_tokens: 100,
                total_output_tokens: 50,
                total_cost: 0.01,
            },
        );

        let cloned = tracker.clone();
        // Both should see the same data.
        assert!(cloned.get_usage("key1").is_some());
        assert_eq!(cloned.get_usage("key1").unwrap().total_input_tokens, 100);

        // Mutation via original visible through clone.
        tracker.set_model_cost("gpt-4", 0.03, 0.06);
        assert_eq!(cloned.get_model_costs().len(), 1);
    }

    // --- Edge-case tests ---

    fn make_resp_ctx_with_header(content_length: Option<&str>) -> ResponseContext {
        let mut headers = HeaderMap::new();
        if let Some(cl) = content_length {
            headers.insert("content-length", HeaderValue::from_str(cl).unwrap());
        }
        ResponseContext {
            status: StatusCode::OK,
            headers,
            upstream: "https://api.openai.com".into(),
            duration: Duration::from_millis(100),
        }
    }

    #[tokio::test]
    async fn test_invalid_utf8_body_no_panic() {
        // #2: Invalid UTF-8 bytes in request body should not cause panics.
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);
        let invalid_utf8: Vec<u8> = vec![0xFF, 0xFE, 0x80, 0x81, 0xC0, 0xC1];
        let mut ctx = make_ctx("sk-test-utf8", &invalid_utf8);

        match tracker.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("should continue with invalid UTF-8 body"),
        }

        let mut resp_ctx = make_resp_ctx(100);
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        // estimate_prompt_tokens uses body.len()/4 = 6/4 = 1
        let entry = tracker.usage.get("sk-test-utf8").unwrap();
        assert_eq!(entry.total_input_tokens, 1);
    }

    #[tokio::test]
    async fn test_cost_exactly_at_budget_limit() {
        // #10: When total_cost == budget_limit, the next request should be rejected (>= check).
        let tracker = CostTracker::new(1.0, 0.01, 0.02, 60);

        tracker.usage.insert(
            "sk-exact".into(),
            KeyUsage {
                total_input_tokens: 1000,
                total_output_tokens: 500,
                total_cost: 1.0, // Exactly at budget_limit
            },
        );

        let mut ctx = make_ctx("sk-exact", b"hello");
        match tracker.on_request(&mut ctx).await {
            Action::Continue => panic!("should reject when cost exactly equals budget limit"),
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
            }
        }
    }

    #[tokio::test]
    async fn test_cost_just_below_budget_limit() {
        // #10 companion: total_cost just below budget_limit should still be allowed.
        let tracker = CostTracker::new(1.0, 0.01, 0.02, 60);

        tracker.usage.insert(
            "sk-almost".into(),
            KeyUsage {
                total_input_tokens: 1000,
                total_output_tokens: 500,
                total_cost: 0.999_999_999, // Just under budget
            },
        );

        let mut ctx = make_ctx("sk-almost", b"hello");
        match tracker.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("should allow when cost is just below budget limit"),
        }
    }

    #[tokio::test]
    async fn test_floating_point_accumulation() {
        // #11: Accumulate many small costs and verify precision doesn't drift significantly.
        let tracker = CostTracker::new(0.0, 0.001, 0.001, 60);

        let iterations: u64 = 10_000;
        for _ in 0..iterations {
            // 4 bytes body => 1 input token; content-length 4 => 1 output token
            let mut ctx = make_ctx("sk-fp", &[b'x'; 4]);
            tracker.on_request(&mut ctx).await;
            let mut resp_ctx = make_resp_ctx(4);
            tracker.on_response(&mut ctx, &mut resp_ctx).await;
        }

        let entry = tracker.usage.get("sk-fp").unwrap();
        assert_eq!(entry.total_input_tokens, iterations);
        assert_eq!(entry.total_output_tokens, iterations);

        // Each request: (1/1000)*0.001 + (1/1000)*0.001 = 0.000002
        // Total expected: 10_000 * 0.000002 = 0.02
        let expected = iterations as f64 * 0.000_002;
        let drift = (entry.total_cost - expected).abs();
        assert!(
            drift < 1e-6,
            "floating-point drift too large: expected {expected}, got {}, drift {drift}",
            entry.total_cost
        );
    }

    #[tokio::test]
    async fn test_malformed_content_length() {
        // #13: Malformed Content-Length values should yield 0 output tokens.
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        let malformed_values = ["abc", "-1", "1.5", "0"];
        for (i, val) in malformed_values.iter().enumerate() {
            let key = format!("sk-malformed-{i}");
            let mut ctx = make_ctx(&key, &[b'x'; 400]); // 100 input tokens
            tracker.on_request(&mut ctx).await;
            let mut resp_ctx = make_resp_ctx_with_header(Some(val));
            tracker.on_response(&mut ctx, &mut resp_ctx).await;

            let entry = tracker.usage.get(key.as_str()).unwrap();
            assert_eq!(
                entry.total_output_tokens, 0,
                "content-length '{val}' should yield 0 output tokens"
            );
        }
    }

    #[tokio::test]
    async fn test_missing_content_length() {
        // #13: Missing Content-Length header should yield 0 output tokens.
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        let mut ctx = make_ctx("sk-no-cl", &[b'x'; 400]);
        tracker.on_request(&mut ctx).await;
        let mut resp_ctx = make_resp_ctx_with_header(None);
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        let entry = tracker.usage.get("sk-no-cl").unwrap();
        assert_eq!(
            entry.total_output_tokens, 0,
            "missing Content-Length should yield 0 output tokens"
        );
    }

    // --- LiteLLM-gap tests ---

    #[tokio::test]
    async fn test_error_response_skips_cost() {
        // Upstream 4xx/5xx responses should NOT accumulate any cost or tokens.
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        for status_code in [400u16, 401, 403, 404, 429, 500, 502, 503] {
            let key = format!("sk-err-{status_code}");
            let mut ctx = make_ctx(&key, &vec![b'x'; 400]); // 100 input tokens
            tracker.on_request(&mut ctx).await;

            let mut headers = HeaderMap::new();
            headers.insert(
                "content-length",
                HeaderValue::from_str("800").unwrap(), // would be 200 output tokens
            );
            let mut resp_ctx = ResponseContext {
                status: StatusCode::from_u16(status_code).unwrap(),
                headers,
                upstream: "https://api.openai.com".into(),
                duration: Duration::from_millis(100),
            };
            tracker.on_response(&mut ctx, &mut resp_ctx).await;

            assert!(
                tracker.usage.get(key.as_str()).is_none(),
                "status {} should not create usage entry",
                status_code
            );
        }
    }

    #[tokio::test]
    async fn test_success_response_still_accumulates_cost() {
        // Verify 2xx responses still record cost after the error-skip fix.
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        let mut ctx = make_ctx("sk-ok-200", &vec![b'x'; 400]); // 100 input tokens
        tracker.on_request(&mut ctx).await;
        let mut resp_ctx = make_resp_ctx(800); // 200 output tokens
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        let entry = tracker.usage.get("sk-ok-200").unwrap();
        assert_eq!(entry.total_input_tokens, 100);
        assert_eq!(entry.total_output_tokens, 200);
        assert!(entry.total_cost > 0.0, "2xx should accumulate cost");
    }

    #[tokio::test]
    async fn test_streaming_response_no_content_length() {
        // Streaming responses typically lack Content-Length. Output tokens should be 0.
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        let mut ctx = make_ctx("sk-stream", &vec![b'x'; 400]); // 100 input tokens
        tracker.on_request(&mut ctx).await;

        // No Content-Length header (simulating chunked/streaming response)
        let mut resp_ctx = ResponseContext {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            upstream: "https://api.openai.com".into(),
            duration: Duration::from_millis(100),
        };
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        let entry = tracker.usage.get("sk-stream").unwrap();
        assert_eq!(entry.total_input_tokens, 100);
        assert_eq!(
            entry.total_output_tokens, 0,
            "streaming (no Content-Length) should record 0 output tokens"
        );
        // Cost should only include input portion
        let expected_cost = (100.0 / 1000.0) * 0.01; // input only
        assert!(
            (entry.total_cost - expected_cost).abs() < 1e-9,
            "expected {expected_cost}, got {}",
            entry.total_cost
        );
    }

    #[test]
    fn test_budget_exceeded_response_openai_format() {
        // Error response JSON must contain message, type, AND code for OpenAI SDK compat.
        let body: &str = BUDGET_EXCEEDED_BODY;
        assert!(body.contains(r#""message":"#), "must have message field");
        assert!(body.contains(r#""type":"#), "must have type field");
        assert!(body.contains(r#""code":"#), "must have code field");
        // Verify it's valid JSON by doing a basic structure check
        assert!(
            body.starts_with(r#"{"error":{"#),
            "must be wrapped in error object"
        );
    }

    #[test]
    fn test_parse_openrouter_scientific_notation() {
        // #25: Scientific notation in pricing values (e.g. "1e-5", "2.5E-3").
        let json = br#"{"data":[
            {"id":"provider/sci-model","pricing":{"prompt":"1e-5","completion":"2.5E-3"}},
            {"id":"provider/sci-model-2","pricing":{"prompt":"1.5e-4","completion":"3E-4"}}
        ]}"#;
        let costs = DashMap::new();
        let count = parse_openrouter_models(json, &costs);

        assert_eq!(count, 2);

        let m1 = costs.get("sci-model").unwrap();
        // 1e-5 * 1000 = 0.01
        assert!(
            (m1.input - 0.01).abs() < 1e-12,
            "expected 0.01, got {}",
            m1.input
        );
        // 2.5E-3 * 1000 = 2.5
        assert!(
            (m1.output - 2.5).abs() < 1e-12,
            "expected 2.5, got {}",
            m1.output
        );

        let m2 = costs.get("sci-model-2").unwrap();
        // 1.5e-4 * 1000 = 0.15
        assert!(
            (m2.input - 0.15).abs() < 1e-12,
            "expected 0.15, got {}",
            m2.input
        );
        // 3E-4 * 1000 = 0.3
        assert!(
            (m2.output - 0.3).abs() < 1e-12,
            "expected 0.3, got {}",
            m2.output
        );
    }

    // --- Gap tests: is_window_expired ---

    #[test]
    fn test_is_window_expired_daily() {
        let start = 1_000_000;
        let one_day = 24 * 3600;
        // Not yet expired (23h59m59s later)
        assert!(!is_window_expired("daily", start, start + one_day - 1));
        // Exactly expired
        assert!(is_window_expired("daily", start, start + one_day));
        // Well past expired
        assert!(is_window_expired("daily", start, start + one_day + 3600));
    }

    #[test]
    fn test_is_window_expired_weekly() {
        let start = 1_000_000;
        let one_week = 7 * 24 * 3600;
        assert!(!is_window_expired("weekly", start, start + one_week - 1));
        assert!(is_window_expired("weekly", start, start + one_week));
    }

    #[test]
    fn test_is_window_expired_monthly() {
        let start = 1_000_000;
        let one_month = 30 * 24 * 3600;
        assert!(!is_window_expired("monthly", start, start + one_month - 1));
        assert!(is_window_expired("monthly", start, start + one_month));
    }

    #[test]
    fn test_is_window_expired_unknown_duration() {
        // Unknown durations should never expire (returns false).
        assert!(!is_window_expired("hourly", 0, 999_999_999));
        assert!(!is_window_expired("yearly", 0, 999_999_999));
        assert!(!is_window_expired("", 0, 999_999_999));
    }

    // --- Gap tests: per-virtual-key budget enforcement ---

    #[tokio::test]
    async fn test_per_virtual_key_budget_takes_precedence() {
        // Global budget is 100.0 but virtual key has budget_limit = 0.05.
        // Usage at 0.06 should be rejected by per-key limit even though global allows it.
        let tracker = CostTracker::new(100.0, 0.01, 0.02, 60);

        let tracking_key = "vk_hash_budget_test";
        tracker.usage.insert(
            tracking_key.into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 0.06, // Over per-key 0.05 limit, under global 100.0
            },
        );

        let mut ctx = make_ctx("sk-unused", b"hello");
        // Simulate virtual key plugin having stashed VirtualKeyMeta
        ctx.extensions.insert(VirtualKeyMeta {
            key_hash: tracking_key.to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: Some(0.05),
            budget_duration: None,
            budget_window_start: None,
            rpm_limit: None,
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });

        match tracker.on_request(&mut ctx).await {
            Action::Continue => panic!("should reject: per-key budget exceeded"),
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
            }
        }
    }

    #[tokio::test]
    async fn test_per_virtual_key_budget_allows_within_limit() {
        // Per-key budget 10.0, usage at 5.0 — should be allowed.
        let tracker = CostTracker::new(0.01, 0.01, 0.02, 60); // tiny global

        let tracking_key = "vk_hash_ok";
        tracker.usage.insert(
            tracking_key.into(),
            KeyUsage {
                total_input_tokens: 1000,
                total_output_tokens: 500,
                total_cost: 5.0,
            },
        );

        let mut ctx = make_ctx("sk-unused", b"hello");
        ctx.extensions.insert(VirtualKeyMeta {
            key_hash: tracking_key.to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: Some(10.0),
            budget_duration: None,
            budget_window_start: None,
            rpm_limit: None,
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });

        match tracker.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("should allow: within per-key budget"),
        }
    }

    #[tokio::test]
    async fn test_time_windowed_budget_reset() {
        // Virtual key has daily budget and the window has expired → usage should be reset.
        let tracker = CostTracker::new(100.0, 0.01, 0.02, 60);

        let tracking_key = "vk_hash_windowed";
        tracker.usage.insert(
            tracking_key.into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 50.0, // Would exceed a 10.0 budget
            },
        );

        // Window started 2 days ago → daily window is expired
        let two_days_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - 2 * 24 * 3600;

        let mut ctx = make_ctx("sk-unused", b"hello");
        ctx.extensions.insert(VirtualKeyMeta {
            key_hash: tracking_key.to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: Some(10.0),
            budget_duration: Some("daily".to_string()),
            budget_window_start: Some(two_days_ago),
            rpm_limit: None,
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });

        // Should reset usage and allow the request
        match tracker.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("should allow after window reset"),
        }

        // Usage should have been cleared (the old 50.0 cost is gone)
        assert!(
            tracker.usage.get(tracking_key).is_none()
                || tracker.usage.get(tracking_key).unwrap().total_cost < 1.0,
            "usage should be reset after window expiry"
        );
    }

    #[tokio::test]
    async fn test_time_windowed_budget_not_expired_enforces() {
        // Virtual key has daily budget, window started 1 hour ago → NOT expired.
        // Usage exceeds budget → should be rejected.
        let tracker = CostTracker::new(100.0, 0.01, 0.02, 60);

        let tracking_key = "vk_hash_not_expired";
        tracker.usage.insert(
            tracking_key.into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 50.0,
            },
        );

        let one_hour_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - 3600;

        let mut ctx = make_ctx("sk-unused", b"hello");
        ctx.extensions.insert(VirtualKeyMeta {
            key_hash: tracking_key.to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: Some(10.0),
            budget_duration: Some("daily".to_string()),
            budget_window_start: Some(one_hour_ago),
            rpm_limit: None,
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });

        match tracker.on_request(&mut ctx).await {
            Action::Continue => panic!("should reject: budget exceeded within current window"),
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
            }
        }
    }

    #[tokio::test]
    async fn test_time_windowed_budget_not_reset_repeatedly_for_stale_meta() {
        let tracker = CostTracker::new(100.0, 0.01, 0.02, 60);
        let tracking_key = "vk_hash_stale_window";
        let two_days_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - 2 * 24 * 3600;

        // First request should reset because the window is expired.
        tracker.usage.insert(
            tracking_key.into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 50.0,
            },
        );
        let mut ctx1 = make_ctx("sk-unused", b"hello");
        ctx1.extensions.insert(VirtualKeyMeta {
            key_hash: tracking_key.to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: Some(10.0),
            budget_duration: Some("daily".to_string()),
            budget_window_start: Some(two_days_ago),
            rpm_limit: None,
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });
        assert!(matches!(
            tracker.on_request(&mut ctx1).await,
            Action::Continue
        ));

        // Re-populate usage as if the key consumed budget in the new window.
        tracker.usage.insert(
            tracking_key.into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 50.0,
            },
        );
        // Reuse stale metadata (same old window_start): should NOT reset again.
        let mut ctx2 = make_ctx("sk-unused", b"hello");
        ctx2.extensions.insert(VirtualKeyMeta {
            key_hash: tracking_key.to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: Some(10.0),
            budget_duration: Some("daily".to_string()),
            budget_window_start: Some(two_days_ago),
            rpm_limit: None,
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });
        match tracker.on_request(&mut ctx2).await {
            Action::Continue => panic!("stale metadata should not trigger repeated window reset"),
            Action::Respond(resp) => assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED),
        }
    }

    #[tokio::test]
    async fn test_time_windowed_budget_persists_window_start_to_store() {
        let store = Arc::new(store::connect("sqlite::memory:").await.unwrap());
        let tracker = CostTracker::new(100.0, 0.01, 0.02, 60).with_store(Arc::clone(&store));
        let tracking_key = "vk_hash_store_window";
        let now_before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let two_days_ago = now_before - 2 * 24 * 3600;

        store
            .upsert_virtual_key(&VirtualKeyRecord {
                key_hash: tracking_key.to_string(),
                project_id: "legacy".to_string(),
                name: "window-test".to_string(),
                provider_name: "openai".to_string(),
                budget_limit: Some(10.0),
                budget_duration: Some("daily".to_string()),
                budget_window_start: Some(two_days_ago),
                rpm_limit: None,
                tpm_limit: None,
                allowed_models: None,
                timeout_secs: None,
                tool_approval_mode: None,
                allowed_tools: None,
                active: true,
                created_at: now_before.to_string(),
                expires_at: None,
            })
            .await
            .unwrap();

        tracker.usage.insert(
            tracking_key.into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 50.0,
            },
        );

        let mut ctx = make_ctx("sk-unused", b"hello");
        ctx.extensions.insert(VirtualKeyMeta {
            key_hash: tracking_key.to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: Some(10.0),
            budget_duration: Some("daily".to_string()),
            budget_window_start: Some(two_days_ago),
            rpm_limit: None,
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });
        assert!(matches!(
            tracker.on_request(&mut ctx).await,
            Action::Continue
        ));

        let updated = store.get_virtual_key(tracking_key).await.unwrap().unwrap();
        let updated_start = updated.budget_window_start.unwrap();
        assert!(
            updated_start >= now_before,
            "window start should be advanced and persisted"
        );
    }

    #[tokio::test]
    async fn test_virtual_key_uses_key_hash_for_tracking() {
        // When VirtualKeyMeta is present, tracking should use key_hash, not the raw API key.
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        let mut ctx = make_ctx("sk-trp-raw-key", &[b'x'; 400]);
        ctx.extensions.insert(VirtualKeyMeta {
            key_hash: "vk_specific_hash".to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: None,
            budget_duration: None,
            budget_window_start: None,
            rpm_limit: None,
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });

        tracker.on_request(&mut ctx).await;
        let mut resp_ctx = make_resp_ctx(800);
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        // Usage should be keyed by the hash, not the raw API key
        assert!(tracker.usage.get("vk_specific_hash").is_some());
        assert!(tracker.usage.get("sk-trp-raw-key").is_none());
    }

    // --- Webhook budget alert tests ---

    #[tokio::test]
    async fn test_webhook_threshold_detection() {
        // Pre-fill usage to 79%, accumulate cost past 80%, verify webhook_fired has entry.
        let tracker = CostTracker::new(10.0, 0.01, 0.02, 60).with_webhook(
            "http://localhost:9999/hook".into(),
            vec![50, 80, 100],
            86400,
        );

        // Pre-fill at 79% of 10.0 budget = 7.9
        tracker.usage.insert(
            "sk-webhook-test".into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 7.9,
            },
        );

        // Request that will add cost to push past 80%
        // 400 bytes body => 100 input tokens; 800 content-length => 200 output tokens
        // Cost: (100/1000)*0.01 + (200/1000)*0.02 = 0.001 + 0.004 = 0.005
        // Total: 7.9 + 0.005 = 7.905 → 79.05% → crosses 50% but not 80% yet?
        // Actually 7.905 / 10.0 = 79.05%. We need to push past 80%.
        // Let's use larger request: 4000 bytes => 1000 tokens, 8000 content-length => 2000 tokens
        // Cost: (1000/1000)*0.01 + (2000/1000)*0.02 = 0.01 + 0.04 = 0.05
        // Total: 7.9 + 0.05 = 7.95 → still not 80%. Let's just pre-fill to 7.9 and use big cost.
        // Pre-fill at 7.5 and use 0.6 cost to cross 80%.
        tracker.usage.insert(
            "sk-webhook-test".into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 7.5, // 75%
            },
        );

        // 4000 bytes => 1000 input tokens, 20000 content-length => 5000 output tokens
        // Cost: (1000/1000)*0.01 + (5000/1000)*0.02 = 0.01 + 0.10 = 0.11
        // Total: 7.5 + 0.11 = 7.61 → 76.1%, still under 80. Need bigger.
        // Pre-fill at 7.9, use big content-length for more output cost.
        tracker.usage.insert(
            "sk-webhook-test".into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 7.9, // 79%
            },
        );

        let body = br#"{"model":"gpt-4","messages":[{"role":"user","content":"x"}]}"#;
        let padded = {
            let mut v = body.to_vec();
            v.resize(4000, b' '); // 1000 input tokens
            v
        };
        let mut ctx = make_ctx("sk-webhook-test", &padded);
        tracker.on_request(&mut ctx).await;

        // 40000 content-length => 10000 output tokens
        // Cost: (1000/1000)*0.01 + (10000/1000)*0.02 = 0.01 + 0.20 = 0.21
        // Total: 7.9 + 0.21 = 8.11 → 81.1% → crosses 80%
        let mut resp_ctx = make_resp_ctx(40000);
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        // Verify that the 80% threshold was recorded in webhook_fired
        assert!(
            tracker
                .webhook_fired
                .contains_key(&("sk-webhook-test".to_string(), 80)),
            "webhook_fired should contain 80% threshold entry"
        );
        // 50% should also be recorded since total is > 50%
        assert!(
            tracker
                .webhook_fired
                .contains_key(&("sk-webhook-test".to_string(), 50)),
            "webhook_fired should contain 50% threshold entry"
        );
    }

    #[tokio::test]
    async fn test_webhook_dedup_within_ttl() {
        // Fire webhook once, accumulate more cost within TTL, verify dedup map doesn't grow.
        let tracker = CostTracker::new(10.0, 0.01, 0.02, 60).with_webhook(
            "http://localhost:9999/hook".into(),
            vec![50, 80, 100],
            86400,
        );

        // Pre-fill at 85% (already past 80%)
        tracker.usage.insert(
            "sk-dedup".into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 8.5,
            },
        );

        // First request — should mark 50% and 80% as fired
        let mut ctx = make_ctx("sk-dedup", &[b'x'; 400]);
        tracker.on_request(&mut ctx).await;
        let mut resp_ctx = make_resp_ctx(800);
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        let fired_count = tracker.webhook_fired.len();

        // Second request — same thresholds already fired, dedup map shouldn't grow
        let mut ctx2 = make_ctx("sk-dedup", &[b'x'; 400]);
        tracker.on_request(&mut ctx2).await;
        let mut resp_ctx2 = make_resp_ctx(800);
        tracker.on_response(&mut ctx2, &mut resp_ctx2).await;

        assert_eq!(
            tracker.webhook_fired.len(),
            fired_count,
            "dedup map should not grow for already-fired thresholds"
        );
    }

    #[tokio::test]
    async fn test_webhook_dedup_cleared_on_window_reset() {
        // Fire a webhook, trigger window expiry via on_request, verify dedup map is cleared.
        let tracker = CostTracker::new(100.0, 0.01, 0.02, 60).with_webhook(
            "http://localhost:9999/hook".into(),
            vec![50, 80, 100],
            86400,
        );

        let tracking_key = "vk_hash_webhook_reset";

        // Pre-fill usage and manually insert a fired entry
        tracker.usage.insert(
            tracking_key.into(),
            KeyUsage {
                total_input_tokens: 5000,
                total_output_tokens: 2500,
                total_cost: 60.0,
            },
        );
        tracker
            .webhook_fired
            .insert((tracking_key.to_string(), 50), std::time::Instant::now());

        assert!(tracker
            .webhook_fired
            .contains_key(&(tracking_key.to_string(), 50)));

        // Set up expired window (2 days ago for daily budget)
        let two_days_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - 2 * 24 * 3600;

        let mut ctx = make_ctx("sk-unused", b"hello");
        ctx.extensions.insert(VirtualKeyMeta {
            key_hash: tracking_key.to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: Some(100.0),
            budget_duration: Some("daily".to_string()),
            budget_window_start: Some(two_days_ago),
            rpm_limit: None,
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });

        // on_request should reset usage AND clear dedup entries for this key
        tracker.on_request(&mut ctx).await;

        assert!(
            !tracker
                .webhook_fired
                .contains_key(&(tracking_key.to_string(), 50)),
            "webhook_fired entries should be cleared after window reset"
        );
    }

    #[tokio::test]
    async fn test_webhook_not_fired_below_threshold() {
        // Usage at 40% with thresholds [50, 80, 100], verify webhook_fired is empty.
        let tracker = CostTracker::new(10.0, 0.01, 0.02, 60).with_webhook(
            "http://localhost:9999/hook".into(),
            vec![50, 80, 100],
            86400,
        );

        // Pre-fill at 35%
        tracker.usage.insert(
            "sk-below".into(),
            KeyUsage {
                total_input_tokens: 1000,
                total_output_tokens: 500,
                total_cost: 3.5,
            },
        );

        // Small request that keeps us under 50%
        let mut ctx = make_ctx("sk-below", &[b'x'; 400]); // 100 input tokens
        tracker.on_request(&mut ctx).await;
        let mut resp_ctx = make_resp_ctx(800); // 200 output tokens
                                               // Cost: (100/1000)*0.01 + (200/1000)*0.02 = 0.001 + 0.004 = 0.005
                                               // Total: 3.5 + 0.005 = 3.505 → 35.05% → under 50%
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        assert!(
            tracker.webhook_fired.is_empty(),
            "webhook_fired should be empty when below all thresholds"
        );
    }

    #[tokio::test]
    async fn test_webhook_multiple_thresholds_crossed() {
        // Jump from 0% to 90% in one request, verify both 50% and 80% entries in webhook_fired.
        let tracker = CostTracker::new(1.0, 0.01, 0.02, 60).with_webhook(
            "http://localhost:9999/hook".into(),
            vec![50, 80, 100],
            86400,
        );

        // No pre-fill — start at 0%
        // 4000 bytes => 1000 input tokens, 160000 content-length => 40000 output tokens
        // Cost: (1000/1000)*0.01 + (40000/1000)*0.02 = 0.01 + 0.80 = 0.81
        // 0.81 / 1.0 = 81% → crosses both 50% and 80%
        let body = br#"{"model":"gpt-4","messages":[{"role":"user","content":"x"}]}"#;
        let padded = {
            let mut v = body.to_vec();
            v.resize(4000, b' ');
            v
        };
        let mut ctx = make_ctx("sk-jump", &padded);
        tracker.on_request(&mut ctx).await;
        let mut resp_ctx = make_resp_ctx(160000);
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        assert!(
            tracker
                .webhook_fired
                .contains_key(&("sk-jump".to_string(), 50)),
            "should fire 50% threshold"
        );
        assert!(
            tracker
                .webhook_fired
                .contains_key(&("sk-jump".to_string(), 80)),
            "should fire 80% threshold"
        );
        // 100% should NOT be fired since we're at 81%
        assert!(
            !tracker
                .webhook_fired
                .contains_key(&("sk-jump".to_string(), 100)),
            "should not fire 100% threshold at 81%"
        );
    }

    #[tokio::test]
    async fn test_sse_response_skips_on_response_cost() {
        // SSE responses should not record cost in on_response (deferred to wrap_response_body).
        let tracker = CostTracker::new(0.0, 0.01, 0.02, 60);

        let mut ctx = make_ctx("sk-sse-test", &[b'x'; 400]);
        tracker.on_request(&mut ctx).await;

        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        let mut resp_ctx = ResponseContext {
            status: StatusCode::OK,
            headers,
            upstream: "https://api.openai.com".into(),
            duration: Duration::from_millis(100),
        };
        tracker.on_response(&mut ctx, &mut resp_ctx).await;

        // No usage should be recorded (meta left in place for wrap_response_body)
        assert!(
            tracker.usage.get("sk-sse-test").is_none(),
            "SSE on_response should not record usage"
        );

        // The LlmRequestMeta should still be in extensions
        assert!(
            ctx.extensions.get::<LlmRequestMeta>().is_some(),
            "meta should remain for wrap_response_body"
        );
    }
}
