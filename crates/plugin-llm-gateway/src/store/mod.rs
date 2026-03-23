#[cfg(feature = "store-mysql")]
pub mod mysql;
#[cfg(feature = "store-postgres")]
pub mod postgres;
pub mod schema;
#[cfg(feature = "store-sqlite")]
pub mod sqlite;

use async_trait::async_trait;

/// Per-key usage record stored in the database.
#[derive(Debug, Clone)]
pub struct KeyUsageRecord {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost: f64,
}

/// Per-model cost record stored in the database.
#[derive(Debug, Clone, Copy)]
pub struct ModelCostRecord {
    pub input_cost_per_1k: f64,
    pub output_cost_per_1k: f64,
}

/// Per-key per-model usage record.
#[derive(Debug, Clone)]
pub struct KeyModelUsageRecord {
    pub api_key: String,
    pub model: String,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost: f64,
}

/// Single request log entry.
#[derive(Debug, Clone)]
pub struct RequestLogEntry {
    pub timestamp_unix: i64,
    pub api_key: String,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub metadata_json: Option<String>,
    pub custom_cost_json: Option<String>,
    pub custom_cost_applied: bool,
    pub provider_name: Option<String>,
    pub prompt_name: Option<String>,
    pub prompt_version: Option<String>,
    pub prompt_environment: Option<String>,
    pub model: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost: f64,
    pub is_streaming: bool,
    pub safety_mode: Option<String>,
    pub safety_matches: Option<String>,
    pub semantic_policy_version: Option<String>,
    pub semantic_index_state: Option<String>,
    pub semantic_degraded_reason: Option<String>,
    pub semantic_findings: Option<String>,
    pub tool_trace: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct RequestLogQuery {
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub metadata_key: Option<String>,
    pub metadata_value: Option<String>,
    pub has_custom_cost: Option<bool>,
    pub custom_cost_applied: Option<bool>,
    pub limit: u32,
}

/// Durable session rollup and optional runtime state.
#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub session_id: String,
    pub project_id: Option<String>,
    pub project_ids_json: Option<String>,
    pub first_request_unix: Option<i64>,
    pub last_request_unix: Option<i64>,
    pub updated_at_unix: i64,
    pub request_count: u64,
    pub streaming_request_count: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost: f64,
    pub providers_json: Option<String>,
    pub models_json: Option<String>,
    pub prompt_names_json: Option<String>,
    pub prompt_versions_json: Option<String>,
    pub tool_names_json: Option<String>,
    pub latest_request_json: Option<String>,
    pub safety_event_count: u64,
    pub semantic_event_count: u64,
    pub semantic_degraded_count: u64,
    pub tool_call_count: u64,
    pub tool_error_count: u64,
    pub status: Option<String>,
    pub owner_id: Option<String>,
    pub owner_acquired_at_unix: Option<i64>,
    pub last_transition_at_unix: Option<i64>,
    pub last_transition_reason: Option<String>,
    pub last_heartbeat_unix: Option<i64>,
    pub lease_expires_at_unix: Option<i64>,
    pub cancel_requested_at_unix: Option<i64>,
    pub cancel_requested_by: Option<String>,
    pub cancel_reason: Option<String>,
    pub handoff_target_owner_id: Option<String>,
    pub handoff_requested_at_unix: Option<i64>,
    pub handoff_reason: Option<String>,
    pub state_json: Option<String>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionEventRecord {
    pub event_seq: i64,
    pub session_id: String,
    pub project_id: Option<String>,
    pub event_kind: String,
    pub actor_id: Option<String>,
    pub reason: Option<String>,
    pub payload_json: Option<String>,
    pub created_at_unix: i64,
}

#[derive(Debug, Clone, Default)]
pub struct SessionListQuery {
    pub project_id: Option<String>,
    pub status: Option<String>,
    pub owner_id: Option<String>,
    pub updated_after_unix: Option<i64>,
    pub limit: u32,
}

/// Virtual key record stored in the database.
#[derive(Debug, Clone)]
pub struct VirtualKeyRecord {
    pub key_hash: String,
    pub project_id: String,
    pub name: String,
    pub provider_name: String,
    pub budget_limit: Option<f64>,
    pub budget_duration: Option<String>,
    pub budget_window_start: Option<i64>,
    pub rpm_limit: Option<u32>,
    pub tpm_limit: Option<u32>,
    pub allowed_models: Option<String>,
    pub timeout_secs: Option<u64>,
    pub tool_approval_mode: Option<String>,
    pub allowed_tools: Option<String>,
    pub active: bool,
    pub created_at: String,
    pub expires_at: Option<String>,
}

/// Project-scoped defaults for runtime governance.
#[derive(Debug, Clone)]
pub struct ProjectPolicyRecord {
    pub project_id: String,
    pub budget_limit: Option<f64>,
    pub budget_duration: Option<String>,
    pub rpm_limit: Option<u32>,
    pub tpm_limit: Option<u32>,
    pub fallback_order: Option<String>,
    pub adaptive_enabled: bool,
    pub timeout_secs: Option<u64>,
    pub provider_rpm_limits: Option<String>,
    pub provider_tpm_limits: Option<String>,
    pub provider_timeouts: Option<String>,
    pub provider_input_costs: Option<String>,
    pub provider_output_costs: Option<String>,
    pub semantic_cache_enabled: Option<bool>,
    pub semantic_cache_ttl_secs: Option<u64>,
    pub semantic_cache_similarity_threshold: Option<f64>,
    pub tool_approval_mode: Option<String>,
    pub allowed_tools: Option<String>,
    pub updated_at: String,
}

/// Store-backed provider override or managed provider definition.
#[derive(Debug, Clone)]
pub struct ManagedProviderRecord {
    pub name: String,
    pub enabled: bool,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub models_json: Option<String>,
    pub api_key_header: Option<String>,
    pub timeout_secs: Option<u64>,
    pub family: Option<String>,
    pub surfaces_json: Option<String>,
    pub routing_metadata_json: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Declarative routing rule persisted in the store.
#[derive(Debug, Clone)]
pub struct RoutingRuleRecord {
    pub rule_id: String,
    pub project_id: String,
    pub name: String,
    pub priority: i32,
    pub enabled: bool,
    pub match_path: Option<String>,
    pub match_model: Option<String>,
    pub match_streaming: Option<bool>,
    pub match_role: Option<String>,
    pub match_headers: Option<String>,
    pub min_prompt_tokens: Option<u32>,
    pub max_prompt_tokens: Option<u32>,
    pub deny_reason: Option<String>,
    pub provider_order: Option<String>,
    pub provider_weights: Option<String>,
    pub timeout_secs: Option<u64>,
    pub created_at: String,
}

/// Sensitive-data policy per project.
#[derive(Debug, Clone)]
pub struct SafetyPolicyRecord {
    pub project_id: String,
    pub mode: String,
    pub rules_json: Option<String>,
    pub updated_at: String,
}

/// Project-scoped semantic safety policy.
#[derive(Debug, Clone)]
pub struct ProjectSemanticPolicyRecord {
    pub project_id: String,
    pub version: String,
    pub enabled: bool,
    pub entities_json: Option<String>,
    pub topics_json: Option<String>,
    pub updated_at: String,
}

/// Durable warmed prompt-cache routing hint.
#[derive(Debug, Clone)]
pub struct PromptCacheRouteRecord {
    pub route_id: String,
    pub project_id: String,
    pub cache_key: String,
    pub provider_name: String,
    pub signal_kind: String,
    pub signal_strength: f64,
    pub observed_at_ms: u64,
    pub expires_at_ms: u64,
}

/// Durable semantic cache entry for restart-safe cache reuse.
#[derive(Debug, Clone)]
pub struct SemanticCacheEntryRecord {
    pub cache_id: String,
    pub project_id: String,
    pub provider_name: String,
    pub model: String,
    pub tokens_json: String,
    pub response_status: u16,
    pub content_type: Option<String>,
    pub response_body: Vec<u8>,
    pub prompt_tokens: u64,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
}

/// Project-scoped registered tool.
#[derive(Debug, Clone)]
pub struct ProjectToolRecord {
    pub project_id: String,
    pub tool_name: String,
    pub description: Option<String>,
    pub input_schema_json: String,
    pub executor_kind: String,
    pub executor_config_json: Option<String>,
    pub enabled: bool,
    pub timeout_ms: Option<u64>,
    pub updated_at: String,
}

/// Project-scoped versioned prompt template.
#[derive(Debug, Clone)]
pub struct ProjectPromptRecord {
    pub project_id: String,
    pub prompt_name: String,
    pub version: String,
    pub environment: String,
    pub description: Option<String>,
    pub target: String,
    pub template_text: String,
    pub variables_schema_json: Option<String>,
    pub rollout_metadata_json: Option<String>,
    pub active: bool,
    pub updated_at: String,
}

/// Project-scoped saved rollout policy for eval comparisons and prompt promotion.
#[derive(Debug, Clone)]
pub struct ProjectRolloutPolicyRecord {
    pub project_id: String,
    pub policy_name: String,
    pub description: Option<String>,
    pub gate_config_json: String,
    pub target_environment: Option<String>,
    pub updated_at: String,
}

/// Persisted prompt rollout workflow derived from eval comparisons and a saved rollout policy.
#[derive(Debug, Clone)]
pub struct ProjectPromptRolloutRecord {
    pub project_id: String,
    pub prompt_name: String,
    pub rollout_id: String,
    pub policy_name: String,
    pub baseline_version: Option<String>,
    pub candidate_version: String,
    pub baseline_run_id: String,
    pub candidate_run_id: String,
    pub target_environment: Option<String>,
    pub status: String,
    pub recommendation_action: Option<String>,
    pub comparison_json: String,
    pub created_at: String,
    pub applied_at: Option<String>,
}

/// Project-scoped dataset metadata for replay/eval workflows.
#[derive(Debug, Clone)]
pub struct ProjectDatasetRecord {
    pub project_id: String,
    pub dataset_name: String,
    pub description: Option<String>,
    pub schema_json: Option<String>,
    pub updated_at: String,
}

/// Single dataset item for replay/eval workflows.
#[derive(Debug, Clone)]
pub struct ProjectDatasetItemRecord {
    pub project_id: String,
    pub dataset_name: String,
    pub item_id: String,
    pub input_json: String,
    pub expected_output_json: Option<String>,
    pub metadata_json: Option<String>,
    pub updated_at: String,
}

/// Summary row for a persisted eval run over a dataset.
#[derive(Debug, Clone)]
pub struct ProjectEvalRunRecord {
    pub run_id: String,
    pub project_id: String,
    pub dataset_name: String,
    pub target_url: String,
    pub status: String,
    pub total_items: u32,
    pub passed_items: u32,
    pub failed_items: u32,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost: f64,
    pub average_latency_ms: f64,
    pub summary_json: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

/// Per-item result for a persisted eval run.
#[derive(Debug, Clone)]
pub struct ProjectEvalRunItemRecord {
    pub run_id: String,
    pub project_id: String,
    pub dataset_name: String,
    pub item_id: String,
    pub passed: bool,
    pub status_code: Option<u16>,
    pub latency_ms: u64,
    pub output_text: Option<String>,
    pub evaluation_json: Option<String>,
    pub error: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost: f64,
    pub created_at: String,
}

/// Durable project governance change event for history/audit views.
#[derive(Debug, Clone)]
pub struct GovernanceChangeRecord {
    pub change_id: String,
    pub project_id: String,
    pub resource_type: String,
    pub resource_id: String,
    pub action: String,
    pub before_json: Option<String>,
    pub after_json: Option<String>,
    pub changed_at: String,
}

/// Trait abstracting the persistent storage backend.
#[async_trait]
pub trait GatewayStore: Send + Sync {
    // Usage
    async fn get_usage(&self, api_key: &str) -> Result<Option<KeyUsageRecord>, StoreError>;
    async fn get_all_usage(&self) -> Result<Vec<(String, KeyUsageRecord)>, StoreError>;
    async fn upsert_usage(&self, api_key: &str, usage: &KeyUsageRecord) -> Result<(), StoreError>;
    async fn delete_usage(&self, api_key: &str) -> Result<bool, StoreError>;
    async fn delete_all_usage(&self) -> Result<(), StoreError>;

    // Model pricing
    async fn get_model_cost(&self, model: &str) -> Result<Option<ModelCostRecord>, StoreError>;
    async fn get_all_model_costs(&self) -> Result<Vec<(String, ModelCostRecord)>, StoreError>;
    async fn upsert_model_cost(
        &self,
        model: &str,
        cost: &ModelCostRecord,
    ) -> Result<(), StoreError>;
    async fn delete_model_cost(&self, model: &str) -> Result<bool, StoreError>;

    // Per-model usage
    async fn get_all_per_model_usage(&self) -> Result<Vec<KeyModelUsageRecord>, StoreError>;
    async fn upsert_per_model_usage(&self, record: &KeyModelUsageRecord) -> Result<(), StoreError>;
    async fn delete_all_per_model_usage(&self) -> Result<(), StoreError>;

    // Request log
    async fn append_request_logs(&self, entries: &[RequestLogEntry]) -> Result<(), StoreError>;
    async fn get_request_logs(
        &self,
        api_key: Option<&str>,
        model: Option<&str>,
        project_id: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RequestLogEntry>, StoreError>;
    async fn get_request_logs_for_session(
        &self,
        session_id: &str,
        project_id: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RequestLogEntry>, StoreError>;
    async fn query_request_logs(
        &self,
        query: &RequestLogQuery,
    ) -> Result<Vec<RequestLogEntry>, StoreError>;
    async fn get_session(&self, session_id: &str) -> Result<Option<SessionRecord>, StoreError>;
    async fn list_sessions(
        &self,
        query: &SessionListQuery,
    ) -> Result<Vec<SessionRecord>, StoreError>;
    async fn list_sessions_for_recovery(
        &self,
        now_unix: i64,
        limit: u32,
    ) -> Result<Vec<String>, StoreError>;
    async fn upsert_session(&self, record: &SessionRecord) -> Result<(), StoreError>;
    async fn append_session_event(&self, record: &SessionEventRecord) -> Result<i64, StoreError>;
    async fn get_session_events(
        &self,
        session_id: &str,
        after_seq: Option<i64>,
        limit: u32,
    ) -> Result<Vec<SessionEventRecord>, StoreError>;

    // Virtual keys
    async fn get_virtual_key(&self, key_hash: &str)
        -> Result<Option<VirtualKeyRecord>, StoreError>;
    async fn get_all_virtual_keys(&self) -> Result<Vec<VirtualKeyRecord>, StoreError>;
    async fn upsert_virtual_key(&self, record: &VirtualKeyRecord) -> Result<(), StoreError>;
    async fn delete_virtual_key(&self, key_hash: &str) -> Result<bool, StoreError>;
    async fn update_virtual_key_budget_window(
        &self,
        key_hash: &str,
        window_start: i64,
    ) -> Result<(), StoreError>;

    // Project policy defaults
    async fn get_project_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<ProjectPolicyRecord>, StoreError>;
    async fn get_all_project_policies(&self) -> Result<Vec<ProjectPolicyRecord>, StoreError>;
    async fn upsert_project_policy(&self, record: &ProjectPolicyRecord) -> Result<(), StoreError>;
    async fn delete_project_policy(&self, project_id: &str) -> Result<bool, StoreError>;

    // Managed provider overlays / dynamic providers
    async fn get_managed_provider(
        &self,
        name: &str,
    ) -> Result<Option<ManagedProviderRecord>, StoreError>;
    async fn get_managed_providers(&self) -> Result<Vec<ManagedProviderRecord>, StoreError>;
    async fn upsert_managed_provider(
        &self,
        record: &ManagedProviderRecord,
    ) -> Result<(), StoreError>;
    async fn delete_managed_provider(&self, name: &str) -> Result<bool, StoreError>;

    // Routing rules
    async fn get_routing_rules(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<RoutingRuleRecord>, StoreError>;
    async fn upsert_routing_rule(&self, record: &RoutingRuleRecord) -> Result<(), StoreError>;
    async fn delete_routing_rule(&self, rule_id: &str) -> Result<bool, StoreError>;

    // Safety policies
    async fn get_safety_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<SafetyPolicyRecord>, StoreError>;
    async fn get_all_safety_policies(&self) -> Result<Vec<SafetyPolicyRecord>, StoreError>;
    async fn upsert_safety_policy(&self, record: &SafetyPolicyRecord) -> Result<(), StoreError>;
    async fn delete_safety_policy(&self, project_id: &str) -> Result<bool, StoreError>;

    // Semantic safety policies
    async fn get_semantic_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<ProjectSemanticPolicyRecord>, StoreError>;
    async fn get_all_semantic_policies(
        &self,
    ) -> Result<Vec<ProjectSemanticPolicyRecord>, StoreError>;
    async fn upsert_semantic_policy(
        &self,
        record: &ProjectSemanticPolicyRecord,
    ) -> Result<(), StoreError>;
    async fn delete_semantic_policy(&self, project_id: &str) -> Result<bool, StoreError>;

    // Prompt-cache routing memory persistence
    async fn get_prompt_cache_routes(
        &self,
        now_ms: u64,
        limit: u32,
    ) -> Result<Vec<PromptCacheRouteRecord>, StoreError>;
    async fn upsert_prompt_cache_route(
        &self,
        record: &PromptCacheRouteRecord,
    ) -> Result<(), StoreError>;
    async fn delete_prompt_cache_route(&self, route_id: &str) -> Result<bool, StoreError>;
    async fn prune_prompt_cache_routes(&self, now_ms: u64) -> Result<(), StoreError>;

    // Semantic cache persistence
    async fn get_semantic_cache_entries(
        &self,
        now_ms: u64,
        limit: u32,
    ) -> Result<Vec<SemanticCacheEntryRecord>, StoreError>;
    async fn upsert_semantic_cache_entry(
        &self,
        record: &SemanticCacheEntryRecord,
    ) -> Result<(), StoreError>;
    async fn prune_semantic_cache_entries(
        &self,
        now_ms: u64,
        max_entries: u32,
    ) -> Result<(), StoreError>;

    // Project tools
    async fn get_project_tool(
        &self,
        project_id: &str,
        tool_name: &str,
    ) -> Result<Option<ProjectToolRecord>, StoreError>;
    async fn get_project_tools(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<ProjectToolRecord>, StoreError>;
    async fn upsert_project_tool(&self, record: &ProjectToolRecord) -> Result<(), StoreError>;
    async fn delete_project_tool(
        &self,
        project_id: &str,
        tool_name: &str,
    ) -> Result<bool, StoreError>;

    // Project prompts
    async fn get_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Result<Option<ProjectPromptRecord>, StoreError>;
    async fn get_project_prompts(
        &self,
        project_id: Option<&str>,
        prompt_name: Option<&str>,
    ) -> Result<Vec<ProjectPromptRecord>, StoreError>;
    async fn upsert_project_prompt(&self, record: &ProjectPromptRecord) -> Result<(), StoreError>;
    async fn delete_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Result<bool, StoreError>;

    // Project rollout policies
    async fn get_project_rollout_policy(
        &self,
        project_id: &str,
        policy_name: &str,
    ) -> Result<Option<ProjectRolloutPolicyRecord>, StoreError>;
    async fn get_project_rollout_policies(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<ProjectRolloutPolicyRecord>, StoreError>;
    async fn upsert_project_rollout_policy(
        &self,
        record: &ProjectRolloutPolicyRecord,
    ) -> Result<(), StoreError>;
    async fn delete_project_rollout_policy(
        &self,
        project_id: &str,
        policy_name: &str,
    ) -> Result<bool, StoreError>;

    // Prompt rollout workflows
    async fn get_project_prompt_rollout(
        &self,
        project_id: &str,
        prompt_name: &str,
        rollout_id: &str,
    ) -> Result<Option<ProjectPromptRolloutRecord>, StoreError>;
    async fn get_project_prompt_rollouts(
        &self,
        project_id: &str,
        prompt_name: &str,
    ) -> Result<Vec<ProjectPromptRolloutRecord>, StoreError>;
    async fn upsert_project_prompt_rollout(
        &self,
        record: &ProjectPromptRolloutRecord,
    ) -> Result<(), StoreError>;

    // Project datasets
    async fn get_project_dataset(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Result<Option<ProjectDatasetRecord>, StoreError>;
    async fn get_project_datasets(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<ProjectDatasetRecord>, StoreError>;
    async fn upsert_project_dataset(&self, record: &ProjectDatasetRecord)
        -> Result<(), StoreError>;
    async fn delete_project_dataset(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Result<bool, StoreError>;

    async fn get_project_dataset_item(
        &self,
        project_id: &str,
        dataset_name: &str,
        item_id: &str,
    ) -> Result<Option<ProjectDatasetItemRecord>, StoreError>;
    async fn get_project_dataset_items(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Result<Vec<ProjectDatasetItemRecord>, StoreError>;
    async fn upsert_project_dataset_item(
        &self,
        record: &ProjectDatasetItemRecord,
    ) -> Result<(), StoreError>;
    async fn delete_project_dataset_item(
        &self,
        project_id: &str,
        dataset_name: &str,
        item_id: &str,
    ) -> Result<bool, StoreError>;

    // Project eval runs
    async fn get_project_eval_run(
        &self,
        project_id: &str,
        run_id: &str,
    ) -> Result<Option<ProjectEvalRunRecord>, StoreError>;
    async fn get_project_eval_runs(
        &self,
        project_id: &str,
        dataset_name: Option<&str>,
    ) -> Result<Vec<ProjectEvalRunRecord>, StoreError>;
    async fn get_project_eval_runs_by_status(
        &self,
        statuses: &[&str],
        limit: u32,
    ) -> Result<Vec<ProjectEvalRunRecord>, StoreError>;
    async fn upsert_project_eval_run(
        &self,
        record: &ProjectEvalRunRecord,
    ) -> Result<(), StoreError>;
    async fn get_project_eval_run_items(
        &self,
        project_id: &str,
        run_id: &str,
    ) -> Result<Vec<ProjectEvalRunItemRecord>, StoreError>;
    async fn upsert_project_eval_run_item(
        &self,
        record: &ProjectEvalRunItemRecord,
    ) -> Result<(), StoreError>;

    // Governance history
    async fn append_governance_change(
        &self,
        record: &GovernanceChangeRecord,
    ) -> Result<(), StoreError>;
    async fn get_governance_changes(
        &self,
        project_id: &str,
        resource_type: Option<&str>,
        limit: u32,
    ) -> Result<Vec<GovernanceChangeRecord>, StoreError>;
}

/// Errors from the storage layer.
#[derive(Debug)]
pub enum StoreError {
    Db(String),
    Other(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Db(msg) => write!(f, "database error: {}", msg),
            StoreError::Other(msg) => write!(f, "store error: {}", msg),
        }
    }
}

impl std::error::Error for StoreError {}

/// Enum dispatch over configured store backends.
pub enum Store {
    #[cfg(feature = "store-sqlite")]
    Sqlite(sqlite::SqliteStore),
    #[cfg(feature = "store-postgres")]
    Postgres(postgres::PostgresStore),
    #[cfg(feature = "store-mysql")]
    Mysql(mysql::MysqlStore),
}

#[async_trait]
impl GatewayStore for Store {
    async fn get_usage(&self, api_key: &str) -> Result<Option<KeyUsageRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_usage(api_key).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_usage(api_key).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_usage(api_key).await,
        }
    }

    async fn get_all_usage(&self) -> Result<Vec<(String, KeyUsageRecord)>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_usage().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_usage().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_usage().await,
        }
    }

    async fn upsert_usage(&self, api_key: &str, usage: &KeyUsageRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_usage(api_key, usage).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_usage(api_key, usage).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_usage(api_key, usage).await,
        }
    }

    async fn delete_usage(&self, api_key: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_usage(api_key).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_usage(api_key).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_usage(api_key).await,
        }
    }

    async fn delete_all_usage(&self) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_all_usage().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_all_usage().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_all_usage().await,
        }
    }

    async fn get_model_cost(&self, model: &str) -> Result<Option<ModelCostRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_model_cost(model).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_model_cost(model).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_model_cost(model).await,
        }
    }

    async fn get_all_model_costs(&self) -> Result<Vec<(String, ModelCostRecord)>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_model_costs().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_model_costs().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_model_costs().await,
        }
    }

    async fn upsert_model_cost(
        &self,
        model: &str,
        cost: &ModelCostRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_model_cost(model, cost).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_model_cost(model, cost).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_model_cost(model, cost).await,
        }
    }

    async fn delete_model_cost(&self, model: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_model_cost(model).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_model_cost(model).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_model_cost(model).await,
        }
    }

    async fn get_all_per_model_usage(&self) -> Result<Vec<KeyModelUsageRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_per_model_usage().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_per_model_usage().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_per_model_usage().await,
        }
    }

    async fn upsert_per_model_usage(&self, record: &KeyModelUsageRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_per_model_usage(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_per_model_usage(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_per_model_usage(record).await,
        }
    }

    async fn delete_all_per_model_usage(&self) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_all_per_model_usage().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_all_per_model_usage().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_all_per_model_usage().await,
        }
    }

    async fn append_request_logs(&self, entries: &[RequestLogEntry]) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.append_request_logs(entries).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.append_request_logs(entries).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.append_request_logs(entries).await,
        }
    }

    async fn get_request_logs(
        &self,
        api_key: Option<&str>,
        model: Option<&str>,
        project_id: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RequestLogEntry>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_request_logs(api_key, model, project_id, limit).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_request_logs(api_key, model, project_id, limit).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_request_logs(api_key, model, project_id, limit).await,
        }
    }

    async fn get_request_logs_for_session(
        &self,
        session_id: &str,
        project_id: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RequestLogEntry>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => {
                s.get_request_logs_for_session(session_id, project_id, limit)
                    .await
            }
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => {
                s.get_request_logs_for_session(session_id, project_id, limit)
                    .await
            }
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => {
                s.get_request_logs_for_session(session_id, project_id, limit)
                    .await
            }
        }
    }

    async fn query_request_logs(
        &self,
        query: &RequestLogQuery,
    ) -> Result<Vec<RequestLogEntry>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.query_request_logs(query).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.query_request_logs(query).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.query_request_logs(query).await,
        }
    }

    async fn get_session(&self, session_id: &str) -> Result<Option<SessionRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_session(session_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_session(session_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_session(session_id).await,
        }
    }

    async fn list_sessions(
        &self,
        query: &SessionListQuery,
    ) -> Result<Vec<SessionRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.list_sessions(query).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.list_sessions(query).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.list_sessions(query).await,
        }
    }

    async fn list_sessions_for_recovery(
        &self,
        now_unix: i64,
        limit: u32,
    ) -> Result<Vec<String>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.list_sessions_for_recovery(now_unix, limit).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.list_sessions_for_recovery(now_unix, limit).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.list_sessions_for_recovery(now_unix, limit).await,
        }
    }

    async fn upsert_session(&self, record: &SessionRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_session(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_session(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_session(record).await,
        }
    }

    async fn append_session_event(&self, record: &SessionEventRecord) -> Result<i64, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.append_session_event(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.append_session_event(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.append_session_event(record).await,
        }
    }

    async fn get_session_events(
        &self,
        session_id: &str,
        after_seq: Option<i64>,
        limit: u32,
    ) -> Result<Vec<SessionEventRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_session_events(session_id, after_seq, limit).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_session_events(session_id, after_seq, limit).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_session_events(session_id, after_seq, limit).await,
        }
    }

    async fn get_virtual_key(
        &self,
        key_hash: &str,
    ) -> Result<Option<VirtualKeyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_virtual_key(key_hash).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_virtual_key(key_hash).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_virtual_key(key_hash).await,
        }
    }

    async fn get_all_virtual_keys(&self) -> Result<Vec<VirtualKeyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_virtual_keys().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_virtual_keys().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_virtual_keys().await,
        }
    }

    async fn upsert_virtual_key(&self, record: &VirtualKeyRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_virtual_key(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_virtual_key(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_virtual_key(record).await,
        }
    }

    async fn delete_virtual_key(&self, key_hash: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_virtual_key(key_hash).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_virtual_key(key_hash).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_virtual_key(key_hash).await,
        }
    }

    async fn update_virtual_key_budget_window(
        &self,
        key_hash: &str,
        window_start: i64,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => {
                s.update_virtual_key_budget_window(key_hash, window_start)
                    .await
            }
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => {
                s.update_virtual_key_budget_window(key_hash, window_start)
                    .await
            }
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => {
                s.update_virtual_key_budget_window(key_hash, window_start)
                    .await
            }
        }
    }

    async fn get_project_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<ProjectPolicyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_policy(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_policy(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_policy(project_id).await,
        }
    }

    async fn get_all_project_policies(&self) -> Result<Vec<ProjectPolicyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_project_policies().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_project_policies().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_project_policies().await,
        }
    }

    async fn upsert_project_policy(&self, record: &ProjectPolicyRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project_policy(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project_policy(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project_policy(record).await,
        }
    }

    async fn delete_project_policy(&self, project_id: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_project_policy(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_project_policy(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_project_policy(project_id).await,
        }
    }

    async fn get_managed_provider(
        &self,
        name: &str,
    ) -> Result<Option<ManagedProviderRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_managed_provider(name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_managed_provider(name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_managed_provider(name).await,
        }
    }

    async fn get_managed_providers(&self) -> Result<Vec<ManagedProviderRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_managed_providers().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_managed_providers().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_managed_providers().await,
        }
    }

    async fn upsert_managed_provider(
        &self,
        record: &ManagedProviderRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_managed_provider(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_managed_provider(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_managed_provider(record).await,
        }
    }

    async fn delete_managed_provider(&self, name: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_managed_provider(name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_managed_provider(name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_managed_provider(name).await,
        }
    }

    async fn get_routing_rules(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<RoutingRuleRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_routing_rules(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_routing_rules(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_routing_rules(project_id).await,
        }
    }

    async fn upsert_routing_rule(&self, record: &RoutingRuleRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_routing_rule(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_routing_rule(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_routing_rule(record).await,
        }
    }

    async fn delete_routing_rule(&self, rule_id: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_routing_rule(rule_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_routing_rule(rule_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_routing_rule(rule_id).await,
        }
    }

    async fn get_safety_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<SafetyPolicyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_safety_policy(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_safety_policy(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_safety_policy(project_id).await,
        }
    }

    async fn get_all_safety_policies(&self) -> Result<Vec<SafetyPolicyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_safety_policies().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_safety_policies().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_safety_policies().await,
        }
    }

    async fn upsert_safety_policy(&self, record: &SafetyPolicyRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_safety_policy(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_safety_policy(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_safety_policy(record).await,
        }
    }

    async fn delete_safety_policy(&self, project_id: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_safety_policy(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_safety_policy(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_safety_policy(project_id).await,
        }
    }

    async fn get_semantic_policy(
        &self,
        project_id: &str,
    ) -> Result<Option<ProjectSemanticPolicyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_semantic_policy(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_semantic_policy(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_semantic_policy(project_id).await,
        }
    }

    async fn get_all_semantic_policies(
        &self,
    ) -> Result<Vec<ProjectSemanticPolicyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_all_semantic_policies().await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_all_semantic_policies().await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_all_semantic_policies().await,
        }
    }

    async fn upsert_semantic_policy(
        &self,
        record: &ProjectSemanticPolicyRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_semantic_policy(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_semantic_policy(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_semantic_policy(record).await,
        }
    }

    async fn delete_semantic_policy(&self, project_id: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_semantic_policy(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_semantic_policy(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_semantic_policy(project_id).await,
        }
    }

    async fn get_prompt_cache_routes(
        &self,
        now_ms: u64,
        limit: u32,
    ) -> Result<Vec<PromptCacheRouteRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_prompt_cache_routes(now_ms, limit).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_prompt_cache_routes(now_ms, limit).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_prompt_cache_routes(now_ms, limit).await,
        }
    }

    async fn upsert_prompt_cache_route(
        &self,
        record: &PromptCacheRouteRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_prompt_cache_route(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_prompt_cache_route(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_prompt_cache_route(record).await,
        }
    }

    async fn delete_prompt_cache_route(&self, route_id: &str) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_prompt_cache_route(route_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_prompt_cache_route(route_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_prompt_cache_route(route_id).await,
        }
    }

    async fn prune_prompt_cache_routes(&self, now_ms: u64) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.prune_prompt_cache_routes(now_ms).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.prune_prompt_cache_routes(now_ms).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.prune_prompt_cache_routes(now_ms).await,
        }
    }

    async fn get_semantic_cache_entries(
        &self,
        now_ms: u64,
        limit: u32,
    ) -> Result<Vec<SemanticCacheEntryRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_semantic_cache_entries(now_ms, limit).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_semantic_cache_entries(now_ms, limit).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_semantic_cache_entries(now_ms, limit).await,
        }
    }

    async fn upsert_semantic_cache_entry(
        &self,
        record: &SemanticCacheEntryRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_semantic_cache_entry(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_semantic_cache_entry(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_semantic_cache_entry(record).await,
        }
    }

    async fn prune_semantic_cache_entries(
        &self,
        now_ms: u64,
        max_entries: u32,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.prune_semantic_cache_entries(now_ms, max_entries).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.prune_semantic_cache_entries(now_ms, max_entries).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.prune_semantic_cache_entries(now_ms, max_entries).await,
        }
    }

    async fn get_project_tool(
        &self,
        project_id: &str,
        tool_name: &str,
    ) -> Result<Option<ProjectToolRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_tool(project_id, tool_name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_tool(project_id, tool_name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_tool(project_id, tool_name).await,
        }
    }

    async fn get_project_tools(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<ProjectToolRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_tools(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_tools(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_tools(project_id).await,
        }
    }

    async fn upsert_project_tool(&self, record: &ProjectToolRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project_tool(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project_tool(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project_tool(record).await,
        }
    }

    async fn delete_project_tool(
        &self,
        project_id: &str,
        tool_name: &str,
    ) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_project_tool(project_id, tool_name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_project_tool(project_id, tool_name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_project_tool(project_id, tool_name).await,
        }
    }

    async fn get_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Result<Option<ProjectPromptRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_prompt(project_id, prompt_name, version).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_prompt(project_id, prompt_name, version).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_prompt(project_id, prompt_name, version).await,
        }
    }

    async fn get_project_prompts(
        &self,
        project_id: Option<&str>,
        prompt_name: Option<&str>,
    ) -> Result<Vec<ProjectPromptRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_prompts(project_id, prompt_name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_prompts(project_id, prompt_name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_prompts(project_id, prompt_name).await,
        }
    }

    async fn upsert_project_prompt(&self, record: &ProjectPromptRecord) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project_prompt(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project_prompt(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project_prompt(record).await,
        }
    }

    async fn delete_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => {
                s.delete_project_prompt(project_id, prompt_name, version)
                    .await
            }
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => {
                s.delete_project_prompt(project_id, prompt_name, version)
                    .await
            }
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => {
                s.delete_project_prompt(project_id, prompt_name, version)
                    .await
            }
        }
    }

    async fn get_project_rollout_policy(
        &self,
        project_id: &str,
        policy_name: &str,
    ) -> Result<Option<ProjectRolloutPolicyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_rollout_policy(project_id, policy_name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_rollout_policy(project_id, policy_name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_rollout_policy(project_id, policy_name).await,
        }
    }

    async fn get_project_rollout_policies(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<ProjectRolloutPolicyRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_rollout_policies(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_rollout_policies(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_rollout_policies(project_id).await,
        }
    }

    async fn upsert_project_rollout_policy(
        &self,
        record: &ProjectRolloutPolicyRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project_rollout_policy(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project_rollout_policy(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project_rollout_policy(record).await,
        }
    }

    async fn delete_project_rollout_policy(
        &self,
        project_id: &str,
        policy_name: &str,
    ) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => {
                s.delete_project_rollout_policy(project_id, policy_name)
                    .await
            }
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => {
                s.delete_project_rollout_policy(project_id, policy_name)
                    .await
            }
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => {
                s.delete_project_rollout_policy(project_id, policy_name)
                    .await
            }
        }
    }

    async fn get_project_prompt_rollout(
        &self,
        project_id: &str,
        prompt_name: &str,
        rollout_id: &str,
    ) -> Result<Option<ProjectPromptRolloutRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => {
                s.get_project_prompt_rollout(project_id, prompt_name, rollout_id)
                    .await
            }
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => {
                s.get_project_prompt_rollout(project_id, prompt_name, rollout_id)
                    .await
            }
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => {
                s.get_project_prompt_rollout(project_id, prompt_name, rollout_id)
                    .await
            }
        }
    }

    async fn get_project_prompt_rollouts(
        &self,
        project_id: &str,
        prompt_name: &str,
    ) -> Result<Vec<ProjectPromptRolloutRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_prompt_rollouts(project_id, prompt_name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_prompt_rollouts(project_id, prompt_name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_prompt_rollouts(project_id, prompt_name).await,
        }
    }

    async fn upsert_project_prompt_rollout(
        &self,
        record: &ProjectPromptRolloutRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project_prompt_rollout(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project_prompt_rollout(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project_prompt_rollout(record).await,
        }
    }

    async fn get_project_dataset(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Result<Option<ProjectDatasetRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_dataset(project_id, dataset_name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_dataset(project_id, dataset_name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_dataset(project_id, dataset_name).await,
        }
    }

    async fn get_project_datasets(
        &self,
        project_id: Option<&str>,
    ) -> Result<Vec<ProjectDatasetRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_datasets(project_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_datasets(project_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_datasets(project_id).await,
        }
    }

    async fn upsert_project_dataset(
        &self,
        record: &ProjectDatasetRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project_dataset(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project_dataset(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project_dataset(record).await,
        }
    }

    async fn delete_project_dataset(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.delete_project_dataset(project_id, dataset_name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.delete_project_dataset(project_id, dataset_name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.delete_project_dataset(project_id, dataset_name).await,
        }
    }

    async fn get_project_dataset_item(
        &self,
        project_id: &str,
        dataset_name: &str,
        item_id: &str,
    ) -> Result<Option<ProjectDatasetItemRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => {
                s.get_project_dataset_item(project_id, dataset_name, item_id)
                    .await
            }
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => {
                s.get_project_dataset_item(project_id, dataset_name, item_id)
                    .await
            }
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => {
                s.get_project_dataset_item(project_id, dataset_name, item_id)
                    .await
            }
        }
    }

    async fn get_project_dataset_items(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Result<Vec<ProjectDatasetItemRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_dataset_items(project_id, dataset_name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_dataset_items(project_id, dataset_name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_dataset_items(project_id, dataset_name).await,
        }
    }

    async fn upsert_project_dataset_item(
        &self,
        record: &ProjectDatasetItemRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project_dataset_item(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project_dataset_item(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project_dataset_item(record).await,
        }
    }

    async fn delete_project_dataset_item(
        &self,
        project_id: &str,
        dataset_name: &str,
        item_id: &str,
    ) -> Result<bool, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => {
                s.delete_project_dataset_item(project_id, dataset_name, item_id)
                    .await
            }
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => {
                s.delete_project_dataset_item(project_id, dataset_name, item_id)
                    .await
            }
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => {
                s.delete_project_dataset_item(project_id, dataset_name, item_id)
                    .await
            }
        }
    }

    async fn get_project_eval_run(
        &self,
        project_id: &str,
        run_id: &str,
    ) -> Result<Option<ProjectEvalRunRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_eval_run(project_id, run_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_eval_run(project_id, run_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_eval_run(project_id, run_id).await,
        }
    }

    async fn get_project_eval_runs(
        &self,
        project_id: &str,
        dataset_name: Option<&str>,
    ) -> Result<Vec<ProjectEvalRunRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_eval_runs(project_id, dataset_name).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_eval_runs(project_id, dataset_name).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_eval_runs(project_id, dataset_name).await,
        }
    }

    async fn get_project_eval_runs_by_status(
        &self,
        statuses: &[&str],
        limit: u32,
    ) -> Result<Vec<ProjectEvalRunRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_eval_runs_by_status(statuses, limit).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_eval_runs_by_status(statuses, limit).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_eval_runs_by_status(statuses, limit).await,
        }
    }

    async fn upsert_project_eval_run(
        &self,
        record: &ProjectEvalRunRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project_eval_run(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project_eval_run(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project_eval_run(record).await,
        }
    }

    async fn get_project_eval_run_items(
        &self,
        project_id: &str,
        run_id: &str,
    ) -> Result<Vec<ProjectEvalRunItemRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.get_project_eval_run_items(project_id, run_id).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.get_project_eval_run_items(project_id, run_id).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.get_project_eval_run_items(project_id, run_id).await,
        }
    }

    async fn upsert_project_eval_run_item(
        &self,
        record: &ProjectEvalRunItemRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.upsert_project_eval_run_item(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.upsert_project_eval_run_item(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.upsert_project_eval_run_item(record).await,
        }
    }

    async fn append_governance_change(
        &self,
        record: &GovernanceChangeRecord,
    ) -> Result<(), StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => s.append_governance_change(record).await,
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => s.append_governance_change(record).await,
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => s.append_governance_change(record).await,
        }
    }

    async fn get_governance_changes(
        &self,
        project_id: &str,
        resource_type: Option<&str>,
        limit: u32,
    ) -> Result<Vec<GovernanceChangeRecord>, StoreError> {
        match self {
            #[cfg(feature = "store-sqlite")]
            Store::Sqlite(s) => {
                s.get_governance_changes(project_id, resource_type, limit)
                    .await
            }
            #[cfg(feature = "store-postgres")]
            Store::Postgres(s) => {
                s.get_governance_changes(project_id, resource_type, limit)
                    .await
            }
            #[cfg(feature = "store-mysql")]
            Store::Mysql(s) => {
                s.get_governance_changes(project_id, resource_type, limit)
                    .await
            }
        }
    }
}

/// Connect to a store from a URL string.
///
/// Supported schemes:
/// - `sqlite://...` (requires `store-sqlite` feature)
/// - `postgres://...` (requires `store-postgres` feature)
/// - `mysql://...` (requires `store-mysql` feature)
pub async fn connect(url: &str) -> Result<Store, StoreError> {
    if url.starts_with("sqlite:") {
        #[cfg(feature = "store-sqlite")]
        {
            let s = sqlite::SqliteStore::connect(url).await?;
            return Ok(Store::Sqlite(s));
        }
        #[cfg(not(feature = "store-sqlite"))]
        return Err(StoreError::Other(
            "SQLite support not compiled in (enable `store-sqlite` feature)".into(),
        ));
    }
    if url.starts_with("postgres:") || url.starts_with("postgresql:") {
        #[cfg(feature = "store-postgres")]
        {
            let s = postgres::PostgresStore::connect(url).await?;
            return Ok(Store::Postgres(s));
        }
        #[cfg(not(feature = "store-postgres"))]
        return Err(StoreError::Other(
            "Postgres support not compiled in (enable `store-postgres` feature)".into(),
        ));
    }
    if url.starts_with("mysql:") {
        #[cfg(feature = "store-mysql")]
        {
            let s = mysql::MysqlStore::connect(url).await?;
            return Ok(Store::Mysql(s));
        }
        #[cfg(not(feature = "store-mysql"))]
        return Err(StoreError::Other(
            "MySQL support not compiled in (enable `store-mysql` feature)".into(),
        ));
    }
    Err(StoreError::Other(format!(
        "unsupported store URL scheme: {}",
        url
    )))
}
