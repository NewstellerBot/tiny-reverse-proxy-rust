//! Typed management and recovery API for gateway state.
//!
//! Export/import and revision snapshots are store-backed control-plane artifacts. They are meant
//! for explicit operator workflows, not as node-local caches.

use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use crate::content_filter::{effective_detector_catalog, EffectiveDetectorConfig};
use crate::cost_tracker::{CostTracker, KeyUsage, ModelCost};
use crate::evals::{
    compare_project_eval_runs, execute_project_eval_run, queue_project_eval_run,
    ProjectEvalRunComparison, ProjectEvalRunComparisonGateRequest, ProjectEvalRunExecution,
    ProjectEvalRunRequest,
};
use crate::governance::{generate_unique_id, GovernanceState};
use crate::prompt_cache::{PromptCache, PromptCacheStatusSnapshot};
use crate::provider_failover::{
    FailedProviderStatus, ProviderConfig, ProviderFailover, ProviderFailureReason,
};
use crate::rate_limiter::TokenRateLimiter;
use crate::semantic_cache::{SemanticCache, SemanticCacheStatusSnapshot};
use crate::semantic_safety::{
    SemanticPolicyDeleteResult, SemanticPolicyMutationResult, SemanticPolicySyncStatus,
    SemanticSafety,
};
use crate::store::{
    GatewayStore, GovernanceChangeRecord, ManagedProviderRecord, ProjectDatasetItemRecord,
    ProjectDatasetRecord, ProjectEvalRunItemRecord, ProjectEvalRunRecord, ProjectPolicyRecord,
    ProjectPromptRecord, ProjectPromptRolloutRecord, ProjectRolloutPolicyRecord,
    ProjectSemanticPolicyRecord, ProjectToolRecord, RequestLogEntry, RequestLogQuery,
    RoutingRuleRecord, SafetyPolicyRecord, SessionEventRecord, SessionListQuery, SessionRecord,
    Store, VirtualKeyRecord,
};
use crate::tool_runtime::{
    ToolRuntime, ToolRuntimeMcpOAuthAuthorizationRequest, ToolRuntimeMcpServerSnapshot,
    ToolRuntimeStatusSnapshot,
};
use crate::virtual_keys::{VirtualKeyLookupError, VirtualKeys};
use proxy_auth::service::AuthService;
use proxy_auth::store::{PrincipalRecord, ProjectRecord, RoleBindingRecord, TokenRecord};
use proxy_auth::{AuthContext, Authenticator, Authorizer, Permission, ProjectId, Role};
use proxy_core::config::{PreviewFeatureSet, PreviewFeatureStatus, ProviderKeyConfig};
use proxy_core::runtime::{RuntimeReliabilitySnapshot, RuntimeReliabilityState};
use serde::{Deserialize, Serialize};

/// Typed Rust API for interacting with LLM gateway plugin state.
///
/// All reads go through DashMaps (always fresh, zero latency).
/// Mutations update both DashMaps and the persistent store (if configured).
#[derive(Clone)]
pub struct LlmGatewayApi {
    cost_tracker: Option<CostTracker>,
    rate_limiter: Option<TokenRateLimiter>,
    provider_failover: Option<ProviderFailover>,
    virtual_keys: Option<VirtualKeys>,
    prompt_cache: Option<PromptCache>,
    semantic_cache: Option<SemanticCache>,
    semantic_safety: Option<SemanticSafety>,
    tool_runtime: Option<ToolRuntime>,
    store: Option<Arc<Store>>,
    auth_service: Option<Arc<AuthService>>,
    governance: Option<Arc<GovernanceState>>,
    preview_features: PreviewFeatureSet,
    runtime_reliability: Option<RuntimeReliabilityState>,
}

#[derive(Clone, Debug)]
pub struct FailedProviderSnapshot {
    pub name: String,
    pub failed_ago_secs: u64,
    pub cooldown_remaining_secs: u64,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct ProviderHealthSnapshot {
    pub name: String,
    pub eligible: bool,
    pub cooldown_remaining_secs: u64,
    pub cooldown_reason: Option<String>,
    pub active_requests: u32,
    pub samples: u64,
    pub ewma_latency_ms: f64,
    pub ewma_error_rate: f64,
    pub ewma_timeout_rate: f64,
    pub ewma_rate_limit_rate: f64,
    pub adaptive_penalty_active_requests: f64,
    pub adaptive_penalty_latency: f64,
    pub adaptive_penalty_error: f64,
    pub adaptive_penalty_timeout: f64,
    pub adaptive_penalty_rate_limit: f64,
    pub adaptive_penalty_total: f64,
}

#[derive(Clone, Debug)]
pub struct RoleCatalogEntry {
    pub role: Role,
    pub description: &'static str,
    pub permissions: Vec<Permission>,
}

#[derive(Clone, Debug)]
pub struct ProjectAccessSnapshot {
    pub project_id: String,
    pub role_bindings: Vec<RoleBindingRecord>,
    pub permissions: Vec<Permission>,
}

#[derive(Clone, Debug)]
pub struct PrincipalAccessSnapshot {
    pub principal: PrincipalRecord,
    pub role_bindings: Vec<RoleBindingRecord>,
    pub instance_permissions: Vec<Permission>,
    pub project_access: Vec<ProjectAccessSnapshot>,
}

static PROJECT_CONFIG_REVISION_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectConfigSnapshot {
    pub project_id: String,
    pub policy: Option<ProjectPolicyRecord>,
    pub routing_rules: Vec<RoutingRuleRecord>,
    pub safety_policy: Option<SafetyPolicyRecord>,
    pub semantic_policy: Option<ProjectSemanticPolicyRecord>,
    pub tools: Vec<ProjectToolRecord>,
    pub prompts: Vec<ProjectPromptRecord>,
    pub rollout_policies: Vec<ProjectRolloutPolicyRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectConfigRevision {
    pub revision_id: String,
    pub project_id: String,
    pub source: String,
    pub snapshot: ProjectConfigSnapshot,
    pub recorded_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectConfigRevisionState {
    pub project_id: String,
    pub active_revision_id: String,
    pub last_known_good_revision_id: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControlPlaneProjectSnapshot {
    pub project_id: String,
    pub name: String,
    pub description: Option<String>,
    pub active: bool,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControlPlaneSnapshot {
    pub exported_at: String,
    pub projects: Vec<ControlPlaneProjectSnapshot>,
    pub managed_providers: Vec<ManagedProviderRecord>,
    pub virtual_keys: Vec<VirtualKeyRecord>,
    pub project_configs: Vec<ProjectConfigSnapshot>,
}

impl LlmGatewayApi {
    pub fn new(
        cost_tracker: Option<CostTracker>,
        rate_limiter: Option<TokenRateLimiter>,
        provider_failover: Option<ProviderFailover>,
        virtual_keys: Option<VirtualKeys>,
        prompt_cache: Option<PromptCache>,
        semantic_cache: Option<SemanticCache>,
        semantic_safety: Option<SemanticSafety>,
        tool_runtime: Option<ToolRuntime>,
        store: Option<Arc<Store>>,
    ) -> Self {
        Self {
            cost_tracker,
            rate_limiter,
            provider_failover,
            virtual_keys,
            prompt_cache,
            semantic_cache,
            semantic_safety,
            tool_runtime,
            store,
            auth_service: None,
            governance: None,
            preview_features: PreviewFeatureSet::default(),
            runtime_reliability: None,
        }
    }

    pub fn with_preview_features(mut self, preview_features: PreviewFeatureSet) -> Self {
        self.preview_features = preview_features;
        self
    }

    pub fn with_governance(
        mut self,
        auth_service: Arc<AuthService>,
        governance: Arc<GovernanceState>,
    ) -> Self {
        self.auth_service = Some(auth_service);
        self.governance = Some(governance);
        self
    }

    pub fn with_runtime_reliability(
        mut self,
        runtime_reliability: RuntimeReliabilityState,
    ) -> Self {
        self.runtime_reliability = Some(runtime_reliability);
        self
    }

    fn validate_project_config_snapshot_inner(
        snapshot: &ProjectConfigSnapshot,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if snapshot.project_id.trim().is_empty() {
            return Err("project config snapshot project_id must not be empty".into());
        }

        let project_id = snapshot.project_id.as_str();
        if snapshot
            .policy
            .as_ref()
            .is_some_and(|record| record.project_id != project_id)
        {
            return Err("project policy project_id does not match snapshot project_id".into());
        }
        if snapshot
            .safety_policy
            .as_ref()
            .is_some_and(|record| record.project_id != project_id)
        {
            return Err("safety policy project_id does not match snapshot project_id".into());
        }
        if snapshot
            .semantic_policy
            .as_ref()
            .is_some_and(|record| record.project_id != project_id)
        {
            return Err("semantic policy project_id does not match snapshot project_id".into());
        }

        for record in &snapshot.routing_rules {
            if record.project_id != project_id {
                return Err("routing rule project_id does not match snapshot project_id".into());
            }
            if let Some(raw) = &record.match_headers {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
            if let Some(raw) = &record.provider_order {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
            if let Some(raw) = &record.provider_weights {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
        }
        for record in &snapshot.tools {
            if record.project_id != project_id {
                return Err("tool project_id does not match snapshot project_id".into());
            }
            let _: serde_json::Value = serde_json::from_str(&record.input_schema_json)?;
            if let Some(raw) = &record.executor_config_json {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
        }
        for record in &snapshot.prompts {
            if record.project_id != project_id {
                return Err("prompt project_id does not match snapshot project_id".into());
            }
            if let Some(raw) = &record.variables_schema_json {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
            if let Some(raw) = &record.rollout_metadata_json {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
        }
        for record in &snapshot.rollout_policies {
            if record.project_id != project_id {
                return Err("rollout policy project_id does not match snapshot project_id".into());
            }
            let _: serde_json::Value = serde_json::from_str(&record.gate_config_json)?;
        }

        if let Some(record) = snapshot.policy.as_ref() {
            for raw in [
                &record.fallback_order,
                &record.provider_rpm_limits,
                &record.provider_tpm_limits,
                &record.provider_timeouts,
                &record.provider_input_costs,
                &record.provider_output_costs,
                &record.allowed_tools,
            ]
            .into_iter()
            .flatten()
            {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
        }
        if let Some(record) = snapshot.safety_policy.as_ref() {
            if let Some(raw) = &record.rules_json {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
        }
        if let Some(record) = snapshot.semantic_policy.as_ref() {
            if let Some(raw) = &record.entities_json {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
            if let Some(raw) = &record.topics_json {
                let _: serde_json::Value = serde_json::from_str(raw)?;
            }
        }

        Ok(())
    }

    fn validate_project_config_snapshot_against_runtime(
        &self,
        snapshot: &ProjectConfigSnapshot,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Self::validate_project_config_snapshot_inner(snapshot)?;

        if let Some(projects) = self.list_projects() {
            if !projects
                .iter()
                .any(|project| project.project_id == snapshot.project_id)
            {
                return Err(format!("project '{}' does not exist", snapshot.project_id).into());
            }
        }

        let configured_provider_names: HashSet<String> = self
            .configured_providers()
            .unwrap_or_default()
            .into_iter()
            .map(|provider| provider.name)
            .collect();
        if configured_provider_names.is_empty() {
            return Ok(());
        }

        Self::validate_project_config_snapshot_provider_references(
            snapshot,
            &configured_provider_names,
        )
    }

    fn validate_project_config_snapshot_provider_references(
        snapshot: &ProjectConfigSnapshot,
        configured_provider_names: &HashSet<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ensure_provider_names = |names: Vec<String>,
                                     field: &str|
         -> Result<(), Box<dyn std::error::Error>> {
            for name in names {
                if !configured_provider_names.contains(&name) {
                    return Err(format!("unknown provider '{name}' referenced by {field}").into());
                }
            }
            Ok(())
        };

        if let Some(policy) = snapshot.policy.as_ref() {
            if let Some(raw) = &policy.fallback_order {
                let names: Vec<String> = serde_json::from_str(raw)?;
                ensure_provider_names(names, "policy.fallback_order")?;
            }
            for (field, raw) in [
                (
                    "policy.provider_rpm_limits",
                    policy.provider_rpm_limits.as_deref(),
                ),
                (
                    "policy.provider_tpm_limits",
                    policy.provider_tpm_limits.as_deref(),
                ),
                (
                    "policy.provider_timeouts",
                    policy.provider_timeouts.as_deref(),
                ),
                (
                    "policy.provider_input_costs",
                    policy.provider_input_costs.as_deref(),
                ),
                (
                    "policy.provider_output_costs",
                    policy.provider_output_costs.as_deref(),
                ),
            ] {
                if let Some(raw) = raw {
                    let keys: serde_json::Map<String, serde_json::Value> =
                        serde_json::from_str(raw)?;
                    ensure_provider_names(keys.into_iter().map(|(key, _)| key).collect(), field)?;
                }
            }
        }

        let mut routing_rule_ids = HashSet::new();
        for rule in &snapshot.routing_rules {
            if !routing_rule_ids.insert(rule.rule_id.clone()) {
                return Err(format!("duplicate routing rule id '{}'", rule.rule_id).into());
            }
            if let Some(raw) = &rule.provider_order {
                let names: Vec<String> = serde_json::from_str(raw)?;
                ensure_provider_names(names, "routing_rule.provider_order")?;
            }
            if let Some(raw) = &rule.provider_weights {
                let keys: serde_json::Map<String, serde_json::Value> = serde_json::from_str(raw)?;
                ensure_provider_names(
                    keys.into_iter().map(|(key, _)| key).collect(),
                    "routing_rule.provider_weights",
                )?;
            }
        }

        let mut tool_names = HashSet::new();
        for tool in &snapshot.tools {
            if !tool_names.insert(tool.tool_name.clone()) {
                return Err(format!("duplicate tool '{}'", tool.tool_name).into());
            }
        }

        let mut prompt_keys = HashSet::new();
        for prompt in &snapshot.prompts {
            let prompt_key = format!(
                "{}:{}:{}",
                prompt.prompt_name, prompt.version, prompt.environment
            );
            if !prompt_keys.insert(prompt_key.clone()) {
                return Err(format!("duplicate prompt '{prompt_key}'").into());
            }
        }

        let mut rollout_policy_names = HashSet::new();
        for policy in &snapshot.rollout_policies {
            if !rollout_policy_names.insert(policy.policy_name.clone()) {
                return Err(format!("duplicate rollout policy '{}'", policy.policy_name).into());
            }
        }

        Ok(())
    }

    fn validate_control_plane_snapshot_inner(
        &self,
        snapshot: &ControlPlaneSnapshot,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut project_ids = HashSet::new();
        for project in &snapshot.projects {
            if project.project_id.trim().is_empty() {
                return Err(
                    "control plane snapshot contains a project with an empty project_id".into(),
                );
            }
            if !project_ids.insert(project.project_id.clone()) {
                return Err(format!("duplicate project '{}'", project.project_id).into());
            }
        }

        let mut provider_names: HashSet<String> = self
            .static_providers()
            .unwrap_or_default()
            .into_iter()
            .map(|provider| provider.name)
            .collect();
        for provider in &snapshot.managed_providers {
            if provider.name.trim().is_empty() {
                return Err("control plane snapshot contains a provider with an empty name".into());
            }
            if !provider_names.insert(provider.name.clone()) {
                return Err(format!("duplicate provider '{}'", provider.name).into());
            }
        }

        let mut key_hashes = HashSet::new();
        for key in &snapshot.virtual_keys {
            if key.key_hash.trim().is_empty() {
                return Err("control plane snapshot contains a key with an empty key_hash".into());
            }
            if !key_hashes.insert(key.key_hash.clone()) {
                return Err(format!("duplicate virtual key '{}'", key.key_hash).into());
            }
            if !project_ids.contains(&key.project_id) {
                return Err(format!(
                    "virtual key '{}' references unknown project '{}'",
                    key.key_hash, key.project_id
                )
                .into());
            }
            if !provider_names.contains(&key.provider_name) {
                return Err(format!(
                    "virtual key '{}' references unknown provider '{}'",
                    key.key_hash, key.provider_name
                )
                .into());
            }
        }

        let mut config_project_ids = HashSet::new();
        for project_config in &snapshot.project_configs {
            if !project_ids.contains(&project_config.project_id) {
                return Err(format!(
                    "project config for '{}' does not have a matching project record",
                    project_config.project_id
                )
                .into());
            }
            if !config_project_ids.insert(project_config.project_id.clone()) {
                return Err(format!(
                    "duplicate project config snapshot for '{}'",
                    project_config.project_id
                )
                .into());
            }
            Self::validate_project_config_snapshot_inner(project_config)?;
            Self::validate_project_config_snapshot_provider_references(
                project_config,
                &provider_names,
            )?;
        }

        Ok(())
    }

    async fn project_config_snapshot_inner(
        &self,
        project_id: &str,
    ) -> Result<ProjectConfigSnapshot, Box<dyn std::error::Error>> {
        let governance = self.governance.as_ref().ok_or("governance not enabled")?;

        let rollout_policies = match self.list_project_rollout_policies(Some(project_id)).await {
            Some(Ok(records)) => records,
            Some(Err(error)) => return Err(error),
            None => Vec::new(),
        };

        Ok(ProjectConfigSnapshot {
            project_id: project_id.to_string(),
            policy: governance.project_policy(project_id),
            routing_rules: governance.list_routing_rules(Some(project_id)),
            safety_policy: governance.safety_policy(project_id),
            semantic_policy: governance.semantic_policy(project_id),
            tools: governance.list_project_tools(Some(project_id)),
            prompts: governance.list_project_prompts(Some(project_id), None),
            rollout_policies,
        })
    }

    async fn record_project_config_revision(
        &self,
        project_id: &str,
        source: &str,
    ) -> Result<Option<ProjectConfigRevision>, Box<dyn std::error::Error>> {
        let Some(store) = self.store.as_ref() else {
            return Ok(None);
        };
        let snapshot = self.project_config_snapshot_inner(project_id).await?;
        Self::validate_project_config_snapshot_inner(&snapshot)?;
        let revision_id = generate_unique_id("cfg", &PROJECT_CONFIG_REVISION_SEQUENCE);
        let revision = ProjectConfigRevision {
            revision_id: revision_id.clone(),
            project_id: project_id.to_string(),
            source: source.to_string(),
            snapshot,
            recorded_at: crate::governance::current_timestamp_string(),
        };
        let snapshot_json = serde_json::to_string(&revision)?;
        store
            .append_governance_change(&GovernanceChangeRecord {
                change_id: revision_id.clone(),
                project_id: project_id.to_string(),
                resource_type: "project_config_revision".to_string(),
                resource_id: revision_id,
                action: source.to_string(),
                before_json: None,
                after_json: Some(snapshot_json),
                changed_at: revision.recorded_at.clone(),
            })
            .await?;
        let revision_state = ProjectConfigRevisionState {
            project_id: project_id.to_string(),
            active_revision_id: revision.revision_id.clone(),
            last_known_good_revision_id: revision.revision_id.clone(),
            updated_at: revision.recorded_at.clone(),
        };
        store
            .append_governance_change(&GovernanceChangeRecord {
                change_id: generate_unique_id("cfgstate", &PROJECT_CONFIG_REVISION_SEQUENCE),
                project_id: project_id.to_string(),
                resource_type: "project_config_revision_state".to_string(),
                resource_id: project_id.to_string(),
                action: source.to_string(),
                before_json: None,
                after_json: Some(serde_json::to_string(&revision_state)?),
                changed_at: revision.recorded_at.clone(),
            })
            .await?;
        Ok(Some(revision))
    }

    async fn apply_project_config_snapshot_inner(
        &self,
        snapshot: &ProjectConfigSnapshot,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_project_config_snapshot_against_runtime(snapshot)?;
        let governance = self.governance.as_ref().ok_or("governance not enabled")?;
        let store = self.store.as_ref().ok_or("store not enabled")?;

        match snapshot.policy.clone() {
            Some(record) => governance.upsert_project_policy(record).await?,
            None => {
                governance
                    .delete_project_policy(&snapshot.project_id)
                    .await?;
            }
        }

        let existing_rule_ids: HashSet<String> = governance
            .list_routing_rules(Some(&snapshot.project_id))
            .into_iter()
            .map(|record| record.rule_id)
            .collect();
        let desired_rule_ids: HashSet<String> = snapshot
            .routing_rules
            .iter()
            .map(|record| record.rule_id.clone())
            .collect();
        for rule_id in existing_rule_ids.difference(&desired_rule_ids) {
            governance.delete_routing_rule(rule_id).await?;
        }
        for record in &snapshot.routing_rules {
            governance.upsert_routing_rule(record.clone()).await?;
        }

        match snapshot.safety_policy.clone() {
            Some(record) => governance.upsert_safety_policy(record).await?,
            None => {
                governance
                    .delete_safety_policy(&snapshot.project_id)
                    .await?;
            }
        }

        match snapshot.semantic_policy.clone() {
            Some(record) => {
                governance.upsert_semantic_policy(record.clone()).await?;
                if let Some(plugin) = &self.semantic_safety {
                    let result = plugin.sync_policy_record(&record).await;
                    if let Some(error) = result.sync_error {
                        return Err(error.into());
                    }
                }
            }
            None => {
                governance
                    .delete_semantic_policy(&snapshot.project_id)
                    .await?;
                if let Some(plugin) = &self.semantic_safety {
                    plugin
                        .delete_policy(&snapshot.project_id)
                        .await
                        .map_err(|error| error.to_string())?;
                }
            }
        }

        let existing_tool_names: HashSet<String> = governance
            .list_project_tools(Some(&snapshot.project_id))
            .into_iter()
            .map(|record| record.tool_name)
            .collect();
        let desired_tool_names: HashSet<String> = snapshot
            .tools
            .iter()
            .map(|record| record.tool_name.clone())
            .collect();
        for tool_name in existing_tool_names.difference(&desired_tool_names) {
            governance
                .delete_project_tool(&snapshot.project_id, tool_name)
                .await?;
        }
        for record in &snapshot.tools {
            governance.upsert_project_tool(record.clone()).await?;
        }

        let existing_prompt_keys: HashSet<(String, String)> = governance
            .list_project_prompts(Some(&snapshot.project_id), None)
            .into_iter()
            .map(|record| (record.prompt_name, record.version))
            .collect();
        let desired_prompt_keys: HashSet<(String, String)> = snapshot
            .prompts
            .iter()
            .map(|record| (record.prompt_name.clone(), record.version.clone()))
            .collect();
        for (prompt_name, version) in existing_prompt_keys.difference(&desired_prompt_keys) {
            governance
                .delete_project_prompt(&snapshot.project_id, prompt_name, version)
                .await?;
        }
        for record in &snapshot.prompts {
            governance.upsert_project_prompt(record.clone()).await?;
        }

        let existing_rollout_policies: HashSet<String> = store
            .get_project_rollout_policies(Some(&snapshot.project_id))
            .await?
            .into_iter()
            .map(|record| record.policy_name)
            .collect();
        let desired_rollout_policies: HashSet<String> = snapshot
            .rollout_policies
            .iter()
            .map(|record| record.policy_name.clone())
            .collect();
        for policy_name in existing_rollout_policies.difference(&desired_rollout_policies) {
            store
                .delete_project_rollout_policy(&snapshot.project_id, policy_name)
                .await?;
        }
        for record in &snapshot.rollout_policies {
            store.upsert_project_rollout_policy(record).await?;
        }

        Ok(())
    }

    async fn ensure_project_config_revision_recorded(
        &self,
        project_id: &str,
        source: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.record_project_config_revision(project_id, source)
            .await?;
        Ok(())
    }

    pub fn auth_service(&self) -> Option<&Arc<AuthService>> {
        self.auth_service.as_ref()
    }

    pub fn governance(&self) -> Option<&Arc<GovernanceState>> {
        self.governance.as_ref()
    }

    pub fn auth_required(&self) -> bool {
        self.auth_service.is_some()
    }

    pub fn has_admin_access_path(&self) -> bool {
        self.auth_service
            .as_ref()
            .map(|service| service.has_admin_access_path())
            .unwrap_or(false)
    }

    pub async fn authenticate_bearer(&self, token: &str) -> Option<AuthContext> {
        let service = self.auth_service.as_ref()?;
        service.authenticate_bearer(token).await
    }

    pub fn is_allowed(
        &self,
        ctx: &AuthContext,
        permission: Permission,
        project_id: Option<&ProjectId>,
    ) -> bool {
        match self.auth_service.as_ref() {
            Some(service) => service.is_allowed(ctx, permission, project_id),
            None => false,
        }
    }

    pub fn accessible_projects(&self, ctx: &AuthContext) -> Vec<ProjectId> {
        match self.auth_service.as_ref() {
            Some(service) => service.accessible_projects(ctx),
            None => Vec::new(),
        }
    }

    // --- Status ---

    pub fn cost_tracker_enabled(&self) -> bool {
        self.cost_tracker.is_some()
    }

    pub fn rate_limiter_enabled(&self) -> bool {
        self.rate_limiter.is_some()
    }

    pub fn provider_failover_enabled(&self) -> bool {
        self.provider_failover.is_some()
    }

    // --- Cost tracker ---

    pub fn cost_usage(&self) -> Option<Vec<(String, KeyUsage)>> {
        self.cost_tracker.as_ref().map(|ct| ct.get_all_usage())
    }

    pub fn cost_usage_for_key(&self, key: &str) -> Option<Option<KeyUsage>> {
        self.cost_tracker.as_ref().map(|ct| ct.get_usage(key))
    }

    pub fn budget_limit(&self) -> Option<f64> {
        self.cost_tracker.as_ref().map(|ct| ct.budget_limit())
    }

    pub fn model_costs(&self) -> Option<Vec<(String, ModelCost)>> {
        self.cost_tracker.as_ref().map(|ct| ct.get_model_costs())
    }

    /// Per-key per-model usage breakdown from the DashMap.
    pub fn model_usage_breakdown(&self) -> Option<Vec<((String, String), KeyUsage)>> {
        self.cost_tracker
            .as_ref()
            .map(|ct| ct.get_all_model_usage())
    }

    /// Query audit request logs from the store.
    pub async fn get_request_logs(
        &self,
        api_key: Option<&str>,
        model: Option<&str>,
        project_id: Option<&str>,
        limit: u32,
    ) -> Option<Result<Vec<RequestLogEntry>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_request_logs(api_key, model, project_id, limit)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        )
    }

    pub async fn get_request_logs_for_session(
        &self,
        session_id: &str,
        project_id: Option<&str>,
        limit: u32,
    ) -> Option<Result<Vec<RequestLogEntry>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_request_logs_for_session(session_id, project_id, limit)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        )
    }

    pub async fn query_request_logs(
        &self,
        query: &RequestLogQuery,
    ) -> Option<Result<Vec<RequestLogEntry>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .query_request_logs(query)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        )
    }

    pub async fn get_session(
        &self,
        session_id: &str,
    ) -> Option<Result<Option<SessionRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_session(session_id)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        )
    }

    pub async fn list_sessions(
        &self,
        query: &SessionListQuery,
    ) -> Option<Result<Vec<SessionRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .list_sessions(query)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        )
    }

    pub async fn upsert_session(
        &self,
        record: SessionRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .upsert_session(&record)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        )
    }

    pub async fn append_session_event(
        &self,
        record: &SessionEventRecord,
    ) -> Option<Result<i64, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .append_session_event(record)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        )
    }

    pub async fn get_session_events(
        &self,
        session_id: &str,
        after_seq: Option<i64>,
        limit: u32,
    ) -> Option<Result<Vec<SessionEventRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_session_events(session_id, after_seq, limit)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        )
    }

    /// Reset usage for a single key. Returns whether the key existed.
    pub async fn reset_cost_usage(&self, key: &str) -> Option<bool> {
        let ct = self.cost_tracker.as_ref()?;
        let existed = ct.reset_usage(key);
        // Also delete from store.
        if let Some(store) = &self.store {
            if let Err(e) = store.delete_usage(key).await {
                tracing::warn!("failed to delete usage from store: {}", e);
            }
        }
        Some(existed)
    }

    /// Reset all usage.
    pub async fn reset_all_cost_usage(&self) -> Option<()> {
        let ct = self.cost_tracker.as_ref()?;
        ct.reset_all_usage();
        if let Some(store) = &self.store {
            if let Err(e) = store.delete_all_usage().await {
                tracing::warn!("failed to delete all usage from store: {}", e);
            }
        }
        Some(())
    }

    /// Set model cost (updates both DashMap and store).
    pub async fn set_model_cost(&self, model: &str, input: f64, output: f64) -> Option<()> {
        let ct = self.cost_tracker.as_ref()?;
        ct.set_model_cost(model, input, output);
        if let Some(store) = &self.store {
            use crate::store::{GatewayStore, ModelCostRecord};
            let record = ModelCostRecord {
                input_cost_per_1k: input,
                output_cost_per_1k: output,
            };
            if let Err(e) = store.upsert_model_cost(model, &record).await {
                tracing::warn!("failed to upsert model cost in store: {}", e);
            }
        }
        Some(())
    }

    /// Delete model cost.
    pub async fn delete_model_cost(&self, model: &str) -> Option<bool> {
        let ct = self.cost_tracker.as_ref()?;
        let existed = ct.delete_model_cost(model);
        if let Some(store) = &self.store {
            if let Err(e) = store.delete_model_cost(model).await {
                tracing::warn!("failed to delete model cost from store: {}", e);
            }
        }
        Some(existed)
    }

    // --- Rate limiter ---

    pub fn rate_limiter_tracked_keys(&self) -> Option<usize> {
        self.rate_limiter.as_ref().map(|rl| rl.tracked_key_count())
    }

    pub fn rate_limiter_config(&self) -> Option<(f64, f64)> {
        self.rate_limiter
            .as_ref()
            .map(|rl| (rl.rate_per_second(), rl.burst()))
    }

    // --- Provider failover ---

    pub fn providers(&self) -> Option<Vec<ProviderConfig>> {
        self.provider_failover.as_ref().map(|pf| pf.get_providers())
    }

    pub fn configured_providers(&self) -> Option<Vec<ProviderKeyConfig>> {
        self.virtual_keys.as_ref().map(|vk| vk.provider_configs())
    }

    pub fn static_providers(&self) -> Option<Vec<ProviderKeyConfig>> {
        self.virtual_keys
            .as_ref()
            .map(|virtual_keys| virtual_keys.static_provider_configs())
    }

    pub fn managed_providers(&self) -> Option<Vec<ManagedProviderRecord>> {
        self.virtual_keys
            .as_ref()
            .map(|virtual_keys| virtual_keys.managed_provider_records())
    }

    pub fn managed_provider(&self, name: &str) -> Option<ManagedProviderRecord> {
        self.virtual_keys
            .as_ref()
            .and_then(|virtual_keys| virtual_keys.managed_provider_record(name))
    }

    pub fn configured_provider(&self, name: &str) -> Option<ProviderKeyConfig> {
        self.virtual_keys
            .as_ref()
            .and_then(|virtual_keys| virtual_keys.effective_provider_config(name))
    }

    pub fn provider_preview(
        &self,
        name: &str,
    ) -> Option<Result<Option<ProviderKeyConfig>, String>> {
        Some(self.virtual_keys.as_ref()?.resolve_provider_preview(name))
    }

    fn refresh_provider_dependents(&self) {
        let Some(virtual_keys) = self.virtual_keys.as_ref() else {
            return;
        };
        let providers = virtual_keys.provider_configs();
        if let Some(prompt_cache) = &self.prompt_cache {
            prompt_cache.set_provider_configs(&providers);
        }
        if let Some(tool_runtime) = &self.tool_runtime {
            tool_runtime.set_provider_configs(&providers);
        }
    }

    pub async fn upsert_managed_provider(
        &self,
        record: ManagedProviderRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let result = self
            .virtual_keys
            .as_ref()?
            .upsert_managed_provider(record)
            .await;
        if result.is_ok() {
            self.refresh_provider_dependents();
        }
        Some(result)
    }

    pub async fn delete_managed_provider(
        &self,
        name: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let result = self
            .virtual_keys
            .as_ref()?
            .delete_managed_provider(name)
            .await;
        if matches!(result, Ok(true)) {
            self.refresh_provider_dependents();
        }
        Some(result)
    }

    fn cooldown_snapshot(
        status: &FailedProviderStatus,
        cooldown: Duration,
    ) -> Option<(u64, u64, ProviderFailureReason)> {
        let elapsed = status.failed_at.elapsed();
        let remaining = cooldown.checked_sub(elapsed)?;
        Some((
            elapsed.as_secs(),
            remaining.as_secs(),
            status.reason.clone(),
        ))
    }

    pub fn failed_providers(&self) -> Option<Vec<FailedProviderSnapshot>> {
        let failover = self.provider_failover.as_ref()?;
        let cooldown = failover.cooldown();
        let mut failed: Vec<FailedProviderSnapshot> = failover
            .get_failed_providers()
            .into_iter()
            .filter_map(|(name, status)| {
                let (failed_ago_secs, cooldown_remaining_secs, reason) =
                    Self::cooldown_snapshot(&status, cooldown)?;
                Some(FailedProviderSnapshot {
                    name,
                    failed_ago_secs,
                    cooldown_remaining_secs,
                    reason: reason.as_str().to_string(),
                })
            })
            .collect();
        failed.sort_by(|left, right| left.name.cmp(&right.name));
        Some(failed)
    }

    pub fn provider_cooldown(&self) -> Option<std::time::Duration> {
        self.provider_failover.as_ref().map(|pf| pf.cooldown())
    }

    pub fn clear_failed_provider(&self, name: &str) -> Option<bool> {
        self.provider_failover
            .as_ref()
            .map(|pf| pf.clear_failed(name))
    }

    pub fn clear_all_failed_providers(&self) -> Option<()> {
        self.provider_failover
            .as_ref()
            .map(|pf| pf.clear_all_failed())
    }

    pub fn runtime_reliability(&self) -> Option<RuntimeReliabilitySnapshot> {
        self.runtime_reliability
            .as_ref()
            .map(RuntimeReliabilityState::snapshot)
    }

    pub fn provider_health(&self) -> Option<Vec<ProviderHealthSnapshot>> {
        let governance = self.governance.as_ref();
        let failover = self.provider_failover.as_ref();

        let mut provider_names = std::collections::BTreeSet::new();
        if let Some(failover) = failover {
            for provider in failover.get_providers() {
                provider_names.insert(provider.name);
            }
        }
        if let Some(governance) = governance {
            for provider_name in governance.provider_stats_keys() {
                provider_names.insert(provider_name);
            }
        }

        let cooldown = failover.map(|pf| pf.cooldown()).unwrap_or_default();
        let failed_states: std::collections::HashMap<String, FailedProviderStatus> = failover
            .map(|pf| pf.get_failed_providers().into_iter().collect())
            .unwrap_or_default();

        Some(
            provider_names
                .into_iter()
                .map(|name| {
                    let stats = governance
                        .map(|state| state.provider_health_stats(&name))
                        .unwrap_or_default();
                    let cooldown_snapshot = failed_states
                        .get(&name)
                        .and_then(|status| Self::cooldown_snapshot(status, cooldown));

                    let cooldown_remaining_secs = cooldown_snapshot
                        .as_ref()
                        .map(|(_, remaining, _)| *remaining)
                        .unwrap_or(0);
                    let cooldown_reason = cooldown_snapshot
                        .as_ref()
                        .map(|(_, _, reason)| reason.as_str().to_string());

                    ProviderHealthSnapshot {
                        name,
                        eligible: cooldown_remaining_secs == 0,
                        cooldown_remaining_secs,
                        cooldown_reason,
                        active_requests: stats.active_requests,
                        samples: stats.samples,
                        ewma_latency_ms: stats.ewma_latency_ms,
                        ewma_error_rate: stats.ewma_error_rate,
                        ewma_timeout_rate: stats.ewma_timeout_rate,
                        ewma_rate_limit_rate: stats.ewma_rate_limit_rate,
                        adaptive_penalty_active_requests: stats.penalties.active_requests,
                        adaptive_penalty_latency: stats.penalties.latency,
                        adaptive_penalty_error: stats.penalties.error,
                        adaptive_penalty_timeout: stats.penalties.timeout,
                        adaptive_penalty_rate_limit: stats.penalties.rate_limit,
                        adaptive_penalty_total: stats.penalties.total(),
                    }
                })
                .collect(),
        )
    }

    pub fn tool_runtime_status(&self) -> Option<ToolRuntimeStatusSnapshot> {
        self.tool_runtime.as_ref().map(|runtime| runtime.status())
    }

    pub fn preview_feature_statuses(&self) -> Vec<PreviewFeatureStatus> {
        self.preview_features.statuses()
    }

    pub fn preview_feature_enabled(&self, feature: proxy_core::config::PreviewFeature) -> bool {
        self.preview_features.contains(feature)
    }

    pub async fn refresh_tool_runtime_mcp_server(
        &self,
        server_name: &str,
    ) -> Option<Result<ToolRuntimeMcpServerSnapshot, Box<dyn std::error::Error>>> {
        let runtime = self.tool_runtime.as_ref()?;
        Some(
            runtime
                .refresh_mcp_server(server_name)
                .await
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() }),
        )
    }

    pub async fn reset_tool_runtime_mcp_session(
        &self,
        server_name: &str,
    ) -> Option<Result<ToolRuntimeMcpServerSnapshot, Box<dyn std::error::Error>>> {
        let runtime = self.tool_runtime.as_ref()?;
        Some(
            runtime
                .reset_mcp_server_session(server_name)
                .await
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() }),
        )
    }

    pub async fn disable_tool_runtime_mcp_server(
        &self,
        server_name: &str,
        actor: Option<String>,
        reason: Option<String>,
    ) -> Option<Result<ToolRuntimeMcpServerSnapshot, Box<dyn std::error::Error>>> {
        let runtime = self.tool_runtime.as_ref()?;
        Some(
            runtime
                .disable_mcp_server(server_name, actor, reason)
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() }),
        )
    }

    pub async fn enable_tool_runtime_mcp_server(
        &self,
        server_name: &str,
        actor: Option<String>,
        reason: Option<String>,
    ) -> Option<Result<ToolRuntimeMcpServerSnapshot, Box<dyn std::error::Error>>> {
        let runtime = self.tool_runtime.as_ref()?;
        Some(
            runtime
                .enable_mcp_server(server_name, actor, reason)
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() }),
        )
    }

    pub async fn begin_tool_runtime_mcp_oauth_authorization(
        &self,
        server_name: &str,
    ) -> Option<Result<ToolRuntimeMcpOAuthAuthorizationRequest, Box<dyn std::error::Error>>> {
        let runtime = self.tool_runtime.as_ref()?;
        Some(
            runtime
                .begin_mcp_oauth_authorization(server_name)
                .await
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() }),
        )
    }

    pub async fn complete_tool_runtime_mcp_oauth_authorization(
        &self,
        server_name: &str,
        state: &str,
        code: &str,
    ) -> Option<Result<ToolRuntimeMcpServerSnapshot, Box<dyn std::error::Error>>> {
        let runtime = self.tool_runtime.as_ref()?;
        Some(
            runtime
                .complete_mcp_oauth_authorization(server_name, state, code)
                .await
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() }),
        )
    }

    pub fn prompt_cache_status(&self) -> Option<PromptCacheStatusSnapshot> {
        self.prompt_cache.as_ref().map(|runtime| runtime.status())
    }

    pub fn semantic_cache_status(&self) -> Option<SemanticCacheStatusSnapshot> {
        self.semantic_cache.as_ref().map(|runtime| runtime.status())
    }

    /// Perform a final flush of cost tracker state to the store.
    pub async fn flush(&self) {
        if let Some(ct) = &self.cost_tracker {
            if let Err(e) = ct.flush_to_store().await {
                tracing::warn!("final flush error: {}", e);
            }
        }
    }

    // --- Virtual keys ---

    pub fn virtual_keys_enabled(&self) -> bool {
        self.virtual_keys.is_some()
    }

    pub fn list_virtual_keys(&self) -> Option<Vec<VirtualKeyRecord>> {
        self.virtual_keys.as_ref().map(|vk| vk.get_all_keys())
    }

    pub fn list_virtual_keys_for_projects(
        &self,
        projects: &[ProjectId],
    ) -> Option<Vec<VirtualKeyRecord>> {
        let allowed: Vec<&str> = projects.iter().map(|project| project.0.as_str()).collect();
        Some(
            self.virtual_keys
                .as_ref()?
                .get_all_keys()
                .into_iter()
                .filter(|record| allowed.iter().any(|project| record.project_id == *project))
                .collect(),
        )
    }

    pub fn get_virtual_key(
        &self,
        hash_prefix: &str,
    ) -> Option<Result<Option<VirtualKeyRecord>, VirtualKeyLookupError>> {
        self.virtual_keys
            .as_ref()
            .map(|vk| vk.get_key_by_prefix(hash_prefix))
    }

    pub fn provider_timeout_secs(&self, provider_name: &str) -> Option<u64> {
        self.virtual_keys
            .as_ref()
            .and_then(|vk| vk.provider_timeout_secs(provider_name))
    }

    pub async fn create_virtual_key(
        &self,
        project_id: Option<&str>,
        name: &str,
        provider_name: &str,
        budget_limit: Option<f64>,
        budget_duration: Option<String>,
        rpm_limit: Option<u32>,
        tpm_limit: Option<u32>,
        allowed_models: Option<Vec<String>>,
        expires_at: Option<String>,
    ) -> Option<Result<(String, String), Box<dyn std::error::Error>>> {
        self.create_virtual_key_with_timeout(
            project_id,
            name,
            provider_name,
            budget_limit,
            budget_duration,
            rpm_limit,
            tpm_limit,
            allowed_models,
            expires_at,
            None,
        )
        .await
    }

    pub async fn create_virtual_key_with_timeout(
        &self,
        project_id: Option<&str>,
        name: &str,
        provider_name: &str,
        budget_limit: Option<f64>,
        budget_duration: Option<String>,
        rpm_limit: Option<u32>,
        tpm_limit: Option<u32>,
        allowed_models: Option<Vec<String>>,
        expires_at: Option<String>,
        timeout_secs: Option<u64>,
    ) -> Option<Result<(String, String), Box<dyn std::error::Error>>> {
        self.create_virtual_key_with_runtime_policy(
            project_id,
            name,
            provider_name,
            budget_limit,
            budget_duration,
            rpm_limit,
            tpm_limit,
            allowed_models,
            expires_at,
            timeout_secs,
            None,
            None,
        )
        .await
    }

    pub async fn create_virtual_key_with_runtime_policy(
        &self,
        project_id: Option<&str>,
        name: &str,
        provider_name: &str,
        budget_limit: Option<f64>,
        budget_duration: Option<String>,
        rpm_limit: Option<u32>,
        tpm_limit: Option<u32>,
        allowed_models: Option<Vec<String>>,
        expires_at: Option<String>,
        timeout_secs: Option<u64>,
        tool_approval_mode: Option<String>,
        allowed_tools: Option<Vec<String>>,
    ) -> Option<Result<(String, String), Box<dyn std::error::Error>>> {
        let vk = self.virtual_keys.as_ref()?;
        Some(
            vk.create_key_for_project_with_runtime_policy(
                project_id,
                name,
                provider_name,
                budget_limit,
                budget_duration,
                rpm_limit,
                tpm_limit,
                allowed_models,
                expires_at,
                timeout_secs,
                tool_approval_mode,
                allowed_tools,
            )
            .await,
        )
    }

    pub async fn update_virtual_key(
        &self,
        hash_prefix: &str,
        budget_limit: Option<Option<f64>>,
        rpm_limit: Option<Option<u32>>,
        tpm_limit: Option<Option<u32>>,
        active: Option<bool>,
        allowed_models: Option<Option<Vec<String>>>,
        expires_at: Option<Option<String>>,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        self.update_virtual_key_with_timeout(
            hash_prefix,
            budget_limit,
            rpm_limit,
            tpm_limit,
            active,
            allowed_models,
            expires_at,
            None,
        )
        .await
    }

    pub async fn update_virtual_key_with_timeout(
        &self,
        hash_prefix: &str,
        budget_limit: Option<Option<f64>>,
        rpm_limit: Option<Option<u32>>,
        tpm_limit: Option<Option<u32>>,
        active: Option<bool>,
        allowed_models: Option<Option<Vec<String>>>,
        expires_at: Option<Option<String>>,
        timeout_secs: Option<Option<u64>>,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        self.update_virtual_key_with_runtime_policy(
            hash_prefix,
            budget_limit,
            rpm_limit,
            tpm_limit,
            active,
            allowed_models,
            expires_at,
            timeout_secs,
            None,
            None,
        )
        .await
    }

    pub async fn update_virtual_key_with_runtime_policy(
        &self,
        hash_prefix: &str,
        budget_limit: Option<Option<f64>>,
        rpm_limit: Option<Option<u32>>,
        tpm_limit: Option<Option<u32>>,
        active: Option<bool>,
        allowed_models: Option<Option<Vec<String>>>,
        expires_at: Option<Option<String>>,
        timeout_secs: Option<Option<u64>>,
        tool_approval_mode: Option<Option<String>>,
        allowed_tools: Option<Option<Vec<String>>>,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let vk = self.virtual_keys.as_ref()?;
        Some(
            vk.update_key_with_runtime_policy(
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
            .await,
        )
    }

    pub async fn delete_virtual_key(
        &self,
        hash_prefix: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let vk = self.virtual_keys.as_ref()?;
        Some(vk.delete_key(hash_prefix).await)
    }

    pub async fn upsert_virtual_key_record(
        &self,
        record: VirtualKeyRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let vk = self.virtual_keys.as_ref()?;
        Some(vk.upsert_key_record(record).await)
    }

    pub fn list_projects(&self) -> Option<Vec<ProjectRecord>> {
        Some(self.auth_service.as_ref()?.list_projects())
    }

    pub async fn create_project(
        &self,
        project_id: &str,
        name: &str,
        description: Option<String>,
    ) -> Option<Result<ProjectRecord, Box<dyn std::error::Error>>> {
        Some(
            self.auth_service
                .as_ref()?
                .create_project(project_id, name, description)
                .await,
        )
    }

    pub async fn delete_project(
        &self,
        project_id: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        Some(self.auth_service.as_ref()?.delete_project(project_id).await)
    }

    pub fn list_principals(&self) -> Option<Vec<PrincipalRecord>> {
        Some(self.auth_service.as_ref()?.list_principals())
    }

    pub async fn create_principal(
        &self,
        name: &str,
    ) -> Option<Result<PrincipalRecord, Box<dyn std::error::Error>>> {
        Some(self.auth_service.as_ref()?.create_principal(name).await)
    }

    pub async fn delete_principal(
        &self,
        principal_id: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        Some(
            self.auth_service
                .as_ref()?
                .delete_principal(principal_id)
                .await,
        )
    }

    pub fn list_tokens(&self, principal_id: Option<&str>) -> Option<Vec<TokenRecord>> {
        Some(self.auth_service.as_ref()?.list_tokens(principal_id))
    }

    pub async fn create_token(
        &self,
        principal_id: &str,
        name: &str,
    ) -> Option<Result<(String, TokenRecord), Box<dyn std::error::Error>>> {
        Some(
            self.auth_service
                .as_ref()?
                .create_token(principal_id, name)
                .await,
        )
    }

    pub async fn delete_token(
        &self,
        token_hash: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        Some(self.auth_service.as_ref()?.delete_token(token_hash).await)
    }

    pub fn list_role_bindings(
        &self,
        principal_id: Option<&str>,
        project_id: Option<&str>,
    ) -> Option<Vec<RoleBindingRecord>> {
        Some(
            self.auth_service
                .as_ref()?
                .list_role_bindings(principal_id, project_id),
        )
    }

    pub async fn create_role_binding(
        &self,
        principal_id: &str,
        role: Role,
        project_id: Option<String>,
    ) -> Option<Result<RoleBindingRecord, Box<dyn std::error::Error>>> {
        Some(
            self.auth_service
                .as_ref()?
                .create_role_binding(principal_id, role, project_id)
                .await,
        )
    }

    pub fn role_catalog(&self) -> Option<Vec<RoleCatalogEntry>> {
        self.auth_service.as_ref()?;
        Some(
            Role::all()
                .iter()
                .cloned()
                .map(|role| RoleCatalogEntry {
                    description: role.description(),
                    permissions: role.permissions(),
                    role,
                })
                .collect(),
        )
    }

    pub fn principal_access(
        &self,
        principal_id: &str,
        project_id: Option<&str>,
    ) -> Option<PrincipalAccessSnapshot> {
        let auth_service = self.auth_service.as_ref()?;
        let principal = auth_service.get_principal(principal_id)?;
        let ctx = auth_service.build_auth_context(principal_id)?;

        let mut role_bindings = auth_service.list_role_bindings(Some(principal_id), None);
        role_bindings.sort_by(|left, right| {
            left.project_id
                .cmp(&right.project_id)
                .then_with(|| left.role.cmp(&right.role))
                .then_with(|| left.binding_id.cmp(&right.binding_id))
        });

        let mut instance_permissions = Permission::all()
            .iter()
            .filter(|permission| auth_service.is_allowed(&ctx, (*permission).clone(), None))
            .cloned()
            .collect::<Vec<_>>();
        instance_permissions.sort_by(|left, right| left.as_str().cmp(right.as_str()));

        let project_ids = match project_id {
            Some(project_id) => vec![ProjectId(project_id.to_string())],
            None => auth_service.accessible_projects(&ctx),
        };

        let mut project_access = project_ids
            .into_iter()
            .map(|project_id| {
                let project_id_value = project_id.0.clone();
                let mut project_role_bindings = role_bindings
                    .iter()
                    .filter(|binding| {
                        binding.project_id.as_deref() == Some(project_id_value.as_str())
                            || (binding.project_id.is_none()
                                && binding.role == Role::InstanceAdmin.as_str())
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                project_role_bindings.sort_by(|left, right| {
                    left.project_id
                        .cmp(&right.project_id)
                        .then_with(|| left.role.cmp(&right.role))
                        .then_with(|| left.binding_id.cmp(&right.binding_id))
                });

                let mut permissions = Permission::all()
                    .iter()
                    .filter(|permission| {
                        auth_service.is_allowed(&ctx, (*permission).clone(), Some(&project_id))
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                permissions.sort_by(|left, right| left.as_str().cmp(right.as_str()));

                ProjectAccessSnapshot {
                    project_id: project_id_value,
                    role_bindings: project_role_bindings,
                    permissions,
                }
            })
            .collect::<Vec<_>>();
        project_access.sort_by(|left, right| left.project_id.cmp(&right.project_id));

        Some(PrincipalAccessSnapshot {
            principal,
            role_bindings,
            instance_permissions,
            project_access,
        })
    }

    pub async fn delete_role_binding(
        &self,
        binding_id: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        Some(
            self.auth_service
                .as_ref()?
                .delete_role_binding(binding_id)
                .await,
        )
    }

    pub async fn export_control_plane_snapshot(
        &self,
    ) -> Option<Result<ControlPlaneSnapshot, Box<dyn std::error::Error>>> {
        let _ = self.auth_service.as_ref()?;
        let _ = self.virtual_keys.as_ref()?;
        let projects = self
            .list_projects()?
            .into_iter()
            .map(|project| ControlPlaneProjectSnapshot {
                project_id: project.project_id,
                name: project.name,
                description: project.description,
                active: project.active,
                created_at: project.created_at,
            })
            .collect::<Vec<_>>();
        Some(
            async {
                let mut project_configs = Vec::with_capacity(projects.len());
                for project in &projects {
                    project_configs.push(
                        self.project_config_snapshot_inner(&project.project_id)
                            .await?,
                    );
                }
                Ok(ControlPlaneSnapshot {
                    exported_at: crate::governance::current_timestamp_string(),
                    projects,
                    managed_providers: self.managed_providers().unwrap_or_default(),
                    virtual_keys: self.list_virtual_keys().unwrap_or_default(),
                    project_configs,
                })
            }
            .await,
        )
    }

    pub fn validate_control_plane_snapshot(
        &self,
        snapshot: &ControlPlaneSnapshot,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        self.auth_service.as_ref()?;
        self.virtual_keys.as_ref()?;
        Some(self.validate_control_plane_snapshot_inner(snapshot))
    }

    pub async fn import_control_plane_snapshot(
        &self,
        snapshot: ControlPlaneSnapshot,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let auth_service = self.auth_service.as_ref()?;
        self.governance.as_ref()?;
        self.store.as_ref()?;
        self.virtual_keys.as_ref()?;
        Some(
            async {
                self.validate_control_plane_snapshot_inner(&snapshot)?;

                for project in snapshot.projects {
                    auth_service
                        .upsert_project(proxy_auth::store::ProjectRecord {
                            project_id: project.project_id,
                            name: project.name,
                            description: project.description,
                            active: project.active,
                            created_at: project.created_at,
                        })
                        .await?;
                }

                for provider in snapshot.managed_providers {
                    match self.upsert_managed_provider(provider).await {
                        Some(Ok(())) => {}
                        Some(Err(error)) => return Err(error),
                        None => return Err("virtual keys not enabled".into()),
                    }
                }

                for project_config in snapshot.project_configs {
                    match self
                        .apply_project_config(project_config, "import_control_plane_snapshot")
                        .await
                    {
                        Some(Ok(_revision)) => {}
                        Some(Err(error)) => return Err(error),
                        None => return Err("governance or store not enabled".into()),
                    }
                }

                for key in snapshot.virtual_keys {
                    match self.upsert_virtual_key_record(key).await {
                        Some(Ok(())) => {}
                        Some(Err(error)) => return Err(error),
                        None => return Err("virtual keys not enabled".into()),
                    }
                }

                Ok(())
            }
            .await,
        )
    }

    pub fn list_project_policies(&self) -> Option<Vec<ProjectPolicyRecord>> {
        Some(self.governance.as_ref()?.list_project_policies())
    }

    pub async fn export_project_config(
        &self,
        project_id: &str,
    ) -> Option<Result<ProjectConfigSnapshot, Box<dyn std::error::Error>>> {
        Some(self.project_config_snapshot_inner(project_id).await)
    }

    pub fn validate_project_config_snapshot(
        &self,
        snapshot: &ProjectConfigSnapshot,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        self.governance.as_ref()?;
        Some(self.validate_project_config_snapshot_against_runtime(snapshot))
    }

    pub async fn list_project_config_revisions(
        &self,
        project_id: &str,
        limit: u32,
    ) -> Option<Result<Vec<ProjectConfigRevision>, Box<dyn std::error::Error>>> {
        let changes = match self
            .get_governance_changes(project_id, Some("project_config_revision"), limit)
            .await
        {
            Some(Ok(changes)) => changes,
            Some(Err(error)) => return Some(Err(error)),
            None => return None,
        };
        Some(Ok(changes
            .into_iter()
            .filter_map(|change| change.after_json)
            .filter_map(|raw| serde_json::from_str::<ProjectConfigRevision>(&raw).ok())
            .collect()))
    }

    pub async fn get_project_config_revision(
        &self,
        project_id: &str,
        revision_id: &str,
    ) -> Option<Result<Option<ProjectConfigRevision>, Box<dyn std::error::Error>>> {
        let revisions = match self.list_project_config_revisions(project_id, 500).await {
            Some(Ok(revisions)) => revisions,
            Some(Err(error)) => return Some(Err(error)),
            None => return None,
        };
        Some(Ok(revisions
            .into_iter()
            .find(|revision| revision.revision_id == revision_id)))
    }

    pub async fn get_project_config_revision_state(
        &self,
        project_id: &str,
    ) -> Option<Result<Option<ProjectConfigRevisionState>, Box<dyn std::error::Error>>> {
        let changes = match self
            .get_governance_changes(project_id, Some("project_config_revision_state"), 1)
            .await
        {
            Some(Ok(changes)) => changes,
            Some(Err(error)) => return Some(Err(error)),
            None => return None,
        };
        Some(Ok(changes
            .into_iter()
            .find_map(|change| change.after_json)
            .and_then(|raw| {
                serde_json::from_str::<ProjectConfigRevisionState>(&raw).ok()
            })))
    }

    pub async fn apply_project_config(
        &self,
        snapshot: ProjectConfigSnapshot,
        source: &str,
    ) -> Option<Result<ProjectConfigRevision, Box<dyn std::error::Error>>> {
        self.governance.as_ref()?;
        self.store.as_ref()?;
        Some(async {
            self.validate_project_config_snapshot_against_runtime(&snapshot)?;
            let previous_snapshot = self.project_config_snapshot_inner(&snapshot.project_id).await?;
            let apply_result = self
                .apply_project_config_snapshot_inner(&snapshot)
                .await
                .map_err(|error| error.to_string());
            if let Err(apply_error) = apply_result {
                let restore_error = self
                    .apply_project_config_snapshot_inner(&previous_snapshot)
                    .await
                    .err()
                    .map(|error| error.to_string());
                let error_message = match restore_error {
                    Some(restore_error) => format!(
                        "project config apply failed ({apply_error}); automatic rollback also failed ({restore_error})"
                    ),
                    None => format!(
                        "project config apply failed ({apply_error}); previous snapshot restored"
                    ),
                };
                return Err(error_message.into());
            }
            match self
                .record_project_config_revision(&snapshot.project_id, source)
                .await?
            {
                Some(revision) => Ok(revision),
                None => Err("store not enabled".into()),
            }
        }
        .await)
    }

    pub async fn rollback_project_config(
        &self,
        project_id: &str,
        revision_id: &str,
    ) -> Option<Result<ProjectConfigRevision, Box<dyn std::error::Error>>> {
        self.governance.as_ref()?;
        self.store.as_ref()?;
        Some(
            async {
                let revision = match self
                    .get_project_config_revision(project_id, revision_id)
                    .await
                {
                    Some(Ok(Some(revision))) => revision,
                    Some(Ok(None)) => {
                        return Err(format!("revision '{revision_id}' not found").into())
                    }
                    Some(Err(error)) => return Err(error),
                    None => return Err("store not enabled".into()),
                };
                let source = format!("rollback_project_config:{revision_id}");
                match self.apply_project_config(revision.snapshot, &source).await {
                    Some(result) => result,
                    None => Err("store not enabled".into()),
                }
            }
            .await,
        )
    }

    pub fn get_project_policy(&self, project_id: &str) -> Option<ProjectPolicyRecord> {
        self.governance.as_ref()?.project_policy(project_id)
    }

    pub async fn upsert_project_policy(
        &self,
        record: ProjectPolicyRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let project_id = record.project_id.clone();
        let upsert_result = governance
            .upsert_project_policy(record)
            .await
            .map_err(|error| error.to_string());
        Some(match upsert_result {
            Ok(()) => {
                match self
                    .ensure_project_config_revision_recorded(&project_id, "upsert_project_policy")
                    .await
                {
                    Ok(()) => Ok(()),
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error.into()),
        })
    }

    pub async fn delete_project_policy(
        &self,
        project_id: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let delete_result = governance
            .delete_project_policy(project_id)
            .await
            .map_err(|error| error.to_string());
        Some(match delete_result {
            Ok(result) => {
                if result {
                    if let Err(error) = self
                        .ensure_project_config_revision_recorded(
                            project_id,
                            "delete_project_policy",
                        )
                        .await
                    {
                        return Some(Err(error));
                    }
                }
                Ok(result)
            }
            Err(error) => Err(error.into()),
        })
    }

    pub async fn get_governance_changes(
        &self,
        project_id: &str,
        resource_type: Option<&str>,
        limit: u32,
    ) -> Option<Result<Vec<GovernanceChangeRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_governance_changes(project_id, resource_type, limit)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        )
    }

    pub fn list_routing_rules(&self, project_id: Option<&str>) -> Option<Vec<RoutingRuleRecord>> {
        Some(self.governance.as_ref()?.list_routing_rules(project_id))
    }

    pub async fn upsert_routing_rule(
        &self,
        record: RoutingRuleRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let project_id = record.project_id.clone();
        let upsert_result = governance
            .upsert_routing_rule(record)
            .await
            .map_err(|error| error.to_string());
        Some(match upsert_result {
            Ok(()) => {
                match self
                    .ensure_project_config_revision_recorded(&project_id, "upsert_routing_rule")
                    .await
                {
                    Ok(()) => Ok(()),
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error.into()),
        })
    }

    pub async fn delete_routing_rule(
        &self,
        rule_id: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let project_id = governance
            .list_routing_rules(None)
            .into_iter()
            .find(|record| record.rule_id == rule_id)
            .map(|record| record.project_id);
        let delete_result = governance
            .delete_routing_rule(rule_id)
            .await
            .map_err(|error| error.to_string());
        Some(match delete_result {
            Ok(result) => {
                if result {
                    if let Some(project_id) = project_id.as_deref() {
                        if let Err(error) = self
                            .ensure_project_config_revision_recorded(
                                project_id,
                                "delete_routing_rule",
                            )
                            .await
                        {
                            return Some(Err(error));
                        }
                    }
                }
                Ok(result)
            }
            Err(error) => Err(error.into()),
        })
    }

    pub fn list_safety_policies(&self) -> Option<Vec<SafetyPolicyRecord>> {
        Some(self.governance.as_ref()?.list_safety_policies())
    }

    pub fn get_safety_policy(&self, project_id: &str) -> Option<SafetyPolicyRecord> {
        self.governance.as_ref()?.safety_policy(project_id)
    }

    pub async fn upsert_safety_policy(
        &self,
        record: SafetyPolicyRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let project_id = record.project_id.clone();
        let upsert_result = governance
            .upsert_safety_policy(record)
            .await
            .map_err(|error| error.to_string());
        Some(match upsert_result {
            Ok(()) => {
                match self
                    .ensure_project_config_revision_recorded(&project_id, "upsert_safety_policy")
                    .await
                {
                    Ok(()) => Ok(()),
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error.into()),
        })
    }

    pub async fn delete_safety_policy(
        &self,
        project_id: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let delete_result = governance
            .delete_safety_policy(project_id)
            .await
            .map_err(|error| error.to_string());
        Some(match delete_result {
            Ok(result) => {
                if result {
                    if let Err(error) = self
                        .ensure_project_config_revision_recorded(project_id, "delete_safety_policy")
                        .await
                    {
                        return Some(Err(error));
                    }
                }
                Ok(result)
            }
            Err(error) => Err(error.into()),
        })
    }

    pub fn list_safety_detectors(
        &self,
        project_id: Option<&str>,
    ) -> Option<Result<Vec<EffectiveDetectorConfig>, Box<dyn std::error::Error>>> {
        Some(effective_detector_catalog(
            self.governance.as_deref(),
            project_id,
        ))
    }

    pub fn list_semantic_policies(&self) -> Option<Vec<ProjectSemanticPolicyRecord>> {
        Some(self.governance.as_ref()?.list_semantic_policies())
    }

    pub fn get_semantic_policy(&self, project_id: &str) -> Option<ProjectSemanticPolicyRecord> {
        self.governance.as_ref()?.semantic_policy(project_id)
    }

    pub async fn upsert_semantic_policy(
        &self,
        record: ProjectSemanticPolicyRecord,
    ) -> Option<Result<SemanticPolicyMutationResult, Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        if let Err(error) = governance.upsert_semantic_policy(record.clone()).await {
            return Some(Err(error));
        }
        let result = match &self.semantic_safety {
            Some(plugin) => plugin.sync_policy_record(&record).await,
            None => SemanticPolicyMutationResult {
                policy_version: record.version.clone(),
                synced: false,
                sync_error: Some("semantic safety plugin not enabled".to_string()),
            },
        };
        if result.synced || result.sync_error.is_some() {
            if let Err(error) = self
                .ensure_project_config_revision_recorded(
                    &record.project_id,
                    "upsert_semantic_policy",
                )
                .await
            {
                return Some(Err(error));
            }
        }
        Some(Ok(result))
    }

    pub async fn delete_semantic_policy(
        &self,
        project_id: &str,
    ) -> Option<Result<SemanticPolicyDeleteResult, Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let existed = match governance.delete_semantic_policy(project_id).await {
            Ok(existed) => existed,
            Err(error) => return Some(Err(error)),
        };
        let sync = match &self.semantic_safety {
            Some(plugin) => match plugin.delete_policy(project_id).await {
                Ok(()) => SemanticPolicyDeleteResult {
                    existed,
                    synced: true,
                    sync_error: None,
                },
                Err(error) => SemanticPolicyDeleteResult {
                    existed,
                    synced: false,
                    sync_error: Some(error.to_string()),
                },
            },
            None => SemanticPolicyDeleteResult {
                existed,
                synced: false,
                sync_error: Some("semantic safety plugin not enabled".to_string()),
            },
        };
        if existed {
            if let Err(error) = self
                .ensure_project_config_revision_recorded(project_id, "delete_semantic_policy")
                .await
            {
                return Some(Err(error));
            }
        }
        Some(Ok(sync))
    }

    pub fn list_project_tools(&self, project_id: Option<&str>) -> Option<Vec<ProjectToolRecord>> {
        Some(self.governance.as_ref()?.list_project_tools(project_id))
    }

    pub fn get_project_tool(&self, project_id: &str, tool_name: &str) -> Option<ProjectToolRecord> {
        self.governance
            .as_ref()?
            .project_tool(project_id, tool_name)
    }

    pub async fn upsert_project_tool(
        &self,
        record: ProjectToolRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let project_id = record.project_id.clone();
        let upsert_result = governance
            .upsert_project_tool(record)
            .await
            .map_err(|error| error.to_string());
        Some(match upsert_result {
            Ok(()) => {
                match self
                    .ensure_project_config_revision_recorded(&project_id, "upsert_project_tool")
                    .await
                {
                    Ok(()) => Ok(()),
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error.into()),
        })
    }

    pub async fn delete_project_tool(
        &self,
        project_id: &str,
        tool_name: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let delete_result = governance
            .delete_project_tool(project_id, tool_name)
            .await
            .map_err(|error| error.to_string());
        Some(match delete_result {
            Ok(result) => {
                if result {
                    if let Err(error) = self
                        .ensure_project_config_revision_recorded(project_id, "delete_project_tool")
                        .await
                    {
                        return Some(Err(error));
                    }
                }
                Ok(result)
            }
            Err(error) => Err(error.into()),
        })
    }

    pub fn list_project_prompts(
        &self,
        project_id: Option<&str>,
        prompt_name: Option<&str>,
    ) -> Option<Vec<ProjectPromptRecord>> {
        Some(
            self.governance
                .as_ref()?
                .list_project_prompts(project_id, prompt_name),
        )
    }

    pub fn get_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Option<ProjectPromptRecord> {
        self.governance
            .as_ref()?
            .project_prompt(project_id, prompt_name, version)
    }

    pub async fn upsert_project_prompt(
        &self,
        record: ProjectPromptRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        let project_id = record.project_id.clone();
        let upsert_result = governance
            .upsert_project_prompt(record)
            .await
            .map_err(|error| error.to_string());
        Some(match upsert_result {
            Ok(()) => {
                match self
                    .ensure_project_config_revision_recorded(&project_id, "upsert_project_prompt")
                    .await
                {
                    Ok(()) => Ok(()),
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error.into()),
        })
    }

    pub async fn delete_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let governance = self.governance.as_ref()?;
        Some(
            match governance
                .delete_project_prompt(project_id, prompt_name, version)
                .await
                .map_err(|error| error.to_string())
            {
                Ok(result) => {
                    if result {
                        if let Err(error) = self
                            .ensure_project_config_revision_recorded(
                                project_id,
                                "delete_project_prompt",
                            )
                            .await
                        {
                            return Some(Err(error));
                        }
                    }
                    Ok(result)
                }
                Err(error) => Err(error.into()),
            },
        )
    }

    pub async fn list_project_rollout_policies(
        &self,
        project_id: Option<&str>,
    ) -> Option<Result<Vec<ProjectRolloutPolicyRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_project_rollout_policies(project_id)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn get_project_rollout_policy(
        &self,
        project_id: &str,
        policy_name: &str,
    ) -> Option<Result<Option<ProjectRolloutPolicyRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_project_rollout_policy(project_id, policy_name)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn upsert_project_rollout_policy(
        &self,
        record: ProjectRolloutPolicyRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        let upsert_result = store
            .upsert_project_rollout_policy(&record)
            .await
            .map_err(|error| error.to_string());
        Some(match upsert_result {
            Ok(()) => {
                match self
                    .ensure_project_config_revision_recorded(
                        &record.project_id,
                        "upsert_project_rollout_policy",
                    )
                    .await
                {
                    Ok(()) => Ok(()),
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error.into()),
        })
    }

    pub async fn delete_project_rollout_policy(
        &self,
        project_id: &str,
        policy_name: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        let delete_result = store
            .delete_project_rollout_policy(project_id, policy_name)
            .await
            .map_err(|error| error.to_string());
        Some(match delete_result {
            Ok(result) => {
                if result {
                    if let Err(error) = self
                        .ensure_project_config_revision_recorded(
                            project_id,
                            "delete_project_rollout_policy",
                        )
                        .await
                    {
                        return Some(Err(error));
                    }
                }
                Ok(result)
            }
            Err(error) => Err(error.into()),
        })
    }

    pub async fn list_project_prompt_rollouts(
        &self,
        project_id: &str,
        prompt_name: &str,
    ) -> Option<Result<Vec<ProjectPromptRolloutRecord>, Box<dyn std::error::Error>>> {
        Some(Ok(self
            .governance
            .as_ref()?
            .list_project_prompt_rollouts(project_id, prompt_name)))
    }

    pub async fn get_project_prompt_rollout(
        &self,
        project_id: &str,
        prompt_name: &str,
        rollout_id: &str,
    ) -> Option<Result<Option<ProjectPromptRolloutRecord>, Box<dyn std::error::Error>>> {
        Some(Ok(self.governance.as_ref()?.project_prompt_rollout(
            project_id,
            prompt_name,
            rollout_id,
        )))
    }

    pub async fn upsert_project_prompt_rollout(
        &self,
        record: ProjectPromptRolloutRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        Some(
            self.governance
                .as_ref()?
                .upsert_project_prompt_rollout(record)
                .await,
        )
    }

    pub async fn list_project_datasets(
        &self,
        project_id: Option<&str>,
    ) -> Option<Result<Vec<ProjectDatasetRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_project_datasets(project_id)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn get_project_dataset(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Option<Result<Option<ProjectDatasetRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_project_dataset(project_id, dataset_name)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn upsert_project_dataset(
        &self,
        record: ProjectDatasetRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .upsert_project_dataset(&record)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn delete_project_dataset(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .delete_project_dataset(project_id, dataset_name)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn list_project_dataset_items(
        &self,
        project_id: &str,
        dataset_name: &str,
    ) -> Option<Result<Vec<ProjectDatasetItemRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_project_dataset_items(project_id, dataset_name)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn get_project_dataset_item(
        &self,
        project_id: &str,
        dataset_name: &str,
        item_id: &str,
    ) -> Option<Result<Option<ProjectDatasetItemRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_project_dataset_item(project_id, dataset_name, item_id)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn upsert_project_dataset_item(
        &self,
        record: ProjectDatasetItemRecord,
    ) -> Option<Result<(), Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .upsert_project_dataset_item(&record)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn delete_project_dataset_item(
        &self,
        project_id: &str,
        dataset_name: &str,
        item_id: &str,
    ) -> Option<Result<bool, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .delete_project_dataset_item(project_id, dataset_name, item_id)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn list_project_eval_runs(
        &self,
        project_id: &str,
        dataset_name: Option<&str>,
    ) -> Option<Result<Vec<ProjectEvalRunRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_project_eval_runs(project_id, dataset_name)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn get_project_eval_run(
        &self,
        project_id: &str,
        run_id: &str,
    ) -> Option<Result<Option<ProjectEvalRunRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_project_eval_run(project_id, run_id)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn list_project_eval_run_items(
        &self,
        project_id: &str,
        run_id: &str,
    ) -> Option<Result<Vec<ProjectEvalRunItemRecord>, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            store
                .get_project_eval_run_items(project_id, run_id)
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>),
        )
    }

    pub async fn execute_project_eval_run(
        &self,
        project_id: &str,
        request: ProjectEvalRunRequest,
    ) -> Option<Result<ProjectEvalRunExecution, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(execute_project_eval_run(Arc::clone(store), project_id, request).await)
    }

    pub async fn queue_project_eval_run(
        &self,
        project_id: &str,
        request: ProjectEvalRunRequest,
    ) -> Option<Result<ProjectEvalRunRecord, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(queue_project_eval_run(Arc::clone(store), project_id, request).await)
    }

    pub async fn compare_project_eval_runs(
        &self,
        project_id: &str,
        baseline_run_id: &str,
        candidate_run_id: &str,
        gate_request: Option<ProjectEvalRunComparisonGateRequest>,
    ) -> Option<Result<ProjectEvalRunComparison, Box<dyn std::error::Error>>> {
        let store = self.store.as_ref()?;
        Some(
            compare_project_eval_runs(
                Arc::clone(store),
                project_id,
                baseline_run_id,
                candidate_run_id,
                gate_request,
            )
            .await,
        )
    }

    pub async fn get_semantic_policy_sync_status(
        &self,
        project_id: &str,
    ) -> Option<SemanticPolicySyncStatus> {
        Some(match &self.semantic_safety {
            Some(plugin) => plugin.get_sync_status(project_id).await,
            None => SemanticPolicySyncStatus {
                project_id: project_id.to_string(),
                policy_version: String::new(),
                index_state: "disabled".to_string(),
                updated_at: String::new(),
                stored_exemplar_count: 0,
                available: false,
                ready: false,
                backend: None,
                message: None,
                error: Some("semantic safety plugin not enabled".to_string()),
            },
        })
    }
}
