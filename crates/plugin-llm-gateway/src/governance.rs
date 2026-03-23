use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use hyper::header::HeaderMap;
use proxy_auth::Role;

use crate::prompt_cache::PromptCacheRoutingAffinity;
use crate::store::{
    GatewayStore, GovernanceChangeRecord, ProjectPolicyRecord, ProjectPromptRecord,
    ProjectPromptRolloutRecord, ProjectSemanticPolicyRecord, ProjectToolRecord, RoutingRuleRecord,
    SafetyPolicyRecord, Store,
};

static GOVERNANCE_CHANGE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SafetyMode {
    RedactAndForward,
    Block,
    ObserveOnly,
}

impl SafetyMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RedactAndForward => "redact_and_forward",
            Self::Block => "block",
            Self::ObserveOnly => "observe_only",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "redact_and_forward" => Some(Self::RedactAndForward),
            "block" => Some(Self::Block),
            "observe_only" | "log" => Some(Self::ObserveOnly),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetyRule {
    pub pattern: Option<String>,
    pub description: Option<String>,
    pub detector_class: Option<String>,
    pub action: Option<String>,
    pub verification: Option<String>,
    pub replacement: Option<String>,
    pub path_patterns: Vec<String>,
    pub allowlist_patterns: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct RoutingDecision {
    pub deny_reason: Option<String>,
    pub ordered_providers: Vec<String>,
    pub provider_weights: HashMap<String, u32>,
    pub timeout_override: Option<Duration>,
    pub cache_affinity_provider: Option<String>,
    pub cache_affinity_applied: bool,
    pub matched_rule_id: Option<String>,
    pub matched_rule_name: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedProjectPrompt {
    pub record: ProjectPromptRecord,
    pub rollout_id: Option<String>,
    pub rollout_mode: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ProviderStats {
    pub ewma_latency_ms: f64,
    pub ewma_error_rate: f64,
    pub ewma_timeout_rate: f64,
    pub ewma_rate_limit_rate: f64,
    pub active_requests: u32,
    pub samples: u64,
}

impl Default for ProviderStats {
    fn default() -> Self {
        Self {
            ewma_latency_ms: 0.0,
            ewma_error_rate: 0.0,
            ewma_timeout_rate: 0.0,
            ewma_rate_limit_rate: 0.0,
            active_requests: 0,
            samples: 0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderPenaltyBreakdown {
    pub active_requests: f64,
    pub latency: f64,
    pub error: f64,
    pub timeout: f64,
    pub rate_limit: f64,
}

impl ProviderPenaltyBreakdown {
    pub fn total(&self) -> f64 {
        self.active_requests + self.latency + self.error + self.timeout + self.rate_limit
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProviderHealthStats {
    pub active_requests: u32,
    pub samples: u64,
    pub ewma_latency_ms: f64,
    pub ewma_error_rate: f64,
    pub ewma_timeout_rate: f64,
    pub ewma_rate_limit_rate: f64,
    pub penalties: ProviderPenaltyBreakdown,
}

#[derive(Clone, Copy, Debug)]
pub enum ProviderOutcome {
    Success { latency: Duration },
    ErrorStatus { latency: Duration },
    RateLimited { latency: Duration },
    Timeout,
}

#[derive(Clone)]
pub struct GovernanceState {
    store: Option<Arc<Store>>,
    project_policies: Arc<DashMap<String, ProjectPolicyRecord>>,
    routing_rules: Arc<DashMap<String, Vec<RoutingRuleRecord>>>,
    safety_policies: Arc<DashMap<String, SafetyPolicyRecord>>,
    semantic_policies: Arc<DashMap<String, ProjectSemanticPolicyRecord>>,
    project_tools: Arc<DashMap<String, Vec<ProjectToolRecord>>>,
    project_prompts: Arc<DashMap<String, Vec<ProjectPromptRecord>>>,
    project_prompt_rollouts: Arc<DashMap<String, Vec<ProjectPromptRolloutRecord>>>,
    provider_stats: Arc<DashMap<String, ProviderStats>>,
}

impl GovernanceState {
    pub fn new(store: Option<Arc<Store>>) -> Self {
        Self {
            store,
            project_policies: Arc::new(DashMap::new()),
            routing_rules: Arc::new(DashMap::new()),
            safety_policies: Arc::new(DashMap::new()),
            semantic_policies: Arc::new(DashMap::new()),
            project_tools: Arc::new(DashMap::new()),
            project_prompts: Arc::new(DashMap::new()),
            project_prompt_rollouts: Arc::new(DashMap::new()),
            provider_stats: Arc::new(DashMap::new()),
        }
    }

    pub async fn load_from_store(&self) -> Result<(), Box<dyn std::error::Error>> {
        let store = match &self.store {
            Some(store) => store,
            None => return Ok(()),
        };

        self.project_policies.clear();
        self.routing_rules.clear();
        self.safety_policies.clear();
        self.semantic_policies.clear();
        self.project_tools.clear();
        self.project_prompts.clear();
        self.project_prompt_rollouts.clear();

        for record in store.get_all_project_policies().await? {
            self.project_policies
                .insert(record.project_id.clone(), record);
        }

        for record in store.get_routing_rules(None).await? {
            self.routing_rules
                .entry(record.project_id.clone())
                .or_default()
                .push(record);
        }
        for mut entry in self.routing_rules.iter_mut() {
            entry.sort_by(|left, right| {
                right
                    .priority
                    .cmp(&left.priority)
                    .then_with(|| left.created_at.cmp(&right.created_at))
            });
        }

        for record in store.get_all_safety_policies().await? {
            self.safety_policies
                .insert(record.project_id.clone(), record);
        }
        for record in store.get_all_semantic_policies().await? {
            self.semantic_policies
                .insert(record.project_id.clone(), record);
        }
        for record in store.get_project_tools(None).await? {
            self.project_tools
                .entry(record.project_id.clone())
                .or_default()
                .push(record);
        }
        for mut entry in self.project_tools.iter_mut() {
            entry.sort_by(|left, right| left.tool_name.cmp(&right.tool_name));
        }
        let mut prompt_keys = Vec::new();
        for record in store.get_project_prompts(None, None).await? {
            prompt_keys.push((record.project_id.clone(), record.prompt_name.clone()));
            self.project_prompts
                .entry(record.project_id.clone())
                .or_default()
                .push(record);
        }
        for mut entry in self.project_prompts.iter_mut() {
            entry.sort_by(|left, right| {
                left.prompt_name
                    .cmp(&right.prompt_name)
                    .then_with(|| left.environment.cmp(&right.environment))
                    .then_with(|| left.version.cmp(&right.version))
            });
        }

        prompt_keys.sort();
        prompt_keys.dedup();
        for (project_id, prompt_name) in prompt_keys {
            for record in store
                .get_project_prompt_rollouts(&project_id, &prompt_name)
                .await?
            {
                self.project_prompt_rollouts
                    .entry(record.project_id.clone())
                    .or_default()
                    .push(record);
            }
        }
        for mut entry in self.project_prompt_rollouts.iter_mut() {
            entry.sort_by(|left, right| {
                left.prompt_name
                    .cmp(&right.prompt_name)
                    .then_with(|| right.created_at.cmp(&left.created_at))
                    .then_with(|| right.rollout_id.cmp(&left.rollout_id))
            });
        }

        Ok(())
    }

    pub fn project_policy(&self, project_id: &str) -> Option<ProjectPolicyRecord> {
        self.project_policies
            .get(project_id)
            .map(|entry| entry.value().clone())
    }

    pub fn safety_policy(&self, project_id: &str) -> Option<SafetyPolicyRecord> {
        self.safety_policies
            .get(project_id)
            .map(|entry| entry.value().clone())
    }

    pub fn safety_mode_for_project(&self, project_id: &str) -> SafetyMode {
        self.safety_policy(project_id)
            .and_then(|record| SafetyMode::parse(&record.mode))
            .unwrap_or(SafetyMode::RedactAndForward)
    }

    pub fn safety_rules_for_project(&self, project_id: &str) -> Vec<SafetyRule> {
        self.safety_policy(project_id)
            .and_then(|record| record.rules_json)
            .and_then(|json| parse_safety_rules(&json))
            .unwrap_or_default()
    }

    pub fn list_project_policies(&self) -> Vec<ProjectPolicyRecord> {
        self.project_policies
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub async fn upsert_project_policy(
        &self,
        record: ProjectPolicyRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let before = self.project_policy(&record.project_id);
        if let Some(store) = &self.store {
            store.upsert_project_policy(&record).await?;
        }
        self.project_policies
            .insert(record.project_id.clone(), record.clone());
        self.record_governance_change(
            &record.project_id,
            "project_policy",
            "policy",
            "upsert",
            before.as_ref().map(project_policy_json),
            Some(project_policy_json(&record)),
        )
        .await?;
        Ok(())
    }

    pub async fn delete_project_policy(
        &self,
        project_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let before = self.project_policy(project_id);
        if let Some(store) = &self.store {
            let deleted = store.delete_project_policy(project_id).await?;
            self.project_policies.remove(project_id);
            if deleted {
                self.record_governance_change(
                    project_id,
                    "project_policy",
                    "policy",
                    "delete",
                    before.as_ref().map(project_policy_json),
                    None,
                )
                .await?;
            }
            return Ok(deleted);
        }
        let deleted = self.project_policies.remove(project_id).is_some();
        if deleted {
            self.record_governance_change(
                project_id,
                "project_policy",
                "policy",
                "delete",
                before.as_ref().map(project_policy_json),
                None,
            )
            .await?;
        }
        Ok(deleted)
    }

    pub fn list_routing_rules(&self, project_id: Option<&str>) -> Vec<RoutingRuleRecord> {
        match project_id {
            Some(project_id) => self
                .routing_rules
                .get(project_id)
                .map(|entry| entry.value().clone())
                .unwrap_or_default(),
            None => self
                .routing_rules
                .iter()
                .flat_map(|entry| entry.value().clone())
                .collect(),
        }
    }

    pub async fn upsert_routing_rule(
        &self,
        record: RoutingRuleRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let before = self.routing_rule(&record.rule_id);
        if let Some(store) = &self.store {
            store.upsert_routing_rule(&record).await?;
        }
        let mut rules = self
            .routing_rules
            .entry(record.project_id.clone())
            .or_default();
        if let Some(existing) = rules.iter_mut().find(|rule| rule.rule_id == record.rule_id) {
            *existing = record.clone();
        } else {
            rules.push(record.clone());
        }
        rules.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.created_at.cmp(&right.created_at))
        });
        drop(rules);
        self.record_governance_change(
            &record.project_id,
            "routing_rule",
            &record.rule_id,
            "upsert",
            before.as_ref().map(routing_rule_json),
            Some(routing_rule_json(&record)),
        )
        .await?;
        Ok(())
    }

    pub async fn delete_routing_rule(
        &self,
        rule_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let before = self.routing_rule(rule_id);
        if let Some(store) = &self.store {
            let deleted = store.delete_routing_rule(rule_id).await?;
            if deleted {
                self.remove_routing_rule(rule_id);
                if let Some(before) = before.as_ref() {
                    self.record_governance_change(
                        &before.project_id,
                        "routing_rule",
                        rule_id,
                        "delete",
                        Some(routing_rule_json(before)),
                        None,
                    )
                    .await?;
                }
            }
            return Ok(deleted);
        }
        let deleted = self.remove_routing_rule(rule_id);
        if deleted {
            if let Some(before) = before.as_ref() {
                self.record_governance_change(
                    &before.project_id,
                    "routing_rule",
                    rule_id,
                    "delete",
                    Some(routing_rule_json(before)),
                    None,
                )
                .await?;
            }
        }
        Ok(deleted)
    }

    pub fn list_safety_policies(&self) -> Vec<SafetyPolicyRecord> {
        self.safety_policies
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub fn semantic_policy(&self, project_id: &str) -> Option<ProjectSemanticPolicyRecord> {
        self.semantic_policies
            .get(project_id)
            .map(|entry| entry.value().clone())
    }

    pub fn list_semantic_policies(&self) -> Vec<ProjectSemanticPolicyRecord> {
        self.semantic_policies
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub fn project_tool(&self, project_id: &str, tool_name: &str) -> Option<ProjectToolRecord> {
        self.project_tools.get(project_id).and_then(|entry| {
            entry
                .iter()
                .find(|record| record.tool_name == tool_name)
                .cloned()
        })
    }

    pub fn list_project_tools(&self, project_id: Option<&str>) -> Vec<ProjectToolRecord> {
        match project_id {
            Some(project_id) => self
                .project_tools
                .get(project_id)
                .map(|entry| entry.value().clone())
                .unwrap_or_default(),
            None => self
                .project_tools
                .iter()
                .flat_map(|entry| entry.value().clone())
                .collect(),
        }
    }

    pub fn project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Option<ProjectPromptRecord> {
        self.project_prompts.get(project_id).and_then(|entry| {
            entry
                .iter()
                .find(|record| record.prompt_name == prompt_name && record.version == version)
                .cloned()
        })
    }

    pub fn resolve_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: Option<&str>,
        environment: &str,
        rollout_seed: Option<&str>,
    ) -> Option<ResolvedProjectPrompt> {
        let prompts = self.project_prompts.get(project_id)?;
        match version {
            Some(version) => prompts
                .iter()
                .find(|record| record.prompt_name == prompt_name && record.version == version)
                .cloned()
                .map(|record| ResolvedProjectPrompt {
                    record,
                    rollout_id: None,
                    rollout_mode: None,
                }),
            None => {
                let baseline = prompts.iter().find(|record| {
                    record.prompt_name == prompt_name
                        && record.environment == environment
                        && record.active
                })?;
                if let Some(rollout) =
                    self.active_canary_prompt_rollout(project_id, prompt_name, environment)
                {
                    let use_candidate = rollout_seed
                        .map(|seed| {
                            rollout_bucket(&rollout.rollout_id, seed) < rollout.traffic_percent
                        })
                        .unwrap_or(false);
                    let selected = if use_candidate {
                        prompts
                            .iter()
                            .find(|record| {
                                record.prompt_name == prompt_name
                                    && record.environment == environment
                                    && record.version == rollout.candidate_version
                            })
                            .unwrap_or(baseline)
                    } else {
                        baseline
                    };
                    return Some(ResolvedProjectPrompt {
                        record: selected.clone(),
                        rollout_id: Some(rollout.rollout_id),
                        rollout_mode: Some("canary".to_string()),
                    });
                }
                Some(ResolvedProjectPrompt {
                    record: baseline.clone(),
                    rollout_id: None,
                    rollout_mode: None,
                })
            }
        }
    }

    pub fn list_project_prompts(
        &self,
        project_id: Option<&str>,
        prompt_name: Option<&str>,
    ) -> Vec<ProjectPromptRecord> {
        let mut prompts = match project_id {
            Some(project_id) => self
                .project_prompts
                .get(project_id)
                .map(|entry| entry.value().clone())
                .unwrap_or_default(),
            None => self
                .project_prompts
                .iter()
                .flat_map(|entry| entry.value().clone())
                .collect(),
        };
        if let Some(prompt_name) = prompt_name {
            prompts.retain(|record| record.prompt_name == prompt_name);
        }
        prompts
    }

    pub fn project_prompt_rollout(
        &self,
        project_id: &str,
        prompt_name: &str,
        rollout_id: &str,
    ) -> Option<ProjectPromptRolloutRecord> {
        self.project_prompt_rollouts
            .get(project_id)
            .and_then(|entry| {
                entry
                    .iter()
                    .find(|record| {
                        record.prompt_name == prompt_name && record.rollout_id == rollout_id
                    })
                    .cloned()
            })
    }

    pub fn list_project_prompt_rollouts(
        &self,
        project_id: &str,
        prompt_name: &str,
    ) -> Vec<ProjectPromptRolloutRecord> {
        self.project_prompt_rollouts
            .get(project_id)
            .map(|entry| {
                entry
                    .iter()
                    .filter(|record| record.prompt_name == prompt_name)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub async fn upsert_safety_policy(
        &self,
        record: SafetyPolicyRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let before = self.safety_policy(&record.project_id);
        if let Some(store) = &self.store {
            store.upsert_safety_policy(&record).await?;
        }
        self.safety_policies
            .insert(record.project_id.clone(), record.clone());
        self.record_governance_change(
            &record.project_id,
            "safety_policy",
            "safety",
            "upsert",
            before.as_ref().map(safety_policy_json),
            Some(safety_policy_json(&record)),
        )
        .await?;
        Ok(())
    }

    pub async fn delete_safety_policy(
        &self,
        project_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let before = self.safety_policy(project_id);
        if let Some(store) = &self.store {
            let deleted = store.delete_safety_policy(project_id).await?;
            self.safety_policies.remove(project_id);
            if deleted {
                self.record_governance_change(
                    project_id,
                    "safety_policy",
                    "safety",
                    "delete",
                    before.as_ref().map(safety_policy_json),
                    None,
                )
                .await?;
            }
            return Ok(deleted);
        }
        let deleted = self.safety_policies.remove(project_id).is_some();
        if deleted {
            self.record_governance_change(
                project_id,
                "safety_policy",
                "safety",
                "delete",
                before.as_ref().map(safety_policy_json),
                None,
            )
            .await?;
        }
        Ok(deleted)
    }

    pub async fn upsert_semantic_policy(
        &self,
        record: ProjectSemanticPolicyRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let before = self.semantic_policy(&record.project_id);
        if let Some(store) = &self.store {
            store.upsert_semantic_policy(&record).await?;
        }
        self.semantic_policies
            .insert(record.project_id.clone(), record.clone());
        self.record_governance_change(
            &record.project_id,
            "semantic_policy",
            "semantic-safety",
            "upsert",
            before.as_ref().map(semantic_policy_json),
            Some(semantic_policy_json(&record)),
        )
        .await?;
        Ok(())
    }

    pub async fn delete_semantic_policy(
        &self,
        project_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let before = self.semantic_policy(project_id);
        if let Some(store) = &self.store {
            let deleted = store.delete_semantic_policy(project_id).await?;
            self.semantic_policies.remove(project_id);
            if deleted {
                self.record_governance_change(
                    project_id,
                    "semantic_policy",
                    "semantic-safety",
                    "delete",
                    before.as_ref().map(semantic_policy_json),
                    None,
                )
                .await?;
            }
            return Ok(deleted);
        }
        let deleted = self.semantic_policies.remove(project_id).is_some();
        if deleted {
            self.record_governance_change(
                project_id,
                "semantic_policy",
                "semantic-safety",
                "delete",
                before.as_ref().map(semantic_policy_json),
                None,
            )
            .await?;
        }
        Ok(deleted)
    }

    pub async fn upsert_project_tool(
        &self,
        record: ProjectToolRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let before = self.project_tool(&record.project_id, &record.tool_name);
        if let Some(store) = &self.store {
            store.upsert_project_tool(&record).await?;
        }
        let mut tools = self
            .project_tools
            .entry(record.project_id.clone())
            .or_default();
        if let Some(existing) = tools
            .iter_mut()
            .find(|tool| tool.tool_name == record.tool_name)
        {
            *existing = record.clone();
        } else {
            tools.push(record.clone());
        }
        tools.sort_by(|left, right| left.tool_name.cmp(&right.tool_name));
        drop(tools);
        self.record_governance_change(
            &record.project_id,
            "project_tool",
            &record.tool_name,
            "upsert",
            before.as_ref().map(project_tool_json),
            Some(project_tool_json(&record)),
        )
        .await?;
        Ok(())
    }

    pub async fn upsert_project_prompt(
        &self,
        record: ProjectPromptRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let before = self.project_prompt(&record.project_id, &record.prompt_name, &record.version);
        let mut deactivated = Vec::new();
        let mut prompts = self
            .project_prompts
            .entry(record.project_id.clone())
            .or_default();
        if record.active {
            for prompt in prompts.iter_mut() {
                if prompt.prompt_name == record.prompt_name
                    && prompt.environment == record.environment
                    && prompt.version != record.version
                    && prompt.active
                {
                    prompt.active = false;
                    prompt.updated_at = current_timestamp_string();
                    deactivated.push(prompt.clone());
                }
            }
        }
        if let Some(existing) = prompts.iter_mut().find(|prompt| {
            prompt.prompt_name == record.prompt_name && prompt.version == record.version
        }) {
            *existing = record.clone();
        } else {
            prompts.push(record.clone());
        }
        prompts.sort_by(|left, right| {
            left.prompt_name
                .cmp(&right.prompt_name)
                .then_with(|| left.environment.cmp(&right.environment))
                .then_with(|| left.version.cmp(&right.version))
        });
        drop(prompts);

        if let Some(store) = &self.store {
            for deactivated_record in &deactivated {
                store.upsert_project_prompt(deactivated_record).await?;
            }
            store.upsert_project_prompt(&record).await?;
        }
        for deactivated_record in &deactivated {
            self.record_governance_change(
                &deactivated_record.project_id,
                "project_prompt",
                &format!(
                    "{}:{}",
                    deactivated_record.prompt_name, deactivated_record.version
                ),
                "deactivate",
                Some(project_prompt_json_with_active(deactivated_record, true)),
                Some(project_prompt_json(deactivated_record)),
            )
            .await?;
        }
        self.record_governance_change(
            &record.project_id,
            "project_prompt",
            &format!("{}:{}", record.prompt_name, record.version),
            "upsert",
            before
                .as_ref()
                .map(|before| project_prompt_json_with_active(before, before.active)),
            Some(project_prompt_json(&record)),
        )
        .await?;
        Ok(())
    }

    pub async fn upsert_project_prompt_rollout(
        &self,
        record: ProjectPromptRolloutRecord,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(store) = &self.store {
            store.upsert_project_prompt_rollout(&record).await?;
        }
        let mut rollouts = self
            .project_prompt_rollouts
            .entry(record.project_id.clone())
            .or_default();
        if let Some(existing) = rollouts.iter_mut().find(|rollout| {
            rollout.prompt_name == record.prompt_name && rollout.rollout_id == record.rollout_id
        }) {
            *existing = record;
        } else {
            rollouts.push(record);
        }
        rollouts.sort_by(|left, right| {
            left.prompt_name
                .cmp(&right.prompt_name)
                .then_with(|| right.created_at.cmp(&left.created_at))
                .then_with(|| right.rollout_id.cmp(&left.rollout_id))
        });
        Ok(())
    }

    pub async fn delete_project_tool(
        &self,
        project_id: &str,
        tool_name: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let before = self.project_tool(project_id, tool_name);
        if let Some(store) = &self.store {
            let deleted = store.delete_project_tool(project_id, tool_name).await?;
            if deleted {
                self.remove_project_tool(project_id, tool_name);
                self.record_governance_change(
                    project_id,
                    "project_tool",
                    tool_name,
                    "delete",
                    before.as_ref().map(project_tool_json),
                    None,
                )
                .await?;
            }
            return Ok(deleted);
        }
        let deleted = self.remove_project_tool(project_id, tool_name);
        if deleted {
            self.record_governance_change(
                project_id,
                "project_tool",
                tool_name,
                "delete",
                before.as_ref().map(project_tool_json),
                None,
            )
            .await?;
        }
        Ok(deleted)
    }

    pub async fn delete_project_prompt(
        &self,
        project_id: &str,
        prompt_name: &str,
        version: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let before = self.project_prompt(project_id, prompt_name, version);
        if let Some(store) = &self.store {
            let deleted = store
                .delete_project_prompt(project_id, prompt_name, version)
                .await?;
            if deleted {
                self.remove_project_prompt(project_id, prompt_name, version);
                self.record_governance_change(
                    project_id,
                    "project_prompt",
                    &format!("{prompt_name}:{version}"),
                    "delete",
                    before.as_ref().map(project_prompt_json),
                    None,
                )
                .await?;
            }
            return Ok(deleted);
        }
        let deleted = self.remove_project_prompt(project_id, prompt_name, version);
        if deleted {
            self.record_governance_change(
                project_id,
                "project_prompt",
                &format!("{prompt_name}:{version}"),
                "delete",
                before.as_ref().map(project_prompt_json),
                None,
            )
            .await?;
        }
        Ok(deleted)
    }

    pub fn begin_provider_request(&self, provider_name: &str) {
        let mut stats = self
            .provider_stats
            .entry(provider_name.to_string())
            .or_default();
        stats.active_requests = stats.active_requests.saturating_add(1);
    }

    pub fn record_provider_outcome(&self, provider_name: &str, outcome: ProviderOutcome) {
        let mut stats = self
            .provider_stats
            .entry(provider_name.to_string())
            .or_default();

        stats.active_requests = stats.active_requests.saturating_sub(1);
        stats.samples = stats.samples.saturating_add(1);

        let alpha = 0.25;
        let update = |current: &mut f64, value: f64| {
            if *current == 0.0 {
                *current = value;
            } else {
                *current = (*current * (1.0 - alpha)) + (value * alpha);
            }
        };

        match outcome {
            ProviderOutcome::Success { latency } => {
                update(&mut stats.ewma_latency_ms, latency.as_secs_f64() * 1000.0);
                update(&mut stats.ewma_error_rate, 0.0);
                update(&mut stats.ewma_timeout_rate, 0.0);
                update(&mut stats.ewma_rate_limit_rate, 0.0);
            }
            ProviderOutcome::ErrorStatus { latency } => {
                update(&mut stats.ewma_latency_ms, latency.as_secs_f64() * 1000.0);
                update(&mut stats.ewma_error_rate, 1.0);
                update(&mut stats.ewma_timeout_rate, 0.0);
                update(&mut stats.ewma_rate_limit_rate, 0.0);
            }
            ProviderOutcome::RateLimited { latency } => {
                update(&mut stats.ewma_latency_ms, latency.as_secs_f64() * 1000.0);
                update(&mut stats.ewma_error_rate, 1.0);
                update(&mut stats.ewma_timeout_rate, 0.0);
                update(&mut stats.ewma_rate_limit_rate, 1.0);
            }
            ProviderOutcome::Timeout => {
                update(&mut stats.ewma_error_rate, 1.0);
                update(&mut stats.ewma_timeout_rate, 1.0);
            }
        }
    }

    pub fn evaluate_routing(
        &self,
        project_id: &str,
        role: Option<&Role>,
        path: &str,
        model: Option<&str>,
        streaming: bool,
        headers: &HeaderMap,
        prompt_tokens: u64,
        available_providers: &[String],
    ) -> RoutingDecision {
        self.evaluate_routing_with_cache_affinity(
            project_id,
            role,
            path,
            model,
            streaming,
            headers,
            prompt_tokens,
            available_providers,
            None,
        )
    }

    pub(crate) fn evaluate_routing_with_cache_affinity(
        &self,
        project_id: &str,
        role: Option<&Role>,
        path: &str,
        model: Option<&str>,
        streaming: bool,
        headers: &HeaderMap,
        prompt_tokens: u64,
        available_providers: &[String],
        cache_affinity: Option<&PromptCacheRoutingAffinity>,
    ) -> RoutingDecision {
        let project_policy = self.project_policy(project_id);
        let mut decision = RoutingDecision {
            deny_reason: None,
            ordered_providers: project_policy
                .as_ref()
                .and_then(|record| parse_string_list(record.fallback_order.as_deref()?))
                .unwrap_or_else(|| available_providers.to_vec()),
            provider_weights: HashMap::new(),
            timeout_override: None,
            cache_affinity_provider: cache_affinity
                .and_then(|affinity| affinity.preferred_provider.clone()),
            cache_affinity_applied: false,
            matched_rule_id: None,
            matched_rule_name: None,
        };

        if let Some(rule) = self.routing_rules.get(project_id).and_then(|rules| {
            rules
                .iter()
                .find(|rule| {
                    rule.enabled
                        && routing_rule_matches(
                            rule,
                            role,
                            path,
                            model,
                            streaming,
                            headers,
                            prompt_tokens,
                        )
                })
                .cloned()
        }) {
            if rule.deny_reason.is_some() {
                decision.matched_rule_id = Some(rule.rule_id.clone());
                decision.matched_rule_name = Some(rule.name.clone());
                decision.deny_reason = rule.deny_reason.clone();
                decision.ordered_providers.clear();
                return decision;
            }
            if let Some(order) = rule.provider_order.as_deref().and_then(parse_string_list) {
                decision.ordered_providers = order;
            }
            if let Some(weights) = rule.provider_weights.as_deref().and_then(parse_weight_map) {
                decision.provider_weights = weights;
            }
            decision.timeout_override = rule.timeout_secs.map(Duration::from_secs);
            decision.matched_rule_id = Some(rule.rule_id.clone());
            decision.matched_rule_name = Some(rule.name.clone());
        }

        let adaptive_enabled = project_policy
            .as_ref()
            .map(|record| record.adaptive_enabled)
            .unwrap_or(true);
        let provider_input_costs = project_policy
            .as_ref()
            .and_then(|record| record.provider_input_costs.as_deref())
            .and_then(parse_float_map);
        let baseline_order = self.rank_providers(
            available_providers,
            &decision.ordered_providers,
            &decision.provider_weights,
            adaptive_enabled,
            prompt_tokens,
            provider_input_costs.as_ref(),
            None,
        );
        decision.ordered_providers = self.rank_providers(
            available_providers,
            &decision.ordered_providers,
            &decision.provider_weights,
            adaptive_enabled,
            prompt_tokens,
            provider_input_costs.as_ref(),
            cache_affinity,
        );
        decision.cache_affinity_applied = cache_affinity
            .map(|affinity| !affinity.is_empty())
            .unwrap_or(false)
            && decision.ordered_providers != baseline_order;
        decision
    }

    fn rank_providers(
        &self,
        available_providers: &[String],
        preferred_order: &[String],
        provider_weights: &HashMap<String, u32>,
        adaptive_enabled: bool,
        prompt_tokens: u64,
        provider_input_costs: Option<&HashMap<String, f64>>,
        cache_affinity: Option<&PromptCacheRoutingAffinity>,
    ) -> Vec<String> {
        let preferred_index: HashMap<&str, usize> = preferred_order
            .iter()
            .enumerate()
            .map(|(idx, provider)| (provider.as_str(), idx))
            .collect();

        let mut providers = available_providers.to_vec();
        providers.sort_by(|left, right| {
            let left_score = self.routing_score(
                left,
                preferred_index.get(left.as_str()).copied(),
                provider_weights.get(left).copied(),
                adaptive_enabled,
                prompt_tokens,
                provider_input_costs,
                cache_affinity,
            );
            let right_score = self.routing_score(
                right,
                preferred_index.get(right.as_str()).copied(),
                provider_weights.get(right).copied(),
                adaptive_enabled,
                prompt_tokens,
                provider_input_costs,
                cache_affinity,
            );
            left_score
                .partial_cmp(&right_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        providers
    }

    fn routing_score(
        &self,
        provider_name: &str,
        preferred_index: Option<usize>,
        weight: Option<u32>,
        adaptive_enabled: bool,
        prompt_tokens: u64,
        provider_input_costs: Option<&HashMap<String, f64>>,
        cache_affinity: Option<&PromptCacheRoutingAffinity>,
    ) -> f64 {
        let mut score = preferred_index.unwrap_or(10_000) as f64 * 1000.0;
        score += weight.map(|v| 1000.0 / v.max(1) as f64).unwrap_or(100.0);
        if let Some(cache_affinity) = cache_affinity {
            score -= cache_affinity
                .positive_bonuses
                .get(provider_name)
                .copied()
                .unwrap_or(0.0);
            score += cache_affinity
                .negative_penalties
                .get(provider_name)
                .copied()
                .unwrap_or(0.0);
        }

        if adaptive_enabled {
            score += self.provider_penalty_breakdown(provider_name).total();
            score += provider_input_costs
                .and_then(|costs| costs.get(provider_name).copied())
                .map(|input_cost_per_1k| {
                    (prompt_tokens as f64 / 1000.0) * input_cost_per_1k * 10_000.0
                })
                .unwrap_or(0.0);
        }

        score
    }

    pub fn provider_stats_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .provider_stats
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        keys.sort();
        keys
    }

    pub fn provider_health_stats(&self, provider_name: &str) -> ProviderHealthStats {
        let stats = self
            .provider_stats
            .get(provider_name)
            .map(|entry| entry.clone())
            .unwrap_or_default();
        ProviderHealthStats {
            active_requests: stats.active_requests,
            samples: stats.samples,
            ewma_latency_ms: stats.ewma_latency_ms,
            ewma_error_rate: stats.ewma_error_rate,
            ewma_timeout_rate: stats.ewma_timeout_rate,
            ewma_rate_limit_rate: stats.ewma_rate_limit_rate,
            penalties: self.provider_penalty_breakdown(provider_name),
        }
    }

    pub fn provider_penalty_breakdown(&self, provider_name: &str) -> ProviderPenaltyBreakdown {
        self.provider_stats
            .get(provider_name)
            .map(|stats| ProviderPenaltyBreakdown {
                active_requests: stats.active_requests as f64 * 10.0,
                latency: stats.ewma_latency_ms,
                error: stats.ewma_error_rate * 10_000.0,
                timeout: stats.ewma_timeout_rate * 15_000.0,
                rate_limit: stats.ewma_rate_limit_rate * 12_000.0,
            })
            .unwrap_or_default()
    }

    fn routing_rule(&self, rule_id: &str) -> Option<RoutingRuleRecord> {
        self.routing_rules
            .iter()
            .flat_map(|entry| entry.value().clone().into_iter())
            .find(|rule| rule.rule_id == rule_id)
    }

    async fn record_governance_change(
        &self,
        project_id: &str,
        resource_type: &str,
        resource_id: &str,
        action: &str,
        before_json: Option<String>,
        after_json: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(store) = &self.store {
            store
                .append_governance_change(&GovernanceChangeRecord {
                    change_id: next_governance_change_id(),
                    project_id: project_id.to_string(),
                    resource_type: resource_type.to_string(),
                    resource_id: resource_id.to_string(),
                    action: action.to_string(),
                    before_json,
                    after_json,
                    changed_at: current_timestamp_string(),
                })
                .await?;
        }
        Ok(())
    }

    fn remove_routing_rule(&self, rule_id: &str) -> bool {
        let keys: Vec<String> = self
            .routing_rules
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys {
            if let Some(mut rules) = self.routing_rules.get_mut(&key) {
                let before = rules.len();
                rules.retain(|rule| rule.rule_id != rule_id);
                if rules.len() != before {
                    return true;
                }
            }
        }
        false
    }

    fn remove_project_tool(&self, project_id: &str, tool_name: &str) -> bool {
        if let Some(mut tools) = self.project_tools.get_mut(project_id) {
            let before = tools.len();
            tools.retain(|tool| tool.tool_name != tool_name);
            let removed = before != tools.len();
            if tools.is_empty() {
                drop(tools);
                self.project_tools.remove(project_id);
            }
            return removed;
        }
        false
    }

    fn remove_project_prompt(&self, project_id: &str, prompt_name: &str, version: &str) -> bool {
        if let Some(mut prompts) = self.project_prompts.get_mut(project_id) {
            let before = prompts.len();
            prompts
                .retain(|prompt| !(prompt.prompt_name == prompt_name && prompt.version == version));
            let removed = before != prompts.len();
            if prompts.is_empty() {
                drop(prompts);
                self.project_prompts.remove(project_id);
            }
            return removed;
        }
        false
    }
}

fn routing_rule_matches(
    rule: &RoutingRuleRecord,
    role: Option<&Role>,
    path: &str,
    model: Option<&str>,
    streaming: bool,
    headers: &HeaderMap,
    prompt_tokens: u64,
) -> bool {
    if let Some(match_path) = &rule.match_path {
        if !wildcard_match(path, match_path) {
            return false;
        }
    }
    if let Some(match_model) = &rule.match_model {
        if model
            .map(|value| !wildcard_match(value, match_model))
            .unwrap_or(true)
        {
            return false;
        }
    }
    if let Some(expected_streaming) = rule.match_streaming {
        if expected_streaming != streaming {
            return false;
        }
    }
    if let Some(match_role) = &rule.match_role {
        if role
            .map(|value| value.as_str() != match_role)
            .unwrap_or(true)
        {
            return false;
        }
    }
    if let Some(min_prompt_tokens) = rule.min_prompt_tokens {
        if prompt_tokens < min_prompt_tokens as u64 {
            return false;
        }
    }
    if let Some(max_prompt_tokens) = rule.max_prompt_tokens {
        if prompt_tokens > max_prompt_tokens as u64 {
            return false;
        }
    }
    if let Some(match_headers) = &rule.match_headers {
        if !header_predicates_match(headers, match_headers) {
            return false;
        }
    }
    true
}

fn header_predicates_match(headers: &HeaderMap, raw: &str) -> bool {
    let value: serde_json::Value = match serde_json::from_str(raw) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let object = match value.as_object() {
        Some(object) => object,
        None => return false,
    };

    object.iter().all(|(name, expected)| {
        let header_value = headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        expected
            .as_str()
            .map(|expected| header_value.contains(expected))
            .unwrap_or(false)
    })
}

fn wildcard_match(value: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some((prefix, suffix)) = pattern.split_once('*') {
        return value.starts_with(prefix) && value.ends_with(suffix);
    }
    value == pattern
}

fn parse_string_list(raw: &str) -> Option<Vec<String>> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    Some(
        value
            .as_array()?
            .iter()
            .filter_map(|entry| entry.as_str().map(ToString::to_string))
            .collect(),
    )
}

fn parse_weight_map(raw: &str) -> Option<HashMap<String, u32>> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    let object = value.as_object()?;
    let mut weights = HashMap::new();
    for (provider, weight) in object {
        if let Some(weight) = weight.as_u64() {
            weights.insert(provider.clone(), weight as u32);
        }
    }
    Some(weights)
}

fn parse_float_map(raw: &str) -> Option<HashMap<String, f64>> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    let object = value.as_object()?;
    let mut values = HashMap::new();
    for (key, value) in object {
        values.insert(key.clone(), value.as_f64()?);
    }
    Some(values)
}

fn parse_safety_rules(raw: &str) -> Option<Vec<SafetyRule>> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    let array = value.as_array()?;
    let mut rules = Vec::with_capacity(array.len());
    for entry in array {
        let object = entry.as_object()?;
        let pattern = object
            .get("pattern")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let description = object
            .get("description")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let detector_class = object
            .get("detector_class")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        if pattern.is_none() && detector_class.is_none() {
            return None;
        }
        let action = object
            .get("action")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let verification = object
            .get("verification")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let replacement = object
            .get("replacement")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let path_patterns =
            parse_string_array(object.get("path_patterns").or_else(|| object.get("paths")))?;
        let allowlist_patterns = parse_string_array(
            object
                .get("allowlist_patterns")
                .or_else(|| object.get("allowlist")),
        )?;
        rules.push(SafetyRule {
            pattern,
            description,
            detector_class,
            action,
            verification,
            replacement,
            path_patterns,
            allowlist_patterns,
        });
    }
    Some(rules)
}

fn parse_string_array(value: Option<&serde_json::Value>) -> Option<Vec<String>> {
    let Some(value) = value else {
        return Some(Vec::new());
    };
    let array = value.as_array()?;
    Some(
        array
            .iter()
            .map(|entry| entry.as_str().map(ToString::to_string))
            .collect::<Option<Vec<_>>>()?,
    )
}

fn project_policy_json(record: &ProjectPolicyRecord) -> String {
    serde_json::json!({
        "project_id": record.project_id.clone(),
        "budget_limit": record.budget_limit,
        "budget_duration": record.budget_duration.clone(),
        "rpm_limit": record.rpm_limit,
        "tpm_limit": record.tpm_limit,
        "fallback_order": record
            .fallback_order
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "adaptive_enabled": record.adaptive_enabled,
        "timeout_secs": record.timeout_secs,
        "provider_rpm_limits": record
            .provider_rpm_limits
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "provider_tpm_limits": record
            .provider_tpm_limits
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "provider_timeouts": record
            .provider_timeouts
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "provider_input_costs": record
            .provider_input_costs
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "provider_output_costs": record
            .provider_output_costs
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "tool_approval_mode": record.tool_approval_mode.clone(),
        "allowed_tools": record
            .allowed_tools
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "updated_at": record.updated_at.clone(),
    })
    .to_string()
}

fn routing_rule_json(record: &RoutingRuleRecord) -> String {
    serde_json::json!({
        "rule_id": record.rule_id.clone(),
        "project_id": record.project_id.clone(),
        "name": record.name.clone(),
        "priority": record.priority,
        "enabled": record.enabled,
        "match_path": record.match_path.clone(),
        "match_model": record.match_model.clone(),
        "match_streaming": record.match_streaming,
        "match_role": record.match_role.clone(),
        "match_headers": record
            .match_headers
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "min_prompt_tokens": record.min_prompt_tokens,
        "max_prompt_tokens": record.max_prompt_tokens,
        "deny_reason": record.deny_reason.clone(),
        "provider_order": record
            .provider_order
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "provider_weights": record
            .provider_weights
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "timeout_secs": record.timeout_secs,
        "created_at": record.created_at.clone(),
    })
    .to_string()
}

fn safety_policy_json(record: &SafetyPolicyRecord) -> String {
    serde_json::json!({
        "project_id": record.project_id.clone(),
        "mode": record.mode.clone(),
        "rules": record
            .rules_json
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "updated_at": record.updated_at.clone(),
    })
    .to_string()
}

fn semantic_policy_json(record: &ProjectSemanticPolicyRecord) -> String {
    serde_json::json!({
        "project_id": record.project_id.clone(),
        "version": record.version.clone(),
        "enabled": record.enabled,
        "entities": record
            .entities_json
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "topics": record
            .topics_json
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "updated_at": record.updated_at.clone(),
    })
    .to_string()
}

fn project_tool_json(record: &ProjectToolRecord) -> String {
    serde_json::json!({
        "project_id": record.project_id.clone(),
        "tool_name": record.tool_name.clone(),
        "description": record.description.clone(),
        "input_schema": serde_json::from_str::<serde_json::Value>(&record.input_schema_json)
            .unwrap_or(serde_json::Value::Null),
        "executor_kind": record.executor_kind.clone(),
        "executor_config": record
            .executor_config_json
            .as_ref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or(serde_json::Value::Null),
        "enabled": record.enabled,
        "timeout_ms": record.timeout_ms,
        "updated_at": record.updated_at.clone(),
    })
    .to_string()
}

fn project_prompt_json(record: &ProjectPromptRecord) -> String {
    project_prompt_json_with_active(record, record.active)
}

fn project_prompt_json_with_active(record: &ProjectPromptRecord, active: bool) -> String {
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
        "active": active,
        "updated_at": record.updated_at.clone(),
    })
    .to_string()
}

fn next_governance_change_id() -> String {
    let sequence = GOVERNANCE_CHANGE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("chg-{}-{sequence:016x}", current_timestamp_string())
}

pub fn current_timestamp_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string()
}

#[derive(Clone, Debug)]
struct ActivePromptCanaryRollout {
    rollout_id: String,
    candidate_version: String,
    traffic_percent: u8,
}

impl GovernanceState {
    fn active_canary_prompt_rollout(
        &self,
        project_id: &str,
        prompt_name: &str,
        environment: &str,
    ) -> Option<ActivePromptCanaryRollout> {
        let rollouts = self.project_prompt_rollouts.get(project_id)?;
        rollouts.iter().find_map(|record| {
            if record.prompt_name != prompt_name || record.status != "applied_canary" {
                return None;
            }
            let runtime = parse_runtime_prompt_rollout_config(&record.comparison_json)?;
            if !runtime.mode.eq_ignore_ascii_case("canary") {
                return None;
            }
            if let Some(target_environment) = record.target_environment.as_deref() {
                if target_environment != environment {
                    return None;
                }
            }
            Some(ActivePromptCanaryRollout {
                rollout_id: record.rollout_id.clone(),
                candidate_version: record.candidate_version.clone(),
                traffic_percent: runtime.traffic_percent,
            })
        })
    }
}

#[derive(Clone, Debug)]
struct RuntimePromptRolloutConfig {
    mode: String,
    traffic_percent: u8,
}

fn parse_runtime_prompt_rollout_config(raw: &str) -> Option<RuntimePromptRolloutConfig> {
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
    if traffic_percent == 0 {
        return None;
    }
    Some(RuntimePromptRolloutConfig {
        mode: mode.to_string(),
        traffic_percent,
    })
}

fn rollout_bucket(rollout_id: &str, seed: &str) -> u8 {
    let mut hasher = DefaultHasher::new();
    rollout_id.hash(&mut hasher);
    seed.hash(&mut hasher);
    (hasher.finish() % 100) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::header::HeaderValue;

    fn providers() -> Vec<String> {
        vec!["openai".to_string(), "anthropic".to_string()]
    }

    #[tokio::test]
    async fn evaluate_routing_uses_project_policy_fallback_order() {
        let governance = GovernanceState::new(None);
        governance
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: "project-a".to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["anthropic","openai"]"#.to_string()),
                adaptive_enabled: false,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: None,
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();

        let decision = governance.evaluate_routing(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            true,
            &HeaderMap::new(),
            120,
            &providers(),
        );

        assert_eq!(
            decision.ordered_providers,
            vec!["anthropic".to_string(), "openai".to_string()]
        );
        assert!(decision.deny_reason.is_none());
    }

    #[tokio::test]
    async fn evaluate_routing_prefers_highest_priority_matching_rule() {
        let governance = GovernanceState::new(None);

        governance
            .upsert_routing_rule(RoutingRuleRecord {
                rule_id: "low".to_string(),
                project_id: "project-a".to_string(),
                name: "low".to_string(),
                priority: 10,
                enabled: true,
                match_path: Some("/v1/*".to_string()),
                match_model: Some("gpt-*".to_string()),
                match_streaming: None,
                match_role: None,
                match_headers: None,
                min_prompt_tokens: None,
                max_prompt_tokens: None,
                deny_reason: None,
                provider_order: Some(r#"["openai","anthropic"]"#.to_string()),
                provider_weights: None,
                timeout_secs: Some(5),
                created_at: "1".to_string(),
            })
            .await
            .unwrap();
        governance
            .upsert_routing_rule(RoutingRuleRecord {
                rule_id: "high".to_string(),
                project_id: "project-a".to_string(),
                name: "high".to_string(),
                priority: 50,
                enabled: true,
                match_path: Some("/v1/chat/*".to_string()),
                match_model: Some("gpt-4o".to_string()),
                match_streaming: Some(true),
                match_role: Some(Role::ProjectRuntime.as_str().to_string()),
                match_headers: Some(r#"{"x-route":"beta"}"#.to_string()),
                min_prompt_tokens: Some(100),
                max_prompt_tokens: Some(500),
                deny_reason: None,
                provider_order: Some(r#"["anthropic","openai"]"#.to_string()),
                provider_weights: Some(r#"{"anthropic":90}"#.to_string()),
                timeout_secs: Some(30),
                created_at: "2".to_string(),
            })
            .await
            .unwrap();

        let mut headers = HeaderMap::new();
        headers.insert("x-route", HeaderValue::from_static("beta-rollout"));

        let decision = governance.evaluate_routing(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            true,
            &headers,
            150,
            &providers(),
        );

        assert_eq!(
            decision.ordered_providers,
            vec!["anthropic".to_string(), "openai".to_string()]
        );
        assert_eq!(decision.provider_weights.get("anthropic"), Some(&90));
        assert_eq!(decision.timeout_override, Some(Duration::from_secs(30)));
        assert_eq!(decision.matched_rule_id.as_deref(), Some("high"));
        assert_eq!(decision.matched_rule_name.as_deref(), Some("high"));
    }

    #[tokio::test]
    async fn evaluate_routing_deny_rule_short_circuits_provider_selection() {
        let governance = GovernanceState::new(None);
        governance
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: "project-a".to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["openai","anthropic"]"#.to_string()),
                adaptive_enabled: false,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: None,
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();
        governance
            .upsert_routing_rule(RoutingRuleRecord {
                rule_id: "deny".to_string(),
                project_id: "project-a".to_string(),
                name: "deny".to_string(),
                priority: 100,
                enabled: true,
                match_path: Some("/v1/chat/*".to_string()),
                match_model: Some("gpt-4o".to_string()),
                match_streaming: None,
                match_role: None,
                match_headers: None,
                min_prompt_tokens: None,
                max_prompt_tokens: None,
                deny_reason: Some("manual review required".to_string()),
                provider_order: None,
                provider_weights: None,
                timeout_secs: None,
                created_at: "1".to_string(),
            })
            .await
            .unwrap();

        let decision = governance.evaluate_routing(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            false,
            &HeaderMap::new(),
            20,
            &providers(),
        );

        assert_eq!(
            decision.deny_reason.as_deref(),
            Some("manual review required")
        );
        assert!(decision.ordered_providers.is_empty());
    }

    #[tokio::test]
    async fn evaluate_routing_requires_all_match_criteria() {
        let governance = GovernanceState::new(None);
        governance
            .upsert_routing_rule(RoutingRuleRecord {
                rule_id: "strict".to_string(),
                project_id: "project-a".to_string(),
                name: "strict".to_string(),
                priority: 10,
                enabled: true,
                match_path: Some("/v1/chat/*".to_string()),
                match_model: Some("gpt-*".to_string()),
                match_streaming: Some(true),
                match_role: Some(Role::ProjectRuntime.as_str().to_string()),
                match_headers: Some(r#"{"x-route":"beta"}"#.to_string()),
                min_prompt_tokens: Some(50),
                max_prompt_tokens: Some(500),
                deny_reason: None,
                provider_order: Some(r#"["anthropic","openai"]"#.to_string()),
                provider_weights: None,
                timeout_secs: None,
                created_at: "1".to_string(),
            })
            .await
            .unwrap();

        let mut headers = HeaderMap::new();
        headers.insert("x-route", HeaderValue::from_static("beta-canary"));

        let matched = governance.evaluate_routing(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            true,
            &headers,
            100,
            &providers(),
        );
        assert_eq!(
            matched.ordered_providers,
            vec!["anthropic".to_string(), "openai".to_string()]
        );

        let missed = governance.evaluate_routing(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            true,
            &headers,
            10,
            &providers(),
        );
        assert_eq!(missed.ordered_providers, providers());
    }

    #[test]
    fn adaptive_ranking_prefers_healthier_provider() {
        let governance = GovernanceState::new(None);
        governance.record_provider_outcome("openai", ProviderOutcome::Timeout);
        governance.record_provider_outcome(
            "anthropic",
            ProviderOutcome::Success {
                latency: Duration::from_millis(15),
            },
        );

        let decision = governance.evaluate_routing(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            true,
            &HeaderMap::new(),
            64,
            &providers(),
        );

        assert_eq!(
            decision.ordered_providers,
            vec!["anthropic".to_string(), "openai".to_string()]
        );
    }

    #[tokio::test]
    async fn cache_affinity_bonus_can_override_single_step_fallback_order() {
        let governance = GovernanceState::new(None);
        governance
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: "project-a".to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["beta","alpha"]"#.to_string()),
                adaptive_enabled: false,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: None,
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();

        let available = vec!["alpha".to_string(), "beta".to_string()];
        let affinity = PromptCacheRoutingAffinity {
            preferred_provider: Some("alpha".to_string()),
            positive_bonuses: HashMap::from([("alpha".to_string(), 2_000.0)]),
            negative_penalties: HashMap::new(),
        };
        let decision = governance.evaluate_routing_with_cache_affinity(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            false,
            &HeaderMap::new(),
            32,
            &available,
            Some(&affinity),
        );

        assert_eq!(
            decision.ordered_providers,
            vec!["alpha".to_string(), "beta".to_string()]
        );
        assert_eq!(decision.cache_affinity_provider.as_deref(), Some("alpha"));
        assert!(decision.cache_affinity_applied);
    }

    #[tokio::test]
    async fn negative_cache_affinity_penalty_can_override_single_step_fallback_order() {
        let governance = GovernanceState::new(None);
        governance
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: "project-a".to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["alpha","beta"]"#.to_string()),
                adaptive_enabled: false,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: None,
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();

        let available = vec!["alpha".to_string(), "beta".to_string()];
        let affinity = PromptCacheRoutingAffinity {
            preferred_provider: None,
            positive_bonuses: HashMap::new(),
            negative_penalties: HashMap::from([("alpha".to_string(), 2_500.0)]),
        };
        let decision = governance.evaluate_routing_with_cache_affinity(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            false,
            &HeaderMap::new(),
            32,
            &available,
            Some(&affinity),
        );

        assert_eq!(
            decision.ordered_providers,
            vec!["beta".to_string(), "alpha".to_string()]
        );
        assert!(decision.cache_affinity_applied);
        assert_eq!(decision.cache_affinity_provider, None);
    }

    #[tokio::test]
    async fn adaptive_ranking_prefers_cheaper_provider_for_large_prompt() {
        let governance = GovernanceState::new(None);
        governance
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: "project-a".to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["openai","anthropic"]"#.to_string()),
                adaptive_enabled: true,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: Some(r#"{"openai":0.03,"anthropic":0.01}"#.to_string()),
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();

        let decision = governance.evaluate_routing(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            false,
            &HeaderMap::new(),
            10_000,
            &providers(),
        );

        assert_eq!(
            decision.ordered_providers,
            vec!["anthropic".to_string(), "openai".to_string()]
        );
    }

    #[test]
    fn provider_health_stats_expose_penalty_breakdown() {
        let governance = GovernanceState::new(None);
        governance.begin_provider_request("openai");
        governance.record_provider_outcome("openai", ProviderOutcome::Timeout);

        let stats = governance.provider_health_stats("openai");
        assert_eq!(stats.active_requests, 0);
        assert_eq!(stats.samples, 1);
        assert!(stats.ewma_error_rate > 0.0);
        assert!(stats.ewma_timeout_rate > 0.0);
        assert_eq!(stats.penalties.latency, 0.0);
        assert!(stats.penalties.error > 0.0);
        assert!(stats.penalties.timeout > 0.0);
        assert!(stats.penalties.total() > stats.penalties.error);
    }

    #[tokio::test]
    async fn project_policy_can_disable_adaptive_reordering() {
        let governance = GovernanceState::new(None);
        governance
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: "project-a".to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["openai","anthropic"]"#.to_string()),
                adaptive_enabled: false,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: None,
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "1".to_string(),
            })
            .await
            .unwrap();
        governance.record_provider_outcome("openai", ProviderOutcome::Timeout);
        governance.record_provider_outcome(
            "anthropic",
            ProviderOutcome::Success {
                latency: Duration::from_millis(15),
            },
        );

        let decision = governance.evaluate_routing(
            "project-a",
            Some(&Role::ProjectRuntime),
            "/v1/chat/completions",
            Some("gpt-4o"),
            true,
            &HeaderMap::new(),
            64,
            &providers(),
        );

        assert_eq!(
            decision.ordered_providers,
            vec!["openai".to_string(), "anthropic".to_string()]
        );
    }
}
