use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::estimate_prompt_tokens;
use crate::extract_model;
use crate::governance::GovernanceState;
use crate::metrics::LlmMetrics;
use crate::prompt_cache::PromptCacheRoutingAffinity;
use crate::store::{GatewayStore, ProjectPolicyRecord, SemanticCacheEntryRecord, Store};
use crate::virtual_keys::{RoutingDebugTrace, VirtualKeyMeta};
use crate::{cached_request_json_mut, sync_cached_request_json_body, update_request_body};
use async_trait::async_trait;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::header::{HeaderValue, CONTENT_LENGTH, CONTENT_TYPE};
use hyper::{Response, StatusCode};
use proxy_core::plugin::{Action, Plugin, RequestContext};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const SEMANTIC_CACHE_ROUTE_MAX_BONUS: f64 = 3_500.0;
const SEMANTIC_CACHE_ROUTE_MIN_FRESHNESS: f64 = 0.2;

#[derive(Clone, Debug)]
struct SemanticCacheOptIn {
    enabled: bool,
    ttl_secs: u64,
    similarity_threshold: f64,
}

#[derive(Clone, Debug)]
struct SemanticCacheRequestMeta {
    project_id: String,
    provider_name: String,
    model: String,
    normalized_prompt: String,
    prompt_tokens: u64,
    ttl_secs: u64,
}

#[derive(Clone, Debug)]
struct SemanticCacheRoutingPreference {
    model: String,
    normalized_prompt: String,
    similarity_threshold: f64,
}

#[derive(Clone, Debug)]
struct SemanticCacheEntry {
    cache_id: String,
    project_id: String,
    provider_name: String,
    model: String,
    tokens: HashSet<String>,
    response_status: u16,
    content_type: Option<String>,
    response_body: Bytes,
    prompt_tokens: u64,
    created_at_ms: u64,
    expires_at_ms: u64,
}

#[derive(Clone, Debug)]
pub struct SemanticCacheStatusSnapshot {
    pub default_ttl_secs: u64,
    pub default_similarity_threshold: f64,
    pub max_entries: usize,
    pub store_backed: bool,
    pub entry_count: u64,
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
    pub skips: u64,
    pub saved_prompt_tokens: u64,
}

#[derive(Clone)]
pub struct SemanticCache {
    default_ttl_secs: u64,
    default_similarity_threshold: f64,
    max_entries: usize,
    entries: Arc<RwLock<Vec<SemanticCacheEntry>>>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    stores: Arc<AtomicU64>,
    skips: Arc<AtomicU64>,
    saved_prompt_tokens: Arc<AtomicU64>,
    metrics: Option<LlmMetrics>,
    governance: Option<Arc<GovernanceState>>,
    store: Option<Arc<Store>>,
}

impl SemanticCache {
    pub fn new(config: &toml::Value) -> Result<Self, Box<dyn std::error::Error>> {
        let default_ttl_secs = config
            .get("default_ttl_secs")
            .and_then(|value| value.as_integer())
            .map(|value| value as u64)
            .unwrap_or(300);
        let default_similarity_threshold = config
            .get("default_similarity_threshold")
            .and_then(|value| value.as_float())
            .unwrap_or(0.85);
        if !(0.0..=1.0).contains(&default_similarity_threshold) {
            return Err(
                "semantic_cache.default_similarity_threshold must be between 0.0 and 1.0".into(),
            );
        }
        let max_entries = config
            .get("max_entries")
            .and_then(|value| value.as_integer())
            .map(|value| value as usize)
            .unwrap_or(1024);
        Ok(Self {
            default_ttl_secs,
            default_similarity_threshold,
            max_entries,
            entries: Arc::new(RwLock::new(Vec::new())),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            stores: Arc::new(AtomicU64::new(0)),
            skips: Arc::new(AtomicU64::new(0)),
            saved_prompt_tokens: Arc::new(AtomicU64::new(0)),
            metrics: None,
            governance: None,
            store: None,
        })
    }

    pub fn with_metrics(mut self, metrics: LlmMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn with_governance(mut self, governance: Arc<GovernanceState>) -> Self {
        self.governance = Some(governance);
        self
    }

    pub fn with_store(mut self, store: Arc<Store>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn status(&self) -> SemanticCacheStatusSnapshot {
        SemanticCacheStatusSnapshot {
            default_ttl_secs: self.default_ttl_secs,
            default_similarity_threshold: self.default_similarity_threshold,
            max_entries: self.max_entries,
            store_backed: self.store.is_some(),
            entry_count: self
                .entries
                .read()
                .expect("semantic cache lock poisoned")
                .len()
                .try_into()
                .unwrap_or(u64::MAX),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            stores: self.stores.load(Ordering::Relaxed),
            skips: self.skips.load(Ordering::Relaxed),
            saved_prompt_tokens: self.saved_prompt_tokens.load(Ordering::Relaxed),
        }
    }

    pub async fn load_from_store(&self) -> Result<(), crate::store::StoreError> {
        let Some(store) = self.store.as_ref() else {
            return Ok(());
        };

        let records = store
            .get_semantic_cache_entries(now_ms(), self.max_entries.try_into().unwrap_or(u32::MAX))
            .await?;
        let mut loaded = Vec::with_capacity(records.len());
        for record in records {
            match entry_from_record(record) {
                Some(entry) => loaded.push(entry),
                None => {
                    tracing::warn!("skipping malformed semantic cache entry loaded from store");
                }
            }
        }
        let loaded_len = loaded.len();
        *self.entries.write().expect("semantic cache lock poisoned") = loaded;
        if let Some(metrics) = self.metrics.as_ref() {
            metrics
                .semantic_cache_entries
                .set(loaded_len.try_into().unwrap_or(i64::MAX) as i64);
        }
        tracing::info!(entries = loaded_len, "semantic cache restored from store");
        Ok(())
    }

    async fn insert_entry(&self, entry: SemanticCacheEntry) {
        let now = now_ms();
        let store_record = entry_to_record(&entry);
        let entry_count = {
            let mut entries = self.entries.write().expect("semantic cache lock poisoned");
            entries.retain(|candidate| {
                candidate.expires_at_ms > now && candidate.cache_id != entry.cache_id
            });
            entries.push(entry);
            if entries.len() > self.max_entries {
                entries.sort_by_key(|candidate| candidate.created_at_ms);
                let remove_count = entries.len().saturating_sub(self.max_entries);
                entries.drain(0..remove_count);
            }
            entries.len()
        };
        if let Some(metrics) = self.metrics.as_ref() {
            metrics
                .semantic_cache_entries
                .set(entry_count.try_into().unwrap_or(i64::MAX));
        }
        if let Some(store) = self.store.as_ref() {
            if let Err(err) = store.upsert_semantic_cache_entry(&store_record).await {
                tracing::warn!(error = %err, "failed to persist semantic cache entry");
            }
            if let Err(err) = store
                .prune_semantic_cache_entries(now, self.max_entries.try_into().unwrap_or(u32::MAX))
                .await
            {
                tracing::warn!(error = %err, "failed to prune semantic cache store");
            }
        }
    }

    async fn find_hit(
        &self,
        project_id: &str,
        provider_name: &str,
        model: &str,
        normalized_prompt: &str,
        similarity_threshold: f64,
    ) -> Option<(SemanticCacheEntry, f64)> {
        let prompt_tokens = tokenize(normalized_prompt);
        let now = now_ms();
        let mut entries = self.entries.write().expect("semantic cache lock poisoned");
        entries.retain(|candidate| candidate.expires_at_ms > now);

        let mut best: Option<(SemanticCacheEntry, f64)> = None;
        for entry in entries.iter() {
            if entry.project_id != project_id
                || entry.provider_name != provider_name
                || entry.model != model
            {
                continue;
            }
            let similarity = jaccard_similarity(&prompt_tokens, &entry.tokens);
            if similarity < similarity_threshold {
                continue;
            }
            if best
                .as_ref()
                .map(|(_, best_similarity)| similarity > *best_similarity)
                .unwrap_or(true)
            {
                best = Some((entry.clone(), similarity));
            }
        }
        best
    }

    pub(crate) fn routing_affinity_for_request(
        &self,
        project_id: &str,
        body: Option<&[u8]>,
        available_providers: &[String],
        project_policy: Option<&ProjectPolicyRecord>,
    ) -> Result<Option<PromptCacheRoutingAffinity>, String> {
        let Some(preference) = self.routing_preference_from_body(body, project_policy)? else {
            return Ok(None);
        };
        self.routing_affinity_from_preference(project_id, &preference, available_providers)
    }

    pub(crate) fn routing_affinity_for_request_json(
        &self,
        project_id: &str,
        request_json: &Value,
        available_providers: &[String],
        project_policy: Option<&ProjectPolicyRecord>,
    ) -> Result<Option<PromptCacheRoutingAffinity>, String> {
        let Some(preference) = self.routing_preference_from_json(request_json, project_policy)?
        else {
            return Ok(None);
        };
        self.routing_affinity_from_preference(project_id, &preference, available_providers)
    }

    fn routing_affinity_from_preference(
        &self,
        project_id: &str,
        preference: &SemanticCacheRoutingPreference,
        available_providers: &[String],
    ) -> Result<Option<PromptCacheRoutingAffinity>, String> {
        let affinity = self.routing_affinity(
            project_id,
            &preference.model,
            &preference.normalized_prompt,
            preference.similarity_threshold,
            available_providers,
        );
        Ok((!affinity.is_empty()).then_some(affinity))
    }

    fn routing_preference_from_body(
        &self,
        body: Option<&[u8]>,
        project_policy: Option<&ProjectPolicyRecord>,
    ) -> Result<Option<SemanticCacheRoutingPreference>, String> {
        let Some(body) = body else {
            return Ok(None);
        };
        let request_json = match serde_json::from_slice::<Value>(body) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        self.routing_preference_from_json(&request_json, project_policy)
    }

    fn routing_preference_from_json(
        &self,
        request_json: &Value,
        project_policy: Option<&ProjectPolicyRecord>,
    ) -> Result<Option<SemanticCacheRoutingPreference>, String> {
        let mut request_json = request_json.clone();
        let project_cache_enabled = project_policy.and_then(|policy| policy.semantic_cache_enabled);
        let effective_default_ttl_secs = project_policy
            .and_then(|policy| policy.semantic_cache_ttl_secs)
            .unwrap_or(self.default_ttl_secs);
        let effective_default_similarity_threshold = project_policy
            .and_then(|policy| policy.semantic_cache_similarity_threshold)
            .filter(|value| (0.0..=1.0).contains(value))
            .unwrap_or(self.default_similarity_threshold);
        let Some(opt_in) = extract_opt_in(
            &mut request_json,
            project_cache_enabled.unwrap_or(false),
            effective_default_ttl_secs,
            effective_default_similarity_threshold,
        )?
        else {
            return Ok(None);
        };
        if matches!(project_cache_enabled, Some(false)) && opt_in.enabled {
            return Ok(None);
        }
        if !opt_in.enabled {
            return Ok(None);
        }
        if request_json
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || request_json.get("trp_tools").is_some()
            || request_json.get("tools").is_some()
        {
            return Ok(None);
        }
        let body = Bytes::from(request_json.to_string());
        let Some(model) = extract_model(&body) else {
            return Ok(None);
        };
        let Some(normalized_prompt) = extract_request_prompt(&request_json) else {
            return Ok(None);
        };
        Ok(Some(SemanticCacheRoutingPreference {
            model,
            normalized_prompt,
            similarity_threshold: opt_in.similarity_threshold,
        }))
    }

    fn routing_affinity(
        &self,
        project_id: &str,
        model: &str,
        normalized_prompt: &str,
        similarity_threshold: f64,
        available_providers: &[String],
    ) -> PromptCacheRoutingAffinity {
        let prompt_tokens = tokenize(normalized_prompt);
        let now = now_ms();
        let available_provider_names = available_providers
            .iter()
            .map(|provider| provider.as_str())
            .collect::<HashSet<_>>();
        let mut entries = self.entries.write().expect("semantic cache lock poisoned");
        entries.retain(|candidate| candidate.expires_at_ms > now);

        let mut affinity = PromptCacheRoutingAffinity::default();
        let mut best_bonus_by_provider = std::collections::HashMap::<String, f64>::new();
        for entry in entries.iter() {
            if entry.project_id != project_id
                || entry.model != model
                || !available_provider_names.contains(entry.provider_name.as_str())
            {
                continue;
            }
            let similarity = jaccard_similarity(&prompt_tokens, &entry.tokens);
            if similarity < similarity_threshold {
                continue;
            }
            let bonus = semantic_cache_route_bonus(entry, similarity, now);
            let current = best_bonus_by_provider
                .entry(entry.provider_name.clone())
                .or_insert(0.0);
            if bonus > *current {
                *current = bonus;
            }
        }
        for (provider_name, bonus) in best_bonus_by_provider {
            affinity.add_positive_bonus(provider_name, bonus);
        }
        affinity
    }
}

#[async_trait]
impl Plugin for SemanticCache {
    fn name(&self) -> &str {
        "semantic_cache"
    }

    async fn on_request(&self, ctx: &mut RequestContext) -> Action {
        let Some(virtual_key) = ctx.extensions.get::<VirtualKeyMeta>().cloned() else {
            return Action::Continue;
        };
        if ctx.body.is_none() {
            return Action::Continue;
        }

        let Some(request_json) = cached_request_json_mut(ctx) else {
            return Action::Continue;
        };

        let project_policy = self
            .governance
            .as_ref()
            .and_then(|governance| governance.project_policy(&virtual_key.project_id));
        let project_cache_enabled = project_policy
            .as_ref()
            .and_then(|policy| policy.semantic_cache_enabled);
        let effective_default_ttl_secs = project_policy
            .as_ref()
            .and_then(|policy| policy.semantic_cache_ttl_secs)
            .unwrap_or(self.default_ttl_secs);
        let effective_default_similarity_threshold = project_policy
            .as_ref()
            .and_then(|policy| policy.semantic_cache_similarity_threshold)
            .filter(|value| (0.0..=1.0).contains(value))
            .unwrap_or(self.default_similarity_threshold);
        let opt_in = match extract_opt_in(
            request_json,
            project_cache_enabled.unwrap_or(false),
            effective_default_ttl_secs,
            effective_default_similarity_threshold,
        ) {
            Ok(opt_in) => opt_in,
            Err(error) => return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error)),
        };
        let Some(opt_in) = opt_in else {
            return Action::Continue;
        };
        if matches!(project_cache_enabled, Some(false)) && opt_in.enabled {
            if let Err(error) = sync_cached_request_json_body(ctx) {
                return Action::Respond(json_error(StatusCode::INTERNAL_SERVER_ERROR, &error));
            }
            self.record_skip(&virtual_key.provider_name, "project_disabled");
            return Action::Continue;
        }
        if !opt_in.enabled {
            if let Err(error) = sync_cached_request_json_body(ctx) {
                return Action::Respond(json_error(StatusCode::INTERNAL_SERVER_ERROR, &error));
            }
            self.record_skip(&virtual_key.provider_name, "disabled");
            return Action::Continue;
        }

        if request_json
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || request_json.get("trp_tools").is_some()
            || request_json.get("tools").is_some()
        {
            if let Err(error) = sync_cached_request_json_body(ctx) {
                return Action::Respond(json_error(StatusCode::INTERNAL_SERVER_ERROR, &error));
            }
            self.record_skip(&virtual_key.provider_name, "unsupported");
            return Action::Continue;
        }

        let body = match serde_json::to_vec(&request_json) {
            Ok(bytes) => Bytes::from(bytes),
            Err(error) => {
                return Action::Respond(json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("failed to encode semantic cache request: {error}"),
                ))
            }
        };
        let Some(model) = extract_model(&body) else {
            update_request_body(ctx, body);
            self.record_skip(&virtual_key.provider_name, "missing_model");
            return Action::Continue;
        };
        let Some(normalized_prompt) = extract_request_prompt(request_json) else {
            update_request_body(ctx, body);
            self.record_skip(&virtual_key.provider_name, "unsupported_shape");
            return Action::Continue;
        };

        if let Some((entry, similarity)) = self
            .find_hit(
                &virtual_key.project_id,
                &virtual_key.provider_name,
                &model,
                &normalized_prompt,
                opt_in.similarity_threshold,
            )
            .await
        {
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.saved_prompt_tokens
                .fetch_add(entry.prompt_tokens, Ordering::Relaxed);
            if let Some(metrics) = self.metrics.as_ref() {
                metrics
                    .semantic_cache_requests_total
                    .with_label_values(&[virtual_key.provider_name.as_str(), "hit"])
                    .inc();
                metrics
                    .semantic_cache_saved_prompt_tokens_total
                    .with_label_values(&[virtual_key.provider_name.as_str()])
                    .inc_by(entry.prompt_tokens);
            }
            let mut response = cached_response(&entry, similarity);
            if let Some(trace) = ctx.extensions.get::<RoutingDebugTrace>() {
                if trace.enabled {
                    attach_routing_debug_headers(&mut response, trace);
                }
            }
            return Action::Respond(response);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        if let Some(metrics) = self.metrics.as_ref() {
            metrics
                .semantic_cache_requests_total
                .with_label_values(&[virtual_key.provider_name.as_str(), "miss"])
                .inc();
        }
        ctx.extensions.insert(SemanticCacheRequestMeta {
            project_id: virtual_key.project_id,
            provider_name: virtual_key.provider_name,
            model,
            normalized_prompt,
            prompt_tokens: estimate_prompt_tokens(&body),
            ttl_secs: opt_in.ttl_secs,
        });
        update_request_body(ctx, body);
        Action::Continue
    }

    async fn transform_response(
        &self,
        ctx: &mut RequestContext,
        resp: Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>,
    ) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
        let Some(request_meta) = ctx.extensions.get::<SemanticCacheRequestMeta>().cloned() else {
            return resp;
        };
        let is_sse = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.contains("text/event-stream"))
            .unwrap_or(false);
        if !resp.status().is_success() || is_sse {
            self.record_skip(&request_meta.provider_name, "uncacheable_response");
            return with_cache_headers(resp, "skip", None);
        }

        let (mut parts, body) = resp.into_parts();
        let body_bytes = match body.collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(_) => {
                let resp = Response::from_parts(
                    parts,
                    Full::new(Bytes::new())
                        .map_err(|never| match never {})
                        .boxed(),
                );
                self.record_skip(&request_meta.provider_name, "response_read_error");
                return with_cache_headers(resp, "skip", None);
            }
        };
        let content_type = parts
            .headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string);
        let now = now_ms();
        let entry = SemanticCacheEntry {
            cache_id: semantic_cache_id(
                &request_meta.project_id,
                &request_meta.provider_name,
                &request_meta.model,
                &request_meta.normalized_prompt,
            ),
            project_id: request_meta.project_id.clone(),
            provider_name: request_meta.provider_name.clone(),
            model: request_meta.model.clone(),
            tokens: tokenize(&request_meta.normalized_prompt),
            response_status: parts.status.as_u16(),
            content_type,
            response_body: body_bytes.clone(),
            prompt_tokens: request_meta.prompt_tokens,
            created_at_ms: now,
            expires_at_ms: now.saturating_add(request_meta.ttl_secs.saturating_mul(1000)),
        };
        self.insert_entry(entry).await;
        self.stores.fetch_add(1, Ordering::Relaxed);
        if let Some(metrics) = self.metrics.as_ref() {
            metrics
                .semantic_cache_requests_total
                .with_label_values(&[request_meta.provider_name.as_str(), "store"])
                .inc();
        }
        parts.headers.remove(CONTENT_LENGTH);
        let resp = Response::from_parts(
            parts,
            Full::new(body_bytes)
                .map_err(|never| match never {})
                .boxed(),
        );
        with_cache_headers(resp, "miss", None)
    }
}

pub fn create_plugin(config: &toml::Value) -> Result<SemanticCache, Box<dyn std::error::Error>> {
    SemanticCache::new(config)
}

pub fn create(config: &toml::Value) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>> {
    Ok(Box::new(create_plugin(config)?))
}

impl SemanticCache {
    fn record_skip(&self, provider_name: &str, reason: &str) {
        self.skips.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(provider = provider_name, reason, "semantic cache skipped");
        if let Some(metrics) = self.metrics.as_ref() {
            metrics
                .semantic_cache_requests_total
                .with_label_values(&[provider_name, "skip"])
                .inc();
        }
    }
}

fn semantic_cache_route_bonus(entry: &SemanticCacheEntry, similarity: f64, now_ms: u64) -> f64 {
    let remaining_ms = entry.expires_at_ms.saturating_sub(now_ms);
    if remaining_ms == 0 {
        return 0.0;
    }
    let total_lifetime_ms = entry.expires_at_ms.saturating_sub(entry.created_at_ms);
    let freshness = if total_lifetime_ms == 0 {
        1.0
    } else {
        (remaining_ms as f64 / total_lifetime_ms as f64).clamp(0.0, 1.0)
    }
    .max(SEMANTIC_CACHE_ROUTE_MIN_FRESHNESS);
    SEMANTIC_CACHE_ROUTE_MAX_BONUS * similarity.clamp(0.0, 1.0) * freshness
}

fn with_cache_headers(
    mut resp: Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>,
    status: &str,
    similarity: Option<f64>,
) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
    resp.headers_mut().insert(
        "x-trp-semantic-cache",
        HeaderValue::from_str(status).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );
    if let Some(similarity) = similarity {
        if let Ok(value) = HeaderValue::from_str(&format!("{similarity:.3}")) {
            resp.headers_mut()
                .insert("x-trp-semantic-cache-similarity", value);
        }
    }
    resp
}

fn attach_routing_debug_headers(
    resp: &mut Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>>,
    trace: &RoutingDebugTrace,
) {
    if let Some(selected_provider) = trace.selected_provider.as_deref() {
        if let Ok(value) = HeaderValue::from_str(selected_provider) {
            resp.headers_mut().insert("x-trp-provider-selected", value);
        }
    }
    if let Some(preferred_provider) = trace.prompt_cache_preferred_provider.as_deref() {
        if let Ok(value) = HeaderValue::from_str(preferred_provider) {
            resp.headers_mut().insert("x-trp-prompt-cache-route", value);
        }
    }
    if let Some(preferred_provider) = trace.semantic_cache_preferred_provider.as_deref() {
        if let Ok(value) = HeaderValue::from_str(preferred_provider) {
            resp.headers_mut()
                .insert("x-trp-semantic-cache-route", value);
        }
    }
    if !trace.prompt_cache_negative_providers.is_empty() {
        if let Ok(value) = HeaderValue::from_str(&trace.prompt_cache_negative_providers.join(",")) {
            resp.headers_mut()
                .insert("x-trp-prompt-cache-negative", value);
        }
    }
    if trace.prompt_cache_preferred_provider.is_some()
        || !trace.prompt_cache_negative_providers.is_empty()
    {
        let affinity_value = if trace.prompt_cache_affinity_applied {
            "applied"
        } else {
            "observed"
        };
        resp.headers_mut().insert(
            "x-trp-prompt-cache-affinity",
            HeaderValue::from_static(affinity_value),
        );
    }
    if !trace.ordered_providers.is_empty() {
        if let Ok(value) = HeaderValue::from_str(&trace.ordered_providers.join(",")) {
            resp.headers_mut().insert("x-trp-provider-order", value);
        }
    }
    if !trace.attempted_providers.is_empty() {
        if let Ok(value) = HeaderValue::from_str(&trace.attempted_providers.join(",")) {
            resp.headers_mut().insert("x-trp-provider-attempts", value);
        }
    }
    if let Some(rule_name) = trace
        .matched_rule_name
        .as_deref()
        .or(trace.matched_rule_id.as_deref())
    {
        if let Ok(value) = HeaderValue::from_str(rule_name) {
            resp.headers_mut().insert("x-trp-routing-rule", value);
        }
    }
}

fn cached_response(
    entry: &SemanticCacheEntry,
    similarity: f64,
) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
    let mut builder = Response::builder().status(entry.response_status);
    if let Some(content_type) = entry.content_type.as_deref() {
        builder = builder.header(CONTENT_TYPE, content_type);
    }
    builder = builder.header(CONTENT_LENGTH, entry.response_body.len().to_string());
    let response = builder
        .body(
            Full::new(entry.response_body.clone())
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap_or_else(|_| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build semantic cache response",
            )
        });
    with_cache_headers(response, "hit", Some(similarity))
}

fn extract_opt_in(
    request_json: &mut Value,
    default_enabled: bool,
    default_ttl_secs: u64,
    default_similarity_threshold: f64,
) -> Result<Option<SemanticCacheOptIn>, String> {
    let object = request_json
        .as_object_mut()
        .ok_or_else(|| "semantic_cache request body must be a JSON object".to_string())?;
    let Some(raw_value) = object.remove("trp_semantic_cache") else {
        return Ok(default_enabled.then_some(SemanticCacheOptIn {
            enabled: true,
            ttl_secs: default_ttl_secs,
            similarity_threshold: default_similarity_threshold,
        }));
    };
    let opt_in = raw_value
        .as_object()
        .ok_or_else(|| "trp_semantic_cache must be an object".to_string())?;
    let enabled = opt_in
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let ttl_secs = opt_in
        .get("ttl_secs")
        .and_then(Value::as_u64)
        .unwrap_or(default_ttl_secs);
    let similarity_threshold = opt_in
        .get("similarity_threshold")
        .and_then(Value::as_f64)
        .unwrap_or(default_similarity_threshold);
    if !(0.0..=1.0).contains(&similarity_threshold) {
        return Err("trp_semantic_cache.similarity_threshold must be between 0.0 and 1.0".into());
    }
    Ok(Some(SemanticCacheOptIn {
        enabled,
        ttl_secs,
        similarity_threshold,
    }))
}

fn extract_request_prompt(request_json: &Value) -> Option<String> {
    let mut segments = Vec::new();
    if let Some(system) = request_json.get("system") {
        collect_value_text(system, &mut segments);
    }
    if let Some(messages) = request_json.get("messages").and_then(Value::as_array) {
        for message in messages {
            if let Some(role) = message.get("role").and_then(Value::as_str) {
                segments.push(role.to_string());
            }
            if let Some(content) = message.get("content") {
                collect_value_text(content, &mut segments);
            }
        }
    }
    if let Some(input) = request_json.get("input") {
        collect_value_text(input, &mut segments);
    }
    let joined = segments.join("\n");
    let normalized = normalize_prompt(&joined);
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn collect_value_text(value: &Value, segments: &mut Vec<String>) {
    match value {
        Value::String(text) => segments.push(text.clone()),
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(text) => segments.push(text.clone()),
                    Value::Object(map) => {
                        if let Some(text) = map.get("text").and_then(Value::as_str) {
                            segments.push(text.to_string());
                        } else if let Some(text) = map.get("content").and_then(Value::as_str) {
                            segments.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                segments.push(text.to_string());
            } else if let Some(text) = map.get("content").and_then(Value::as_str) {
                segments.push(text.to_string());
            }
        }
        _ => {}
    }
}

fn normalize_prompt(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn tokenize(normalized_prompt: &str) -> HashSet<String> {
    normalized_prompt
        .split_whitespace()
        .map(ToString::to_string)
        .collect()
}

fn serialize_tokens(tokens: &HashSet<String>) -> String {
    let mut values = tokens.iter().cloned().collect::<Vec<_>>();
    values.sort();
    serde_json::to_string(&values).unwrap_or_else(|_| "[]".to_string())
}

fn deserialize_tokens(tokens_json: &str) -> Option<HashSet<String>> {
    serde_json::from_str::<Vec<String>>(tokens_json)
        .ok()
        .map(|tokens| tokens.into_iter().collect())
}

fn semantic_cache_id(
    project_id: &str,
    provider_name: &str,
    model: &str,
    normalized_prompt: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(project_id.as_bytes());
    hasher.update([0]);
    hasher.update(provider_name.as_bytes());
    hasher.update([0]);
    hasher.update(model.as_bytes());
    hasher.update([0]);
    hasher.update(normalized_prompt.as_bytes());
    format!("semcache-{}", hex::encode(hasher.finalize()))
}

fn entry_to_record(entry: &SemanticCacheEntry) -> SemanticCacheEntryRecord {
    SemanticCacheEntryRecord {
        cache_id: entry.cache_id.clone(),
        project_id: entry.project_id.clone(),
        provider_name: entry.provider_name.clone(),
        model: entry.model.clone(),
        tokens_json: serialize_tokens(&entry.tokens),
        response_status: entry.response_status,
        content_type: entry.content_type.clone(),
        response_body: entry.response_body.to_vec(),
        prompt_tokens: entry.prompt_tokens,
        created_at_ms: entry.created_at_ms,
        expires_at_ms: entry.expires_at_ms,
    }
}

fn entry_from_record(record: SemanticCacheEntryRecord) -> Option<SemanticCacheEntry> {
    let tokens = deserialize_tokens(&record.tokens_json)?;
    Some(SemanticCacheEntry {
        cache_id: record.cache_id,
        project_id: record.project_id,
        provider_name: record.provider_name,
        model: record.model,
        tokens,
        response_status: record.response_status,
        content_type: record.content_type,
        response_body: Bytes::from(record.response_body),
        prompt_tokens: record.prompt_tokens,
        created_at_ms: record.created_at_ms,
        expires_at_ms: record.expires_at_ms,
    })
}

fn jaccard_similarity(left: &HashSet<String>, right: &HashSet<String>) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count() as f64;
    let union = left.union(right).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn json_error(
    status: StatusCode,
    message: &str,
) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(
            Full::new(Bytes::from(json!({ "error": message }).to_string()))
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_prompt_canonicalizes_whitespace_and_punctuation() {
        assert_eq!(
            normalize_prompt("Reset password, please!\nReset password."),
            "reset password please reset password"
        );
    }

    #[test]
    fn jaccard_similarity_scores_related_prompts_higher_than_zero() {
        let left = tokenize("reset password help");
        let right = tokenize("need password reset help");
        assert!(jaccard_similarity(&left, &right) > 0.7);
    }
}
