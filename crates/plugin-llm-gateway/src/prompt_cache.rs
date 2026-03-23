use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Frame;
use hyper::header::{HeaderValue, CONTENT_LENGTH};
use hyper::{Response, StatusCode};
use proxy_core::config::{
    PromptCacheProtocol as SharedPromptCacheProtocol, ProviderExtraCapability, ProviderKeyConfig,
    ProviderPromptCacheSemantics, ProviderRuntimeSemantics, ProviderSurfaceCatalog,
};
use proxy_core::plugin::{Action, Plugin, RequestContext, ResponseContext};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::oneshot;

use crate::metrics::LlmMetrics;
use crate::store::{GatewayStore, PromptCacheRouteRecord, Store};
use crate::virtual_keys::VirtualKeyMeta;
use crate::{cached_request_json_mut, sync_cached_request_json_body};

const PROMPT_CACHE_WARM_MAX_BONUS: f64 = 2_000.0;
const PROMPT_CACHE_WARM_MIN_FRESHNESS: f64 = 0.15;
const PROMPT_CACHE_NEGATIVE_MAX_PENALTY: f64 = 2_500.0;
const PROMPT_CACHE_NEGATIVE_TTL_SECS: u64 = 90;
const DEFAULT_ROUTING_FLUSH_INTERVAL_MS: u64 = 2_000;
const DEFAULT_ROUTING_PRUNE_INTERVAL_SECS: u64 = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PromptCacheScope {
    Auto,
    System,
    Tools,
    Messages,
}

impl PromptCacheScope {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::System => "system",
            Self::Tools => "tools",
            Self::Messages => "messages",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "system" => Some(Self::System),
            "tools" => Some(Self::Tools),
            "messages" => Some(Self::Messages),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct PromptCacheOptIn {
    enabled: bool,
    ttl: Option<String>,
    scope: PromptCacheScope,
    key: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PromptCacheRoutingRequirement {
    AnyPromptCache,
    OpenAiControls,
    AnthropicControls,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptCacheRoutingHint {
    pub enabled: bool,
    requirement: PromptCacheRoutingRequirement,
}

impl PromptCacheRoutingHint {
    pub(crate) fn required_capability(self) -> ProviderExtraCapability {
        match self.requirement {
            PromptCacheRoutingRequirement::AnyPromptCache => {
                ProviderExtraCapability::PromptCacheRequestControls
            }
            PromptCacheRoutingRequirement::OpenAiControls => {
                ProviderExtraCapability::PromptCacheOpenAi
            }
            PromptCacheRoutingRequirement::AnthropicControls => {
                ProviderExtraCapability::PromptCacheAnthropic
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PromptCacheRoutingPreference {
    pub hint: PromptCacheRoutingHint,
    pub key: Option<String>,
}

#[derive(Clone, Debug)]
enum PromptCacheRouteSignalKind {
    Warm,
    Negative,
}

#[derive(Clone, Debug)]
enum PromptCacheRouteMutation {
    Upsert(PromptCacheRouteRecord),
    Delete { route_id: String },
}

impl PromptCacheRouteMutation {
    fn route_id(&self) -> &str {
        match self {
            Self::Upsert(record) => &record.route_id,
            Self::Delete { route_id } => route_id,
        }
    }
}

impl PromptCacheRouteSignalKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Negative => "negative",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "warm" => Some(Self::Warm),
            "negative" => Some(Self::Negative),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct PromptCacheRouteSignal {
    provider_name: String,
    kind: PromptCacheRouteSignalKind,
    signal_strength: f64,
    observed_at_ms: u64,
    expires_at_ms: u64,
}

#[derive(Clone, Debug, Default)]
struct PromptCacheRoutingState {
    warm: Option<PromptCacheRouteSignal>,
    negatives: HashMap<String, PromptCacheRouteSignal>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PromptCacheRoutingAffinity {
    pub preferred_provider: Option<String>,
    pub positive_bonuses: HashMap<String, f64>,
    pub negative_penalties: HashMap<String, f64>,
}

impl PromptCacheRoutingAffinity {
    pub fn is_empty(&self) -> bool {
        self.positive_bonuses.is_empty() && self.negative_penalties.is_empty()
    }

    pub fn add_positive_bonus(&mut self, provider_name: impl Into<String>, bonus: f64) {
        if bonus <= 0.0 {
            return;
        }
        let provider_name = provider_name.into();
        *self.positive_bonuses.entry(provider_name).or_insert(0.0) += bonus;
        self.refresh_preferred_provider();
    }

    pub fn add_negative_penalty(&mut self, provider_name: impl Into<String>, penalty: f64) {
        if penalty <= 0.0 {
            return;
        }
        let provider_name = provider_name.into();
        *self.negative_penalties.entry(provider_name).or_insert(0.0) += penalty;
    }

    pub fn merge(&mut self, other: &Self) {
        for (provider, bonus) in &other.positive_bonuses {
            self.add_positive_bonus(provider.clone(), *bonus);
        }
        for (provider, penalty) in &other.negative_penalties {
            self.add_negative_penalty(provider.clone(), *penalty);
        }
        self.refresh_preferred_provider();
    }

    fn refresh_preferred_provider(&mut self) {
        self.preferred_provider = self
            .positive_bonuses
            .iter()
            .filter(|(_, bonus)| **bonus > 0.0)
            .max_by(|(left_name, left_bonus), (right_name, right_bonus)| {
                left_bonus
                    .partial_cmp(right_bonus)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right_name.cmp(left_name))
            })
            .map(|(provider, _)| provider.clone());
    }
}

#[derive(Clone, Debug, Default)]
pub struct PromptCacheRoutingMemory {
    routes: Arc<DashMap<String, PromptCacheRoutingState>>,
}

impl PromptCacheRoutingMemory {
    fn composite_key(project_id: &str, cache_key: &str) -> String {
        format!("{project_id}\u{001f}{cache_key}")
    }

    fn cleanup_state(&self, storage_key: &str, now: u64) -> Option<PromptCacheRoutingState> {
        let mut entry = self.routes.get_mut(storage_key)?;
        if entry
            .warm
            .as_ref()
            .map(|signal| signal.expires_at_ms <= now)
            .unwrap_or(false)
        {
            entry.warm = None;
        }
        entry
            .negatives
            .retain(|_, signal| signal.expires_at_ms > now);
        let state = entry.clone();
        let should_remove = state.warm.is_none() && state.negatives.is_empty();
        drop(entry);
        if should_remove {
            self.routes.remove(storage_key);
            None
        } else {
            Some(state)
        }
    }

    #[cfg(test)]
    pub(crate) fn remember(
        &self,
        project_id: &str,
        cache_key: &str,
        provider_name: &str,
        signal_strength: f64,
        ttl_secs: u64,
    ) {
        let now = now_ms();
        let expires_at_ms = now.saturating_add(ttl_secs.saturating_mul(1000));
        self.remember_until(
            project_id,
            cache_key,
            provider_name,
            signal_strength,
            now,
            expires_at_ms,
        );
    }

    pub(crate) fn remember_until(
        &self,
        project_id: &str,
        cache_key: &str,
        provider_name: &str,
        signal_strength: f64,
        observed_at_ms: u64,
        expires_at_ms: u64,
    ) {
        let storage_key = Self::composite_key(project_id, cache_key);
        let mut state = self
            .cleanup_state(&storage_key, now_ms())
            .unwrap_or_default();
        state.warm = Some(PromptCacheRouteSignal {
            provider_name: provider_name.to_string(),
            kind: PromptCacheRouteSignalKind::Warm,
            signal_strength,
            observed_at_ms,
            expires_at_ms,
        });
        state.negatives.remove(provider_name);
        self.routes.insert(storage_key, state);
    }

    #[cfg(test)]
    pub(crate) fn remember_negative(
        &self,
        project_id: &str,
        cache_key: &str,
        provider_name: &str,
        signal_strength: f64,
        ttl_secs: u64,
    ) {
        let now = now_ms();
        let expires_at_ms = now.saturating_add(ttl_secs.saturating_mul(1000));
        self.remember_negative_until(
            project_id,
            cache_key,
            provider_name,
            signal_strength,
            now,
            expires_at_ms,
        );
    }

    pub(crate) fn remember_negative_until(
        &self,
        project_id: &str,
        cache_key: &str,
        provider_name: &str,
        signal_strength: f64,
        observed_at_ms: u64,
        expires_at_ms: u64,
    ) {
        let storage_key = Self::composite_key(project_id, cache_key);
        let mut state = self
            .cleanup_state(&storage_key, now_ms())
            .unwrap_or_default();
        state.negatives.insert(
            provider_name.to_string(),
            PromptCacheRouteSignal {
                provider_name: provider_name.to_string(),
                kind: PromptCacheRouteSignalKind::Negative,
                signal_strength,
                observed_at_ms,
                expires_at_ms,
            },
        );
        self.routes.insert(storage_key, state);
    }

    pub(crate) fn forget_warm(
        &self,
        project_id: &str,
        cache_key: &str,
        provider_name: &str,
    ) -> bool {
        let storage_key = Self::composite_key(project_id, cache_key);
        let Some(mut entry) = self.routes.get_mut(&storage_key) else {
            return false;
        };
        let mut removed = false;
        if entry
            .warm
            .as_ref()
            .map(|signal| signal.provider_name == provider_name)
            .unwrap_or(false)
        {
            entry.warm = None;
            removed = true;
        }
        let should_remove = entry.warm.is_none() && entry.negatives.is_empty();
        drop(entry);
        if should_remove {
            self.routes.remove(&storage_key);
        }
        removed
    }

    pub(crate) fn routing_affinity(
        &self,
        project_id: &str,
        cache_key: &str,
        available_providers: &[String],
    ) -> PromptCacheRoutingAffinity {
        let storage_key = Self::composite_key(project_id, cache_key);
        let Some(state) = self.cleanup_state(&storage_key, now_ms()) else {
            return PromptCacheRoutingAffinity::default();
        };
        let now = now_ms();
        let mut affinity = PromptCacheRoutingAffinity::default();
        if let Some(warm) = state.warm.as_ref().filter(|signal| {
            available_providers
                .iter()
                .any(|provider| provider == &signal.provider_name)
        }) {
            affinity.add_positive_bonus(
                warm.provider_name.clone(),
                decayed_signal_value(warm, PROMPT_CACHE_WARM_MAX_BONUS),
            );
        }
        for provider in available_providers {
            if let Some(negative) = state.negatives.get(provider) {
                if negative.expires_at_ms > now {
                    affinity.add_negative_penalty(
                        provider.clone(),
                        decayed_signal_value(negative, PROMPT_CACHE_NEGATIVE_MAX_PENALTY),
                    );
                }
            }
        }
        affinity
    }

    pub(crate) fn warmed_entry_count(&self) -> u64 {
        let now = now_ms();
        let keys = self
            .routes
            .iter()
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        let mut count = 0u64;
        for key in keys {
            if let Some(state) = self.cleanup_state(&key, now) {
                if state.warm.is_some() {
                    count = count.saturating_add(1);
                }
            }
        }
        count
    }

    pub(crate) fn negative_entry_count(&self) -> u64 {
        let now = now_ms();
        let keys = self
            .routes
            .iter()
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        let mut count = 0u64;
        for key in keys {
            if let Some(state) = self.cleanup_state(&key, now) {
                count = count.saturating_add(state.negatives.len().try_into().unwrap_or(u64::MAX));
            }
        }
        count
    }
}

fn decayed_signal_value(signal: &PromptCacheRouteSignal, max_value: f64) -> f64 {
    let strength = signal.signal_strength.clamp(0.0, 1.0);
    if strength <= 0.0 {
        return 0.0;
    }
    let remaining_ms = signal.expires_at_ms.saturating_sub(now_ms());
    if remaining_ms == 0 {
        return 0.0;
    }
    let total_lifetime_ms = signal.expires_at_ms.saturating_sub(signal.observed_at_ms);
    let freshness = if total_lifetime_ms == 0 {
        1.0
    } else {
        (remaining_ms as f64 / total_lifetime_ms as f64).clamp(0.0, 1.0)
    };
    let freshness = match signal.kind {
        PromptCacheRouteSignalKind::Warm => freshness.max(PROMPT_CACHE_WARM_MIN_FRESHNESS),
        PromptCacheRouteSignalKind::Negative => freshness,
    };
    max_value * strength * freshness
}

#[derive(Clone, Debug)]
struct PromptCacheRequestMeta {
    project_id: String,
    provider_name: String,
    protocol: Option<SharedPromptCacheProtocol>,
    opt_in_requested: bool,
    routing_cache_key: Option<String>,
    routing_ttl_secs: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct PromptCacheUsage {
    pub provider_name: String,
    pub protocol: String,
    pub read_tokens: u64,
    pub write_tokens: u64,
    pub outcome: String,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PromptCacheProviderSnapshot {
    pub name: String,
    pub family: String,
    pub surfaces: ProviderSurfaceCatalog,
    #[serde(flatten)]
    pub semantics: ProviderRuntimeSemantics,
    #[serde(flatten)]
    pub prompt_cache: ProviderPromptCacheSemantics,
}

#[derive(Clone, Debug)]
pub struct PromptCacheStatusSnapshot {
    pub anthropic_default_scope: String,
    pub store_backed: bool,
    pub routing_hint_persistence_enabled: bool,
    pub routing_flush_interval_ms: u64,
    pub routing_prune_interval_secs: u64,
    pub warmed_route_count: u64,
    pub negative_route_count: u64,
    pub pending_route_updates: u64,
    pub last_route_flush_unix_ms: Option<u64>,
    pub providers: Vec<PromptCacheProviderSnapshot>,
}

#[derive(Clone)]
struct PromptCachePersistence {
    store: Arc<Store>,
    pending: Arc<DashMap<String, PromptCacheRouteMutation>>,
    flush_interval_ms: u64,
    prune_interval_secs: u64,
    last_flush_unix_ms: Arc<AtomicU64>,
}

impl PromptCachePersistence {
    fn new(store: Arc<Store>, flush_interval_ms: u64, prune_interval_secs: u64) -> Self {
        Self {
            store,
            pending: Arc::new(DashMap::new()),
            flush_interval_ms,
            prune_interval_secs,
            last_flush_unix_ms: Arc::new(AtomicU64::new(0)),
        }
    }

    fn enqueue(&self, mutation: PromptCacheRouteMutation) {
        self.pending
            .insert(mutation.route_id().to_string(), mutation);
    }

    fn pending_update_count(&self) -> u64 {
        self.pending.len().try_into().unwrap_or(u64::MAX)
    }

    fn last_flush_unix_ms(&self) -> Option<u64> {
        match self.last_flush_unix_ms.load(Ordering::Relaxed) {
            0 => None,
            value => Some(value),
        }
    }

    async fn flush_pending(&self) -> Result<u64, crate::store::StoreError> {
        let keys = self
            .pending
            .iter()
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        if keys.is_empty() {
            return Ok(0);
        }

        let mut drained = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some((_, mutation)) = self.pending.remove(&key) {
                drained.push(mutation);
            }
        }
        if drained.is_empty() {
            return Ok(0);
        }

        let mut applied = 0u64;
        for index in 0..drained.len() {
            let mutation = &drained[index];
            let result = match mutation {
                PromptCacheRouteMutation::Upsert(record) => {
                    self.store.upsert_prompt_cache_route(record).await
                }
                PromptCacheRouteMutation::Delete { route_id } => self
                    .store
                    .delete_prompt_cache_route(route_id)
                    .await
                    .map(|_| ()),
            };
            if let Err(error) = result {
                for pending in drained.into_iter().skip(index) {
                    self.enqueue(pending);
                }
                return Err(error);
            }
            applied = applied.saturating_add(1);
        }

        self.last_flush_unix_ms.store(now_ms(), Ordering::Relaxed);
        Ok(applied)
    }

    async fn prune_expired(&self) -> Result<(), crate::store::StoreError> {
        self.store.prune_prompt_cache_routes(now_ms()).await
    }

    fn spawn_task(&self) -> tokio::task::JoinHandle<()> {
        let persistence = self.clone();
        tokio::spawn(async move {
            let mut last_prune_ms = now_ms();
            loop {
                tokio::time::sleep(Duration::from_millis(persistence.flush_interval_ms)).await;
                if let Err(error) = persistence.flush_pending().await {
                    tracing::warn!(
                        error = %error,
                        "prompt cache routing hint flush error"
                    );
                }

                let now = now_ms();
                if now.saturating_sub(last_prune_ms)
                    >= persistence.prune_interval_secs.saturating_mul(1000)
                {
                    if let Err(error) = persistence.prune_expired().await {
                        tracing::warn!(
                            error = %error,
                            "prompt cache routing hint prune error"
                        );
                    } else {
                        last_prune_ms = now;
                    }
                }
            }
        })
    }
}

#[derive(Clone)]
pub struct PromptCache {
    providers: Arc<RwLock<HashMap<String, ProviderKeyConfig>>>,
    anthropic_default_scope: PromptCacheScope,
    metrics: Option<LlmMetrics>,
    routing_memory: PromptCacheRoutingMemory,
    store: Option<Arc<Store>>,
    persist_routing_hints: bool,
    routing_flush_interval_ms: u64,
    routing_prune_interval_secs: u64,
    persistence: Option<PromptCachePersistence>,
}

impl PromptCache {
    pub fn new(
        config: &toml::Value,
        providers: &[ProviderKeyConfig],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let anthropic_default_scope = config
            .get("anthropic_default_scope")
            .and_then(|value| value.as_str())
            .map(|value| {
                PromptCacheScope::parse(value).ok_or_else(|| {
                    format!(
                        "prompt_cache.anthropic_default_scope must be one of: auto, system, tools, messages"
                    )
                })
            })
            .transpose()?
            .unwrap_or(PromptCacheScope::Auto);
        let persist_routing_hints = config
            .get("persist_routing_hints")
            .and_then(|value| value.as_bool())
            .unwrap_or(true);
        let routing_flush_interval_ms = parse_positive_u64(
            config,
            "routing_flush_interval_ms",
            DEFAULT_ROUTING_FLUSH_INTERVAL_MS,
        )?;
        let routing_prune_interval_secs = parse_positive_u64(
            config,
            "routing_prune_interval_secs",
            DEFAULT_ROUTING_PRUNE_INTERVAL_SECS,
        )?;
        Ok(Self {
            providers: Arc::new(RwLock::new(
                providers
                    .iter()
                    .map(|provider| {
                        let mut provider = provider.clone();
                        provider.refresh_derived_semantics();
                        (provider.name.clone(), provider)
                    })
                    .collect(),
            )),
            anthropic_default_scope,
            metrics: None,
            routing_memory: PromptCacheRoutingMemory::default(),
            store: None,
            persist_routing_hints,
            routing_flush_interval_ms,
            routing_prune_interval_secs,
            persistence: None,
        })
    }

    pub fn with_metrics(mut self, metrics: LlmMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn with_store(mut self, store: Arc<Store>) -> Self {
        if self.persist_routing_hints {
            self.persistence = Some(PromptCachePersistence::new(
                Arc::clone(&store),
                self.routing_flush_interval_ms,
                self.routing_prune_interval_secs,
            ));
        }
        self.store = Some(store);
        self
    }

    pub fn set_provider_configs(&self, providers: &[ProviderKeyConfig]) {
        *self
            .providers
            .write()
            .unwrap_or_else(|poison| poison.into_inner()) = providers
            .iter()
            .map(|provider| {
                let mut provider = provider.clone();
                provider.refresh_derived_semantics();
                (provider.name.clone(), provider)
            })
            .collect();
    }

    pub fn status(&self) -> PromptCacheStatusSnapshot {
        let provider_configs = self
            .providers
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut providers = provider_configs
            .values()
            .map(|provider| PromptCacheProviderSnapshot {
                name: provider.name.clone(),
                family: provider.family_kind().as_str().to_string(),
                surfaces: provider.surfaces().clone(),
                semantics: provider.runtime_semantics(),
                prompt_cache: provider.prompt_cache_semantics(),
            })
            .collect::<Vec<_>>();
        providers.sort_by(|left, right| left.name.cmp(&right.name));
        PromptCacheStatusSnapshot {
            anthropic_default_scope: self.anthropic_default_scope.as_str().to_string(),
            store_backed: self.persistence.is_some(),
            routing_hint_persistence_enabled: self.persistence.is_some(),
            routing_flush_interval_ms: self.routing_flush_interval_ms,
            routing_prune_interval_secs: self.routing_prune_interval_secs,
            warmed_route_count: self.routing_memory.warmed_entry_count(),
            negative_route_count: self.routing_memory.negative_entry_count(),
            pending_route_updates: self
                .persistence
                .as_ref()
                .map(|persistence| persistence.pending_update_count())
                .unwrap_or(0),
            last_route_flush_unix_ms: self
                .persistence
                .as_ref()
                .and_then(|persistence| persistence.last_flush_unix_ms()),
            providers,
        }
    }

    pub async fn load_from_store(&self) -> Result<(), crate::store::StoreError> {
        let Some(persistence) = self.persistence.as_ref() else {
            return Ok(());
        };

        let records = persistence
            .store
            .get_prompt_cache_routes(now_ms(), 10_000)
            .await?;
        let restore_now = now_ms();
        for record in records {
            let observed_at_ms = if record.observed_at_ms == 0 {
                restore_now.min(record.expires_at_ms)
            } else {
                record.observed_at_ms
            };
            match PromptCacheRouteSignalKind::parse(&record.signal_kind) {
                Some(PromptCacheRouteSignalKind::Warm) => self.routing_memory.remember_until(
                    &record.project_id,
                    &record.cache_key,
                    &record.provider_name,
                    record.signal_strength,
                    observed_at_ms,
                    record.expires_at_ms,
                ),
                Some(PromptCacheRouteSignalKind::Negative) => {
                    self.routing_memory.remember_negative_until(
                        &record.project_id,
                        &record.cache_key,
                        &record.provider_name,
                        record.signal_strength,
                        observed_at_ms,
                        record.expires_at_ms,
                    )
                }
                None => {}
            }
        }
        tracing::info!(
            routes = self.routing_memory.warmed_entry_count(),
            negative_routes = self.routing_memory.negative_entry_count(),
            "prompt cache routing memory restored from store"
        );
        Ok(())
    }

    pub(crate) fn anthropic_default_scope_for_routing(&self) -> PromptCacheScope {
        self.anthropic_default_scope
    }

    pub(crate) fn routing_memory(&self) -> PromptCacheRoutingMemory {
        self.routing_memory.clone()
    }

    pub fn spawn_persistence_task(&self) -> Option<tokio::task::JoinHandle<()>> {
        self.persistence
            .as_ref()
            .map(|persistence| persistence.spawn_task())
    }

    #[cfg(test)]
    async fn flush_pending_route_updates(&self) -> Result<u64, crate::store::StoreError> {
        let Some(persistence) = self.persistence.as_ref() else {
            return Ok(0);
        };
        persistence.flush_pending().await
    }
}

#[async_trait]
impl Plugin for PromptCache {
    fn name(&self) -> &str {
        "prompt_cache"
    }

    async fn on_request(&self, ctx: &mut RequestContext) -> Action {
        let Some(virtual_key) = ctx.extensions.get::<VirtualKeyMeta>().cloned() else {
            return Action::Continue;
        };
        let providers = self
            .providers
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(provider) = providers.get(&virtual_key.provider_name) else {
            return Action::Continue;
        };
        let protocol = infer_prompt_cache_protocol(provider);

        if ctx.body.is_none() {
            ctx.extensions.insert(PromptCacheRequestMeta {
                project_id: virtual_key.project_id,
                provider_name: virtual_key.provider_name,
                protocol,
                opt_in_requested: false,
                routing_cache_key: None,
                routing_ttl_secs: None,
            });
            return Action::Continue;
        }

        let Some(request_json) = cached_request_json_mut(ctx) else {
            ctx.extensions.insert(PromptCacheRequestMeta {
                project_id: virtual_key.project_id,
                provider_name: virtual_key.provider_name,
                protocol,
                opt_in_requested: false,
                routing_cache_key: None,
                routing_ttl_secs: None,
            });
            return Action::Continue;
        };

        let opt_in = match extract_opt_in(request_json, self.anthropic_default_scope) {
            Ok(opt_in) => opt_in,
            Err(error) => return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error)),
        };

        let opt_in_requested = opt_in.is_some();
        let routing_cache_key = opt_in
            .as_ref()
            .filter(|opt_in| opt_in.enabled)
            .and_then(|opt_in| opt_in.key.clone());
        let routing_ttl_secs = opt_in.as_ref().and_then(|opt_in| {
            opt_in
                .enabled
                .then(|| prompt_cache_routing_ttl_secs(protocol, opt_in))
        });
        if protocol.is_none() {
            if let Some(ref opt_in) = opt_in {
                if let Err(error) = sync_cached_request_json_body(ctx) {
                    return Action::Respond(json_error(StatusCode::INTERNAL_SERVER_ERROR, &error));
                }
                if opt_in.enabled {
                    return Action::Respond(json_error(
                        StatusCode::BAD_REQUEST,
                        "provider does not support gateway prompt cache controls",
                    ));
                }
            }
            ctx.extensions.insert(PromptCacheRequestMeta {
                project_id: virtual_key.project_id,
                provider_name: virtual_key.provider_name,
                protocol,
                opt_in_requested,
                routing_cache_key,
                routing_ttl_secs,
            });
            return Action::Continue;
        }

        if let Some(opt_in) = opt_in {
            if opt_in.enabled {
                let result = match protocol {
                    Some(SharedPromptCacheProtocol::OpenAi) => apply_openai_request_controls(
                        request_json,
                        &opt_in,
                        request_controls_supported(provider),
                    ),
                    Some(SharedPromptCacheProtocol::Anthropic) => {
                        apply_anthropic_request_controls(request_json, &opt_in)
                    }
                    None => Ok(()),
                };
                if let Err(error) = result {
                    return Action::Respond(json_error(StatusCode::BAD_REQUEST, &error));
                }
            }

            if let Err(error) = sync_cached_request_json_body(ctx) {
                return Action::Respond(json_error(StatusCode::INTERNAL_SERVER_ERROR, &error));
            }
        }

        ctx.extensions.insert(PromptCacheRequestMeta {
            project_id: virtual_key.project_id,
            provider_name: virtual_key.provider_name,
            protocol,
            opt_in_requested,
            routing_cache_key,
            routing_ttl_secs,
        });
        Action::Continue
    }

    async fn transform_response(
        &self,
        ctx: &mut RequestContext,
        resp: Response<BoxBody<Bytes, hyper::Error>>,
    ) -> Response<BoxBody<Bytes, hyper::Error>> {
        let Some(meta) = ctx.extensions.get::<PromptCacheRequestMeta>().cloned() else {
            return resp;
        };

        let is_sse = resp
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .map(|content_type| content_type.contains("text/event-stream"))
            .unwrap_or(false);
        if is_sse {
            return resp;
        }

        let (parts, body) = resp.into_parts();
        let bytes = match body.collect().await {
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

        let usage = extract_prompt_cache_usage_from_tail(&meta, &bytes);
        let final_usage = usage.or_else(|| {
            if meta.opt_in_requested {
                Some(PromptCacheUsage {
                    provider_name: meta.provider_name.clone(),
                    protocol: meta
                        .protocol
                        .map(|protocol| protocol.as_str().to_string())
                        .unwrap_or_else(|| "none".to_string()),
                    read_tokens: 0,
                    write_tokens: 0,
                    outcome: "miss".to_string(),
                })
            } else {
                None
            }
        });
        if let Some(usage) = final_usage.as_ref() {
            apply_prompt_cache_routing_memory_update(
                &self.routing_memory,
                self.persistence.as_ref(),
                &meta,
                usage,
            );
            ctx.extensions.insert(usage.clone());
        }

        let mut response = Response::from_parts(
            parts,
            Full::new(bytes.clone())
                .map_err(|never| match never {})
                .boxed(),
        );
        if let Ok(value) = HeaderValue::from_str(&bytes.len().to_string()) {
            response.headers_mut().insert(CONTENT_LENGTH, value);
        }
        if let Some(usage) = final_usage {
            attach_response_headers(&mut response, &usage);
        }
        response
    }

    async fn on_response(&self, ctx: &mut RequestContext, resp: &mut ResponseContext) -> Action {
        let is_sse = resp
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .map(|content_type| content_type.contains("text/event-stream"))
            .unwrap_or(false);
        if is_sse {
            return Action::Continue;
        }
        if let Some(usage) = ctx.extensions.get::<PromptCacheUsage>().cloned() {
            record_prompt_cache_observation(self.metrics.as_ref(), clone_otel_span(ctx), &usage);
        }
        Action::Continue
    }

    fn wrap_response_body(
        &self,
        ctx: &RequestContext,
        resp: &ResponseContext,
        body: BoxBody<Bytes, hyper::Error>,
    ) -> BoxBody<Bytes, hyper::Error> {
        let is_sse = resp
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .map(|content_type| content_type.contains("text/event-stream"))
            .unwrap_or(false);
        if !is_sse {
            return body;
        }

        let Some(meta) = ctx.extensions.get::<PromptCacheRequestMeta>().cloned() else {
            return body;
        };
        let (wrapper, rx) = PromptCacheExtractorBody::new(body, meta.clone());
        let metrics = self.metrics.clone();
        let routing_memory = self.routing_memory.clone();
        let persistence = self.persistence.clone();
        let otel = clone_otel_span(ctx);
        tokio::spawn(async move {
            let fallback_provider_name = meta.provider_name.clone();
            let usage = rx.await.ok().flatten().or_else(|| {
                if meta.opt_in_requested {
                    Some(PromptCacheUsage {
                        provider_name: fallback_provider_name,
                        protocol: meta
                            .protocol
                            .map(|protocol| protocol.as_str().to_string())
                            .unwrap_or_else(|| "none".to_string()),
                        read_tokens: 0,
                        write_tokens: 0,
                        outcome: "miss".to_string(),
                    })
                } else {
                    None
                }
            });
            if let Some(usage) = usage {
                apply_prompt_cache_routing_memory_update(
                    &routing_memory,
                    persistence.as_ref(),
                    &meta,
                    &usage,
                );
                record_prompt_cache_observation(metrics.as_ref(), otel, &usage);
            }
        });

        BoxBody::new(wrapper)
    }
}

pub fn create_plugin(
    config: &toml::Value,
    providers: &[ProviderKeyConfig],
) -> Result<PromptCache, Box<dyn std::error::Error>> {
    PromptCache::new(config, providers)
}

pub fn create(_config: &toml::Value) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>> {
    Err("prompt_cache requires provider context; use create_plugins_with_options".into())
}

fn infer_prompt_cache_protocol(provider: &ProviderKeyConfig) -> Option<SharedPromptCacheProtocol> {
    match provider
        .prompt_cache_semantics()
        .prompt_cache_protocol
        .as_str()
    {
        "anthropic" => Some(SharedPromptCacheProtocol::Anthropic),
        "openai" => Some(SharedPromptCacheProtocol::OpenAi),
        _ => None,
    }
}

pub(crate) fn provider_supports_prompt_cache(provider: &ProviderKeyConfig) -> bool {
    provider.prompt_cache_semantics().supports_prompt_cache
}

fn request_controls_supported(provider: &ProviderKeyConfig) -> bool {
    provider.prompt_cache_semantics().request_controls_supported
}

pub(crate) fn prompt_cache_routing_preference_from_body(
    body: Option<&[u8]>,
    anthropic_default_scope: PromptCacheScope,
) -> Result<Option<PromptCacheRoutingPreference>, String> {
    let Some(body) = body else {
        return Ok(None);
    };
    let request_json = match serde_json::from_slice::<Value>(body) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    prompt_cache_routing_preference_from_json(&request_json, anthropic_default_scope)
}

pub(crate) fn prompt_cache_routing_preference_from_json(
    request_json: &Value,
    anthropic_default_scope: PromptCacheScope,
) -> Result<Option<PromptCacheRoutingPreference>, String> {
    let Some(opt_in_value) = request_json.get("trp_prompt_cache") else {
        return Ok(None);
    };
    let opt_in = parse_opt_in_value(opt_in_value, anthropic_default_scope)?;
    Ok(Some(PromptCacheRoutingPreference {
        hint: PromptCacheRoutingHint {
            enabled: opt_in.enabled,
            requirement: routing_requirement_for_opt_in(&opt_in),
        },
        key: opt_in.key,
    }))
}

pub(crate) fn provider_matches_prompt_cache_hint(
    provider: &ProviderKeyConfig,
    hint: PromptCacheRoutingHint,
) -> bool {
    if !hint.enabled {
        return true;
    }
    match hint.requirement {
        PromptCacheRoutingRequirement::AnyPromptCache => provider_supports_prompt_cache(provider),
        PromptCacheRoutingRequirement::OpenAiControls => {
            matches!(
                infer_prompt_cache_protocol(provider),
                Some(SharedPromptCacheProtocol::OpenAi)
            ) && request_controls_supported(provider)
        }
        PromptCacheRoutingRequirement::AnthropicControls => {
            matches!(
                infer_prompt_cache_protocol(provider),
                Some(SharedPromptCacheProtocol::Anthropic)
            )
        }
    }
}

fn extract_opt_in(
    request_json: &mut Value,
    anthropic_default_scope: PromptCacheScope,
) -> Result<Option<PromptCacheOptIn>, String> {
    let object = request_json
        .as_object_mut()
        .ok_or_else(|| "prompt cache request body must be a JSON object".to_string())?;
    let Some(opt_in_value) = object.remove("trp_prompt_cache") else {
        return Ok(None);
    };
    parse_opt_in_value(&opt_in_value, anthropic_default_scope).map(Some)
}

fn parse_opt_in_value(
    opt_in_value: &Value,
    anthropic_default_scope: PromptCacheScope,
) -> Result<PromptCacheOptIn, String> {
    let opt_in = opt_in_value
        .as_object()
        .ok_or_else(|| "trp_prompt_cache must be an object".to_string())?;
    let enabled = opt_in
        .get("enabled")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let ttl = opt_in
        .get("ttl")
        .and_then(|value| value.as_str())
        .map(ToString::to_string);
    let scope = match opt_in.get("scope").and_then(|value| value.as_str()) {
        Some(value) => PromptCacheScope::parse(value).ok_or_else(|| {
            "trp_prompt_cache.scope must be one of: auto, system, tools, messages".to_string()
        })?,
        None => anthropic_default_scope,
    };
    let key = opt_in
        .get("key")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    Ok(PromptCacheOptIn {
        enabled,
        ttl,
        scope,
        key,
    })
}

fn routing_requirement_for_opt_in(opt_in: &PromptCacheOptIn) -> PromptCacheRoutingRequirement {
    if opt_in.key.is_some() {
        return PromptCacheRoutingRequirement::OpenAiControls;
    }
    if !matches!(opt_in.scope, PromptCacheScope::Auto) {
        return PromptCacheRoutingRequirement::AnthropicControls;
    }
    match opt_in.ttl.as_deref() {
        Some("in_memory") | Some("24h") => PromptCacheRoutingRequirement::OpenAiControls,
        Some("5m") | Some("1h") => PromptCacheRoutingRequirement::AnthropicControls,
        _ => PromptCacheRoutingRequirement::AnyPromptCache,
    }
}

fn prompt_cache_routing_ttl_secs(
    protocol: Option<SharedPromptCacheProtocol>,
    opt_in: &PromptCacheOptIn,
) -> u64 {
    match protocol {
        Some(SharedPromptCacheProtocol::OpenAi) => match opt_in.ttl.as_deref() {
            Some("24h") => 24 * 60 * 60,
            Some("in_memory") | None => 5 * 60,
            Some(_) => 5 * 60,
        },
        Some(SharedPromptCacheProtocol::Anthropic) => match opt_in.ttl.as_deref() {
            Some("1h") => 60 * 60,
            Some("5m") | None => 5 * 60,
            Some(_) => 5 * 60,
        },
        None => 5 * 60,
    }
}

fn parse_positive_u64(
    config: &toml::Value,
    field: &str,
    default: u64,
) -> Result<u64, Box<dyn std::error::Error>> {
    let Some(value) = config.get(field) else {
        return Ok(default);
    };
    let integer = value
        .as_integer()
        .ok_or_else(|| format!("prompt_cache.{field} must be a positive integer"))?;
    let parsed = u64::try_from(integer)
        .map_err(|_| format!("prompt_cache.{field} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("prompt_cache.{field} must be greater than 0").into());
    }
    Ok(parsed)
}

fn apply_prompt_cache_routing_memory_update(
    memory: &PromptCacheRoutingMemory,
    persistence: Option<&PromptCachePersistence>,
    meta: &PromptCacheRequestMeta,
    usage: &PromptCacheUsage,
) {
    let Some(cache_key) = meta.routing_cache_key.as_deref() else {
        return;
    };
    let Some(ttl_secs) = meta.routing_ttl_secs else {
        return;
    };
    let warm_route_id = prompt_cache_warm_route_id(&meta.project_id, cache_key);
    let negative_route_id =
        prompt_cache_negative_route_id(&meta.project_id, cache_key, &meta.provider_name);
    if usage.read_tokens > 0 || usage.write_tokens > 0 {
        let observed_at_ms = now_ms();
        let expires_at_ms = now_ms().saturating_add(ttl_secs.saturating_mul(1000));
        memory.remember_until(
            &meta.project_id,
            cache_key,
            &meta.provider_name,
            1.0,
            observed_at_ms,
            expires_at_ms,
        );
        if let Some(persistence) = persistence {
            let record = PromptCacheRouteRecord {
                route_id: warm_route_id,
                project_id: meta.project_id.clone(),
                cache_key: cache_key.to_string(),
                provider_name: meta.provider_name.clone(),
                signal_kind: PromptCacheRouteSignalKind::Warm.as_str().to_string(),
                signal_strength: 1.0,
                observed_at_ms,
                expires_at_ms,
            };
            persistence.enqueue(PromptCacheRouteMutation::Upsert(record));
            persistence.enqueue(PromptCacheRouteMutation::Delete {
                route_id: negative_route_id,
            });
        }
    } else if usage.outcome == "miss" {
        let removed_warm = memory.forget_warm(&meta.project_id, cache_key, &meta.provider_name);
        let observed_at_ms = now_ms();
        let expires_at_ms =
            observed_at_ms.saturating_add(PROMPT_CACHE_NEGATIVE_TTL_SECS.saturating_mul(1000));
        memory.remember_negative_until(
            &meta.project_id,
            cache_key,
            &meta.provider_name,
            1.0,
            observed_at_ms,
            expires_at_ms,
        );
        if let Some(persistence) = persistence {
            if removed_warm {
                persistence.enqueue(PromptCacheRouteMutation::Delete {
                    route_id: warm_route_id,
                });
            }
            let record = PromptCacheRouteRecord {
                route_id: negative_route_id,
                project_id: meta.project_id.clone(),
                cache_key: cache_key.to_string(),
                provider_name: meta.provider_name.clone(),
                signal_kind: PromptCacheRouteSignalKind::Negative.as_str().to_string(),
                signal_strength: 1.0,
                observed_at_ms,
                expires_at_ms,
            };
            persistence.enqueue(PromptCacheRouteMutation::Upsert(record));
        }
    }
}

fn prompt_cache_warm_route_id(project_id: &str, cache_key: &str) -> String {
    prompt_cache_route_id(project_id, cache_key, "warm", None)
}

fn prompt_cache_negative_route_id(
    project_id: &str,
    cache_key: &str,
    provider_name: &str,
) -> String {
    prompt_cache_route_id(project_id, cache_key, "negative", Some(provider_name))
}

fn prompt_cache_route_id(
    project_id: &str,
    cache_key: &str,
    signal_kind: &str,
    provider_name: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(project_id.as_bytes());
    hasher.update([0]);
    hasher.update(cache_key.as_bytes());
    hasher.update([0]);
    hasher.update(signal_kind.as_bytes());
    if let Some(provider_name) = provider_name {
        hasher.update([0]);
        hasher.update(provider_name.as_bytes());
    }
    format!("pcroute-{}", hex::encode(hasher.finalize()))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn apply_openai_request_controls(
    request_json: &mut Value,
    opt_in: &PromptCacheOptIn,
    request_controls_supported: bool,
) -> Result<(), String> {
    let object = request_json
        .as_object_mut()
        .ok_or_else(|| "OpenAI prompt cache request must be a JSON object".to_string())?;
    if !matches!(opt_in.scope, PromptCacheScope::Auto) {
        return Err("OpenAI prompt cache only supports scope=auto".to_string());
    }
    if object.contains_key("prompt_cache_key") || object.contains_key("prompt_cache_retention") {
        return Err(
            "request already contains prompt_cache_key or prompt_cache_retention; use one interface"
                .to_string(),
        );
    }
    if (opt_in.key.is_some() || opt_in.ttl.is_some()) && !request_controls_supported {
        return Err(
            "provider supports prompt cache reporting but not OpenAI prompt cache request controls"
                .to_string(),
        );
    }
    if let Some(key) = opt_in.key.as_ref() {
        object.insert("prompt_cache_key".to_string(), Value::String(key.clone()));
    }
    if let Some(ttl) = opt_in.ttl.as_deref() {
        match ttl {
            "in_memory" => {}
            "24h" => {
                object.insert(
                    "prompt_cache_retention".to_string(),
                    Value::String("24h".to_string()),
                );
            }
            other => {
                return Err(format!(
                    "OpenAI prompt cache ttl must be one of: in_memory, 24h (got '{}')",
                    other
                ))
            }
        }
    }
    Ok(())
}

fn apply_anthropic_request_controls(
    request_json: &mut Value,
    opt_in: &PromptCacheOptIn,
) -> Result<(), String> {
    if opt_in.key.is_some() {
        return Err("Anthropic prompt cache does not support a gateway cache key".to_string());
    }
    if has_existing_anthropic_cache_control(request_json) {
        return Err(
            "request already contains provider-native cache_control; use one interface".to_string(),
        );
    }
    let cache_control = match opt_in.ttl.as_deref() {
        None | Some("5m") => json!({ "type": "ephemeral" }),
        Some("1h") => json!({ "type": "ephemeral", "ttl": "1h" }),
        Some(other) => {
            return Err(format!(
                "Anthropic prompt cache ttl must be one of: 5m, 1h (got '{}')",
                other
            ))
        }
    };
    let scope = resolve_anthropic_scope(request_json, opt_in.scope);
    match scope {
        PromptCacheScope::System => {
            insert_anthropic_system_cache_control(request_json, cache_control)
        }
        PromptCacheScope::Tools => insert_anthropic_tool_cache_control(request_json, cache_control),
        PromptCacheScope::Messages | PromptCacheScope::Auto => {
            insert_anthropic_message_cache_control(request_json, cache_control)
        }
    }
}

fn resolve_anthropic_scope(request_json: &Value, requested: PromptCacheScope) -> PromptCacheScope {
    if !matches!(requested, PromptCacheScope::Auto) {
        return requested;
    }
    if request_json.get("system").is_some() {
        PromptCacheScope::System
    } else if request_json
        .get("tools")
        .and_then(|value| value.as_array())
        .map(|tools| !tools.is_empty())
        .unwrap_or(false)
    {
        PromptCacheScope::Tools
    } else {
        PromptCacheScope::Messages
    }
}

fn insert_anthropic_system_cache_control(
    request_json: &mut Value,
    cache_control: Value,
) -> Result<(), String> {
    let system = request_json
        .as_object_mut()
        .ok_or_else(|| "Anthropic prompt cache request must be a JSON object".to_string())?
        .get_mut("system")
        .ok_or_else(|| {
            "Anthropic request is missing system content for prompt cache scope=system".to_string()
        })?;
    match system {
        Value::String(text) => {
            *system = Value::Array(vec![json!({
                "type": "text",
                "text": text.clone(),
                "cache_control": cache_control,
            })]);
        }
        Value::Array(blocks) => {
            let last = blocks
                .last_mut()
                .ok_or_else(|| "Anthropic system content is empty".to_string())?;
            insert_cache_control_into_block(last, cache_control)?;
        }
        _ => return Err("Anthropic system content must be a string or array".to_string()),
    }
    Ok(())
}

fn insert_anthropic_tool_cache_control(
    request_json: &mut Value,
    cache_control: Value,
) -> Result<(), String> {
    let tools = request_json
        .as_object_mut()
        .ok_or_else(|| "Anthropic prompt cache request must be a JSON object".to_string())?
        .get_mut("tools")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| {
            "Anthropic request is missing tools for prompt cache scope=tools".to_string()
        })?;
    let last = tools
        .last_mut()
        .ok_or_else(|| "Anthropic request has no tool definitions to cache".to_string())?;
    let object = last
        .as_object_mut()
        .ok_or_else(|| "Anthropic tool definition must be an object".to_string())?;
    if object.contains_key("cache_control") {
        return Err("Anthropic tool definition already contains cache_control".to_string());
    }
    object.insert("cache_control".to_string(), cache_control);
    Ok(())
}

fn insert_anthropic_message_cache_control(
    request_json: &mut Value,
    cache_control: Value,
) -> Result<(), String> {
    let messages = request_json
        .as_object_mut()
        .ok_or_else(|| "Anthropic prompt cache request must be a JSON object".to_string())?
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "Anthropic request is missing messages".to_string())?;
    let last_message = messages
        .last_mut()
        .ok_or_else(|| "Anthropic request has no messages to cache".to_string())?;
    let content = last_message
        .as_object_mut()
        .ok_or_else(|| "Anthropic message must be an object".to_string())?
        .get_mut("content")
        .ok_or_else(|| "Anthropic message is missing content".to_string())?;
    match content {
        Value::String(text) => {
            *content = Value::Array(vec![json!({
                "type": "text",
                "text": text.clone(),
                "cache_control": cache_control,
            })]);
        }
        Value::Array(blocks) => {
            let last = blocks
                .last_mut()
                .ok_or_else(|| "Anthropic message content is empty".to_string())?;
            insert_cache_control_into_block(last, cache_control)?;
        }
        _ => return Err("Anthropic message content must be a string or array".to_string()),
    }
    Ok(())
}

fn insert_cache_control_into_block(block: &mut Value, cache_control: Value) -> Result<(), String> {
    match block {
        Value::String(text) => {
            *block = json!({
                "type": "text",
                "text": text.clone(),
                "cache_control": cache_control,
            });
            Ok(())
        }
        Value::Object(object) => {
            if object.contains_key("cache_control") {
                return Err("Anthropic content block already contains cache_control".to_string());
            }
            object.insert("cache_control".to_string(), cache_control);
            Ok(())
        }
        _ => Err("Anthropic content block must be a string or object".to_string()),
    }
}

fn has_existing_anthropic_cache_control(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            object.contains_key("cache_control")
                || object.values().any(has_existing_anthropic_cache_control)
        }
        Value::Array(values) => values.iter().any(has_existing_anthropic_cache_control),
        _ => false,
    }
}

fn prompt_cache_outcome(read_tokens: u64, write_tokens: u64) -> &'static str {
    match (read_tokens > 0, write_tokens > 0) {
        (true, true) => "hit_write",
        (true, false) => "hit",
        (false, true) => "write",
        (false, false) => "miss",
    }
}

fn attach_response_headers(
    resp: &mut Response<BoxBody<Bytes, hyper::Error>>,
    usage: &PromptCacheUsage,
) {
    if let Ok(value) = HeaderValue::from_str(&usage.provider_name) {
        resp.headers_mut()
            .insert("x-trp-prompt-cache-provider", value);
    }
    if let Ok(value) = HeaderValue::from_str(&usage.protocol) {
        resp.headers_mut()
            .insert("x-trp-prompt-cache-protocol", value);
    }
    if let Ok(value) = HeaderValue::from_str(&usage.outcome) {
        resp.headers_mut()
            .insert("x-trp-prompt-cache-status", value);
    }
    if let Ok(value) = HeaderValue::from_str(&usage.read_tokens.to_string()) {
        resp.headers_mut()
            .insert("x-trp-prompt-cache-read-tokens", value);
    }
    if let Ok(value) = HeaderValue::from_str(&usage.write_tokens.to_string()) {
        resp.headers_mut()
            .insert("x-trp-prompt-cache-write-tokens", value);
    }
}

fn record_prompt_cache_observation(
    metrics: Option<&LlmMetrics>,
    otel: MaybeOtelSpan,
    usage: &PromptCacheUsage,
) {
    if let Some(metrics) = metrics {
        metrics
            .prompt_cache_requests_total
            .with_label_values(&[usage.provider_name.as_str(), usage.outcome.as_str()])
            .inc();
        metrics
            .prompt_cache_read_tokens_total
            .with_label_values(&[usage.provider_name.as_str()])
            .inc_by(usage.read_tokens);
        metrics
            .prompt_cache_write_tokens_total
            .with_label_values(&[usage.provider_name.as_str()])
            .inc_by(usage.write_tokens);
    }
    record_prompt_cache_otel(otel, usage);
}

#[cfg(feature = "opentelemetry")]
type MaybeOtelSpan = Option<proxy_core::otel::OtelSpan>;
#[cfg(not(feature = "opentelemetry"))]
type MaybeOtelSpan = Option<()>;

#[cfg(feature = "opentelemetry")]
fn clone_otel_span(ctx: &RequestContext) -> MaybeOtelSpan {
    ctx.extensions.get::<proxy_core::otel::OtelSpan>().cloned()
}

#[cfg(not(feature = "opentelemetry"))]
fn clone_otel_span(_ctx: &RequestContext) -> MaybeOtelSpan {
    None
}

#[cfg(feature = "opentelemetry")]
fn record_prompt_cache_otel(otel: MaybeOtelSpan, usage: &PromptCacheUsage) {
    if let Some(otel) = otel {
        otel.0
            .record("llm.prompt_cache_protocol", usage.protocol.as_str());
        otel.0
            .record("llm.prompt_cache_status", usage.outcome.as_str());
        otel.0
            .record("llm.prompt_cache_read_tokens", usage.read_tokens);
        otel.0
            .record("llm.prompt_cache_write_tokens", usage.write_tokens);
    }
}

#[cfg(not(feature = "opentelemetry"))]
fn record_prompt_cache_otel(_otel: MaybeOtelSpan, _usage: &PromptCacheUsage) {}

fn json_error(status: StatusCode, message: &str) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(
            Full::new(Bytes::from(
                json!({ "error": message }).to_string().into_bytes(),
            ))
            .map_err(|never| match never {})
            .boxed(),
        )
        .unwrap()
}

struct PromptCacheExtractorBody {
    inner: BoxBody<Bytes, hyper::Error>,
    tail_buf: Vec<u8>,
    meta: PromptCacheRequestMeta,
    tx: Option<oneshot::Sender<Option<PromptCacheUsage>>>,
}

const TAIL_CAPACITY: usize = 4096;

impl PromptCacheExtractorBody {
    fn new(
        inner: BoxBody<Bytes, hyper::Error>,
        meta: PromptCacheRequestMeta,
    ) -> (Self, oneshot::Receiver<Option<PromptCacheUsage>>) {
        let (tx, rx) = oneshot::channel();
        (
            Self {
                inner,
                tail_buf: Vec::with_capacity(TAIL_CAPACITY),
                meta,
                tx: Some(tx),
            },
            rx,
        )
    }

    fn append_to_tail(&mut self, data: &[u8]) {
        self.tail_buf.extend_from_slice(data);
        if self.tail_buf.len() > TAIL_CAPACITY {
            let excess = self.tail_buf.len() - TAIL_CAPACITY;
            self.tail_buf.drain(..excess);
        }
    }

    fn extract_and_send(&mut self) {
        let usage = extract_prompt_cache_usage_from_tail(&self.meta, &self.tail_buf);
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(usage);
        }
    }
}

impl hyper::body::Body for PromptCacheExtractorBody {
    type Data = Bytes;
    type Error = hyper::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.append_to_tail(data);
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(error))) => {
                this.extract_and_send();
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                this.extract_and_send();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn extract_prompt_cache_usage_from_tail(
    meta: &PromptCacheRequestMeta,
    tail: &[u8],
) -> Option<PromptCacheUsage> {
    let text = std::str::from_utf8(tail).ok()?;
    let usage_idx = text.rfind("\"usage\"")?;
    let after_usage = &text[usage_idx..];
    let brace_start = after_usage.find('{')?;
    let usage_obj = &after_usage[brace_start..];
    let brace_end = usage_obj.find('}')?;
    let usage_str = &usage_obj[..=brace_end];

    let (read_tokens, write_tokens) = match meta.protocol {
        Some(SharedPromptCacheProtocol::OpenAi) => (
            extract_int_field(usage_str, "cached_tokens").unwrap_or(0),
            extract_int_field(usage_str, "cache_write_tokens").unwrap_or(0),
        ),
        Some(SharedPromptCacheProtocol::Anthropic) => (
            extract_int_field(usage_str, "cache_read_input_tokens").unwrap_or(0),
            extract_int_field(usage_str, "cache_creation_input_tokens").unwrap_or(0),
        ),
        None => (0, 0),
    };

    if read_tokens == 0 && write_tokens == 0 && !meta.opt_in_requested {
        return None;
    }
    Some(PromptCacheUsage {
        provider_name: meta.provider_name.clone(),
        protocol: meta
            .protocol
            .map(|protocol| protocol.as_str().to_string())
            .unwrap_or_else(|| "none".to_string()),
        read_tokens,
        write_tokens,
        outcome: prompt_cache_outcome(read_tokens, write_tokens).to_string(),
    })
}

fn extract_int_field(text: &str, field: &str) -> Option<u64> {
    let pattern = format!("\"{}\"", field);
    let idx = text.find(&pattern)?;
    let after = &text[idx + pattern.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();
    let end = after
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after.len());
    if end == 0 {
        return None;
    }
    after[..end].parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::GatewayStore;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    #[test]
    fn routing_memory_prefers_warmed_provider_and_tracks_negative_signals() {
        let memory = PromptCacheRoutingMemory::default();
        let available = vec!["alpha".to_string(), "beta".to_string()];

        assert!(memory
            .routing_affinity("project-a", "tenant:123", &available)
            .is_empty());

        memory.remember("project-a", "tenant:123", "beta", 1.0, 60);
        let affinity = memory.routing_affinity("project-a", "tenant:123", &available);
        assert_eq!(affinity.preferred_provider.as_deref(), Some("beta"));
        assert!(
            affinity
                .positive_bonuses
                .get("beta")
                .copied()
                .unwrap_or_default()
                > 0.0
        );
        assert!(affinity.negative_penalties.is_empty());

        memory.remember_negative("project-a", "tenant:123", "alpha", 1.0, 60);
        let affinity = memory.routing_affinity("project-a", "tenant:123", &available);
        assert_eq!(affinity.preferred_provider.as_deref(), Some("beta"));
        assert!(affinity.negative_penalties.contains_key("alpha"));

        assert!(!memory.forget_warm("project-a", "tenant:123", "alpha"));
        let affinity = memory.routing_affinity("project-a", "tenant:123", &available);
        assert_eq!(affinity.preferred_provider.as_deref(), Some("beta"));
        assert!(affinity.negative_penalties.contains_key("alpha"));

        assert!(memory.forget_warm("project-a", "tenant:123", "beta"));
        let affinity = memory.routing_affinity("project-a", "tenant:123", &available);
        assert!(affinity.preferred_provider.is_none());
        assert!(affinity.negative_penalties.contains_key("alpha"));
    }

    #[test]
    fn anthropic_auto_scope_prefers_system_then_tools_then_messages() {
        let request = json!({
            "system": "Be precise",
            "tools": [{"name": "search"}],
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert_eq!(
            resolve_anthropic_scope(&request, PromptCacheScope::Auto),
            PromptCacheScope::System
        );

        let request = json!({
            "tools": [{"name": "search"}],
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert_eq!(
            resolve_anthropic_scope(&request, PromptCacheScope::Auto),
            PromptCacheScope::Tools
        );
    }

    #[test]
    fn extract_prompt_cache_usage_from_openai_response_tail() {
        let meta = PromptCacheRequestMeta {
            project_id: "project-a".to_string(),
            provider_name: "openai".to_string(),
            protocol: Some(SharedPromptCacheProtocol::OpenAi),
            opt_in_requested: true,
            routing_cache_key: Some("tenant:123".to_string()),
            routing_ttl_secs: Some(60),
        };
        let body = br#"{
            "id":"chatcmpl_123",
            "usage":{
                "prompt_tokens":12,
                "completion_tokens":15,
                "prompt_tokens_details":{
                    "cached_tokens":64,
                    "cache_write_tokens":16
                }
            }
        }"#;

        let usage = extract_prompt_cache_usage_from_tail(&meta, body).unwrap();
        assert_eq!(usage.read_tokens, 64);
        assert_eq!(usage.write_tokens, 16);
        assert_eq!(usage.outcome, "hit_write");
    }

    #[test]
    fn extract_prompt_cache_usage_from_anthropic_response_tail() {
        let meta = PromptCacheRequestMeta {
            project_id: "project-a".to_string(),
            provider_name: "anthropic".to_string(),
            protocol: Some(SharedPromptCacheProtocol::Anthropic),
            opt_in_requested: true,
            routing_cache_key: None,
            routing_ttl_secs: Some(60),
        };
        let body = br#"{
            "type":"message",
            "usage":{
                "input_tokens":12,
                "output_tokens":15,
                "cache_read_input_tokens":32,
                "cache_creation_input_tokens":8
            }
        }"#;

        let usage = extract_prompt_cache_usage_from_tail(&meta, body).unwrap();
        assert_eq!(usage.read_tokens, 32);
        assert_eq!(usage.write_tokens, 8);
        assert_eq!(usage.outcome, "hit_write");
    }

    #[test]
    fn inserts_anthropic_cache_control_into_string_system() {
        let mut request = json!({
            "system": "Be precise",
            "messages": [{"role": "user", "content": "hi"}]
        });
        apply_anthropic_request_controls(
            &mut request,
            &PromptCacheOptIn {
                enabled: true,
                ttl: Some("1h".to_string()),
                scope: PromptCacheScope::System,
                key: None,
            },
        )
        .unwrap();
        assert_eq!(
            request["system"][0]["cache_control"]["ttl"].as_str(),
            Some("1h")
        );
    }

    #[tokio::test]
    async fn prompt_cache_persistence_flushes_and_coalesces_mutations() {
        let temp_db = NamedTempFile::new().unwrap();
        let store_url = format!("sqlite://{}", temp_db.path().display());
        let store = Arc::new(crate::store::connect(&store_url).await.unwrap());
        let prompt_cache = PromptCache::new(&toml::Value::Table(toml::value::Map::new()), &[])
            .unwrap()
            .with_store(Arc::clone(&store));

        let meta = PromptCacheRequestMeta {
            project_id: "project-a".to_string(),
            provider_name: "alpha".to_string(),
            protocol: Some(SharedPromptCacheProtocol::OpenAi),
            opt_in_requested: true,
            routing_cache_key: Some("tenant:123".to_string()),
            routing_ttl_secs: Some(60),
        };
        let warm_usage = PromptCacheUsage {
            provider_name: "alpha".to_string(),
            protocol: "openai".to_string(),
            read_tokens: 64,
            write_tokens: 16,
            outcome: "hit_write".to_string(),
        };
        apply_prompt_cache_routing_memory_update(
            &prompt_cache.routing_memory,
            prompt_cache.persistence.as_ref(),
            &meta,
            &warm_usage,
        );
        assert_eq!(
            prompt_cache.status().pending_route_updates,
            2,
            "warm write should queue a warm upsert plus negative-route delete"
        );
        prompt_cache.flush_pending_route_updates().await.unwrap();
        let routes = store.get_prompt_cache_routes(now_ms(), 10).await.unwrap();
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].provider_name, "alpha");
        assert_eq!(routes[0].signal_kind, "warm");

        let miss_usage = PromptCacheUsage {
            provider_name: "alpha".to_string(),
            protocol: "openai".to_string(),
            read_tokens: 0,
            write_tokens: 0,
            outcome: "miss".to_string(),
        };
        apply_prompt_cache_routing_memory_update(
            &prompt_cache.routing_memory,
            prompt_cache.persistence.as_ref(),
            &meta,
            &miss_usage,
        );
        prompt_cache.flush_pending_route_updates().await.unwrap();
        let routes = store.get_prompt_cache_routes(now_ms(), 10).await.unwrap();
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].provider_name, "alpha");
        assert_eq!(routes[0].signal_kind, "negative");

        let persistence = prompt_cache.persistence.as_ref().unwrap();
        persistence.enqueue(PromptCacheRouteMutation::Delete {
            route_id: "custom-route".to_string(),
        });
        persistence.enqueue(PromptCacheRouteMutation::Upsert(PromptCacheRouteRecord {
            route_id: "custom-route".to_string(),
            project_id: "project-a".to_string(),
            cache_key: "tenant:custom".to_string(),
            provider_name: "beta".to_string(),
            signal_kind: "warm".to_string(),
            signal_strength: 1.0,
            observed_at_ms: now_ms(),
            expires_at_ms: now_ms().saturating_add(60_000),
        }));
        prompt_cache.flush_pending_route_updates().await.unwrap();
        let routes = store.get_prompt_cache_routes(now_ms(), 10).await.unwrap();
        assert!(routes.iter().any(|route| {
            route.route_id == "custom-route"
                && route.provider_name == "beta"
                && route.signal_kind == "warm"
        }));
    }

    #[tokio::test]
    async fn prompt_cache_background_task_prunes_expired_routes() {
        let temp_db = NamedTempFile::new().unwrap();
        let store_url = format!("sqlite://{}", temp_db.path().display());
        let store = Arc::new(crate::store::connect(&store_url).await.unwrap());
        let config = toml::Value::Table({
            let mut table = toml::value::Map::new();
            table.insert("routing_flush_interval_ms".into(), toml::Value::Integer(20));
            table.insert(
                "routing_prune_interval_secs".into(),
                toml::Value::Integer(1),
            );
            table
        });
        let prompt_cache = PromptCache::new(&config, &[])
            .unwrap()
            .with_store(Arc::clone(&store));
        let _task = prompt_cache.spawn_persistence_task().unwrap();
        store
            .upsert_prompt_cache_route(&PromptCacheRouteRecord {
                route_id: "expired-route".to_string(),
                project_id: "project-a".to_string(),
                cache_key: "tenant:expired".to_string(),
                provider_name: "alpha".to_string(),
                signal_kind: "warm".to_string(),
                signal_strength: 1.0,
                observed_at_ms: now_ms().saturating_sub(5_000),
                expires_at_ms: now_ms().saturating_sub(1_000),
            })
            .await
            .unwrap();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let routes = store.get_prompt_cache_routes(now_ms(), 10).await.unwrap();
            if routes.is_empty() {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for background prune"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }
}
