pub mod api;
pub mod content_filter;
pub mod cost_tracker;
pub mod evals;
pub mod governance;
pub mod management_server;
pub mod metrics;
pub mod prompt_cache;
pub mod prompt_registry;
pub mod provider_failover;
pub mod rate_limiter;
pub mod semantic_cache;
pub mod semantic_safety;
pub mod session_recovery;
pub mod store;
pub mod streaming;
pub mod tool_runtime;
pub mod virtual_keys;

use std::sync::Arc;

use bytes::Bytes;
use hyper::header::HeaderMap;
use hyper::header::{HeaderValue, CONTENT_LENGTH};
use proxy_auth::service::AuthService;
use proxy_core::config::{ModelAliasConfig, PluginConfig, ProviderKeyConfig};
use proxy_core::plugin::{Plugin, PluginRegistry, ProviderCandidates, RequestContext};
use serde::de::{MapAccess, Visitor};
use serde::Deserialize;
use serde_json::Value;

use api::LlmGatewayApi;
use governance::GovernanceState;

#[derive(Clone, Debug)]
pub(crate) struct CachedRequestJson {
    pub value: Value,
}

pub(crate) const REQUEST_METADATA_HEADER: &str = "cf-aig-metadata";
pub(crate) const REQUEST_CUSTOM_COST_HEADER: &str = "cf-aig-custom-cost";

#[derive(Clone, Debug)]
pub(crate) struct RequestMetadata {
    pub value: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct RequestCustomCost {
    pub per_token_in: f64,
    pub per_token_out: f64,
}

impl RequestCustomCost {
    pub fn to_json_value(&self) -> Value {
        serde_json::json!({
            "per_token_in": self.per_token_in,
            "per_token_out": self.per_token_out,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestCustomCostPayload {
    per_token_in: f64,
    per_token_out: f64,
}

struct OrderedRequestMetadataEntries(Vec<(String, Value)>);

impl<'de> serde::Deserialize<'de> for OrderedRequestMetadataEntries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OrderedRequestMetadataVisitor;

        impl<'de> Visitor<'de> for OrderedRequestMetadataVisitor {
            type Value = OrderedRequestMetadataEntries;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, Value>()? {
                    entries.push((key, value));
                }
                Ok(OrderedRequestMetadataEntries(entries))
            }
        }

        deserializer.deserialize_map(OrderedRequestMetadataVisitor)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CreatePluginsOptions {
    pub bootstrap_admin_token: Option<String>,
    pub allow_direct_provider_keys: bool,
}

fn request_metadata_value_is_scalar(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_))
}

pub(crate) fn parse_request_metadata_header_value(raw: &str) -> Result<RequestMetadata, String> {
    let OrderedRequestMetadataEntries(entries) = serde_json::from_str(raw).map_err(|error| {
        format!("{REQUEST_METADATA_HEADER} must be a valid JSON object: {error}")
    })?;

    let mut kept_entries = Vec::new();
    for (key, value) in entries {
        if !request_metadata_value_is_scalar(&value) {
            return Err(format!(
                "{REQUEST_METADATA_HEADER} values must be strings, numbers, or booleans"
            ));
        }
        if kept_entries.len() < 5 {
            kept_entries.push((key, value));
        }
    }

    let mut object = serde_json::Map::new();
    for (key, value) in kept_entries {
        object.insert(key, value);
    }
    Ok(RequestMetadata {
        value: Value::Object(object),
    })
}

pub(crate) fn take_request_metadata(
    headers: &mut HeaderMap,
) -> Result<Option<RequestMetadata>, String> {
    let Some(raw_value) = headers.remove(REQUEST_METADATA_HEADER) else {
        return Ok(None);
    };
    let raw = raw_value
        .to_str()
        .map_err(|_| format!("{REQUEST_METADATA_HEADER} must be valid UTF-8"))?;
    parse_request_metadata_header_value(raw).map(Some)
}

pub(crate) fn parse_request_custom_cost_header_value(
    raw: &str,
) -> Result<RequestCustomCost, String> {
    let payload: RequestCustomCostPayload = serde_json::from_str(raw).map_err(|error| {
        format!(
            "{REQUEST_CUSTOM_COST_HEADER} must be a valid JSON object with per_token_in and per_token_out: {error}"
        )
    })?;
    if !payload.per_token_in.is_finite() || payload.per_token_in < 0.0 {
        return Err(format!(
            "{REQUEST_CUSTOM_COST_HEADER}.per_token_in must be a finite non-negative number"
        ));
    }
    if !payload.per_token_out.is_finite() || payload.per_token_out < 0.0 {
        return Err(format!(
            "{REQUEST_CUSTOM_COST_HEADER}.per_token_out must be a finite non-negative number"
        ));
    }
    Ok(RequestCustomCost {
        per_token_in: payload.per_token_in,
        per_token_out: payload.per_token_out,
    })
}

pub(crate) fn take_request_custom_cost(
    headers: &mut HeaderMap,
) -> Result<Option<RequestCustomCost>, String> {
    let Some(raw_value) = headers.remove(REQUEST_CUSTOM_COST_HEADER) else {
        return Ok(None);
    };
    let raw = raw_value
        .to_str()
        .map_err(|_| format!("{REQUEST_CUSTOM_COST_HEADER} must be valid UTF-8"))?;
    parse_request_custom_cost_header_value(raw).map(Some)
}

/// Register all LLM gateway plugins with the registry.
pub fn register_all(registry: &mut PluginRegistry) {
    registry.register("rate_limiter", rate_limiter::create);
    registry.register("provider_failover", provider_failover::create);
    registry.register("cost_tracker", cost_tracker::create);
    registry.register("prompt_registry", prompt_registry::create);
    registry.register("prompt_cache", prompt_cache::create);
    registry.register("semantic_cache", semantic_cache::create);
    registry.register("semantic_safety", semantic_safety::create);
    registry.register("content_filter", content_filter::create);
}

/// Create all enabled LLM gateway plugins and return both the plugin list and an API handle.
///
/// This is the preferred entry point when using the management API.  It creates
/// concrete plugin instances, clones them into an `LlmGatewayApi`, and returns
/// the boxed plugins for use in a `PluginChain`.
///
/// When `providers` is non-empty, a `VirtualKeys` plugin is created first in the chain.
///
/// If `registry` is provided, LLM-specific Prometheus metrics are registered and
/// wired into each plugin.
pub async fn create_plugins(
    configs: &[PluginConfig],
    store_url: Option<&str>,
    providers: &[ProviderKeyConfig],
    model_aliases: &[ModelAliasConfig],
    registry: Option<&prometheus::Registry>,
) -> Result<(Vec<Box<dyn Plugin>>, LlmGatewayApi), Box<dyn std::error::Error>> {
    create_plugins_with_options(
        configs,
        store_url,
        providers,
        model_aliases,
        CreatePluginsOptions::default(),
        registry,
    )
    .await
}

pub async fn create_plugins_with_options(
    configs: &[PluginConfig],
    store_url: Option<&str>,
    providers: &[ProviderKeyConfig],
    model_aliases: &[ModelAliasConfig],
    options: CreatePluginsOptions,
    registry: Option<&prometheus::Registry>,
) -> Result<(Vec<Box<dyn Plugin>>, LlmGatewayApi), Box<dyn std::error::Error>> {
    // Connect to the store if a URL is provided.
    let store = match store_url {
        Some(url) => {
            let s = store::connect(url).await?;
            Some(Arc::new(s))
        }
        None => None,
    };
    let auth_store = match store_url {
        Some(url) => {
            let s = proxy_auth::store::connect(url).await?;
            Some(Arc::new(s))
        }
        None => None,
    };
    let bootstrap_token = options
        .bootstrap_admin_token
        .or_else(|| std::env::var("TRP_BOOTSTRAP_ADMIN_TOKEN").ok());
    let auth_service = Arc::new(AuthService::new(auth_store, bootstrap_token));
    auth_service.load_from_store().await?;

    let governance = Arc::new(GovernanceState::new(store.clone()));
    governance.load_from_store().await?;

    let llm_metrics = registry.map(metrics::LlmMetrics::new);

    let mut plugins: Vec<Box<dyn Plugin>> = Vec::new();
    let mut cost_tracker_handle: Option<cost_tracker::CostTracker> = None;
    let mut rate_limiter_handle: Option<rate_limiter::TokenRateLimiter> = None;
    let mut failover_handle: Option<provider_failover::ProviderFailover> = None;
    let mut virtual_keys_handle: Option<virtual_keys::VirtualKeys> = None;
    let mut prompt_cache_handle: Option<prompt_cache::PromptCache> = None;
    let mut semantic_cache_handle: Option<semantic_cache::SemanticCache> = None;
    let mut semantic_safety_handle: Option<semantic_safety::SemanticSafety> = None;
    let mut tool_runtime_handle: Option<tool_runtime::ToolRuntime> = None;

    // Virtual keys plugin goes FIRST in the chain so it runs before cost/rate.
    if !providers.is_empty() {
        let mut vk = virtual_keys::VirtualKeys::new_with_governance(
            providers,
            model_aliases,
            store.clone(),
            Arc::clone(&governance),
            options.allow_direct_provider_keys,
        );
        if let Some(ref m) = llm_metrics {
            vk = vk.with_metrics(m.clone());
        }
        vk.load_from_store().await?;
        virtual_keys_handle = Some(vk.clone());
        plugins.push(Box::new(vk));
        tracing::info!(
            providers = providers.len(),
            "virtual_keys: loaded {} providers",
            providers.len()
        );
    }

    let semantic_safety_position = configs
        .iter()
        .position(|pc| pc.enabled && pc.name == "semantic_safety");
    let content_filter_position = configs
        .iter()
        .position(|pc| pc.enabled && pc.name == "content_filter");
    let prompt_registry_position = configs
        .iter()
        .position(|pc| pc.enabled && pc.name == "prompt_registry");
    let prompt_cache_position = configs
        .iter()
        .position(|pc| pc.enabled && pc.name == "prompt_cache");
    let semantic_cache_position = configs
        .iter()
        .position(|pc| pc.enabled && pc.name == "semantic_cache");
    let tool_runtime_position = configs
        .iter()
        .position(|pc| pc.enabled && pc.name == "tool_runtime");

    if let (Some(semantic_index), Some(filter_index)) =
        (semantic_safety_position, content_filter_position)
    {
        if semantic_index < filter_index {
            return Err(
                "semantic_safety must be configured after content_filter so local redaction runs before remote export"
                    .into(),
            );
        }
    }
    if let Some(prompt_registry_index) = prompt_registry_position {
        for (plugin_name, plugin_index) in [
            ("semantic_cache", semantic_cache_position),
            ("prompt_cache", prompt_cache_position),
            ("content_filter", content_filter_position),
            ("semantic_safety", semantic_safety_position),
            ("tool_runtime", tool_runtime_position),
        ] {
            if let Some(plugin_index) = plugin_index {
                if prompt_registry_index > plugin_index {
                    return Err(format!(
                        "prompt_registry must be configured before {plugin_name} so resolved prompt text is included in downstream caching, safety, and tool processing"
                    )
                    .into());
                }
            }
        }
    }
    if let (Some(filter_index), Some(cache_index)) =
        (content_filter_position, semantic_cache_position)
    {
        if cache_index < filter_index {
            return Err(
                "semantic_cache must be configured after content_filter so local redaction runs before cache matching"
                    .into(),
            );
        }
    }
    if let (Some(semantic_index), Some(cache_index)) =
        (semantic_safety_position, semantic_cache_position)
    {
        if cache_index < semantic_index {
            return Err(
                "semantic_cache must be configured after semantic_safety so request safety checks still run on cache hits"
                    .into(),
            );
        }
    }
    if let (Some(cache_index), Some(prompt_cache_index)) =
        (semantic_cache_position, prompt_cache_position)
    {
        if cache_index > prompt_cache_index {
            return Err(
                "semantic_cache must be configured before prompt_cache so gateway cache checks happen before provider-specific cache controls"
                    .into(),
            );
        }
    }

    for pc in configs {
        if !pc.enabled {
            continue;
        }
        match pc.name.as_str() {
            "prompt_registry" => {
                let registry = prompt_registry::create_plugin(&pc.config, Arc::clone(&governance))?;
                plugins.push(Box::new(registry));
                tracing::info!("prompt_registry: enabled");
            }
            "content_filter" => {
                let filter = content_filter::create_filter(&pc.config)?
                    .with_governance(Arc::clone(&governance));
                plugins.push(Box::new(filter));
                tracing::info!("content_filter: enabled");
            }
            "prompt_cache" => {
                let effective_providers = virtual_keys_handle
                    .as_ref()
                    .map(|virtual_keys| virtual_keys.provider_configs())
                    .unwrap_or_else(|| providers.to_vec());
                let mut prompt_cache =
                    prompt_cache::create_plugin(&pc.config, &effective_providers)?;
                if let Some(ref m) = llm_metrics {
                    prompt_cache = prompt_cache.with_metrics(m.clone());
                }
                if let Some(ref s) = store {
                    prompt_cache = prompt_cache.with_store(Arc::clone(s));
                    prompt_cache.load_from_store().await?;
                    prompt_cache.spawn_persistence_task();
                }
                if let Some(ref virtual_keys) = virtual_keys_handle {
                    virtual_keys.set_prompt_cache_anthropic_default_scope(
                        prompt_cache.anthropic_default_scope_for_routing(),
                    );
                    virtual_keys.set_prompt_cache_routing_memory(prompt_cache.routing_memory());
                }
                prompt_cache_handle = Some(prompt_cache.clone());
                plugins.push(Box::new(prompt_cache));
                tracing::info!("prompt_cache: enabled");
            }
            "semantic_cache" => {
                let mut semantic_cache = semantic_cache::create_plugin(&pc.config)?
                    .with_governance(Arc::clone(&governance));
                if let Some(ref m) = llm_metrics {
                    semantic_cache = semantic_cache.with_metrics(m.clone());
                }
                if let Some(ref s) = store {
                    semantic_cache = semantic_cache.with_store(Arc::clone(s));
                    semantic_cache.load_from_store().await?;
                }
                if let Some(ref virtual_keys) = virtual_keys_handle {
                    virtual_keys.set_semantic_cache_handle(semantic_cache.clone());
                }
                semantic_cache_handle = Some(semantic_cache.clone());
                plugins.push(Box::new(semantic_cache));
                tracing::info!("semantic_cache: enabled");
            }
            "semantic_safety" => {
                let mut semantic =
                    semantic_safety::create_plugin(&pc.config, Arc::clone(&governance))?;
                if let Some(ref m) = llm_metrics {
                    semantic = semantic.with_metrics(m.clone());
                }
                semantic.spawn_reconciliation_task();
                semantic_safety_handle = Some(semantic.clone());
                plugins.push(Box::new(semantic));
                tracing::info!("semantic_safety: enabled");
            }
            "tool_runtime" => {
                let effective_providers = virtual_keys_handle
                    .as_ref()
                    .map(|virtual_keys| virtual_keys.provider_configs())
                    .unwrap_or_else(|| providers.to_vec());
                let runtime = tool_runtime::create_plugin(
                    &pc.config,
                    &effective_providers,
                    Arc::clone(&governance),
                )
                .await?;
                tool_runtime_handle = Some(runtime.clone());
                plugins.push(Box::new(runtime));
                tracing::info!("tool_runtime: enabled");
            }
            "cost_tracker" => {
                let mut tracker = cost_tracker::create_tracker(&pc.config)?;
                if let Some(ref m) = llm_metrics {
                    tracker = tracker.with_metrics(m.clone());
                }
                if let Some(ref s) = store {
                    tracker = tracker.with_store(Arc::clone(s));
                    tracker.load_from_store().await?;
                    tracker.spawn_flush_task(30);
                    tracker.spawn_audit_drain_task();
                }

                let model_count = tracker.get_model_costs().len();
                if model_count == 0 {
                    tracing::info!(
                        "cost_tracker: no per-model pricing in config, will use defaults"
                    );
                } else {
                    tracing::info!(
                        models = model_count,
                        "cost_tracker loaded pricing for {} models",
                        model_count
                    );
                }
                tracker.spawn_logger_task();

                cost_tracker_handle = Some(tracker.clone());
                plugins.push(Box::new(tracker));
            }
            "rate_limiter" | "token_rate_limiter" => {
                let mut limiter = rate_limiter::create_limiter(&pc.config)?;
                if let Some(ref m) = llm_metrics {
                    limiter = limiter.with_metrics(m.clone());
                }
                rate_limiter_handle = Some(limiter.clone());
                plugins.push(Box::new(limiter));
            }
            "provider_failover" => {
                let mut failover = provider_failover::create_failover(&pc.config)?;
                if let Some(ref m) = llm_metrics {
                    failover = failover.with_metrics(m.clone());
                }
                failover_handle = Some(failover.clone());
                plugins.push(Box::new(failover));
            }
            other => {
                tracing::warn!("unknown LLM gateway plugin: {}", other);
            }
        }
    }

    let recovery_store = store.clone();
    let api = LlmGatewayApi::new(
        cost_tracker_handle,
        rate_limiter_handle,
        failover_handle,
        virtual_keys_handle,
        prompt_cache_handle,
        semantic_cache_handle,
        semantic_safety_handle,
        tool_runtime_handle,
        store,
    )
    .with_governance(auth_service, governance);

    if let Some(ref store) = recovery_store {
        evals::spawn_eval_recovery_task(Arc::clone(store));
        session_recovery::spawn_session_recovery_task(Arc::clone(store));
    }

    Ok((plugins, api))
}

/// Extract the API key from `Authorization: Bearer <key>` (OpenAI)
/// or `x-api-key: <key>` (Anthropic).
pub fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    // OpenAI-style: Authorization: Bearer <key>
    if let Some(value) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(token) = value.strip_prefix("Bearer ") {
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    // Anthropic-style: x-api-key: <key>
    if let Some(value) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

/// Estimate prompt tokens from the request body using the ~4 chars/token heuristic.
pub fn estimate_prompt_tokens(body: &[u8]) -> u64 {
    (body.len() as u64) / 4
}

/// Extract the model name from a JSON body by byte-scanning for `"model":"..."`.
/// Avoids pulling in serde_json.
pub fn extract_model(body: &[u8]) -> Option<String> {
    // Look for "model" followed by optional whitespace, colon, optional whitespace, quote.
    let key = b"\"model\"";
    let body_len = body.len();
    let key_len = key.len();

    let mut i = 0;
    while i + key_len < body_len {
        if &body[i..i + key_len] == key {
            // Skip past key and find the colon.
            let mut j = i + key_len;
            // Skip whitespace.
            while j < body_len
                && (body[j] == b' ' || body[j] == b'\t' || body[j] == b'\n' || body[j] == b'\r')
            {
                j += 1;
            }
            if j >= body_len || body[j] != b':' {
                i += 1;
                continue;
            }
            j += 1; // skip colon
                    // Skip whitespace.
            while j < body_len
                && (body[j] == b' ' || body[j] == b'\t' || body[j] == b'\n' || body[j] == b'\r')
            {
                j += 1;
            }
            if j >= body_len || body[j] != b'"' {
                i += 1;
                continue;
            }
            j += 1; // skip opening quote
            let start = j;
            while j < body_len && body[j] != b'"' {
                j += 1;
            }
            if j < body_len {
                return Some(String::from_utf8_lossy(&body[start..j]).into_owned());
            }
        }
        i += 1;
    }
    None
}

pub(crate) fn cached_request_json<'a>(ctx: &'a mut RequestContext) -> Option<&'a Value> {
    if !ensure_cached_request_json(ctx) {
        return None;
    }
    ctx.extensions
        .get::<CachedRequestJson>()
        .map(|cached| &cached.value)
}

pub(crate) fn cached_request_json_mut<'a>(ctx: &'a mut RequestContext) -> Option<&'a mut Value> {
    if !ensure_cached_request_json(ctx) {
        return None;
    }
    ctx.extensions
        .get_mut::<CachedRequestJson>()
        .map(|cached| &mut cached.value)
}

pub(crate) fn sync_cached_request_json_body(ctx: &mut RequestContext) -> Result<(), String> {
    let Some(cached) = ctx.extensions.get::<CachedRequestJson>() else {
        return Ok(());
    };
    let body = serde_json::to_vec(&cached.value)
        .map(Bytes::from)
        .map_err(|error| format!("failed to encode request body: {error}"))?;
    apply_request_body_update(ctx, body, None);
    Ok(())
}

pub(crate) fn update_request_body(ctx: &mut RequestContext, body: Bytes) {
    let parsed = serde_json::from_slice::<Value>(&body).ok();
    apply_request_body_update(ctx, body, parsed);
}

fn ensure_cached_request_json(ctx: &mut RequestContext) -> bool {
    if ctx.extensions.get::<CachedRequestJson>().is_some() {
        return true;
    }
    let Some(body) = ctx.body.as_ref() else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return false;
    };
    ctx.extensions.insert(CachedRequestJson { value });
    true
}

fn apply_request_body_update(ctx: &mut RequestContext, body: Bytes, parsed: Option<Value>) {
    if let Ok(value) = HeaderValue::from_str(&body.len().to_string()) {
        ctx.headers.insert(CONTENT_LENGTH, value.clone());
        if let Some(candidates) = ctx.extensions.get_mut::<ProviderCandidates>() {
            for candidate in &mut candidates.0 {
                candidate.headers.insert(CONTENT_LENGTH, value.clone());
            }
        }
    }
    ctx.body = Some(body);
    match parsed {
        Some(value) => ctx.extensions.insert(CachedRequestJson { value }),
        None => ctx.extensions.remove::<CachedRequestJson>(),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use hyper::header::{HeaderMap, HeaderValue};
    use hyper::http::Extensions;
    use hyper::{Method, Uri, Version};
    use proxy_core::plugin::RequestContext;
    use std::sync::Arc;

    #[test]
    fn test_extract_api_key_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer sk-test-123"),
        );
        assert_eq!(extract_api_key(&headers), Some("sk-test-123".into()));
    }

    #[test]
    fn test_extract_api_key_missing() {
        let headers = HeaderMap::new();
        assert_eq!(extract_api_key(&headers), None);
    }

    #[test]
    fn test_extract_api_key_no_bearer_prefix() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Basic abc123"));
        assert_eq!(extract_api_key(&headers), None);
    }

    #[test]
    fn test_extract_api_key_empty_token() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer "));
        assert_eq!(extract_api_key(&headers), None);
    }

    #[test]
    fn test_extract_api_key_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("sk-ant-abc123"));
        assert_eq!(extract_api_key(&headers), Some("sk-ant-abc123".into()));
    }

    #[test]
    fn test_extract_api_key_x_api_key_empty() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static(""));
        assert_eq!(extract_api_key(&headers), None);
    }

    #[test]
    fn test_extract_api_key_bearer_takes_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer sk-openai"),
        );
        headers.insert("x-api-key", HeaderValue::from_static("sk-anthropic"));
        assert_eq!(extract_api_key(&headers), Some("sk-openai".into()));
    }

    #[test]
    fn test_estimate_prompt_tokens() {
        // 100 bytes => 25 tokens
        let body = vec![b'a'; 100];
        assert_eq!(estimate_prompt_tokens(&body), 25);
    }

    #[test]
    fn test_estimate_prompt_tokens_empty() {
        assert_eq!(estimate_prompt_tokens(b""), 0);
    }

    #[test]
    fn test_extract_model_simple() {
        let body = br#"{"model":"gpt-4","messages":[]}"#;
        assert_eq!(extract_model(body), Some("gpt-4".into()));
    }

    #[test]
    fn test_extract_model_with_spaces() {
        let body = br#"{"model" : "claude-3-opus" , "messages":[]}"#;
        assert_eq!(extract_model(body), Some("claude-3-opus".into()));
    }

    #[test]
    fn test_extract_model_missing() {
        let body = br#"{"messages":[]}"#;
        assert_eq!(extract_model(body), None);
    }

    #[test]
    fn test_extract_model_not_first_field() {
        let body = br#"{"messages":[],"model":"gpt-3.5-turbo","temperature":0.7}"#;
        assert_eq!(extract_model(body), Some("gpt-3.5-turbo".into()));
    }

    // --- Edge-case tests ---

    #[test]
    fn test_extract_api_key_bearer_empty_after_prefix() {
        // #14: "Bearer " (trailing space, no key) should return None.
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer "));
        assert_eq!(extract_api_key(&headers), None);
    }

    #[test]
    fn test_extract_api_key_bearer_whitespace_only() {
        // #14 edge: "Bearer  " (space as key) — strip_prefix("Bearer ") yields " ",
        // which is not empty, so it returns Some(" ").
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer  "));
        // A single space is technically a non-empty key after stripping the prefix.
        assert_eq!(extract_api_key(&headers), Some(" ".into()));
    }

    #[test]
    fn test_very_long_api_key() {
        // #15: Keys >10KB used as DashMap keys — should not panic.
        let long_key = "sk-".to_string() + &"a".repeat(10_240);
        let header_val = format!("Bearer {long_key}");
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_str(&header_val).unwrap());
        let result = extract_api_key(&headers);
        assert_eq!(result, Some(long_key.clone()));

        // Verify masking works for long keys (first 8 chars + "...").
        let masked = if long_key.len() > 8 {
            format!("{}...", &long_key[..8])
        } else {
            long_key.clone()
        };
        assert_eq!(masked, "sk-aaaaa...");
    }

    #[test]
    fn test_very_long_api_key_as_dashmap_key() {
        // #15: Verify DashMap can store and retrieve very long keys without panic.
        let long_key = "sk-".to_string() + &"b".repeat(10_240);
        let map: dashmap::DashMap<String, u64> = dashmap::DashMap::new();
        map.insert(long_key.clone(), 42);
        assert_eq!(*map.get(&long_key).unwrap(), 42);
    }

    #[test]
    fn test_extract_model_invalid_utf8() {
        // #2: extract_model should handle invalid UTF-8 gracefully (uses from_utf8_lossy).
        let mut body = br#"{"model":""#.to_vec();
        body.extend_from_slice(&[0xFF, 0xFE]);
        body.extend_from_slice(br#"","messages":[]}"#);
        // Should not panic. The model name will contain replacement characters.
        let result = extract_model(&body);
        assert!(result.is_some());
    }

    #[test]
    fn test_cached_request_json_syncs_body() {
        let mut ctx = RequestContext {
            peer_addr: None,
            method: Method::POST,
            uri: Uri::from_static("/v1/chat/completions"),
            version: Version::HTTP_11,
            headers: HeaderMap::new(),
            body: Some(Bytes::from_static(br#"{"model":"gpt-4o","stream":false}"#)),
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        };

        let cached = cached_request_json_mut(&mut ctx).expect("request json");
        cached["stream"] = Value::Bool(true);
        sync_cached_request_json_body(&mut ctx).expect("sync request body");

        let body = std::str::from_utf8(ctx.body.as_ref().unwrap()).unwrap();
        assert!(body.contains(r#""stream":true"#));
        assert_eq!(
            cached_request_json(&mut ctx)
                .and_then(|value| value.get("stream"))
                .and_then(Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn test_parse_request_metadata_header_value_accepts_scalars_and_limits_entries() {
        let metadata = parse_request_metadata_header_value(
            r#"{"first":"a","second":2,"third":true,"fourth":"d","fifth":5,"sixth":"ignored"}"#,
        )
        .expect("request metadata");

        let object = metadata.value.as_object().expect("metadata object");
        assert_eq!(object.len(), 5);
        assert_eq!(object.get("first").and_then(Value::as_str), Some("a"));
        assert_eq!(object.get("second").and_then(Value::as_i64), Some(2));
        assert_eq!(object.get("third").and_then(Value::as_bool), Some(true));
        assert!(object.get("sixth").is_none());
    }

    #[test]
    fn test_parse_request_metadata_header_value_rejects_non_object_and_nested_values() {
        for raw in [
            r#"[]"#,
            r#"{"nested":{"x":1}}"#,
            r#"{"list":[1,2]}"#,
            r#"{"empty":null}"#,
        ] {
            assert!(
                parse_request_metadata_header_value(raw).is_err(),
                "expected metadata parse failure for {raw}"
            );
        }
    }

    #[test]
    fn test_take_request_metadata_removes_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            REQUEST_METADATA_HEADER,
            HeaderValue::from_static(r#"{"trace_id":"abc","sampled":true}"#),
        );

        let metadata = take_request_metadata(&mut headers)
            .expect("take metadata")
            .expect("metadata present");
        assert!(!headers.contains_key(REQUEST_METADATA_HEADER));
        assert_eq!(
            metadata.value.get("trace_id").and_then(Value::as_str),
            Some("abc")
        );
    }

    #[test]
    fn test_parse_request_custom_cost_header_value_accepts_non_negative_numbers() {
        let custom_cost = parse_request_custom_cost_header_value(
            r#"{"per_token_in":0.000001,"per_token_out":0.000002}"#,
        )
        .expect("custom cost");
        assert!((custom_cost.per_token_in - 0.000001).abs() < f64::EPSILON);
        assert!((custom_cost.per_token_out - 0.000002).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_request_custom_cost_header_value_rejects_invalid_shapes() {
        for raw in [
            r#"[]"#,
            r#"{"per_token_in":"1","per_token_out":2}"#,
            r#"{"per_token_in":1,"per_token_out":null}"#,
            r#"{"per_token_in":-1,"per_token_out":2}"#,
            r#"{"per_token_in":1,"per_token_out":2,"extra":3}"#,
        ] {
            assert!(
                parse_request_custom_cost_header_value(raw).is_err(),
                "expected custom cost parse failure for {raw}"
            );
        }
    }

    #[test]
    fn test_take_request_custom_cost_removes_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            REQUEST_CUSTOM_COST_HEADER,
            HeaderValue::from_static(r#"{"per_token_in":0.1,"per_token_out":0.2}"#),
        );

        let custom_cost = take_request_custom_cost(&mut headers)
            .expect("take custom cost")
            .expect("custom cost present");
        assert!(!headers.contains_key(REQUEST_CUSTOM_COST_HEADER));
        assert!((custom_cost.per_token_in - 0.1).abs() < f64::EPSILON);
        assert!((custom_cost.per_token_out - 0.2).abs() < f64::EPSILON);
    }
}
