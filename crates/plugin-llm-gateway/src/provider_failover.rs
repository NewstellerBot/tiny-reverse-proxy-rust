use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use dashmap::DashMap;

use proxy_core::plugin::{Action, Plugin, ProviderCandidates, ProxyError, RequestContext};

use crate::metrics::LlmMetrics;
use crate::virtual_keys::RoutingDebugTrace;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderFailureReason {
    Timeout,
    RateLimited,
    Upstream5xx,
    TransportError,
}

impl ProviderFailureReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::RateLimited => "rate_limited",
            Self::Upstream5xx => "upstream_5xx",
            Self::TransportError => "transport_error",
        }
    }
}

#[derive(Clone, Debug)]
pub struct FailedProviderStatus {
    pub failed_at: Instant,
    pub reason: ProviderFailureReason,
}

#[derive(Clone)]
pub struct ProviderConfig {
    pub name: String,
    pub pattern: String,
}

#[derive(Clone)]
pub struct ProviderFailover {
    providers: Vec<ProviderConfig>,
    /// Maps provider name -> cooldown state when it was marked failed.
    failed_providers: Arc<DashMap<String, FailedProviderStatus>>,
    cooldown: Duration,
    metrics: Option<LlmMetrics>,
}

impl ProviderFailover {
    pub fn new(providers: Vec<(String, String)>, cooldown_secs: u64) -> Self {
        Self {
            providers: providers
                .into_iter()
                .map(|(name, pattern)| ProviderConfig { name, pattern })
                .collect(),
            failed_providers: Arc::new(DashMap::new()),
            cooldown: Duration::from_secs(cooldown_secs),
            metrics: None,
        }
    }

    /// Attach LLM Prometheus metrics.
    pub fn with_metrics(mut self, metrics: LlmMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Find the provider name whose pattern matches the given upstream URL.
    fn provider_for_upstream(&self, upstream: &str) -> Option<&str> {
        self.providers
            .iter()
            .find(|p| upstream.contains(&p.pattern))
            .map(|p| p.name.as_str())
    }

    // --- Accessor methods for the API layer ---

    pub fn get_providers(&self) -> Vec<ProviderConfig> {
        self.providers.clone()
    }

    pub fn get_failed_providers(&self) -> Vec<(String, FailedProviderStatus)> {
        self.failed_providers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    pub fn cooldown(&self) -> Duration {
        self.cooldown
    }

    pub fn clear_failed(&self, name: &str) -> bool {
        let cleared = self.failed_providers.remove(name).is_some();
        if cleared {
            if let Some(ref m) = self.metrics {
                m.provider_cooldown_active.with_label_values(&[name]).set(0);
            }
        }
        cleared
    }

    pub fn clear_all_failed(&self) {
        if let Some(ref m) = self.metrics {
            for provider in self.failed_providers.iter() {
                m.provider_cooldown_active
                    .with_label_values(&[provider.key()])
                    .set(0);
            }
        }
        self.failed_providers.clear();
    }

    fn cooldown_remaining(&self, failed_at: Instant, now: Instant) -> Option<Duration> {
        let elapsed = now.checked_duration_since(failed_at)?;
        self.cooldown.checked_sub(elapsed)
    }

    fn failure_reason(error: &ProxyError) -> ProviderFailureReason {
        match error {
            ProxyError::Timeout => ProviderFailureReason::Timeout,
            ProxyError::UpstreamStatus(status)
                if *status == hyper::StatusCode::TOO_MANY_REQUESTS =>
            {
                ProviderFailureReason::RateLimited
            }
            ProxyError::UpstreamStatus(status) if status.is_server_error() => {
                ProviderFailureReason::Upstream5xx
            }
            ProxyError::ConnectError(_)
            | ProxyError::UpstreamError(_)
            | ProxyError::ServerError(_) => ProviderFailureReason::TransportError,
            ProxyError::UpstreamStatus(_) => ProviderFailureReason::TransportError,
        }
    }
}

#[async_trait]
impl Plugin for ProviderFailover {
    fn name(&self) -> &str {
        "provider_failover"
    }

    async fn on_upstream_select(
        &self,
        ctx: &mut RequestContext,
        servers: &mut Vec<&String>,
    ) -> Action {
        let now = Instant::now();

        // Remove servers whose provider is still in cooldown.
        servers.retain(|server| {
            if let Some(provider_name) = self.provider_for_upstream(server) {
                if let Some(entry) = self.failed_providers.get(provider_name) {
                    let status = entry.clone();
                    if self.cooldown_remaining(status.failed_at, now).is_some() {
                        tracing::debug!(
                            provider = provider_name,
                            server = %server,
                            "skipping failed provider"
                        );
                        return false;
                    }
                    // Cooldown elapsed — remove from failed set.
                    drop(entry);
                    self.failed_providers.remove(provider_name);
                    if let Some(ref m) = self.metrics {
                        m.provider_cooldown_active
                            .with_label_values(&[provider_name])
                            .set(0);
                    }
                }
            }
            true
        });

        // Also filter ProviderCandidates if present.
        if let Some(mut candidates) = ctx.extensions.remove::<ProviderCandidates>() {
            candidates.0.retain(|candidate| {
                if let Some(provider_name) = self.provider_for_upstream(&candidate.upstream) {
                    if let Some(entry) = self.failed_providers.get(provider_name) {
                        let status = entry.clone();
                        if self.cooldown_remaining(status.failed_at, now).is_some() {
                            tracing::debug!(
                                provider = provider_name,
                                upstream = %candidate.upstream,
                                "filtering failed provider from candidates"
                            );
                            return false;
                        }
                        drop(entry);
                        self.failed_providers.remove(provider_name);
                        if let Some(ref m) = self.metrics {
                            m.provider_cooldown_active
                                .with_label_values(&[provider_name])
                                .set(0);
                        }
                    }
                }
                true
            });
            if let Some(trace) = ctx.extensions.get_mut::<RoutingDebugTrace>() {
                trace.ordered_providers = candidates
                    .0
                    .iter()
                    .filter_map(|candidate| {
                        self.provider_for_upstream(&candidate.upstream)
                            .map(ToString::to_string)
                    })
                    .collect();
            }
            ctx.extensions.insert(candidates);
        } else if let Some(trace) = ctx.extensions.get_mut::<RoutingDebugTrace>() {
            trace.ordered_providers = servers
                .iter()
                .filter_map(|server| self.provider_for_upstream(server).map(ToString::to_string))
                .collect();
        }

        Action::Continue
    }

    async fn on_error(&self, ctx: &mut RequestContext, _error: &ProxyError) -> Action {
        let error = _error;
        if let Some(ref upstream) = ctx.selected_upstream {
            if let Some(provider_name) = self.provider_for_upstream(upstream) {
                let reason = Self::failure_reason(error);
                tracing::warn!(
                    provider = provider_name,
                    upstream = %upstream,
                    reason = reason.as_str(),
                    "marking provider as failed"
                );
                self.failed_providers.insert(
                    provider_name.to_string(),
                    FailedProviderStatus {
                        failed_at: Instant::now(),
                        reason: reason.clone(),
                    },
                );
                if let Some(ref m) = self.metrics {
                    m.provider_errors_total
                        .with_label_values(&[provider_name])
                        .inc();
                    m.provider_cooldowns_total
                        .with_label_values(&[provider_name, reason.as_str()])
                        .inc();
                    m.provider_cooldown_active
                        .with_label_values(&[provider_name])
                        .set(1);
                }
            }
        }
        Action::Continue
    }
}

/// Create a ProviderFailover directly (not boxed) for use with the API layer.
pub fn create_failover(
    config: &toml::Value,
) -> Result<ProviderFailover, Box<dyn std::error::Error>> {
    let cooldown_secs = config
        .get("cooldown_secs")
        .and_then(|v| v.as_integer())
        .unwrap_or(30) as u64;

    let providers = config
        .get("providers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|entry| {
                    let name = entry.get("name")?.as_str()?.to_string();
                    let pattern = entry.get("pattern")?.as_str()?.to_string();
                    Some((name, pattern))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(ProviderFailover::new(providers, cooldown_secs))
}

/// Factory function for creating a ProviderFailover from TOML config.
pub fn create(config: &toml::Value) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>> {
    Ok(Box::new(create_failover(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::header::HeaderMap;
    use hyper::http::Extensions;
    use hyper::{Method, Uri, Version};
    use proxy_core::plugin::{ProviderCandidate, ProviderCandidates};
    use std::sync::Arc;

    fn make_failover() -> ProviderFailover {
        ProviderFailover::new(
            vec![
                ("openai".into(), "api.openai.com".into()),
                ("anthropic".into(), "api.anthropic.com".into()),
            ],
            30,
        )
    }

    fn make_ctx() -> RequestContext {
        RequestContext {
            peer_addr: None,
            method: Method::POST,
            uri: Uri::from_static("http://localhost/v1/chat/completions"),
            version: Version::HTTP_11,
            headers: HeaderMap::new(),
            body: None,
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        }
    }

    fn failed_status(reason: ProviderFailureReason) -> FailedProviderStatus {
        FailedProviderStatus {
            failed_at: Instant::now(),
            reason,
        }
    }

    #[tokio::test]
    async fn test_no_failures_passes_all() {
        let failover = make_failover();
        let mut ctx = make_ctx();
        let s1 = "https://api.openai.com".to_string();
        let s2 = "https://api.anthropic.com".to_string();
        let mut servers = vec![&s1, &s2];

        failover.on_upstream_select(&mut ctx, &mut servers).await;
        assert_eq!(servers.len(), 2);
    }

    #[tokio::test]
    async fn test_failed_provider_removed() {
        let failover = make_failover();
        let mut ctx = make_ctx();

        // Mark openai as failed.
        failover.failed_providers.insert(
            "openai".into(),
            failed_status(ProviderFailureReason::Timeout),
        );

        let s1 = "https://api.openai.com".to_string();
        let s2 = "https://api.anthropic.com".to_string();
        let mut servers = vec![&s1, &s2];

        failover.on_upstream_select(&mut ctx, &mut servers).await;
        assert_eq!(servers.len(), 1);
        assert!(servers[0].contains("anthropic"));
    }

    #[tokio::test]
    async fn test_cooldown_expired_restores_provider() {
        let failover = ProviderFailover::new(
            vec![("openai".into(), "api.openai.com".into())],
            0, // 0-second cooldown => immediately restored
        );
        let mut ctx = make_ctx();

        // Mark as failed in the past.
        failover.failed_providers.insert(
            "openai".into(),
            FailedProviderStatus {
                failed_at: Instant::now() - Duration::from_secs(1),
                reason: ProviderFailureReason::Timeout,
            },
        );

        let s1 = "https://api.openai.com".to_string();
        let mut servers = vec![&s1];

        failover.on_upstream_select(&mut ctx, &mut servers).await;
        assert_eq!(
            servers.len(),
            1,
            "provider should be restored after cooldown"
        );
        assert!(
            !failover.failed_providers.contains_key("openai"),
            "entry should be removed after cooldown"
        );
    }

    #[tokio::test]
    async fn test_on_error_marks_provider_failed() {
        let failover = make_failover();
        let mut ctx = make_ctx();
        ctx.selected_upstream = Some("https://api.openai.com".into());

        let err = ProxyError::Timeout;
        failover.on_error(&mut ctx, &err).await;

        assert!(failover.failed_providers.contains_key("openai"));
        assert_eq!(
            failover
                .failed_providers
                .get("openai")
                .as_deref()
                .map(|status| status.reason.as_str()),
            Some("timeout")
        );
    }

    #[tokio::test]
    async fn test_unmatched_server_not_affected() {
        let failover = make_failover();
        let mut ctx = make_ctx();

        // Mark both known providers as failed.
        failover.failed_providers.insert(
            "openai".into(),
            failed_status(ProviderFailureReason::Timeout),
        );
        failover.failed_providers.insert(
            "anthropic".into(),
            failed_status(ProviderFailureReason::TransportError),
        );

        let s1 = "https://api.openai.com".to_string();
        let s2 = "https://custom-llm.example.com".to_string();
        let mut servers = vec![&s1, &s2];

        failover.on_upstream_select(&mut ctx, &mut servers).await;
        assert_eq!(servers.len(), 1);
        assert!(servers[0].contains("custom-llm"));
    }

    #[test]
    fn test_provider_for_upstream() {
        let failover = make_failover();
        assert_eq!(
            failover.provider_for_upstream("https://api.openai.com/v1/chat"),
            Some("openai")
        );
        assert_eq!(
            failover.provider_for_upstream("https://api.anthropic.com/v1/messages"),
            Some("anthropic")
        );
        assert_eq!(
            failover.provider_for_upstream("https://custom.example.com"),
            None
        );
    }

    #[test]
    fn test_clone_shares_failed_state() {
        let failover = make_failover();
        failover.failed_providers.insert(
            "openai".into(),
            failed_status(ProviderFailureReason::Timeout),
        );

        let cloned = failover.clone();
        assert_eq!(cloned.get_failed_providers().len(), 1);

        // Mutation via original visible through clone.
        failover.clear_all_failed();
        assert_eq!(cloned.get_failed_providers().len(), 0);
    }

    // --- Edge-case tests ---

    #[tokio::test]
    async fn test_all_providers_failed() {
        // #5: When ALL providers are marked as failed, on_upstream_select should filter
        // out every server, leaving an empty list (caller is responsible for 502).
        let failover = make_failover();
        let mut ctx = make_ctx();

        failover.failed_providers.insert(
            "openai".into(),
            failed_status(ProviderFailureReason::Timeout),
        );
        failover.failed_providers.insert(
            "anthropic".into(),
            failed_status(ProviderFailureReason::TransportError),
        );

        let s1 = "https://api.openai.com".to_string();
        let s2 = "https://api.anthropic.com".to_string();
        let mut servers = vec![&s1, &s2];

        let action = failover.on_upstream_select(&mut ctx, &mut servers).await;
        assert!(servers.is_empty(), "all failed providers should be removed");
        assert!(matches!(action, Action::Continue));
    }

    #[tokio::test]
    async fn test_single_provider_fails() {
        // #16: Only one provider configured. When it fails, the server list becomes empty
        // since there's nowhere to fail over to.
        let failover = ProviderFailover::new(vec![("openai".into(), "api.openai.com".into())], 30);
        let mut ctx = make_ctx();

        failover.failed_providers.insert(
            "openai".into(),
            failed_status(ProviderFailureReason::Timeout),
        );

        let s1 = "https://api.openai.com".to_string();
        let mut servers = vec![&s1];

        failover.on_upstream_select(&mut ctx, &mut servers).await;
        assert!(
            servers.is_empty(),
            "single failed provider should leave empty server list"
        );
    }

    #[tokio::test]
    async fn test_empty_providers_list() {
        // #17: ProviderFailover with an empty providers vec should not panic.
        let failover = ProviderFailover::new(vec![], 30);
        let mut ctx = make_ctx();

        // Unmatched servers should pass through since no providers are configured.
        let s1 = "https://api.openai.com".to_string();
        let mut servers = vec![&s1];
        failover.on_upstream_select(&mut ctx, &mut servers).await;
        assert_eq!(servers.len(), 1, "unmatched server should remain");

        // Also test with an empty servers list.
        let mut empty_servers: Vec<&String> = vec![];
        failover
            .on_upstream_select(&mut ctx, &mut empty_servers)
            .await;
        assert!(empty_servers.is_empty());
    }

    #[tokio::test]
    async fn provider_failover_filters_candidates_list() {
        let failover = make_failover();
        let mut ctx = make_ctx();

        // Insert candidates.
        ctx.extensions.insert(ProviderCandidates(vec![
            ProviderCandidate {
                upstream: "https://api.openai.com".to_string(),
                headers: HeaderMap::new(),
            },
            ProviderCandidate {
                upstream: "https://api.anthropic.com".to_string(),
                headers: HeaderMap::new(),
            },
        ]));

        // Mark openai as failed.
        failover.failed_providers.insert(
            "openai".into(),
            failed_status(ProviderFailureReason::Timeout),
        );

        let s1 = "https://api.openai.com".to_string();
        let s2 = "https://api.anthropic.com".to_string();
        let mut servers = vec![&s1, &s2];
        failover.on_upstream_select(&mut ctx, &mut servers).await;

        // Candidates should be filtered too.
        let candidates = ctx.extensions.get::<ProviderCandidates>().unwrap();
        assert_eq!(candidates.0.len(), 1);
        assert!(candidates.0[0].upstream.contains("anthropic"));
    }
}
