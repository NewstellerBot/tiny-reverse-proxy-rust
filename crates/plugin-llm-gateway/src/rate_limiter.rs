use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use http_body_util::{BodyExt, Full};
use hyper::header::HeaderValue;
use hyper::{Response, StatusCode};
use std::sync::Arc;

use proxy_core::plugin::{Action, Plugin, RequestContext};
use proxy_core::rate_limit::TokenBucketMap;

use crate::metrics::LlmMetrics;
use crate::virtual_keys::VirtualKeyMeta;
use crate::{estimate_prompt_tokens, extract_api_key};

#[derive(Clone)]
pub struct TokenRateLimiter {
    /// Per-key token-per-minute buckets.
    tpm_buckets: TokenBucketMap<String>,
    /// Per-key request-per-minute buckets (1 token = 1 request).
    rpm_buckets: Option<TokenBucketMap<String>>,
    /// Per-virtual-key RPM override buckets, grouped by configured RPM limit.
    per_key_rpm_buckets: Arc<DashMap<u32, TokenBucketMap<String>>>,
    /// Per-virtual-key TPM override buckets, grouped by configured TPM limit.
    per_key_tpm_buckets: Arc<DashMap<u32, TokenBucketMap<String>>>,
    metrics: Option<LlmMetrics>,
}

impl TokenRateLimiter {
    pub fn new(tokens_per_minute: f64, burst_tokens: f64) -> Self {
        Self {
            tpm_buckets: TokenBucketMap::new(tokens_per_minute / 60.0, burst_tokens),
            rpm_buckets: None,
            per_key_rpm_buckets: Arc::new(DashMap::new()),
            per_key_tpm_buckets: Arc::new(DashMap::new()),
            metrics: None,
        }
    }

    pub fn with_rpm(mut self, requests_per_minute: f64, burst_requests: f64) -> Self {
        self.rpm_buckets = Some(TokenBucketMap::new(
            requests_per_minute / 60.0,
            burst_requests,
        ));
        self
    }

    /// Attach LLM Prometheus metrics.
    pub fn with_metrics(mut self, metrics: LlmMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    // --- Accessor methods for the API layer ---

    pub fn tracked_key_count(&self) -> usize {
        self.tpm_buckets.len()
    }

    pub fn rate_per_second(&self) -> f64 {
        self.tpm_buckets.rate()
    }

    pub fn burst(&self) -> f64 {
        self.tpm_buckets.burst()
    }

    pub fn rpm_config(&self) -> Option<(f64, f64)> {
        self.rpm_buckets.as_ref().map(|b| (b.rate(), b.burst()))
    }
}

const RATE_LIMIT_BODY: &str = r#"{"error":{"message":"Token rate limit exceeded","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;

/// Info about which rate limit was hit, used to populate response headers.
struct RateLimitHit {
    /// "requests" or "tokens"
    limit_type: &'static str,
    /// Configured limit per minute.
    limit_per_minute: f64,
    /// Remaining tokens/requests in the bucket right now.
    remaining: f64,
    /// Rate at which the bucket refills (per second).
    rate_per_sec: f64,
}

fn rate_limit_response_with_headers(
    hit: Option<&RateLimitHit>,
) -> Response<http_body_util::combinators::BoxBody<Bytes, hyper::Error>> {
    let mut resp = Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .body(
            Full::new(Bytes::from(RATE_LIMIT_BODY))
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap();
    resp.headers_mut()
        .insert("content-type", HeaderValue::from_static("application/json"));
    if let Some(info) = hit {
        let (limit_hdr, remaining_hdr, reset_hdr) = match info.limit_type {
            "requests" => (
                "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-reset-requests",
            ),
            _ => (
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-tokens",
                "x-ratelimit-reset-tokens",
            ),
        };

        let remaining = info.remaining.max(0.0) as u64;
        let reset_secs = if info.rate_per_sec > 0.0 {
            ((1.0 - info.remaining.max(0.0)) / info.rate_per_sec)
                .ceil()
                .max(1.0) as u64
        } else {
            60
        };

        if let Ok(v) = HeaderValue::from_str(&format!("{}", info.limit_per_minute as u64)) {
            resp.headers_mut().insert(limit_hdr, v);
        }
        if let Ok(v) = HeaderValue::from_str(&format!("{}", remaining)) {
            resp.headers_mut().insert(remaining_hdr, v);
        }
        if let Ok(v) = HeaderValue::from_str(&format!("{}s", reset_secs)) {
            resp.headers_mut().insert(reset_hdr, v);
        }
        if let Ok(v) = HeaderValue::from_str(&format!("{}", reset_secs)) {
            resp.headers_mut().insert("retry-after", v);
        }
    }
    resp
}

#[async_trait]
impl Plugin for TokenRateLimiter {
    fn name(&self) -> &str {
        "rate_limiter"
    }

    async fn on_request(&self, ctx: &mut RequestContext) -> Action {
        // If a virtual key was validated upstream, use its hash as the rate limit key.
        let (rate_key, per_key_rpm, per_key_tpm) =
            if let Some(vk_meta) = ctx.extensions.get::<VirtualKeyMeta>() {
                (
                    vk_meta.key_hash.clone(),
                    vk_meta.rpm_limit,
                    vk_meta.tpm_limit,
                )
            } else {
                match extract_api_key(&ctx.headers) {
                    Some(key) => (key, None, None),
                    None => return Action::Continue, // No API key — skip rate limiting
                }
            };

        // RPM check (1 request = 1 token consumed). Per-key override wins over global.
        if let Some(rpm_limit) = per_key_rpm {
            let bucket = self
                .per_key_rpm_buckets
                .entry(rpm_limit)
                .or_insert_with(|| TokenBucketMap::new(rpm_limit as f64 / 60.0, rpm_limit as f64));
            if !bucket.try_consume(rate_key.clone(), 1.0) {
                if let Some(ref m) = self.metrics {
                    m.rate_limit_rejections_total
                        .with_label_values(&["rpm"])
                        .inc();
                }
                #[cfg(feature = "opentelemetry")]
                if let Some(otel) = ctx.extensions.get::<proxy_core::otel::OtelSpan>() {
                    otel.0.record("llm.rate_limited", true);
                }
                let remaining = bucket.remaining_for_key(&rate_key).unwrap_or(0.0);
                return Action::Respond(rate_limit_response_with_headers(Some(&RateLimitHit {
                    limit_type: "requests",
                    limit_per_minute: rpm_limit as f64,
                    remaining,
                    rate_per_sec: bucket.rate(),
                })));
            }
        } else if let Some(ref rpm) = self.rpm_buckets {
            if !rpm.try_consume(rate_key.clone(), 1.0) {
                if let Some(ref m) = self.metrics {
                    m.rate_limit_rejections_total
                        .with_label_values(&["rpm"])
                        .inc();
                }
                #[cfg(feature = "opentelemetry")]
                if let Some(otel) = ctx.extensions.get::<proxy_core::otel::OtelSpan>() {
                    otel.0.record("llm.rate_limited", true);
                }
                let remaining = rpm.remaining_for_key(&rate_key).unwrap_or(0.0);
                return Action::Respond(rate_limit_response_with_headers(Some(&RateLimitHit {
                    limit_type: "requests",
                    limit_per_minute: rpm.rate() * 60.0,
                    remaining,
                    rate_per_sec: rpm.rate(),
                })));
            }
        }

        // TPM check.
        let estimated_tokens = ctx
            .body
            .as_ref()
            .map(|b| estimate_prompt_tokens(b))
            .unwrap_or(1) as f64;

        let tpm_hit = if let Some(tpm_limit) = per_key_tpm {
            let bucket = self
                .per_key_tpm_buckets
                .entry(tpm_limit)
                .or_insert_with(|| TokenBucketMap::new(tpm_limit as f64 / 60.0, tpm_limit as f64));
            if bucket.try_consume(rate_key.clone(), estimated_tokens) {
                None
            } else {
                Some(RateLimitHit {
                    limit_type: "tokens",
                    limit_per_minute: tpm_limit as f64,
                    remaining: bucket.remaining_for_key(&rate_key).unwrap_or(0.0),
                    rate_per_sec: bucket.rate(),
                })
            }
        } else if self
            .tpm_buckets
            .try_consume(rate_key.clone(), estimated_tokens)
        {
            None
        } else {
            Some(RateLimitHit {
                limit_type: "tokens",
                limit_per_minute: self.tpm_buckets.rate() * 60.0,
                remaining: self.tpm_buckets.remaining_for_key(&rate_key).unwrap_or(0.0),
                rate_per_sec: self.tpm_buckets.rate(),
            })
        };

        if let Some(hit) = tpm_hit {
            if let Some(ref m) = self.metrics {
                m.rate_limit_rejections_total
                    .with_label_values(&["tpm"])
                    .inc();
            }
            #[cfg(feature = "opentelemetry")]
            if let Some(otel) = ctx.extensions.get::<proxy_core::otel::OtelSpan>() {
                otel.0.record("llm.rate_limited", true);
            }
            Action::Respond(rate_limit_response_with_headers(Some(&hit)))
        } else {
            Action::Continue
        }
    }
}

/// Create a TokenRateLimiter directly (not boxed) for use with the API layer.
pub fn create_limiter(
    config: &toml::Value,
) -> Result<TokenRateLimiter, Box<dyn std::error::Error>> {
    let tokens_per_minute = config
        .get("tokens_per_minute")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .unwrap_or(100_000.0);
    let burst_tokens = config
        .get("burst_tokens")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .unwrap_or(20_000.0);

    let mut limiter = TokenRateLimiter::new(tokens_per_minute, burst_tokens);

    // Optional RPM (requests-per-minute) limiting.
    if let Some(rpm) = config
        .get("requests_per_minute")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
    {
        let burst_requests = config
            .get("burst_requests")
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .unwrap_or(rpm / 2.0);
        limiter = limiter.with_rpm(rpm, burst_requests);
    }

    Ok(limiter)
}

/// Factory function for creating a TokenRateLimiter from TOML config.
pub fn create(config: &toml::Value) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>> {
    let plugin = create_limiter(config)?;
    plugin.tpm_buckets.spawn_cleanup_task();
    if let Some(ref rpm) = plugin.rpm_buckets {
        rpm.spawn_cleanup_task();
    }
    Ok(Box::new(plugin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::header::{HeaderMap, HeaderValue};
    use hyper::http::Extensions;
    use hyper::{Method, Uri, Version};
    use std::sync::Arc;

    fn make_ctx(api_key: &str, body_size: usize) -> RequestContext {
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
            body: Some(Bytes::from(vec![b'x'; body_size])),
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        }
    }

    #[tokio::test]
    async fn test_allows_within_burst() {
        let limiter = TokenRateLimiter::new(60_000.0, 1000.0);
        // 400 bytes => 100 tokens, burst is 1000
        let mut ctx = make_ctx("sk-test", 400);
        match limiter.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("should allow within burst"),
        }
    }

    #[tokio::test]
    async fn test_rejects_over_burst() {
        let limiter = TokenRateLimiter::new(60.0, 100.0);
        // 800 bytes => 200 tokens, burst is only 100
        let mut ctx = make_ctx("sk-test", 800);
        match limiter.on_request(&mut ctx).await {
            Action::Continue => panic!("should reject over burst"),
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
            }
        }
    }

    #[tokio::test]
    async fn test_variable_consumption() {
        let limiter = TokenRateLimiter::new(60.0, 100.0);

        // First request: 200 bytes => 50 tokens (within burst of 100)
        let mut ctx1 = make_ctx("sk-key", 200);
        match limiter.on_request(&mut ctx1).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("first request should be allowed"),
        }

        // Second request: 300 bytes => 75 tokens (50 remaining < 75 needed)
        let mut ctx2 = make_ctx("sk-key", 300);
        match limiter.on_request(&mut ctx2).await {
            Action::Continue => panic!("second request should be rejected"),
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
            }
        }
    }

    #[tokio::test]
    async fn test_separate_keys() {
        let limiter = TokenRateLimiter::new(60.0, 100.0);

        // Exhaust key A's budget
        let mut ctx_a = make_ctx("sk-key-a", 800);
        match limiter.on_request(&mut ctx_a).await {
            Action::Continue => panic!("should reject"),
            Action::Respond(_) => {}
        }

        // Key B should still have its own budget
        let mut ctx_b = make_ctx("sk-key-b", 200);
        match limiter.on_request(&mut ctx_b).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("different key should be allowed"),
        }
    }

    #[tokio::test]
    async fn test_no_api_key_passes_through() {
        let limiter = TokenRateLimiter::new(60.0, 10.0);
        let mut ctx = RequestContext {
            peer_addr: None,
            method: Method::POST,
            uri: Uri::from_static("http://localhost/v1/chat/completions"),
            version: Version::HTTP_11,
            headers: HeaderMap::new(),
            body: Some(Bytes::from(vec![b'x'; 10000])),
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        };
        match limiter.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("no API key should pass through"),
        }
    }

    #[tokio::test]
    async fn test_per_virtual_key_rpm_override_persists_across_requests() {
        let limiter = TokenRateLimiter::new(60_000.0, 10_000.0);

        let mut ctx1 = make_ctx("sk-vk", 128);
        ctx1.extensions.insert(VirtualKeyMeta {
            key_hash: "vk_hash_1".to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: None,
            budget_duration: None,
            budget_window_start: None,
            rpm_limit: Some(1),
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });

        match limiter.on_request(&mut ctx1).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("first request should be allowed"),
        }

        let mut ctx2 = make_ctx("sk-vk", 128);
        ctx2.extensions.insert(VirtualKeyMeta {
            key_hash: "vk_hash_1".to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: None,
            budget_duration: None,
            budget_window_start: None,
            rpm_limit: Some(1),
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });

        match limiter.on_request(&mut ctx2).await {
            Action::Continue => panic!("second request should be rate limited"),
            Action::Respond(resp) => assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS),
        }
    }

    #[tokio::test]
    async fn test_per_virtual_key_tpm_override_wins_over_global_limit() {
        let limiter = TokenRateLimiter::new(60_000.0, 10_000.0);

        let mut ctx1 = make_ctx("sk-vk", 120); // 30 tokens
        ctx1.extensions.insert(VirtualKeyMeta {
            key_hash: "vk_hash_tpm".to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: None,
            budget_duration: None,
            budget_window_start: None,
            rpm_limit: None,
            tpm_limit: Some(40),
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });
        assert!(matches!(
            limiter.on_request(&mut ctx1).await,
            Action::Continue
        ));

        let mut ctx2 = make_ctx("sk-vk", 120); // another 30 tokens => exceeds custom 40 TPM burst
        ctx2.extensions.insert(VirtualKeyMeta {
            key_hash: "vk_hash_tpm".to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: None,
            budget_duration: None,
            budget_window_start: None,
            rpm_limit: None,
            tpm_limit: Some(40),
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });
        assert!(matches!(
            limiter.on_request(&mut ctx2).await,
            Action::Respond(_)
        ));
    }

    // --- OpenAI SDK compatibility tests ---

    #[test]
    fn test_rate_limit_response_openai_format() {
        // Error response JSON must contain message, type, AND code for OpenAI SDK compat.
        let body: &str = RATE_LIMIT_BODY;
        assert!(body.contains(r#""message":"#), "must have message field");
        assert!(body.contains(r#""type":"#), "must have type field");
        assert!(body.contains(r#""code":"#), "must have code field");
        assert!(
            body.starts_with(r#"{"error":{"#),
            "must be wrapped in error object"
        );
    }

    // --- Edge-case tests ---

    #[tokio::test]
    async fn test_invalid_utf8_body_no_panic() {
        // #2: Invalid UTF-8 bytes in request body should not cause panics.
        // estimate_prompt_tokens uses body.len() directly, no UTF-8 parsing.
        let limiter = TokenRateLimiter::new(60_000.0, 1000.0);
        let invalid_utf8: Vec<u8> = vec![0xFF, 0xFE, 0x80, 0x81, 0xC0, 0xC1];
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str("Bearer sk-utf8-test").unwrap(),
        );
        let mut ctx = RequestContext {
            peer_addr: None,
            method: Method::POST,
            uri: Uri::from_static("http://localhost/v1/chat/completions"),
            version: Version::HTTP_11,
            headers,
            body: Some(Bytes::from(invalid_utf8)),
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        };

        // 6 bytes => 1 token, well within burst of 1000
        match limiter.on_request(&mut ctx).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("should continue with invalid UTF-8 body"),
        }
    }

    // --- Gap tests: RPM rate limiting ---

    #[tokio::test]
    async fn test_global_rpm_limiting() {
        // 1 RPM with burst of 1 → first request allowed, second rejected
        let limiter = TokenRateLimiter::new(60_000.0, 10_000.0).with_rpm(1.0, 1.0);

        let mut ctx1 = make_ctx("sk-rpm-test", 128);
        match limiter.on_request(&mut ctx1).await {
            Action::Continue => {}
            Action::Respond(_) => panic!("first request should be allowed by RPM"),
        }

        let mut ctx2 = make_ctx("sk-rpm-test", 128);
        match limiter.on_request(&mut ctx2).await {
            Action::Continue => panic!("second request should be rejected by RPM"),
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
            }
        }
    }

    #[tokio::test]
    async fn test_rpm_separate_keys() {
        // Different API keys should have independent RPM buckets.
        let limiter = TokenRateLimiter::new(60_000.0, 10_000.0).with_rpm(1.0, 1.0);

        // Exhaust key A's RPM
        let mut ctx_a = make_ctx("sk-rpm-a", 128);
        assert!(matches!(
            limiter.on_request(&mut ctx_a).await,
            Action::Continue
        ));

        let mut ctx_a2 = make_ctx("sk-rpm-a", 128);
        assert!(matches!(
            limiter.on_request(&mut ctx_a2).await,
            Action::Respond(_)
        ));

        // Key B should still have its own RPM budget
        let mut ctx_b = make_ctx("sk-rpm-b", 128);
        assert!(matches!(
            limiter.on_request(&mut ctx_b).await,
            Action::Continue
        ));
    }

    #[tokio::test]
    async fn test_rpm_checked_before_tpm() {
        // RPM limit of 1 with huge TPM → should reject on RPM, not TPM.
        let limiter = TokenRateLimiter::new(60_000.0, 100_000.0).with_rpm(1.0, 1.0);

        let mut ctx1 = make_ctx("sk-rpm-first", 4); // tiny body
        assert!(matches!(
            limiter.on_request(&mut ctx1).await,
            Action::Continue
        ));

        let mut ctx2 = make_ctx("sk-rpm-first", 4); // tiny body, within TPM
        match limiter.on_request(&mut ctx2).await {
            Action::Continue => panic!("should be rejected by RPM despite being within TPM"),
            Action::Respond(resp) => {
                assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
            }
        }
    }

    #[tokio::test]
    async fn test_virtual_key_uses_key_hash_for_rate_limiting() {
        // When VirtualKeyMeta is present, rate limiting should use key_hash.
        let limiter = TokenRateLimiter::new(60.0, 100.0); // tight burst: 100 tokens

        // Use 80 tokens via virtual key hash "vk_rate_hash"
        let mut ctx1 = make_ctx("sk-raw", 320); // 80 tokens
        ctx1.extensions.insert(VirtualKeyMeta {
            key_hash: "vk_rate_hash".to_string(),
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
        assert!(matches!(
            limiter.on_request(&mut ctx1).await,
            Action::Continue
        ));

        // Try to use 80 more under the SAME hash → should fail (80+80 > 100 burst)
        let mut ctx2 = make_ctx("sk-raw", 320);
        ctx2.extensions.insert(VirtualKeyMeta {
            key_hash: "vk_rate_hash".to_string(),
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
        assert!(matches!(
            limiter.on_request(&mut ctx2).await,
            Action::Respond(_)
        ));

        // Different raw key but SAME virtual key hash → still rejected
        let mut ctx3 = make_ctx("sk-other-raw", 320);
        ctx3.extensions.insert(VirtualKeyMeta {
            key_hash: "vk_rate_hash".to_string(),
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
        assert!(matches!(
            limiter.on_request(&mut ctx3).await,
            Action::Respond(_)
        ));
    }

    #[tokio::test]
    async fn test_per_key_rpm_different_limits() {
        // Two virtual keys with different RPM limits.
        let limiter = TokenRateLimiter::new(60_000.0, 10_000.0);

        // Key with RPM=2 (burst 2): first two requests allowed, third rejected
        for i in 0..2 {
            let mut ctx = make_ctx("sk-vk", 128);
            ctx.extensions.insert(VirtualKeyMeta {
                key_hash: "vk_rpm2".to_string(),
                project_id: "legacy".to_string(),
                provider_name: "openai".to_string(),
                budget_limit: None,
                budget_duration: None,
                budget_window_start: None,
                rpm_limit: Some(2),
                tpm_limit: None,
                tool_approval_mode: Default::default(),
                allowed_tools: None,
            });
            match limiter.on_request(&mut ctx).await {
                Action::Continue => {}
                Action::Respond(_) => panic!("request {} should be allowed (rpm=2)", i),
            }
        }

        let mut ctx3 = make_ctx("sk-vk", 128);
        ctx3.extensions.insert(VirtualKeyMeta {
            key_hash: "vk_rpm2".to_string(),
            project_id: "legacy".to_string(),
            provider_name: "openai".to_string(),
            budget_limit: None,
            budget_duration: None,
            budget_window_start: None,
            rpm_limit: Some(2),
            tpm_limit: None,
            tool_approval_mode: Default::default(),
            allowed_tools: None,
        });
        assert!(
            matches!(limiter.on_request(&mut ctx3).await, Action::Respond(_)),
            "third request should be rejected (rpm=2)"
        );
    }

    #[test]
    fn test_rpm_config_accessor() {
        let limiter = TokenRateLimiter::new(60_000.0, 10_000.0);
        assert!(limiter.rpm_config().is_none(), "no RPM configured");

        let limiter_with_rpm = limiter.with_rpm(120.0, 60.0);
        let (rate, burst) = limiter_with_rpm.rpm_config().unwrap();
        assert!((rate - 2.0).abs() < 1e-9, "120 RPM / 60 = 2.0 per sec");
        assert!((burst - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_rate_limit_response_without_info_has_no_headers() {
        // Passing None should not add rate limit headers.
        let resp = rate_limit_response_with_headers(None);
        assert!(resp.headers().get("x-ratelimit-limit-requests").is_none());
        assert!(resp.headers().get("retry-after").is_none());
    }

    #[test]
    fn test_rate_limit_response_with_rpm_headers() {
        let resp = rate_limit_response_with_headers(Some(&RateLimitHit {
            limit_type: "requests",
            limit_per_minute: 60.0,
            remaining: 0.0,
            rate_per_sec: 1.0,
        }));
        assert!(
            resp.headers().get("x-ratelimit-limit-requests").is_some(),
            "should have limit-requests header"
        );
        assert_eq!(
            resp.headers()
                .get("x-ratelimit-limit-requests")
                .unwrap()
                .to_str()
                .unwrap(),
            "60"
        );
        assert!(
            resp.headers()
                .get("x-ratelimit-remaining-requests")
                .is_some(),
            "should have remaining-requests header"
        );
        assert!(
            resp.headers().get("x-ratelimit-reset-requests").is_some(),
            "should have reset-requests header"
        );
        assert!(
            resp.headers().get("retry-after").is_some(),
            "should have retry-after header"
        );
    }

    #[test]
    fn test_rate_limit_response_with_tpm_headers() {
        let resp = rate_limit_response_with_headers(Some(&RateLimitHit {
            limit_type: "tokens",
            limit_per_minute: 100_000.0,
            remaining: 50.0,
            rate_per_sec: 1666.67,
        }));
        assert_eq!(
            resp.headers()
                .get("x-ratelimit-limit-tokens")
                .unwrap()
                .to_str()
                .unwrap(),
            "100000"
        );
        assert!(resp.headers().get("x-ratelimit-remaining-tokens").is_some());
        assert!(resp.headers().get("x-ratelimit-reset-tokens").is_some());
        assert!(resp.headers().get("retry-after").is_some());
    }
}
