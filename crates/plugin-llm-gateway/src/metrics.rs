use prometheus::{
    CounterVec, GaugeVec, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts, Registry,
};

/// LLM-specific Prometheus metrics.
///
/// All counters/gauges are registered against the provided `Registry` so they
/// appear alongside the core proxy metrics on the `/metrics` endpoint.
#[derive(Clone)]
pub struct LlmMetrics {
    pub tokens_total: IntCounterVec,
    pub cost_dollars_total: CounterVec,
    pub request_tokens: HistogramVec,
    pub rate_limit_rejections_total: IntCounterVec,
    pub budget_rejections_total: IntCounter,
    pub provider_errors_total: IntCounterVec,
    pub provider_cooldowns_total: IntCounterVec,
    pub provider_selected_total: IntCounterVec,
    pub provider_ewma_latency_ms: GaugeVec,
    pub provider_ewma_error_rate: GaugeVec,
    pub provider_ewma_timeout_rate: GaugeVec,
    pub provider_ewma_rate_limit_rate: GaugeVec,
    pub provider_active_requests: IntGaugeVec,
    pub provider_cooldown_active: IntGaugeVec,
    pub active_virtual_keys: IntGauge,
    pub streaming_requests_total: IntCounter,
    pub prompt_cache_requests_total: IntCounterVec,
    pub prompt_cache_read_tokens_total: IntCounterVec,
    pub prompt_cache_write_tokens_total: IntCounterVec,
    pub semantic_cache_requests_total: IntCounterVec,
    pub semantic_cache_saved_prompt_tokens_total: IntCounterVec,
    pub semantic_cache_entries: IntGauge,
    pub semantic_requests_total: IntCounterVec,
    pub semantic_findings_total: IntCounter,
    pub semantic_degraded_total: IntCounter,
    pub semantic_service_latency_ms: HistogramVec,
}

impl LlmMetrics {
    pub fn new(registry: &Registry) -> Self {
        let token_buckets = vec![
            1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 50000.0,
        ];

        let tokens_total = IntCounterVec::new(
            Opts::new("llm_tokens_total", "Total LLM tokens processed"),
            &["key", "model", "direction"],
        )
        .expect("failed to create llm_tokens_total");
        registry
            .register(Box::new(tokens_total.clone()))
            .expect("failed to register llm_tokens_total");

        let cost_dollars_total = CounterVec::new(
            Opts::new("llm_cost_dollars_total", "Total LLM cost in dollars"),
            &["key", "model"],
        )
        .expect("failed to create llm_cost_dollars_total");
        registry
            .register(Box::new(cost_dollars_total.clone()))
            .expect("failed to register llm_cost_dollars_total");

        let request_tokens = HistogramVec::new(
            HistogramOpts::new("llm_request_tokens", "Token count per request")
                .buckets(token_buckets),
            &["direction"],
        )
        .expect("failed to create llm_request_tokens");
        registry
            .register(Box::new(request_tokens.clone()))
            .expect("failed to register llm_request_tokens");

        let rate_limit_rejections_total = IntCounterVec::new(
            Opts::new(
                "llm_rate_limit_rejections_total",
                "Total LLM rate limit rejections",
            ),
            &["limit_type"],
        )
        .expect("failed to create llm_rate_limit_rejections_total");
        registry
            .register(Box::new(rate_limit_rejections_total.clone()))
            .expect("failed to register llm_rate_limit_rejections_total");

        let budget_rejections_total = IntCounter::new(
            "llm_budget_rejections_total",
            "Total LLM budget exceeded rejections",
        )
        .expect("failed to create llm_budget_rejections_total");
        registry
            .register(Box::new(budget_rejections_total.clone()))
            .expect("failed to register llm_budget_rejections_total");

        let provider_errors_total = IntCounterVec::new(
            Opts::new(
                "llm_provider_errors_total",
                "Total LLM provider errors triggering failover",
            ),
            &["provider"],
        )
        .expect("failed to create llm_provider_errors_total");
        registry
            .register(Box::new(provider_errors_total.clone()))
            .expect("failed to register llm_provider_errors_total");

        let provider_cooldowns_total = IntCounterVec::new(
            Opts::new(
                "llm_provider_cooldowns_total",
                "Total provider cooldowns triggered by failover reason",
            ),
            &["provider", "reason"],
        )
        .expect("failed to create llm_provider_cooldowns_total");
        registry
            .register(Box::new(provider_cooldowns_total.clone()))
            .expect("failed to register llm_provider_cooldowns_total");

        let provider_selected_total = IntCounterVec::new(
            Opts::new(
                "llm_provider_selected_total",
                "Total final provider selections returned to clients",
            ),
            &["provider"],
        )
        .expect("failed to create llm_provider_selected_total");
        registry
            .register(Box::new(provider_selected_total.clone()))
            .expect("failed to register llm_provider_selected_total");

        let provider_ewma_latency_ms = GaugeVec::new(
            Opts::new(
                "llm_provider_ewma_latency_ms",
                "Current provider EWMA latency in milliseconds",
            ),
            &["provider"],
        )
        .expect("failed to create llm_provider_ewma_latency_ms");
        registry
            .register(Box::new(provider_ewma_latency_ms.clone()))
            .expect("failed to register llm_provider_ewma_latency_ms");

        let provider_ewma_error_rate = GaugeVec::new(
            Opts::new(
                "llm_provider_ewma_error_rate",
                "Current provider EWMA error rate",
            ),
            &["provider"],
        )
        .expect("failed to create llm_provider_ewma_error_rate");
        registry
            .register(Box::new(provider_ewma_error_rate.clone()))
            .expect("failed to register llm_provider_ewma_error_rate");

        let provider_ewma_timeout_rate = GaugeVec::new(
            Opts::new(
                "llm_provider_ewma_timeout_rate",
                "Current provider EWMA timeout rate",
            ),
            &["provider"],
        )
        .expect("failed to create llm_provider_ewma_timeout_rate");
        registry
            .register(Box::new(provider_ewma_timeout_rate.clone()))
            .expect("failed to register llm_provider_ewma_timeout_rate");

        let provider_ewma_rate_limit_rate = GaugeVec::new(
            Opts::new(
                "llm_provider_ewma_rate_limit_rate",
                "Current provider EWMA rate-limit rate",
            ),
            &["provider"],
        )
        .expect("failed to create llm_provider_ewma_rate_limit_rate");
        registry
            .register(Box::new(provider_ewma_rate_limit_rate.clone()))
            .expect("failed to register llm_provider_ewma_rate_limit_rate");

        let provider_active_requests = IntGaugeVec::new(
            Opts::new(
                "llm_provider_active_requests",
                "Current in-flight request count per provider",
            ),
            &["provider"],
        )
        .expect("failed to create llm_provider_active_requests");
        registry
            .register(Box::new(provider_active_requests.clone()))
            .expect("failed to register llm_provider_active_requests");

        let provider_cooldown_active = IntGaugeVec::new(
            Opts::new(
                "llm_provider_cooldown_active",
                "Whether a provider is currently in cooldown",
            ),
            &["provider"],
        )
        .expect("failed to create llm_provider_cooldown_active");
        registry
            .register(Box::new(provider_cooldown_active.clone()))
            .expect("failed to register llm_provider_cooldown_active");

        let active_virtual_keys =
            IntGauge::new("llm_active_virtual_keys", "Number of active virtual keys")
                .expect("failed to create llm_active_virtual_keys");
        registry
            .register(Box::new(active_virtual_keys.clone()))
            .expect("failed to register llm_active_virtual_keys");

        let streaming_requests_total = IntCounter::new(
            "llm_streaming_requests_total",
            "Total LLM streaming (SSE) requests",
        )
        .expect("failed to create llm_streaming_requests_total");
        registry
            .register(Box::new(streaming_requests_total.clone()))
            .expect("failed to register llm_streaming_requests_total");

        let prompt_cache_requests_total = IntCounterVec::new(
            Opts::new(
                "llm_prompt_cache_requests_total",
                "Total prompt cache observations by provider and outcome",
            ),
            &["provider", "outcome"],
        )
        .expect("failed to create llm_prompt_cache_requests_total");
        registry
            .register(Box::new(prompt_cache_requests_total.clone()))
            .expect("failed to register llm_prompt_cache_requests_total");

        let prompt_cache_read_tokens_total = IntCounterVec::new(
            Opts::new(
                "llm_prompt_cache_read_tokens_total",
                "Total prompt cache read tokens observed by provider",
            ),
            &["provider"],
        )
        .expect("failed to create llm_prompt_cache_read_tokens_total");
        registry
            .register(Box::new(prompt_cache_read_tokens_total.clone()))
            .expect("failed to register llm_prompt_cache_read_tokens_total");

        let prompt_cache_write_tokens_total = IntCounterVec::new(
            Opts::new(
                "llm_prompt_cache_write_tokens_total",
                "Total prompt cache write tokens observed by provider",
            ),
            &["provider"],
        )
        .expect("failed to create llm_prompt_cache_write_tokens_total");
        registry
            .register(Box::new(prompt_cache_write_tokens_total.clone()))
            .expect("failed to register llm_prompt_cache_write_tokens_total");

        let semantic_cache_requests_total = IntCounterVec::new(
            Opts::new(
                "llm_semantic_cache_requests_total",
                "Total semantic cache observations by provider and outcome",
            ),
            &["provider", "outcome"],
        )
        .expect("failed to create llm_semantic_cache_requests_total");
        registry
            .register(Box::new(semantic_cache_requests_total.clone()))
            .expect("failed to register llm_semantic_cache_requests_total");

        let semantic_cache_saved_prompt_tokens_total = IntCounterVec::new(
            Opts::new(
                "llm_semantic_cache_saved_prompt_tokens_total",
                "Estimated prompt tokens saved by semantic cache hits",
            ),
            &["provider"],
        )
        .expect("failed to create llm_semantic_cache_saved_prompt_tokens_total");
        registry
            .register(Box::new(semantic_cache_saved_prompt_tokens_total.clone()))
            .expect("failed to register llm_semantic_cache_saved_prompt_tokens_total");

        let semantic_cache_entries = IntGauge::new(
            "llm_semantic_cache_entries",
            "Current number of semantic cache entries",
        )
        .expect("failed to create llm_semantic_cache_entries");
        registry
            .register(Box::new(semantic_cache_entries.clone()))
            .expect("failed to register llm_semantic_cache_entries");

        let semantic_requests_total = IntCounterVec::new(
            Opts::new(
                "llm_semantic_requests_total",
                "Total semantic safety evaluations by outcome",
            ),
            &["outcome"],
        )
        .expect("failed to create llm_semantic_requests_total");
        registry
            .register(Box::new(semantic_requests_total.clone()))
            .expect("failed to register llm_semantic_requests_total");

        let semantic_findings_total = IntCounter::new(
            "llm_semantic_findings_total",
            "Total semantic safety findings returned by the service",
        )
        .expect("failed to create llm_semantic_findings_total");
        registry
            .register(Box::new(semantic_findings_total.clone()))
            .expect("failed to register llm_semantic_findings_total");

        let semantic_degraded_total = IntCounter::new(
            "llm_semantic_degraded_total",
            "Total semantic safety degraded evaluations",
        )
        .expect("failed to create llm_semantic_degraded_total");
        registry
            .register(Box::new(semantic_degraded_total.clone()))
            .expect("failed to register llm_semantic_degraded_total");

        let semantic_service_latency_ms = HistogramVec::new(
            HistogramOpts::new(
                "llm_semantic_service_latency_ms",
                "Semantic safety service latency in milliseconds",
            )
            .buckets(vec![5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]),
            &["result"],
        )
        .expect("failed to create llm_semantic_service_latency_ms");
        registry
            .register(Box::new(semantic_service_latency_ms.clone()))
            .expect("failed to register llm_semantic_service_latency_ms");

        Self {
            tokens_total,
            cost_dollars_total,
            request_tokens,
            rate_limit_rejections_total,
            budget_rejections_total,
            provider_errors_total,
            provider_cooldowns_total,
            provider_selected_total,
            provider_ewma_latency_ms,
            provider_ewma_error_rate,
            provider_ewma_timeout_rate,
            provider_ewma_rate_limit_rate,
            provider_active_requests,
            provider_cooldown_active,
            active_virtual_keys,
            streaming_requests_total,
            prompt_cache_requests_total,
            prompt_cache_read_tokens_total,
            prompt_cache_write_tokens_total,
            semantic_cache_requests_total,
            semantic_cache_saved_prompt_tokens_total,
            semantic_cache_entries,
            semantic_requests_total,
            semantic_findings_total,
            semantic_degraded_total,
            semantic_service_latency_ms,
        }
    }
}

/// Mask an API key for use as a Prometheus label: first 8 chars + "...".
pub fn mask_key(key: &str) -> String {
    if key.len() > 8 {
        format!("{}...", &key[..8])
    } else {
        key.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_metrics_register_and_use() {
        let registry = Registry::new();
        let m = LlmMetrics::new(&registry);

        m.tokens_total
            .with_label_values(&["sk-abcde...", "gpt-4o", "input"])
            .inc_by(100);
        m.tokens_total
            .with_label_values(&["sk-abcde...", "gpt-4o", "output"])
            .inc_by(50);
        m.cost_dollars_total
            .with_label_values(&["sk-abcde...", "gpt-4o"])
            .inc_by(0.0015);
        m.request_tokens
            .with_label_values(&["input"])
            .observe(100.0);
        m.rate_limit_rejections_total
            .with_label_values(&["tpm"])
            .inc();
        m.budget_rejections_total.inc();
        m.provider_errors_total.with_label_values(&["openai"]).inc();
        m.provider_cooldowns_total
            .with_label_values(&["openai", "timeout"])
            .inc();
        m.provider_selected_total
            .with_label_values(&["anthropic"])
            .inc();
        m.prompt_cache_requests_total
            .with_label_values(&["openai", "hit"])
            .inc();
        m.prompt_cache_read_tokens_total
            .with_label_values(&["openai"])
            .inc_by(128);
        m.prompt_cache_write_tokens_total
            .with_label_values(&["anthropic"])
            .inc_by(64);
        m.provider_ewma_latency_ms
            .with_label_values(&["openai"])
            .set(12.0);
        m.provider_ewma_error_rate
            .with_label_values(&["openai"])
            .set(0.1);
        m.provider_ewma_timeout_rate
            .with_label_values(&["openai"])
            .set(0.05);
        m.provider_ewma_rate_limit_rate
            .with_label_values(&["openai"])
            .set(0.02);
        m.provider_active_requests
            .with_label_values(&["openai"])
            .set(2);
        m.provider_cooldown_active
            .with_label_values(&["openai"])
            .set(1);
        m.active_virtual_keys.set(5);
        m.streaming_requests_total.inc();
        m.semantic_requests_total
            .with_label_values(&["evaluated"])
            .inc();
        m.semantic_findings_total.inc();
        m.semantic_degraded_total.inc();
        m.semantic_service_latency_ms
            .with_label_values(&["ok"])
            .observe(12.0);

        let families = registry.gather();
        let names: Vec<&str> = families.iter().map(|f| f.name()).collect();

        assert!(names.contains(&"llm_tokens_total"));
        assert!(names.contains(&"llm_cost_dollars_total"));
        assert!(names.contains(&"llm_request_tokens"));
        assert!(names.contains(&"llm_rate_limit_rejections_total"));
        assert!(names.contains(&"llm_budget_rejections_total"));
        assert!(names.contains(&"llm_provider_errors_total"));
        assert!(names.contains(&"llm_provider_cooldowns_total"));
        assert!(names.contains(&"llm_provider_selected_total"));
        assert!(names.contains(&"llm_provider_ewma_latency_ms"));
        assert!(names.contains(&"llm_provider_ewma_error_rate"));
        assert!(names.contains(&"llm_provider_ewma_timeout_rate"));
        assert!(names.contains(&"llm_provider_ewma_rate_limit_rate"));
        assert!(names.contains(&"llm_provider_active_requests"));
        assert!(names.contains(&"llm_provider_cooldown_active"));
        assert!(names.contains(&"llm_active_virtual_keys"));
        assert!(names.contains(&"llm_streaming_requests_total"));
        assert!(names.contains(&"llm_semantic_requests_total"));
        assert!(names.contains(&"llm_semantic_findings_total"));
        assert!(names.contains(&"llm_semantic_degraded_total"));
        assert!(names.contains(&"llm_semantic_service_latency_ms"));
    }

    #[test]
    fn test_mask_key() {
        assert_eq!(mask_key("sk-abcdefghijk"), "sk-abcde...");
        assert_eq!(mask_key("short"), "short");
        assert_eq!(mask_key("12345678"), "12345678");
        assert_eq!(mask_key("123456789"), "12345678...");
    }
}
