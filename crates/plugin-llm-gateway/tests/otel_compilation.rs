/// Compilation tests for OpenTelemetry integration.
///
/// These tests verify that the OTEL code paths compile correctly
/// regardless of whether the `opentelemetry` feature is enabled.
/// When the feature is off, the `#[cfg(feature = "opentelemetry")]`
/// blocks are simply absent — this test file ensures the non-OTEL
/// paths still work.

#[cfg(feature = "opentelemetry")]
mod with_otel {
    use proxy_core::otel::OtelSpan;

    #[test]
    fn otel_span_is_clone() {
        let span = tracing::info_span!(
            "test_span",
            llm.model = tracing::field::Empty,
            llm.input_tokens = tracing::field::Empty,
        );
        let otel = OtelSpan(span.clone());
        let _cloned = otel.clone();
    }

    #[test]
    fn otel_span_can_record_fields() {
        let span = tracing::info_span!(
            "test_record",
            llm.model = tracing::field::Empty,
            llm.provider = tracing::field::Empty,
            llm.provider_attempts = tracing::field::Empty,
            llm.routing_rule = tracing::field::Empty,
            llm.upstream_timeout_ms = tracing::field::Empty,
            llm.prompt_cache_protocol = tracing::field::Empty,
            llm.prompt_cache_status = tracing::field::Empty,
            llm.prompt_cache_read_tokens = tracing::field::Empty,
            llm.prompt_cache_write_tokens = tracing::field::Empty,
            llm.input_tokens = tracing::field::Empty,
            llm.output_tokens = tracing::field::Empty,
            llm.cost_usd = tracing::field::Empty,
            llm.rate_limited = tracing::field::Empty,
        );
        let otel = OtelSpan(span);

        otel.0.record("llm.model", "gpt-4o");
        otel.0.record("llm.provider", "openai");
        otel.0.record("llm.provider_attempts", "openai,anthropic");
        otel.0.record("llm.routing_rule", "prefer-anthropic");
        otel.0.record("llm.upstream_timeout_ms", 1500u64);
        otel.0.record("llm.prompt_cache_protocol", "openai");
        otel.0.record("llm.prompt_cache_status", "hit");
        otel.0.record("llm.prompt_cache_read_tokens", 512u64);
        otel.0.record("llm.prompt_cache_write_tokens", 128u64);
        otel.0.record("llm.input_tokens", 100u64);
        otel.0.record("llm.output_tokens", 50u64);
        otel.0.record("llm.cost_usd", "0.001500");
        otel.0.record("llm.rate_limited", true);
    }

    #[test]
    fn otel_span_storable_in_extensions() {
        let span = tracing::info_span!("test_ext");
        let otel = OtelSpan(span);

        let mut ext = hyper::http::Extensions::new();
        ext.insert(otel.clone());
        assert!(ext.get::<OtelSpan>().is_some());
    }
}

/// Tests that compile with or without the opentelemetry feature.
#[test]
fn plugins_work_without_otel() {
    // This test verifies that all plugin constructors work without OTEL.
    // If the code has any unconditional OTEL dependency this would fail to compile.
    let _tracker = plugin_llm_gateway::cost_tracker::CostTracker::new(0.0, 0.01, 0.03, 3600);
    let _limiter = plugin_llm_gateway::rate_limiter::TokenRateLimiter::new(60000.0, 10000.0);
    let _failover = plugin_llm_gateway::provider_failover::ProviderFailover::new(vec![], 30);
    let _vk = plugin_llm_gateway::virtual_keys::VirtualKeys::new(&[], &[], None);
}
