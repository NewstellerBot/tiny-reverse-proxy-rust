use std::time::Duration;

use bytes::Bytes;
use hyper::header::HeaderValue;
use hyper::{Method, StatusCode};

use plugin_llm_gateway::metrics::LlmMetrics;

/// Test that LLM metrics are recorded through the cost_tracker plugin.
#[tokio::test]
async fn cost_tracker_records_prometheus_metrics() {
    let registry = prometheus::Registry::new();
    let llm_metrics = LlmMetrics::new(&registry);

    // Create a cost_tracker with metrics attached.
    let config = toml::Value::Table({
        let mut t = toml::value::Map::new();
        t.insert("budget_limit".into(), toml::Value::Float(0.0));
        t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
        t
    });
    let tracker = plugin_llm_gateway::cost_tracker::create_tracker(&config)
        .unwrap()
        .with_metrics(llm_metrics.clone());

    use hyper::header::HeaderMap;
    use hyper::http::Extensions;
    use proxy_core::plugin::Plugin;
    use std::sync::Arc;

    // Simulate on_request
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer sk-test-key-123"),
    );
    let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hello world"}]}"#;
    let mut ctx = proxy_core::plugin::RequestContext {
        peer_addr: None,
        method: Method::POST,
        uri: hyper::Uri::from_static("http://localhost/v1/chat/completions"),
        version: hyper::Version::HTTP_11,
        headers,
        body: Some(Bytes::from(body.to_vec())),
        route: None,
        selected_upstream: None,
        auth: None,
        connection: Arc::new(Extensions::new()),
        extensions: Extensions::new(),
    };

    match tracker.on_request(&mut ctx).await {
        proxy_core::plugin::Action::Continue => {}
        _ => panic!("on_request should continue"),
    }

    // Simulate on_response (non-SSE)
    let mut resp_headers = HeaderMap::new();
    resp_headers.insert("content-length", HeaderValue::from_static("400"));
    resp_headers.insert("content-type", HeaderValue::from_static("application/json"));
    let mut resp_ctx = proxy_core::plugin::ResponseContext {
        status: StatusCode::OK,
        headers: resp_headers,
        upstream: "http://localhost:8080".to_string(),
        duration: Duration::from_millis(50),
    };

    match tracker.on_response(&mut ctx, &mut resp_ctx).await {
        proxy_core::plugin::Action::Continue => {}
        _ => panic!("on_response should continue"),
    }

    // Verify Prometheus metrics were recorded
    let families = registry.gather();
    let names: Vec<&str> = families.iter().map(|f| f.get_name()).collect();

    assert!(
        names.contains(&"llm_tokens_total"),
        "should have llm_tokens_total"
    );
    assert!(
        names.contains(&"llm_cost_dollars_total"),
        "should have llm_cost_dollars_total"
    );
    assert!(
        names.contains(&"llm_request_tokens"),
        "should have llm_request_tokens"
    );

    // Check that token counts are non-zero
    let tokens = llm_metrics
        .tokens_total
        .with_label_values(&["sk-test-...", "gpt-4o", "input"])
        .get();
    assert!(tokens > 0, "input tokens should be > 0, got {}", tokens);

    let output_tokens = llm_metrics
        .tokens_total
        .with_label_values(&["sk-test-...", "gpt-4o", "output"])
        .get();
    assert!(
        output_tokens > 0,
        "output tokens should be > 0, got {}",
        output_tokens
    );
}

/// Test that budget rejection increments the metric.
#[tokio::test]
async fn budget_rejection_increments_metric() {
    let registry = prometheus::Registry::new();
    let llm_metrics = LlmMetrics::new(&registry);

    let config = toml::Value::Table({
        let mut t = toml::value::Map::new();
        t.insert("budget_limit".into(), toml::Value::Float(0.001));
        t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
        t
    });
    let tracker = plugin_llm_gateway::cost_tracker::create_tracker(&config)
        .unwrap()
        .with_metrics(llm_metrics.clone());

    use hyper::header::HeaderMap;
    use hyper::http::Extensions;
    use proxy_core::plugin::Plugin;
    use std::sync::Arc;

    // First: do a request+response to accumulate cost above the tiny budget.
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer sk-budget"),
    );
    let body = Bytes::from(vec![b'x'; 4000]); // 1000 tokens
    let mut ctx = proxy_core::plugin::RequestContext {
        peer_addr: None,
        method: Method::POST,
        uri: hyper::Uri::from_static("http://localhost/v1/chat/completions"),
        version: hyper::Version::HTTP_11,
        headers: headers.clone(),
        body: Some(body),
        route: None,
        selected_upstream: None,
        auth: None,
        connection: Arc::new(Extensions::new()),
        extensions: Extensions::new(),
    };
    tracker.on_request(&mut ctx).await;

    let mut resp_headers = HeaderMap::new();
    resp_headers.insert("content-length", HeaderValue::from_static("4000"));
    resp_headers.insert("content-type", HeaderValue::from_static("application/json"));
    let mut resp_ctx = proxy_core::plugin::ResponseContext {
        status: StatusCode::OK,
        headers: resp_headers,
        upstream: "http://localhost:8080".to_string(),
        duration: Duration::from_millis(50),
    };
    tracker.on_response(&mut ctx, &mut resp_ctx).await;

    // Second request should be budget-rejected.
    let mut ctx2 = proxy_core::plugin::RequestContext {
        peer_addr: None,
        method: Method::POST,
        uri: hyper::Uri::from_static("http://localhost/v1/chat/completions"),
        version: hyper::Version::HTTP_11,
        headers,
        body: Some(Bytes::from(b"hello".to_vec())),
        route: None,
        selected_upstream: None,
        auth: None,
        connection: Arc::new(Extensions::new()),
        extensions: Extensions::new(),
    };
    match tracker.on_request(&mut ctx2).await {
        proxy_core::plugin::Action::Respond(resp) => {
            assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
        }
        _ => panic!("should be budget-rejected"),
    }

    assert_eq!(
        llm_metrics.budget_rejections_total.get(),
        1,
        "budget_rejections_total should be 1"
    );
}

/// Test that rate limiter records rejection metrics.
#[tokio::test]
async fn rate_limiter_records_rejection_metrics() {
    let registry = prometheus::Registry::new();
    let llm_metrics = LlmMetrics::new(&registry);

    let limiter = plugin_llm_gateway::rate_limiter::TokenRateLimiter::new(60.0, 10.0)
        .with_metrics(llm_metrics.clone());

    use hyper::header::HeaderMap;
    use hyper::http::Extensions;
    use proxy_core::plugin::Plugin;
    use std::sync::Arc;

    // Send a request with body that exceeds burst (10 tokens).
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer sk-rate-test"),
    );
    let mut ctx = proxy_core::plugin::RequestContext {
        peer_addr: None,
        method: Method::POST,
        uri: hyper::Uri::from_static("http://localhost/v1/chat/completions"),
        version: hyper::Version::HTTP_11,
        headers,
        body: Some(Bytes::from(vec![b'x'; 200])), // 50 tokens > 10 burst
        route: None,
        selected_upstream: None,
        auth: None,
        connection: Arc::new(Extensions::new()),
        extensions: Extensions::new(),
    };

    match limiter.on_request(&mut ctx).await {
        proxy_core::plugin::Action::Respond(resp) => {
            assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        }
        _ => panic!("should be rate limited"),
    }

    assert_eq!(
        llm_metrics
            .rate_limit_rejections_total
            .with_label_values(&["tpm"])
            .get(),
        1,
        "tpm rejection counter should be 1"
    );
}

/// Test that RPM rejection records the right metric label.
#[tokio::test]
async fn rpm_rejection_records_metric() {
    let registry = prometheus::Registry::new();
    let llm_metrics = LlmMetrics::new(&registry);

    let limiter = plugin_llm_gateway::rate_limiter::TokenRateLimiter::new(60_000.0, 10_000.0)
        .with_rpm(1.0, 1.0)
        .with_metrics(llm_metrics.clone());

    use hyper::header::HeaderMap;
    use hyper::http::Extensions;
    use proxy_core::plugin::Plugin;
    use std::sync::Arc;

    let make_ctx = || {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer sk-rpm-metric"),
        );
        proxy_core::plugin::RequestContext {
            peer_addr: None,
            method: Method::POST,
            uri: hyper::Uri::from_static("http://localhost/v1/chat/completions"),
            version: hyper::Version::HTTP_11,
            headers,
            body: Some(Bytes::from(b"tiny".to_vec())),
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        }
    };

    // First request passes
    let mut ctx1 = make_ctx();
    assert!(matches!(
        limiter.on_request(&mut ctx1).await,
        proxy_core::plugin::Action::Continue
    ));

    // Second request hits RPM
    let mut ctx2 = make_ctx();
    assert!(matches!(
        limiter.on_request(&mut ctx2).await,
        proxy_core::plugin::Action::Respond(_)
    ));

    assert_eq!(
        llm_metrics
            .rate_limit_rejections_total
            .with_label_values(&["rpm"])
            .get(),
        1,
        "rpm rejection counter should be 1"
    );
}

/// Test that provider failover records error metric.
#[tokio::test]
async fn provider_failover_records_error_metric() {
    let registry = prometheus::Registry::new();
    let llm_metrics = LlmMetrics::new(&registry);

    let failover = plugin_llm_gateway::provider_failover::ProviderFailover::new(
        vec![("openai".into(), "api.openai.com".into())],
        30,
    )
    .with_metrics(llm_metrics.clone());

    use hyper::header::HeaderMap;
    use hyper::http::Extensions;
    use proxy_core::plugin::{Plugin, ProxyError};
    use std::sync::Arc;

    let mut ctx = proxy_core::plugin::RequestContext {
        peer_addr: None,
        method: Method::POST,
        uri: hyper::Uri::from_static("http://localhost/v1/chat/completions"),
        version: hyper::Version::HTTP_11,
        headers: HeaderMap::new(),
        body: None,
        route: None,
        selected_upstream: Some("https://api.openai.com/v1".to_string()),
        auth: None,
        connection: Arc::new(Extensions::new()),
        extensions: Extensions::new(),
    };

    failover.on_error(&mut ctx, &ProxyError::Timeout).await;

    assert_eq!(
        llm_metrics
            .provider_errors_total
            .with_label_values(&["openai"])
            .get(),
        1,
        "provider_errors_total{{openai}} should be 1"
    );
}

/// Test that all LLM metrics and core metrics can coexist on the same registry.
#[test]
fn shared_registry_contains_both_core_and_llm_metrics() {
    let registry = prometheus::Registry::new();

    // Register core metrics
    let core_metrics = proxy_core::metrics::Metrics::new_with_registry(registry.clone());
    core_metrics
        .requests_total
        .with_label_values(&["200", "GET"])
        .inc();

    // Register LLM metrics
    let llm_metrics = LlmMetrics::new(&registry);
    llm_metrics
        .tokens_total
        .with_label_values(&["sk-test...", "gpt-4o", "input"])
        .inc_by(42);

    // Touch a few more metrics so they appear in gather output
    // (Vec-typed metrics only appear after a label set is initialized)
    core_metrics
        .request_duration_seconds
        .with_label_values(&["GET"])
        .observe(0.1);
    llm_metrics.budget_rejections_total.inc();
    llm_metrics.active_virtual_keys.set(3);
    llm_metrics
        .rate_limit_rejections_total
        .with_label_values(&["tpm"])
        .inc();

    let families = registry.gather();
    let names: Vec<&str> = families.iter().map(|f| f.get_name()).collect();

    // Both core and LLM metrics should be present on the same registry
    assert!(
        names.contains(&"proxy_requests_total"),
        "core proxy_requests_total missing: {:?}",
        names
    );
    assert!(
        names.contains(&"proxy_request_duration_seconds"),
        "core proxy_request_duration_seconds missing: {:?}",
        names
    );
    assert!(
        names.contains(&"proxy_active_connections"),
        "core proxy_active_connections missing: {:?}",
        names
    );
    assert!(
        names.contains(&"llm_tokens_total"),
        "llm_tokens_total missing: {:?}",
        names
    );
    assert!(
        names.contains(&"llm_budget_rejections_total"),
        "llm_budget_rejections_total missing: {:?}",
        names
    );
    assert!(
        names.contains(&"llm_active_virtual_keys"),
        "llm_active_virtual_keys missing: {:?}",
        names
    );

    // Encode to text and verify both namespaces are present
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let mut buffer = Vec::new();
    encoder.encode(&families, &mut buffer).unwrap();
    let output = String::from_utf8(buffer).unwrap();

    assert!(output.contains("proxy_requests_total"));
    assert!(output.contains("llm_tokens_total"));
}
