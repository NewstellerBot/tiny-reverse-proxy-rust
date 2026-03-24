use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts,
    Registry, TextEncoder,
};
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

#[derive(Clone)]
pub struct Metrics {
    pub registry: Arc<Registry>,
    pub requests_total: IntCounterVec,
    pub request_duration_seconds: HistogramVec,
    pub active_connections: IntGauge,
    pub upstream_healthy: IntGaugeVec,
    pub retry_attempts_total: IntCounterVec,
    pub retry_exhaustions_total: IntCounterVec,
    pub admission_rejections_total: IntCounter,
    pub brownout_active: IntGauge,
    pub brownout_activations_total: IntCounter,
}

impl Metrics {
    pub fn new() -> Self {
        Self::new_with_registry(Registry::new())
    }

    pub fn new_with_registry(registry: Registry) -> Self {
        let requests_total = IntCounterVec::new(
            Opts::new("proxy_requests_total", "Total number of proxied requests"),
            &["status", "method"],
        )
        .expect("failed to create proxy_requests_total counter");
        registry
            .register(Box::new(requests_total.clone()))
            .expect("failed to register proxy_requests_total");

        let duration_buckets = vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ];
        let request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "proxy_request_duration_seconds",
                "Request duration in seconds",
            )
            .buckets(duration_buckets),
            &["method"],
        )
        .expect("failed to create proxy_request_duration_seconds histogram");
        registry
            .register(Box::new(request_duration_seconds.clone()))
            .expect("failed to register proxy_request_duration_seconds");

        let active_connections = IntGauge::new(
            "proxy_active_connections",
            "Number of currently active connections",
        )
        .expect("failed to create proxy_active_connections gauge");
        registry
            .register(Box::new(active_connections.clone()))
            .expect("failed to register proxy_active_connections");

        let upstream_healthy = IntGaugeVec::new(
            Opts::new(
                "proxy_upstream_healthy",
                "Whether an upstream is healthy (1) or not (0)",
            ),
            &["upstream"],
        )
        .expect("failed to create proxy_upstream_healthy gauge");
        registry
            .register(Box::new(upstream_healthy.clone()))
            .expect("failed to register proxy_upstream_healthy");

        let retry_attempts_total = IntCounterVec::new(
            Opts::new(
                "proxy_retry_attempts_total",
                "Total number of retry or provider failover attempts",
            ),
            &["mode", "reason"],
        )
        .expect("failed to create proxy_retry_attempts_total counter");
        registry
            .register(Box::new(retry_attempts_total.clone()))
            .expect("failed to register proxy_retry_attempts_total");

        let retry_exhaustions_total = IntCounterVec::new(
            Opts::new(
                "proxy_retry_exhaustions_total",
                "Total number of requests that exhausted retry or failover opportunities",
            ),
            &["mode", "reason"],
        )
        .expect("failed to create proxy_retry_exhaustions_total counter");
        registry
            .register(Box::new(retry_exhaustions_total.clone()))
            .expect("failed to register proxy_retry_exhaustions_total");

        let admission_rejections_total = IntCounter::new(
            "proxy_admission_rejections_total",
            "Total number of requests rejected by hard admission control",
        )
        .expect("failed to create proxy_admission_rejections_total counter");
        registry
            .register(Box::new(admission_rejections_total.clone()))
            .expect("failed to register proxy_admission_rejections_total");

        let brownout_active = IntGauge::new(
            "proxy_brownout_active",
            "Whether local brownout mode is currently active (1) or not (0)",
        )
        .expect("failed to create proxy_brownout_active gauge");
        registry
            .register(Box::new(brownout_active.clone()))
            .expect("failed to register proxy_brownout_active");

        let brownout_activations_total = IntCounter::new(
            "proxy_brownout_activations_total",
            "Total number of times local brownout mode was activated",
        )
        .expect("failed to create proxy_brownout_activations_total counter");
        registry
            .register(Box::new(brownout_activations_total.clone()))
            .expect("failed to register proxy_brownout_activations_total");

        Self {
            registry: Arc::new(registry),
            requests_total,
            request_duration_seconds,
            active_connections,
            upstream_healthy,
            retry_attempts_total,
            retry_exhaustions_total,
            admission_rejections_total,
            brownout_active,
            brownout_activations_total,
        }
    }

    pub fn encode(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .expect("failed to encode metrics");
        String::from_utf8(buffer).expect("metrics output is not valid UTF-8")
    }

    pub fn record_retry_attempt(&self, mode: &str, reason: &str) {
        self.retry_attempts_total
            .with_label_values(&[mode, reason])
            .inc();
    }

    pub fn record_retry_exhaustion(&self, mode: &str, reason: &str) {
        self.retry_exhaustions_total
            .with_label_values(&[mode, reason])
            .inc();
    }

    pub fn record_admission_rejection(&self) {
        self.admission_rejections_total.inc();
    }

    pub fn set_brownout_active(&self, active: bool) {
        self.brownout_active.set(if active { 1 } else { 0 });
    }

    pub fn record_brownout_activation(&self) {
        self.brownout_activations_total.inc();
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Start a metrics HTTP server on the given port.
/// This runs a simple TCP listener serving the /metrics endpoint.
pub async fn start_metrics_server(port: u16, metrics: Metrics) {
    let listener = bind_metrics_listener(port)
        .await
        .expect("failed to bind metrics server");
    serve_metrics_listener(listener, metrics).await;
}

pub async fn bind_metrics_listener(port: u16) -> std::io::Result<TcpListener> {
    TcpListener::bind(("0.0.0.0", port)).await
}

pub async fn serve_metrics_listener(listener: TcpListener, metrics: Metrics) {
    tracing::info!(
        port = listener.local_addr().ok().map(|addr| addr.port()),
        "metrics server listening"
    );
    loop {
        let (stream, _addr) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                tracing::error!(error = %e, "failed to accept metrics connection");
                continue;
            }
        };

        let metrics = metrics.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let service = service_fn(move |req: Request<Incoming>| {
                let metrics = metrics.clone();
                async move { handle_metrics_request(req, metrics) }
            });

            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                tracing::error!(error = %e, "metrics connection error");
            }
        });
    }
}

fn handle_metrics_request(
    req: Request<Incoming>,
    metrics: Metrics,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    if req.method() == hyper::Method::GET && req.uri().path() == "/metrics" {
        let body = metrics.encode();
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            .body(Full::new(Bytes::from(body)))
            .unwrap())
    } else {
        Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from("Not Found")))
            .unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_encode_contains_expected_names() {
        let metrics = Metrics::new();

        // Increment some counters
        metrics
            .requests_total
            .with_label_values(&["200", "GET"])
            .inc();
        metrics
            .requests_total
            .with_label_values(&["500", "POST"])
            .inc_by(3);

        // Observe a duration
        metrics
            .request_duration_seconds
            .with_label_values(&["GET"])
            .observe(0.042);

        // Set gauges
        metrics.active_connections.set(5);
        metrics
            .upstream_healthy
            .with_label_values(&["backend-1"])
            .set(1);
        metrics.record_retry_attempt("core", "5xx");
        metrics.record_retry_exhaustion("provider", "timeout");
        metrics.record_admission_rejection();
        metrics.record_brownout_activation();
        metrics.set_brownout_active(true);

        let output = metrics.encode();

        assert!(
            output.contains("proxy_requests_total"),
            "output should contain proxy_requests_total"
        );
        assert!(
            output.contains("proxy_request_duration_seconds"),
            "output should contain proxy_request_duration_seconds"
        );
        assert!(
            output.contains("proxy_active_connections"),
            "output should contain proxy_active_connections"
        );
        assert!(
            output.contains("proxy_upstream_healthy"),
            "output should contain proxy_upstream_healthy"
        );
        assert!(
            output.contains("proxy_retry_attempts_total"),
            "output should contain proxy_retry_attempts_total"
        );
        assert!(
            output.contains("proxy_retry_exhaustions_total"),
            "output should contain proxy_retry_exhaustions_total"
        );
        assert!(
            output.contains("proxy_admission_rejections_total"),
            "output should contain proxy_admission_rejections_total"
        );
        assert!(
            output.contains("proxy_brownout_active"),
            "output should contain proxy_brownout_active"
        );
        assert!(
            output.contains("proxy_brownout_activations_total"),
            "output should contain proxy_brownout_activations_total"
        );

        // Verify specific label values appear
        assert!(
            output.contains(r#"status="200"#),
            "output should contain status=200 label"
        );
        assert!(
            output.contains(r#"method="GET"#),
            "output should contain method=GET label"
        );
        assert!(
            output.contains(r#"upstream="backend-1"#),
            "output should contain upstream=backend-1 label"
        );

        // Verify counter values
        assert!(
            output.contains("proxy_requests_total{method=\"GET\",status=\"200\"} 1"),
            "GET 200 counter should be 1"
        );
        assert!(
            output.contains("proxy_requests_total{method=\"POST\",status=\"500\"} 3"),
            "POST 500 counter should be 3"
        );

        // Verify gauge value
        assert!(
            output.contains("proxy_active_connections 5"),
            "active_connections gauge should be 5"
        );
    }
}
