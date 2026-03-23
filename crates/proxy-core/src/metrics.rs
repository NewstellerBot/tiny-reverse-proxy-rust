use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
    TextEncoder,
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

        Self {
            registry: Arc::new(registry),
            requests_total,
            request_duration_seconds,
            active_connections,
            upstream_healthy,
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
}

/// Start a metrics HTTP server on the given port.
/// This runs a simple TCP listener serving the /metrics endpoint.
pub async fn start_metrics_server(port: u16, metrics: Metrics) {
    let listener = TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("failed to bind metrics server");

    tracing::info!(port, "metrics server listening");

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
