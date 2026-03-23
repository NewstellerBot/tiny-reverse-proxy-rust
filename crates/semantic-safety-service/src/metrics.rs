use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts, Registry, TextEncoder,
};
use tokio::net::TcpListener;

#[derive(Clone)]
pub struct SemanticSafetyServiceMetrics {
    registry: Registry,
    pub evaluate_requests_total: IntCounterVec,
    pub evaluate_latency_ms: HistogramVec,
    pub indexed_projects: IntGauge,
    pub indexed_exemplars: IntGauge,
}

impl SemanticSafetyServiceMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let evaluate_requests_total = IntCounterVec::new(
            Opts::new(
                "semantic_safety_service_evaluate_requests_total",
                "Total semantic safety evaluate RPCs by outcome",
            ),
            &["outcome"],
        )
        .expect("failed to create evaluate_requests_total");
        registry
            .register(Box::new(evaluate_requests_total.clone()))
            .expect("failed to register evaluate_requests_total");

        let evaluate_latency_ms = HistogramVec::new(
            HistogramOpts::new(
                "semantic_safety_service_evaluate_latency_ms",
                "Latency of semantic safety evaluate RPCs in milliseconds",
            )
            .buckets(vec![
                1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
            ]),
            &["outcome"],
        )
        .expect("failed to create evaluate_latency_ms");
        registry
            .register(Box::new(evaluate_latency_ms.clone()))
            .expect("failed to register evaluate_latency_ms");

        let indexed_projects = IntGauge::new(
            "semantic_safety_service_indexed_projects",
            "Number of project indexes currently loaded in semantic safety service",
        )
        .expect("failed to create indexed_projects");
        registry
            .register(Box::new(indexed_projects.clone()))
            .expect("failed to register indexed_projects");

        let indexed_exemplars = IntGauge::new(
            "semantic_safety_service_indexed_exemplars",
            "Number of exemplar embeddings currently loaded in semantic safety service",
        )
        .expect("failed to create indexed_exemplars");
        registry
            .register(Box::new(indexed_exemplars.clone()))
            .expect("failed to register indexed_exemplars");

        Self {
            registry,
            evaluate_requests_total,
            evaluate_latency_ms,
            indexed_projects,
            indexed_exemplars,
        }
    }

    pub fn render(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

impl Default for SemanticSafetyServiceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

pub async fn start_metrics_server(
    addr: SocketAddr,
    metrics: Arc<SemanticSafetyServiceMetrics>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "semantic safety metrics listening");

    loop {
        let (stream, _) = listener.accept().await?;
        let metrics = Arc::clone(&metrics);
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let service = service_fn(move |req: Request<Incoming>| {
                let metrics = Arc::clone(&metrics);
                async move { handle_metrics_request(req, metrics).await }
            });

            if let Err(error) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                tracing::warn!(error = %error, "semantic safety metrics connection error");
            }
        });
    }
}

async fn handle_metrics_request(
    req: Request<Incoming>,
    metrics: Arc<SemanticSafetyServiceMetrics>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    if req.method() != Method::GET || req.uri().path() != "/metrics" {
        return Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from_static(b"not found")))
            .expect("failed to build metrics not-found response"));
    }

    match metrics.render() {
        Ok(body) => Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/plain; version=0.0.4")
            .body(Full::new(Bytes::from(body)))
            .expect("failed to build metrics response")),
        Err(error) => Ok(Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Full::new(Bytes::from(format!(
                "failed to render metrics: {error}"
            ))))
            .expect("failed to build metrics error response")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_includes_semantic_safety_metrics() {
        let metrics = SemanticSafetyServiceMetrics::new();
        metrics
            .evaluate_requests_total
            .with_label_values(&["ready"])
            .inc();
        metrics
            .evaluate_latency_ms
            .with_label_values(&["ready"])
            .observe(12.0);
        metrics.indexed_projects.set(2);
        metrics.indexed_exemplars.set(7);

        let rendered = metrics.render().unwrap();
        assert!(rendered.contains("semantic_safety_service_evaluate_requests_total"));
        assert!(rendered.contains("outcome=\"ready\""));
        assert!(rendered.contains("semantic_safety_service_indexed_projects 2"));
        assert!(rendered.contains("semantic_safety_service_indexed_exemplars 7"));
    }
}
