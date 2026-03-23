use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::Request;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;

use crate::config::HealthCheckConfig;
use crate::metrics::Metrics;

/// HTTPS-capable health check client (same connector as the proxy client).
type HealthClient = Client<
    hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>,
    BoxBody<Bytes, hyper::Error>,
>;

/// Tracks which upstream servers are currently healthy.
#[derive(Clone)]
pub struct HealthState {
    tracked: Arc<RwLock<HashSet<String>>>,
    healthy: Arc<RwLock<HashSet<String>>>,
    metrics: Option<Metrics>,
}

/// Reloadable list of upstreams that should be health-checked.
#[derive(Debug, Clone)]
pub struct HealthCheckTargets {
    upstreams: Arc<RwLock<Vec<String>>>,
}

impl HealthState {
    /// Create a new HealthState with all provided upstreams initially marked healthy.
    pub fn new(upstreams: &[String]) -> Self {
        let set: HashSet<String> = upstreams.iter().cloned().collect();
        Self {
            tracked: Arc::new(RwLock::new(set.clone())),
            healthy: Arc::new(RwLock::new(set)),
            metrics: None,
        }
    }

    /// Attach metrics and seed the current healthy gauges.
    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        for upstream in self.healthy.read().unwrap().iter() {
            metrics
                .upstream_healthy
                .with_label_values(&[upstream])
                .set(1);
        }
        self.metrics = Some(metrics);
        self
    }

    /// Returns true if the given upstream is currently healthy.
    pub fn is_healthy(&self, upstream: &str) -> bool {
        self.healthy.read().unwrap().contains(upstream)
    }

    /// Mark an upstream as healthy.
    pub fn mark_healthy(&self, upstream: &str) {
        self.tracked.write().unwrap().insert(upstream.to_string());
        self.healthy.write().unwrap().insert(upstream.to_string());
        if let Some(ref metrics) = self.metrics {
            metrics
                .upstream_healthy
                .with_label_values(&[upstream])
                .set(1);
        }
    }

    /// Mark an upstream as unhealthy.
    pub fn mark_unhealthy(&self, upstream: &str) {
        self.healthy.write().unwrap().remove(upstream);
        if let Some(ref metrics) = self.metrics {
            metrics
                .upstream_healthy
                .with_label_values(&[upstream])
                .set(0);
        }
    }

    /// Synchronize the tracked upstream set with the current configuration.
    ///
    /// Newly added upstreams start healthy so a route reload does not
    /// immediately strand them behind a stale health set.
    pub fn sync_upstreams(&self, upstreams: &[String]) {
        let desired: HashSet<String> = upstreams.iter().cloned().collect();
        let mut tracked = self.tracked.write().unwrap();
        let mut healthy = self.healthy.write().unwrap();
        let removed: Vec<String> = tracked
            .iter()
            .filter(|upstream| !desired.contains(*upstream))
            .cloned()
            .collect();
        let added: Vec<String> = desired
            .iter()
            .filter(|upstream| !tracked.contains(*upstream))
            .cloned()
            .collect();
        *tracked = desired.clone();
        healthy.retain(|upstream| desired.contains(upstream));
        for upstream in &added {
            healthy.insert(upstream.clone());
        }

        if let Some(ref metrics) = self.metrics {
            for upstream in removed {
                let _ = metrics.upstream_healthy.remove_label_values(&[&upstream]);
            }
            for upstream in added {
                metrics
                    .upstream_healthy
                    .with_label_values(&[&upstream])
                    .set(1);
            }
        }
    }
}

impl HealthCheckTargets {
    pub fn new(upstreams: Vec<String>) -> Self {
        Self {
            upstreams: Arc::new(RwLock::new(upstreams)),
        }
    }

    pub fn snapshot(&self) -> Vec<String> {
        self.upstreams.read().unwrap().clone()
    }

    pub fn replace(&self, upstreams: Vec<String>) {
        *self.upstreams.write().unwrap() = upstreams;
    }
}

fn build_health_client() -> HealthClient {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    let https_connector = hyper_rustls::HttpsConnectorBuilder::new()
        .with_tls_config(tls_config)
        .https_or_http()
        .enable_http1()
        .enable_http2()
        .build();

    Client::builder(TokioExecutor::new()).build(https_connector)
}

/// Determine the correct URI scheme for an upstream address.
fn upstream_health_uri(upstream: &str, path: &str) -> String {
    if upstream.contains("://") {
        format!("{}{}", upstream, path)
    } else {
        format!("http://{}{}", upstream, path)
    }
}

/// Probe a single upstream. Returns true if healthy (2xx response).
async fn check_upstream(
    client: &HealthClient,
    upstream: &str,
    path: &str,
    timeout: Duration,
) -> bool {
    let uri = upstream_health_uri(upstream, path);
    let body: BoxBody<Bytes, hyper::Error> = Full::new(Bytes::new())
        .map_err(|never| match never {})
        .boxed();
    let req = match Request::builder().uri(&uri).body(body) {
        Ok(r) => r,
        Err(_) => return false,
    };

    match tokio::time::timeout(timeout, client.request(req)).await {
        Ok(Ok(resp)) => resp.status().is_success(),
        _ => false,
    }
}

/// Spawn a background task that periodically health-checks all upstreams.
pub fn spawn_health_checker(
    upstreams: Vec<String>,
    state: HealthState,
    config: &HealthCheckConfig,
) -> tokio::task::JoinHandle<()> {
    spawn_health_checker_with_targets(HealthCheckTargets::new(upstreams), state, config)
}

/// Spawn a background task that periodically health-checks the current target set.
pub fn spawn_health_checker_with_targets(
    targets: HealthCheckTargets,
    state: HealthState,
    config: &HealthCheckConfig,
) -> tokio::task::JoinHandle<()> {
    let interval = Duration::from_secs(config.interval_secs);
    let timeout = Duration::from_secs(config.timeout_secs);
    let path = config.path.clone();

    tokio::spawn(async move {
        let client = build_health_client();
        loop {
            let upstreams = targets.snapshot();
            state.sync_upstreams(&upstreams);
            for upstream in &upstreams {
                let healthy = check_upstream(&client, upstream, &path, timeout).await;
                if healthy {
                    if !state.is_healthy(upstream) {
                        tracing::info!(upstream = %upstream, "upstream recovered");
                    }
                    state.mark_healthy(upstream);
                } else {
                    if state.is_healthy(upstream) {
                        tracing::warn!(upstream = %upstream, "upstream failed health check");
                    }
                    state.mark_unhealthy(upstream);
                }
            }
            tokio::time::sleep(interval).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metrics;

    #[test]
    fn initially_all_healthy() {
        let upstreams = vec!["a:80".to_string(), "b:80".to_string()];
        let state = HealthState::new(&upstreams);
        assert!(state.is_healthy("a:80"));
        assert!(state.is_healthy("b:80"));
        assert!(!state.is_healthy("c:80"));
    }

    #[test]
    fn mark_unhealthy_removes() {
        let upstreams = vec!["a:80".to_string(), "b:80".to_string()];
        let state = HealthState::new(&upstreams);
        state.mark_unhealthy("a:80");
        assert!(!state.is_healthy("a:80"));
        assert!(state.is_healthy("b:80"));
    }

    #[test]
    fn mark_healthy_restores() {
        let upstreams = vec!["a:80".to_string()];
        let state = HealthState::new(&upstreams);
        state.mark_unhealthy("a:80");
        assert!(!state.is_healthy("a:80"));
        state.mark_healthy("a:80");
        assert!(state.is_healthy("a:80"));
    }

    #[test]
    fn clone_shares_state() {
        let upstreams = vec!["a:80".to_string()];
        let state = HealthState::new(&upstreams);
        let cloned = state.clone();
        state.mark_unhealthy("a:80");
        assert!(!cloned.is_healthy("a:80"));
    }

    #[test]
    fn sync_upstreams_adds_new_and_removes_old_entries() {
        let state = HealthState::new(&["a:80".to_string(), "b:80".to_string()]);
        state.mark_unhealthy("a:80");

        state.sync_upstreams(&["b:80".to_string(), "c:80".to_string()]);

        assert!(!state.is_healthy("a:80"));
        assert!(state.is_healthy("b:80"));
        assert!(state.is_healthy("c:80"));
    }

    #[test]
    fn metrics_follow_health_transitions_and_target_changes() {
        let metrics = Metrics::new();
        let state = HealthState::new(&["a:80".to_string()]).with_metrics(metrics.clone());

        let encoded = metrics.encode();
        assert!(encoded.contains(r#"upstream="a:80""#));
        assert!(encoded.contains("proxy_upstream_healthy{upstream=\"a:80\"} 1"));

        state.mark_unhealthy("a:80");
        let encoded = metrics.encode();
        assert!(encoded.contains("proxy_upstream_healthy{upstream=\"a:80\"} 0"));

        state.sync_upstreams(&["b:80".to_string()]);
        let encoded = metrics.encode();
        assert!(!encoded.contains(r#"upstream="a:80""#));
        assert!(encoded.contains("proxy_upstream_healthy{upstream=\"b:80\"} 1"));
    }

    #[test]
    fn sync_upstreams_preserves_existing_unhealthy_status() {
        let state = HealthState::new(&["a:80".to_string(), "b:80".to_string()]);
        state.mark_unhealthy("a:80");

        state.sync_upstreams(&["a:80".to_string(), "b:80".to_string(), "c:80".to_string()]);

        assert!(!state.is_healthy("a:80"));
        assert!(state.is_healthy("b:80"));
        assert!(state.is_healthy("c:80"));
    }

    #[test]
    fn health_check_targets_replace_updates_snapshot() {
        let targets = HealthCheckTargets::new(vec!["a:80".to_string()]);
        assert_eq!(targets.snapshot(), vec!["a:80".to_string()]);

        targets.replace(vec!["b:80".to_string(), "c:80".to_string()]);

        assert_eq!(
            targets.snapshot(),
            vec!["b:80".to_string(), "c:80".to_string()]
        );
    }

    #[tokio::test]
    async fn check_upstream_unreachable_returns_false() {
        let client = build_health_client();
        let result =
            check_upstream(&client, "127.0.0.1:1", "/health", Duration::from_secs(1)).await;
        assert!(!result);
    }

    #[tokio::test]
    async fn check_upstream_healthy_returns_true() {
        use hyper::body::Incoming;
        use hyper::service::service_fn;
        use hyper::{Response, StatusCode};
        use hyper_util::rt::TokioIo;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let io = TokioIo::new(stream);
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(
                    io,
                    service_fn(|_req: Request<Incoming>| async {
                        Ok::<_, hyper::Error>(
                            Response::builder()
                                .status(StatusCode::OK)
                                .body(Full::new(Bytes::new()))
                                .unwrap(),
                        )
                    }),
                )
                .await;
        });

        let client = build_health_client();
        let result = check_upstream(&client, &addr, "/health", Duration::from_secs(2)).await;
        assert!(result);
    }

    #[tokio::test]
    async fn check_upstream_500_returns_false() {
        use hyper::body::Incoming;
        use hyper::service::service_fn;
        use hyper::{Response, StatusCode};
        use hyper_util::rt::TokioIo;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let io = TokioIo::new(stream);
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(
                    io,
                    service_fn(|_req: Request<Incoming>| async {
                        Ok::<_, hyper::Error>(
                            Response::builder()
                                .status(StatusCode::INTERNAL_SERVER_ERROR)
                                .body(Full::new(Bytes::new()))
                                .unwrap(),
                        )
                    }),
                )
                .await;
        });

        let client = build_health_client();
        let result = check_upstream(&client, &addr, "/health", Duration::from_secs(2)).await;
        assert!(!result);
    }

    #[test]
    fn test_upstream_health_uri_with_scheme() {
        assert_eq!(
            upstream_health_uri("https://backend.example.com", "/health"),
            "https://backend.example.com/health"
        );
    }

    #[test]
    fn test_upstream_health_uri_without_scheme() {
        assert_eq!(
            upstream_health_uri("backend:8080", "/health"),
            "http://backend:8080/health"
        );
    }
}
