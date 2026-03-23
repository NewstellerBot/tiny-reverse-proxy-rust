use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tempfile::NamedTempFile;
use tokio::net::TcpListener;

use proxy_core::cache::ResponseCache;
use proxy_core::circuit_breaker::CircuitBreaker;
use proxy_core::config::{LbStrategy, RouteConfig};
use proxy_core::handlers::proxy::{build_client, ProxyService};
use proxy_core::metrics::Metrics;
use proxy_core::plugin::PluginChain;
use proxy_core::rate_limit::RateLimiter;
use proxy_core::router::PathResolution;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .unwrap()
        .to_path_buf()
}

fn dotenv_contents() -> Option<&'static str> {
    static DOTENV: OnceLock<Option<String>> = OnceLock::new();
    DOTENV
        .get_or_init(|| std::fs::read_to_string(repo_root().join(".env")).ok())
        .as_deref()
}

fn parse_dotenv_var(dotenv: &str, name: &str) -> Option<String> {
    dotenv.lines().find_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return None;
        }

        let (key, value) = trimmed.split_once('=')?;
        if key.trim() != name {
            return None;
        }

        Some(
            value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string(),
        )
    })
}

pub fn repo_root_dotenv_env(name: &str) -> Option<String> {
    dotenv_contents().and_then(|dotenv| parse_dotenv_var(dotenv, name))
}

pub fn required_test_env(name: &str, reason: &str) -> String {
    std::env::var(name)
        .ok()
        .or_else(|| repo_root_dotenv_env(name))
        .unwrap_or_else(|| panic!("{name} must be set to {reason}"))
}

pub fn optional_test_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .or_else(|| repo_root_dotenv_env(name))
    })
}

fn optional_duration_env(names: &[&str], default_secs: u64) -> Duration {
    optional_test_env(names)
        .and_then(|raw| raw.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(default_secs))
}

pub fn openai_api_key(reason: &str) -> String {
    required_test_env("OPENAI_API_KEY", reason)
}

pub fn openai_organization() -> Option<String> {
    optional_test_env(&["OPENAI_ORGANIZATION", "OPENAI_ORG_ID"])
}

pub fn openai_project() -> Option<String> {
    optional_test_env(&["OPENAI_PROJECT", "OPENAI_PROJECT_ID"])
}

pub fn openai_realtime_model() -> String {
    optional_test_env(&["OPENAI_REALTIME_MODEL"]).unwrap_or_else(|| "gpt-realtime".to_string())
}

pub fn openai_realtime_timeout() -> Duration {
    optional_duration_env(&["OPENAI_REALTIME_TIMEOUT_SECS"], 20)
}

pub fn openai_responses_model() -> String {
    optional_test_env(&["OPENAI_RESPONSES_MODEL"]).unwrap_or_else(|| "gpt-4.1-mini".to_string())
}

pub fn openai_responses_timeout() -> Duration {
    optional_duration_env(&["OPENAI_RESPONSES_TIMEOUT_SECS"], 30)
}

pub fn create_temp_config(content: &str) -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(content.as_bytes()).unwrap();
    temp_file
}

/// Routes all paths to the configured servers.
pub struct CatchAllRouter {
    pub route_config: RouteConfig,
}

impl PathResolution for CatchAllRouter {
    fn get_servers(&self, _path: &str) -> Option<&Vec<String>> {
        Some(&self.route_config.servers)
    }

    fn get_route_config(&self, _path: &str) -> Option<&RouteConfig> {
        Some(&self.route_config)
    }
}

/// Convenience to create a catch-all router for given servers.
pub fn catch_all_router(servers: Vec<String>) -> Arc<CatchAllRouter> {
    Arc::new(CatchAllRouter {
        route_config: RouteConfig {
            servers,
            lb: LbStrategy::RoundRobin,
            weights: None,
        },
    })
}

/// Optional features for the proxy under test.
pub struct TestProxyConfig {
    pub rate_limiter: Option<RateLimiter>,
    pub metrics: Option<Metrics>,
    pub circuit_breaker: Option<CircuitBreaker>,
    pub cache: Option<ResponseCache>,
    pub compression_enabled: bool,
    pub max_request_body_bytes: u64,
    pub plugins: Option<Arc<PluginChain>>,
}

impl Default for TestProxyConfig {
    fn default() -> Self {
        Self {
            rate_limiter: None,
            metrics: None,
            circuit_breaker: None,
            cache: None,
            compression_enabled: true,
            max_request_body_bytes: 10 * 1024 * 1024,
            plugins: None,
        }
    }
}

/// Start a proxy with a custom config. Returns the bound address.
pub async fn start_proxy_with_config(
    router: Arc<CatchAllRouter>,
    config: TestProxyConfig,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();

    let client = build_client();
    let counter = Arc::new(AtomicUsize::new(0));

    tokio::spawn(async move {
        loop {
            let (stream, peer_addr) = match listener.accept().await {
                Ok(conn) => conn,
                Err(_) => break,
            };

            let mut svc =
                ProxyService::new(Arc::clone(&router), client.clone(), Arc::clone(&counter))
                    .with_peer_addr(peer_addr)
                    .with_compression(config.compression_enabled)
                    .with_max_request_body(config.max_request_body_bytes);

            if let Some(ref rl) = config.rate_limiter {
                svc = svc.with_rate_limiter(rl.clone());
            }
            if let Some(ref m) = config.metrics {
                svc = svc.with_metrics(m.clone());
            }
            if let Some(ref cb) = config.circuit_breaker {
                svc = svc.with_circuit_breaker(cb.clone());
            }
            if let Some(ref c) = config.cache {
                svc = svc.with_cache(c.clone());
            }
            if let Some(ref p) = config.plugins {
                svc = svc.with_plugins(Arc::clone(p));
            }

            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .keep_alive(true)
                    .serve_connection(
                        io,
                        service_fn(move |req: Request<Incoming>| {
                            let svc = svc.clone();
                            async move { svc.handle(req).await }
                        }),
                    )
                    .await;
            });
        }
    });

    addr
}

/// Start a proxy with default config. Returns the bound address.
pub async fn start_proxy(router: Arc<CatchAllRouter>) -> String {
    start_proxy_with_config(router, TestProxyConfig::default()).await
}

/// Start a mock upstream that delegates to a handler closure.
pub async fn start_upstream<F>(handler: F) -> String
where
    F: Fn(Request<Incoming>) -> Response<Full<Bytes>> + Send + Sync + Clone + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();

    tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(conn) => conn,
                Err(_) => break,
            };

            let handler = handler.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .keep_alive(true)
                    .serve_connection(
                        io,
                        service_fn(move |req: Request<Incoming>| {
                            let resp = handler(req);
                            async move { Ok::<_, hyper::Error>(resp) }
                        }),
                    )
                    .await;
            });
        }
    });

    addr
}

/// Start a mock upstream that can inspect and consume request bodies asynchronously.
pub async fn start_upstream_async<F, Fut>(handler: F) -> String
where
    F: Fn(Request<Incoming>) -> Fut + Send + Sync + Clone + 'static,
    Fut: std::future::Future<Output = Response<Full<Bytes>>> + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();

    tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(conn) => conn,
                Err(_) => break,
            };

            let handler = handler.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http1::Builder::new()
                    .keep_alive(true)
                    .serve_connection(
                        io,
                        service_fn(move |req: Request<Incoming>| {
                            let handler = handler.clone();
                            async move { Ok::<_, hyper::Error>(handler(req).await) }
                        }),
                    )
                    .await;
            });
        }
    });

    addr
}

/// Start an upstream that always returns 200 "ok".
pub async fn ok_upstream() -> String {
    start_upstream(|_req: Request<Incoming>| {
        Response::builder()
            .status(StatusCode::OK)
            .body(Full::new(Bytes::from("ok")))
            .unwrap()
    })
    .await
}

/// Send a request through the proxy and return the response.
pub async fn send_request(proxy_addr: &str, req: Request<Full<Bytes>>) -> Response<Incoming> {
    let client = build_client();
    let (parts, body) = req.into_parts();

    let uri: hyper::Uri = format!(
        "http://{}{}",
        proxy_addr,
        parts
            .uri
            .path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or("/")
    )
    .parse()
    .unwrap();

    let mut builder = Request::builder().method(parts.method).uri(uri);
    if let Some(headers) = builder.headers_mut() {
        *headers = parts.headers;
    }

    let boxed_body: BoxBody<Bytes, hyper::Error> = body.map_err(|never| match never {}).boxed();
    let req = builder.body(boxed_body).unwrap();
    client.request(req).await.unwrap()
}

/// Send a simple GET request, returning `(status, body_string)`.
pub async fn get(proxy_addr: &str, path: &str) -> (StatusCode, String) {
    let client = build_client();
    let uri = format!("http://{}{}", proxy_addr, path);
    let resp = client
        .request(
            Request::builder()
                .uri(&uri)
                .body(
                    Full::new(Bytes::new())
                        .map_err(|never| match never {})
                        .boxed(),
                )
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let body = resp.collect().await.unwrap().to_bytes();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

#[cfg(test)]
mod tests {
    use super::parse_dotenv_var;

    #[test]
    fn parse_dotenv_var_ignores_comments_and_quotes() {
        let dotenv = r#"
        # ignored
        OPENAI_API_KEY="sk-test"
        OPENAI_PROJECT='proj-123'
        "#;

        assert_eq!(
            parse_dotenv_var(dotenv, "OPENAI_API_KEY").as_deref(),
            Some("sk-test")
        );
        assert_eq!(
            parse_dotenv_var(dotenv, "OPENAI_PROJECT").as_deref(),
            Some("proj-123")
        );
    }

    #[test]
    fn parse_dotenv_var_returns_none_for_missing_names() {
        let dotenv = "OPENAI_API_KEY=sk-test";
        assert!(parse_dotenv_var(dotenv, "OPENAI_ORG_ID").is_none());
    }
}
