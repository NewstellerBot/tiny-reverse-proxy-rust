use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use dashmap::DashMap;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::{HeaderName, HeaderValue, CONTENT_LENGTH, HOST};
use hyper::{Method, Request, Response, StatusCode, Version};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use hyper_util::rt::TokioIo;
use tokio::net::TcpStream;
use tokio_rustls::{rustls, TlsConnector};

use hyper::header::HeaderMap;
use hyper::upgrade::OnUpgrade;

use hyper::http::Extensions;
use tracing::Instrument;

use crate::cache::ResponseCache;
use crate::circuit_breaker::CircuitBreaker;
use crate::compression;
use crate::config::RouteConfig;
use crate::health::HealthState;
use crate::load_balancer;
use crate::metrics::Metrics;
use crate::middleware;
use crate::plugin::{Action, PluginChain, ProxyError, RequestContext, ResponseContext};
use crate::rate_limit::RateLimiter;
use crate::router::RouteResolver;

/// Headers that must be stripped per RFC 9110 §7.6.1.
const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "te",
    "transfer-encoding",
    "upgrade",
    "proxy-authorization",
    "proxy-connection",
];

/// Strip hop-by-hop headers per RFC 9110 §7.6.1.
/// Also removes any headers listed as Connection tokens.
fn strip_hop_by_hop_headers(headers: &mut HeaderMap) {
    // First, collect custom tokens from the Connection header value.
    let custom_tokens: Vec<HeaderName> = headers
        .get_all("connection")
        .iter()
        .flat_map(|val| {
            val.to_str()
                .unwrap_or("")
                .split(',')
                .filter_map(|token| {
                    let token = token.trim().to_lowercase();
                    if token.is_empty() {
                        return None;
                    }
                    HeaderName::try_from(token).ok()
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Remove static hop-by-hop headers.
    for name in HOP_BY_HOP_HEADERS {
        headers.remove(*name);
    }

    // Remove any custom Connection tokens.
    for name in custom_tokens {
        headers.remove(name);
    }
}

/// Conditionally strip hop-by-hop headers — skip for HTTP/2 where Connection is illegal.
fn strip_hop_by_hop_if_needed(headers: &mut HeaderMap, version: Version) {
    if version == Version::HTTP_2 {
        // In HTTP/2, Connection header is illegal. Just remove TE if not "trailers".
        // Preserve TE: trailers for gRPC.
        let keep_te = headers
            .get("te")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.eq_ignore_ascii_case("trailers"))
            .unwrap_or(false);
        headers.remove("connection");
        headers.remove("keep-alive");
        if !keep_te {
            headers.remove("te");
        }
        headers.remove("transfer-encoding");
        headers.remove("upgrade");
        headers.remove("proxy-authorization");
        headers.remove("proxy-connection");
    } else {
        strip_hop_by_hop_headers(headers);
    }
}

/// Append client IP to X-Forwarded-For, set X-Forwarded-Proto and X-Forwarded-Host.
fn add_forwarding_headers(
    headers: &mut HeaderMap,
    peer_addr: SocketAddr,
    original_host: Option<&str>,
    is_tls: bool,
) {
    let client_ip = peer_addr.ip().to_string();

    // X-Forwarded-For: append to existing or create new.
    let xff = match headers.get("x-forwarded-for") {
        Some(existing) => {
            let existing = existing.to_str().unwrap_or("");
            HeaderValue::from_str(&format!("{}, {}", existing, client_ip)).unwrap()
        }
        None => HeaderValue::from_str(&client_ip).unwrap(),
    };
    headers.insert("x-forwarded-for", xff);

    // X-Forwarded-Proto: "https" when TLS is active, "http" otherwise.
    headers.insert(
        "x-forwarded-proto",
        HeaderValue::from_static(if is_tls { "https" } else { "http" }),
    );

    // X-Forwarded-Host: the original Host header value.
    if let Some(host) = original_host {
        if let Ok(val) = HeaderValue::from_str(host) {
            headers.insert("x-forwarded-host", val);
        }
    }
}

/// Format a Via header value — dynamic based on HTTP version.
pub fn via_header_value(version: Version) -> HeaderValue {
    match version {
        Version::HTTP_2 => HeaderValue::from_static("2.0 tiny-reverse-proxy"),
        _ => HeaderValue::from_static("1.1 tiny-reverse-proxy"),
    }
}

/// Returns `true` for HTTP methods that are safe to retry.
fn is_idempotent(method: &Method) -> bool {
    matches!(
        *method,
        Method::GET | Method::HEAD | Method::OPTIONS | Method::PUT | Method::DELETE
    )
}

/// Returns `true` for upstream response status codes that warrant a retry.
fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::BAD_GATEWAY || status == StatusCode::SERVICE_UNAVAILABLE
}

/// Returns `true` for provider response codes that should trigger same-request failover.
fn is_provider_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn status_error(status: StatusCode, has_provider_routing: bool) -> Option<ProxyError> {
    if status.is_server_error() || (has_provider_routing && status == StatusCode::TOO_MANY_REQUESTS)
    {
        Some(ProxyError::UpstreamStatus(status))
    } else {
        None
    }
}

/// Check if a request is a WebSocket upgrade request.
fn is_websocket_upgrade<B>(req: &Request<B>) -> bool {
    let has_upgrade_connection = req.headers().get_all("connection").iter().any(|v| {
        v.to_str()
            .unwrap_or("")
            .split(',')
            .any(|token| token.trim().eq_ignore_ascii_case("upgrade"))
    });

    let has_websocket_upgrade = req
        .headers()
        .get("upgrade")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false);

    has_upgrade_connection && has_websocket_upgrade
}

/// Strip hop-by-hop headers but preserve Connection and Upgrade for WebSocket upgrades.
fn strip_hop_by_hop_headers_for_upgrade(headers: &mut HeaderMap) {
    for name in HOP_BY_HOP_HEADERS {
        if *name == "connection" || *name == "upgrade" {
            continue;
        }
        headers.remove(*name);
    }
}

/// Determine upstream URI scheme based on upstream address string.
fn upstream_uri(upstream: &str, path_and_query: &str) -> String {
    if upstream.contains("://") {
        format!("{}{}", upstream, path_and_query)
    } else if upstream.ends_with(":443") {
        format!("https://{}{}", upstream, path_and_query)
    } else {
        format!("http://{}{}", upstream, path_and_query)
    }
}

/// Extract just the host (and port) for the Host header, stripping any scheme.
fn upstream_host(upstream: &str) -> &str {
    upstream
        .find("://")
        .map(|i| &upstream[i + 3..])
        .unwrap_or(upstream)
}

fn default_root_store() -> rustls::RootCertStore {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    roots
}

#[cfg(test)]
fn websocket_test_tls_config_slot() -> &'static std::sync::Mutex<Option<Arc<rustls::ClientConfig>>>
{
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<Arc<rustls::ClientConfig>>>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

fn build_websocket_tls_config() -> Arc<rustls::ClientConfig> {
    #[cfg(test)]
    if let Some(config) = websocket_test_tls_config_slot().lock().unwrap().clone() {
        return config;
    }

    Arc::new(
        rustls::ClientConfig::builder()
            .with_root_certificates(default_root_store())
            .with_no_client_auth(),
    )
}

struct WebSocketUpstreamTarget {
    connect_addr: String,
    host_header: String,
    server_name: String,
    use_tls: bool,
}

fn parse_websocket_upstream_target(upstream: &str) -> Result<WebSocketUpstreamTarget, String> {
    let upstream_uri = if upstream.contains("://") {
        upstream.to_string()
    } else if upstream.ends_with(":443") {
        format!("https://{upstream}")
    } else {
        format!("http://{upstream}")
    };

    let uri: hyper::Uri = upstream_uri
        .parse()
        .map_err(|e| format!("invalid websocket upstream URI: {e}"))?;
    let authority = uri
        .authority()
        .map(|a| a.as_str().to_string())
        .ok_or_else(|| "websocket upstream missing authority".to_string())?;
    let server_name = uri
        .host()
        .map(ToString::to_string)
        .ok_or_else(|| "websocket upstream missing host".to_string())?;
    let use_tls = matches!(uri.scheme_str(), Some("https" | "wss"));
    let has_explicit_port = uri.port_u16().is_some();
    let default_port = if use_tls { 443 } else { 80 };
    let connect_addr = if has_explicit_port {
        authority.clone()
    } else {
        format!("{authority}:{default_port}")
    };

    Ok(WebSocketUpstreamTarget {
        connect_addr,
        host_header: authority,
        server_name,
        use_tls,
    })
}

#[derive(Debug)]
enum WebSocketUpstreamError {
    Handshake(hyper::Error),
    Request(hyper::Error),
    Timeout,
}

async fn send_websocket_upgrade_request<T>(
    io: TokioIo<T>,
    upstream_req: Request<Incoming>,
) -> Result<Response<Incoming>, WebSocketUpstreamError>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
        .await
        .map_err(WebSocketUpstreamError::Handshake)?;

    tokio::spawn(async move {
        if let Err(e) = conn.with_upgrades().await {
            tracing::debug!("upstream connection closed: {}", e);
        }
    });

    match tokio::time::timeout(Duration::from_secs(30), sender.send_request(upstream_req)).await {
        Ok(Ok(resp)) => Ok(resp),
        Ok(Err(e)) => Err(WebSocketUpstreamError::Request(e)),
        Err(_) => Err(WebSocketUpstreamError::Timeout),
    }
}

fn route_lb_cache_key(route: &RouteConfig) -> String {
    let mut key = format!("{:?}|", route.lb);
    key.push_str(&route.servers.join(","));
    key.push('|');
    if let Some(weights) = &route.weights {
        let joined = weights
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(",");
        key.push_str(&joined);
    }
    key
}

#[derive(Debug)]
enum CollectBodyError {
    TooLarge,
    Read(hyper::Error),
}

async fn collect_body_with_limit(
    mut body: BoxBody<Bytes, hyper::Error>,
    max_bytes: u64,
) -> Result<Bytes, CollectBodyError> {
    let limit = max_bytes.min(usize::MAX as u64) as usize;
    let mut buffered = BytesMut::new();

    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(CollectBodyError::Read)?;
        if let Some(data) = frame.data_ref() {
            if buffered.len().saturating_add(data.len()) > limit {
                return Err(CollectBodyError::TooLarge);
            }
            buffered.extend_from_slice(data);
        }
    }

    Ok(buffered.freeze())
}

/// HTTPS-capable client type using hyper-rustls.
pub type HttpClient = Client<
    hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>,
    BoxBody<Bytes, hyper::Error>,
>;

pub fn build_client() -> HttpClient {
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

    Client::builder(TokioExecutor::new())
        .pool_idle_timeout(Duration::from_secs(90))
        .build(https_connector)
}

pub struct ProxyService<R: RouteResolver> {
    router: Arc<R>,
    client: HttpClient,
    counter: Arc<AtomicUsize>,
    peer_addr: Option<SocketAddr>,
    health_state: Option<HealthState>,
    tls: bool,
    rate_limiter: Option<RateLimiter>,
    metrics: Option<Metrics>,
    circuit_breaker: Option<CircuitBreaker>,
    cache: Option<ResponseCache>,
    compression_enabled: bool,
    max_request_body_bytes: u64,
    upstream_timeout: Duration,
    h3_port: Option<u16>,
    plugins: Option<Arc<PluginChain>>,
    connection_extensions: Arc<Extensions>,
    load_balancers: Arc<DashMap<String, Arc<dyn load_balancer::LoadBalancer>>>,
}

impl<R: RouteResolver> Clone for ProxyService<R> {
    fn clone(&self) -> Self {
        Self {
            router: Arc::clone(&self.router),
            client: self.client.clone(),
            counter: Arc::clone(&self.counter),
            peer_addr: self.peer_addr,
            health_state: self.health_state.clone(),
            tls: self.tls,
            rate_limiter: self.rate_limiter.clone(),
            metrics: self.metrics.clone(),
            circuit_breaker: self.circuit_breaker.clone(),
            cache: self.cache.clone(),
            compression_enabled: self.compression_enabled,
            max_request_body_bytes: self.max_request_body_bytes,
            upstream_timeout: self.upstream_timeout,
            h3_port: self.h3_port,
            plugins: self.plugins.clone(),
            connection_extensions: Arc::clone(&self.connection_extensions),
            load_balancers: Arc::clone(&self.load_balancers),
        }
    }
}

impl<R: RouteResolver> ProxyService<R> {
    pub fn new(router: Arc<R>, client: HttpClient, counter: Arc<AtomicUsize>) -> Self {
        Self {
            router,
            client,
            counter,
            peer_addr: None,
            health_state: None,
            tls: false,
            rate_limiter: None,
            metrics: None,
            circuit_breaker: None,
            cache: None,
            compression_enabled: true,
            max_request_body_bytes: 10 * 1024 * 1024,
            upstream_timeout: Duration::from_secs(600),
            h3_port: None,
            plugins: None,
            connection_extensions: Arc::new(Extensions::new()),
            load_balancers: Arc::new(DashMap::new()),
        }
    }

    pub fn with_peer_addr(mut self, addr: SocketAddr) -> Self {
        self.peer_addr = Some(addr);
        self
    }

    pub fn with_health_state(mut self, state: HealthState) -> Self {
        self.health_state = Some(state);
        self
    }

    pub fn with_tls(mut self, tls: bool) -> Self {
        self.tls = tls;
        self
    }

    pub fn with_rate_limiter(mut self, limiter: RateLimiter) -> Self {
        self.rate_limiter = Some(limiter);
        self
    }

    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn with_circuit_breaker(mut self, cb: CircuitBreaker) -> Self {
        self.circuit_breaker = Some(cb);
        self
    }

    pub fn with_cache(mut self, cache: ResponseCache) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    pub fn with_max_request_body(mut self, max_bytes: u64) -> Self {
        self.max_request_body_bytes = max_bytes;
        self
    }

    pub fn with_upstream_timeout(mut self, secs: u64) -> Self {
        self.upstream_timeout = Duration::from_secs(secs);
        self
    }

    pub fn with_h3_port(mut self, port: u16) -> Self {
        self.h3_port = Some(port);
        self
    }

    pub fn with_plugins(mut self, plugins: Arc<PluginChain>) -> Self {
        self.plugins = Some(plugins);
        self
    }

    pub fn with_connection_extensions(mut self, ext: Arc<Extensions>) -> Self {
        self.connection_extensions = ext;
        self
    }

    pub async fn handle(
        self,
        req: Request<Incoming>,
    ) -> Result<Response<BoxBody<Bytes, hyper::Error>>, Infallible> {
        // WebSocket upgrade must be checked BEFORE converting body (OnUpgrade only on Incoming).
        if is_websocket_upgrade(&req) {
            return self.handle_websocket(req).await;
        }

        // Convert Incoming -> BoxBody and delegate to handle_boxed.
        let (parts, body) = req.into_parts();
        let boxed_req = Request::from_parts(parts, body.boxed());
        self.handle_boxed(boxed_req).await
    }

    /// Handle a request with a generic boxed body. Usable from both HTTP/1+2 and HTTP/3.
    pub async fn handle_boxed(
        self,
        req: Request<BoxBody<Bytes, hyper::Error>>,
    ) -> Result<Response<BoxBody<Bytes, hyper::Error>>, Infallible> {
        let start = std::time::Instant::now();
        let method = req.method().clone();
        let path = req.uri().path().to_string();

        // Create a tracing span for this request.  When the `opentelemetry`
        // feature is active and a subscriber is configured, this becomes a
        // real OTEL span.  Otherwise it compiles to a no-op.
        let span = tracing::info_span!("proxy_request",
            http.method = %method,
            http.target = %path,
            http.status_code = tracing::field::Empty,
            net.upstream = tracing::field::Empty,
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
        let _guard = span.enter();

        let cache_key_path = req
            .uri()
            .path_and_query()
            .map(|pq| pq.as_str().to_string())
            .unwrap_or_else(|| path.clone());
        let version = req.version();
        let accept_encoding = req.headers().get("accept-encoding").cloned();

        // Body size check (before buffering/plugin hooks).
        if let Some(resp) = middleware::check_body_limit(req.headers(), self.max_request_body_bytes)
        {
            return Ok(resp);
        }

        // Buffer body when plugins need it or when Content-Length is missing
        // (chunked/streaming uploads) so the size limit is enforced.
        let has_plugins = self.plugins.as_ref().map_or(false, |p| !p.is_empty());
        let should_buffer_for_limit = !req.headers().contains_key(CONTENT_LENGTH);
        let should_buffer_body = has_plugins || should_buffer_for_limit;
        let (req, plugin_body) = if should_buffer_body {
            let (parts, body) = req.into_parts();
            let collected = match collect_body_with_limit(body, self.max_request_body_bytes).await {
                Ok(c) => c,
                Err(CollectBodyError::TooLarge) => {
                    self.record_metrics(&method, "413", start);
                    return Ok(error_response(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "413 Payload Too Large\n",
                    ));
                }
                Err(CollectBodyError::Read(e)) => {
                    tracing::warn!(error = %e, "failed to buffer request body");
                    self.record_metrics(&method, "502", start);
                    return Ok(error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n"));
                }
            };
            let rebuilt = Request::from_parts(
                parts,
                Full::new(collected.clone())
                    .map_err(|never| match never {})
                    .boxed(),
            );
            let plugin_body = if has_plugins { Some(collected) } else { None };
            (rebuilt, plugin_body)
        } else {
            (req, None)
        };

        // Plugin: on_request hook.
        let mut plugin_ctx = RequestContext {
            peer_addr: self.peer_addr,
            method: method.clone(),
            uri: req.uri().clone(),
            version,
            headers: req.headers().clone(),
            body: plugin_body,
            route: None,
            selected_upstream: None,
            auth: None,
            connection: Arc::clone(&self.connection_extensions),
            extensions: Extensions::new(),
        };

        // Store the span in extensions so plugins can enrich it.
        #[cfg(feature = "opentelemetry")]
        plugin_ctx
            .extensions
            .insert(crate::otel::OtelSpan(span.clone()));

        if let Some(ref plugins) = self.plugins {
            if let Action::Respond(resp) = plugins.run_on_request(&mut plugin_ctx).await {
                return Ok(resp);
            }
        }
        let upstream_path_and_query = plugin_ctx
            .uri
            .path_and_query()
            .map(|pq| pq.as_str().to_string())
            .unwrap_or_else(|| cache_key_path.clone());
        let upstream_method = plugin_ctx.method.clone();

        // Check for per-provider timeout override from plugins.
        let effective_timeout = plugin_ctx
            .extensions
            .get::<crate::plugin::UpstreamTimeout>()
            .map(|t| t.0)
            .unwrap_or(self.upstream_timeout);
        span.record(
            "llm.upstream_timeout_ms",
            effective_timeout.as_millis() as u64,
        );

        // Rate limiting.
        if let Some(ref limiter) = self.rate_limiter {
            if let Some(resp) = middleware::check_rate_limit(
                limiter,
                self.peer_addr,
                self.metrics.as_ref(),
                &method,
            ) {
                return Ok(resp);
            }
        }

        // Route lookup.
        let route_config = match self.router.resolve_route_config(&path) {
            Some(rc) => rc,
            None => {
                self.record_metrics(&method, "404", start);
                return Ok(error_response(StatusCode::NOT_FOUND, "404 Not Found\n"));
            }
        };

        // Serve cache hits before origin health/circuit-breaker gating so
        // fresh cached responses remain available during upstream outages.
        if let Some(ref cache) = self.cache {
            if let Some(mut resp) =
                middleware::check_cache(cache, &method, &cache_key_path, req.headers(), version)
            {
                middleware::add_alt_svc(resp.headers_mut(), self.h3_port);
                self.record_metrics(&method, &resp.status().as_u16().to_string(), start);
                return Ok(resp);
            }
        }

        let servers = &route_config.servers;

        // Filter to healthy upstreams.
        let mut healthy_servers: Vec<&String> = match self.health_state {
            Some(ref hs) => servers.iter().filter(|s| hs.is_healthy(s)).collect(),
            None => servers.iter().collect(),
        };

        // Filter by circuit breaker.
        if let Some(ref cb) = self.circuit_breaker {
            healthy_servers = middleware::filter_by_circuit_breaker(cb, healthy_servers);
        }

        // Plugin: on_upstream_select hook.
        plugin_ctx.route = Some(path.clone());
        if let Some(ref plugins) = self.plugins {
            if let Action::Respond(resp) = plugins
                .run_on_upstream_select(&mut plugin_ctx, &mut healthy_servers)
                .await
            {
                return Ok(resp);
            }
        }

        // Check for ProviderCandidates (cross-provider retry from LLM gateway).
        let provider_candidates = plugin_ctx
            .extensions
            .remove::<crate::plugin::ProviderCandidates>();
        let use_provider_candidates = provider_candidates
            .as_ref()
            .map_or(false, |pc| !pc.0.is_empty());

        let selected_upstream = plugin_ctx.selected_upstream.clone();
        let selected_servers: Vec<String> = match selected_upstream {
            Some(forced_upstream) => vec![forced_upstream],
            None => healthy_servers.iter().map(|s| (*s).clone()).collect(),
        };

        if selected_servers.is_empty() && !use_provider_candidates {
            self.record_metrics(&method, "503", start);
            return Ok(error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "503 Service Unavailable\n",
            ));
        }

        // Load balancer + upstream selection.
        let lb = {
            let key = route_lb_cache_key(&route_config);
            let entry = self.load_balancers.entry(key).or_insert_with(|| {
                load_balancer::create_load_balancer_arc(
                    &route_config.lb,
                    route_config.weights.as_deref(),
                )
            });
            Arc::clone(entry.value())
        };
        let server_refs: Vec<&String> = selected_servers.iter().collect();
        let start_idx = {
            use crate::config::LbStrategy;
            use std::sync::atomic::Ordering;
            match route_config.lb {
                LbStrategy::RoundRobin => {
                    self.counter.fetch_add(1, Ordering::Relaxed) % server_refs.len()
                }
                _ => {
                    let lb_key = self.peer_addr.map(|a| a.ip().to_string());
                    lb.pick(&server_refs, lb_key.as_deref())
                }
            }
        };

        let (mut parts, body) = req.into_parts();

        // Apply plugin-modified headers and body back to the request.
        if has_plugins {
            parts.headers = plugin_ctx.headers.clone();
        }
        let body = if let Some(modified_body) = plugin_ctx.body.take() {
            Full::new(modified_body)
                .map_err(|never| match never {})
                .boxed()
        } else {
            body
        };

        // Capture original Host.
        let original_host = parts
            .headers
            .get(HOST)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        strip_hop_by_hop_if_needed(&mut parts.headers, version);

        if let Some(peer_addr) = self.peer_addr {
            add_forwarding_headers(
                &mut parts.headers,
                peer_addr,
                original_host.as_deref(),
                self.tls,
            );
        }

        parts.headers.append("via", via_header_value(version));

        let path_and_query = parts
            .uri
            .path_and_query()
            .map(|pq| pq.as_str().to_string())
            .unwrap_or_else(|| "/".to_string());

        let max_attempts = if use_provider_candidates {
            provider_candidates.as_ref().unwrap().0.len()
        } else if is_idempotent(&method) && selected_servers.len() > 1 {
            3.min(selected_servers.len())
        } else {
            1
        };

        if use_provider_candidates || max_attempts > 1 {
            let buffered_body =
                match collect_body_with_limit(body, self.max_request_body_bytes).await {
                    Ok(collected) => collected,
                    Err(CollectBodyError::TooLarge) => {
                        self.record_metrics(&method, "413", start);
                        return Ok(error_response(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "413 Payload Too Large\n",
                        ));
                    }
                    Err(CollectBodyError::Read(e)) => {
                        tracing::warn!(error = %e, "failed to buffer request body");
                        self.record_metrics(&method, "502", start);
                        return Ok(error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n"));
                    }
                };

            let mut last_error_response = None;

            for attempt in 0..max_attempts {
                // Resolve upstream and headers from provider candidates or selected_servers.
                let (upstream_str, mut headers) = if use_provider_candidates {
                    let candidates = &provider_candidates.as_ref().unwrap().0;
                    let candidate = &candidates[attempt];
                    (candidate.upstream.clone(), candidate.headers.clone())
                } else {
                    let idx = (start_idx + attempt) % selected_servers.len();
                    (selected_servers[idx].clone(), parts.headers.clone())
                };
                let upstream = &upstream_str;
                plugin_ctx.selected_upstream = Some(upstream.clone());
                let uri = upstream_uri(upstream, &upstream_path_and_query);

                let host_value = match upstream_host(upstream).parse() {
                    Ok(v) => v,
                    Err(_) => {
                        tracing::error!(upstream = %upstream, "malformed upstream host header");
                        self.record_metrics(&method, "502", start);
                        return Ok(error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n"));
                    }
                };
                headers.insert(HOST, host_value);

                let mut upstream_req = Request::builder().method(upstream_method.clone()).uri(&uri);
                if let Some(h) = upstream_req.headers_mut() {
                    *h = headers;
                }
                let upstream_req = match upstream_req.body(
                    Full::new(buffered_body.clone())
                        .map_err(|never| match never {})
                        .boxed(),
                ) {
                    Ok(req) => req,
                    Err(e) => {
                        tracing::error!(upstream = %upstream, error = %e, "failed to build upstream request");
                        self.record_metrics(&method, "502", start);
                        return Ok(error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n"));
                    }
                };

                lb.on_request_start(upstream);
                let attempt_span = tracing::info_span!(
                    "upstream_turn",
                    http.method = %method,
                    http.target = %path,
                    net.upstream = %upstream,
                    proxy.attempt = (attempt + 1) as u64,
                    proxy.timeout_ms = effective_timeout.as_millis() as u64,
                    proxy.provider_retry = use_provider_candidates,
                );
                let result = tokio::time::timeout(
                    effective_timeout,
                    self.client.request(upstream_req).instrument(attempt_span),
                )
                .await;
                lb.on_request_end(upstream);

                match result {
                    Ok(Ok(resp)) => {
                        let status = resp.status();
                        let status_error = status_error(status, use_provider_candidates);
                        if let Some(ref err) = status_error {
                            if let Some(ref plugins) = self.plugins {
                                if let Action::Respond(resp) =
                                    plugins.run_on_error(&mut plugin_ctx, err).await
                                {
                                    return Ok(resp);
                                }
                            }
                        }
                        let should_retry = if use_provider_candidates {
                            is_provider_retryable_status(status)
                        } else {
                            is_retryable_status(status)
                        };
                        if should_retry && attempt + 1 < max_attempts {
                            if status_error.is_some() {
                                if let Some(ref cb) = self.circuit_breaker {
                                    cb.record_failure(upstream);
                                }
                            }
                            tracing::warn!(
                                method = %method, path = %path, upstream = %upstream,
                                status = status.as_u16(), attempt = attempt + 1,
                                "retrying request on next upstream"
                            );
                            last_error_response = Some(resp);
                            continue;
                        }

                        if status.is_success() {
                            if let Some(ref cb) = self.circuit_breaker {
                                cb.record_success(upstream);
                            }
                        }

                        let mut resp = self
                            .finalize_response(
                                resp,
                                version,
                                &method,
                                &path,
                                &parts.headers,
                                accept_encoding,
                                start,
                            )
                            .await;

                        if let Some(ref plugins) = self.plugins {
                            resp = plugins.run_transform_response(&mut plugin_ctx, resp).await;
                        }

                        // Plugin: on_response hook.
                        if let Some(ref plugins) = self.plugins {
                            let mut resp_ctx = ResponseContext {
                                status: resp.status(),
                                headers: resp.headers().clone(),
                                upstream: upstream.clone(),
                                duration: start.elapsed(),
                            };
                            if let Action::Respond(r) = plugins
                                .run_on_response(&mut plugin_ctx, &mut resp_ctx)
                                .await
                            {
                                return Ok(r);
                            }
                        }

                        // Plugin: wrap_response_body hook.
                        if let Some(ref plugins) = self.plugins {
                            let resp_ctx = ResponseContext {
                                status: resp.status(),
                                headers: resp.headers().clone(),
                                upstream: upstream.clone(),
                                duration: start.elapsed(),
                            };
                            let (parts_out, body) = resp.into_parts();
                            let wrapped_body =
                                plugins.run_wrap_response_body(&plugin_ctx, &resp_ctx, body);
                            resp = Response::from_parts(parts_out, wrapped_body);
                        }

                        // Cache the response if applicable.
                        self.maybe_cache_response(
                            &method,
                            &cache_key_path,
                            &parts.headers,
                            &mut resp,
                        )
                        .await;

                        span.record("http.status_code", resp.status().as_u16());
                        span.record("net.upstream", upstream.as_str());
                        tracing::info!(method = %method, path = %path, upstream = %upstream, status = resp.status().as_u16(), "request completed");
                        return Ok(resp);
                    }
                    Ok(Err(e)) => {
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_failure(upstream);
                        }
                        // Plugin: on_error hook.
                        if let Some(ref plugins) = self.plugins {
                            let err = ProxyError::UpstreamError(e.to_string());
                            if let Action::Respond(resp) =
                                plugins.run_on_error(&mut plugin_ctx, &err).await
                            {
                                return Ok(resp);
                            }
                        }
                        if attempt + 1 < max_attempts {
                            tracing::warn!(method = %method, path = %path, upstream = %upstream, error = %e, attempt = attempt + 1, "retrying request on next upstream");
                            continue;
                        }
                        tracing::warn!(method = %method, path = %path, upstream = %upstream, error = %e, "upstream error");
                        self.record_metrics(&method, "502", start);
                        return Ok(error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n"));
                    }
                    Err(_) => {
                        // Plugin: on_error hook (timeout).
                        if let Some(ref plugins) = self.plugins {
                            let err = ProxyError::Timeout;
                            if let Action::Respond(resp) =
                                plugins.run_on_error(&mut plugin_ctx, &err).await
                            {
                                return Ok(resp);
                            }
                        }
                        if use_provider_candidates && attempt + 1 < max_attempts {
                            tracing::warn!(method = %method, path = %path, upstream = %upstream, attempt = attempt + 1, "upstream timeout, retrying on next provider");
                            continue;
                        }
                        tracing::warn!(method = %method, path = %path, upstream = %upstream, "upstream timeout");
                        self.record_metrics(&method, "504", start);
                        return Ok(error_response(
                            StatusCode::GATEWAY_TIMEOUT,
                            "504 Gateway Timeout\n",
                        ));
                    }
                }
            }

            if let Some(resp) = last_error_response {
                let (mut resp_parts, resp_body) = resp.into_parts();
                resp_parts.headers.append("via", via_header_value(version));
                middleware::add_alt_svc(&mut resp_parts.headers, self.h3_port);
                let status_str = resp_parts.status.as_u16().to_string();
                self.record_metrics(&method, &status_str, start);
                tracing::warn!(method = %method, path = %path, status = resp_parts.status.as_u16(), "all retry attempts exhausted");
                return Ok(Response::from_parts(resp_parts, resp_body.boxed()));
            }

            self.record_metrics(&method, "502", start);
            Ok(error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n"))
        } else {
            // Single attempt with streaming body.
            let upstream = &selected_servers[start_idx];
            plugin_ctx.selected_upstream = Some(upstream.clone());
            let uri = upstream_uri(upstream, &path_and_query);

            let host_value: HeaderValue = match upstream_host(upstream).parse() {
                Ok(v) => v,
                Err(_) => {
                    tracing::error!(upstream = %upstream, "malformed upstream host header");
                    self.record_metrics(&method, "502", start);
                    return Ok(error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n"));
                }
            };
            let mut upstream_req = Request::builder().method(parts.method).uri(&uri);
            if let Some(headers) = upstream_req.headers_mut() {
                *headers = parts.headers.clone();
                headers.insert(HOST, host_value);
            }
            let upstream_req = match upstream_req.body(body) {
                Ok(req) => req,
                Err(e) => {
                    tracing::error!(upstream = %upstream, error = %e, "failed to build upstream request");
                    self.record_metrics(&method, "502", start);
                    return Ok(error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n"));
                }
            };

            lb.on_request_start(upstream);
            let attempt_span = tracing::info_span!(
                "upstream_turn",
                http.method = %method,
                http.target = %path,
                net.upstream = %upstream,
                proxy.attempt = 1u64,
                proxy.timeout_ms = effective_timeout.as_millis() as u64,
                proxy.provider_retry = use_provider_candidates,
            );
            let result = tokio::time::timeout(
                effective_timeout,
                self.client.request(upstream_req).instrument(attempt_span),
            )
            .await;
            lb.on_request_end(upstream);

            match result {
                Ok(Ok(resp)) => {
                    let status = resp.status();
                    let status_error = status_error(status, use_provider_candidates);
                    if let Some(ref err) = status_error {
                        if let Some(ref plugins) = self.plugins {
                            if let Action::Respond(resp) =
                                plugins.run_on_error(&mut plugin_ctx, err).await
                            {
                                return Ok(resp);
                            }
                        }
                    }
                    if status.is_success() {
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_success(upstream);
                        }
                    } else if status_error.is_some() {
                        if let Some(ref cb) = self.circuit_breaker {
                            cb.record_failure(upstream);
                        }
                    }

                    let mut resp = self
                        .finalize_response(
                            resp,
                            version,
                            &method,
                            &path,
                            &parts.headers,
                            accept_encoding,
                            start,
                        )
                        .await;

                    if let Some(ref plugins) = self.plugins {
                        resp = plugins.run_transform_response(&mut plugin_ctx, resp).await;
                    }

                    // Plugin: on_response hook.
                    if let Some(ref plugins) = self.plugins {
                        let mut resp_ctx = ResponseContext {
                            status: resp.status(),
                            headers: resp.headers().clone(),
                            upstream: upstream.clone(),
                            duration: start.elapsed(),
                        };
                        if let Action::Respond(r) = plugins
                            .run_on_response(&mut plugin_ctx, &mut resp_ctx)
                            .await
                        {
                            return Ok(r);
                        }
                    }

                    // Plugin: wrap_response_body hook.
                    if let Some(ref plugins) = self.plugins {
                        let resp_ctx = ResponseContext {
                            status: resp.status(),
                            headers: resp.headers().clone(),
                            upstream: upstream.clone(),
                            duration: start.elapsed(),
                        };
                        let (parts_out, body) = resp.into_parts();
                        let wrapped_body =
                            plugins.run_wrap_response_body(&plugin_ctx, &resp_ctx, body);
                        resp = Response::from_parts(parts_out, wrapped_body);
                    }

                    // Cache the response if applicable.
                    self.maybe_cache_response(&method, &cache_key_path, &parts.headers, &mut resp)
                        .await;

                    span.record("http.status_code", resp.status().as_u16());
                    span.record("net.upstream", upstream.as_str());
                    tracing::info!(method = %method, path = %path, upstream = %upstream, status = resp.status().as_u16(), "request completed");
                    Ok(resp)
                }
                Ok(Err(e)) => {
                    if let Some(ref cb) = self.circuit_breaker {
                        cb.record_failure(upstream);
                    }
                    // Plugin: on_error hook.
                    if let Some(ref plugins) = self.plugins {
                        let err = ProxyError::UpstreamError(e.to_string());
                        if let Action::Respond(resp) =
                            plugins.run_on_error(&mut plugin_ctx, &err).await
                        {
                            return Ok(resp);
                        }
                    }
                    tracing::warn!(method = %method, path = %path, upstream = %upstream, error = %e, "upstream error");
                    self.record_metrics(&method, "502", start);
                    Ok(error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n"))
                }
                Err(_) => {
                    // Plugin: on_error hook (timeout).
                    if let Some(ref plugins) = self.plugins {
                        let err = ProxyError::Timeout;
                        if let Action::Respond(resp) =
                            plugins.run_on_error(&mut plugin_ctx, &err).await
                        {
                            return Ok(resp);
                        }
                    }
                    tracing::warn!(method = %method, path = %path, upstream = %upstream, "upstream timeout");
                    self.record_metrics(&method, "504", start);
                    Ok(error_response(
                        StatusCode::GATEWAY_TIMEOUT,
                        "504 Gateway Timeout\n",
                    ))
                }
            }
        }
    }

    /// WebSocket upgrade handler — extracted to keep handle() clean.
    async fn handle_websocket(
        self,
        req: Request<Incoming>,
    ) -> Result<Response<BoxBody<Bytes, hyper::Error>>, Infallible> {
        let start = std::time::Instant::now();
        let method = req.method().clone();
        let path = req.uri().path().to_string();

        let route_config = match self.router.resolve_route_config(&path) {
            Some(rc) => rc,
            None => {
                return Ok(error_response(StatusCode::NOT_FOUND, "404 Not Found\n"));
            }
        };

        let servers = &route_config.servers;
        let mut healthy_servers: Vec<&String> = match self.health_state {
            Some(ref hs) => servers.iter().filter(|s| hs.is_healthy(s)).collect(),
            None => servers.iter().collect(),
        };
        if let Some(ref cb) = self.circuit_breaker {
            healthy_servers = crate::middleware::filter_by_circuit_breaker(cb, healthy_servers);
        }
        if healthy_servers.is_empty() {
            return Ok(error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "503 Service Unavailable\n",
            ));
        }

        let start_idx = {
            use std::sync::atomic::Ordering;
            self.counter.fetch_add(1, Ordering::Relaxed) % healthy_servers.len()
        };
        let upstream = healthy_servers[start_idx];
        let resp = handle_websocket_upgrade(req, upstream, self.peer_addr, self.tls).await;
        self.record_metrics(&method, &resp.status().as_u16().to_string(), start);
        Ok(resp)
    }

    /// Shared response finalization: add headers, record metrics, compress.
    #[allow(clippy::too_many_arguments)]
    async fn finalize_response(
        &self,
        resp: hyper::Response<Incoming>,
        version: Version,
        method: &Method,
        _path: &str,
        _request_headers: &HeaderMap,
        accept_encoding: Option<HeaderValue>,
        start: std::time::Instant,
    ) -> Response<BoxBody<Bytes, hyper::Error>> {
        let (mut resp_parts, resp_body) = resp.into_parts();
        resp_parts.headers.append("via", via_header_value(version));
        resp_parts
            .headers
            .insert("x-cache", HeaderValue::from_static("MISS"));
        crate::middleware::add_alt_svc(&mut resp_parts.headers, self.h3_port);
        self.record_metrics(method, &resp_parts.status.as_u16().to_string(), start);

        let mut resp = Response::from_parts(resp_parts, resp_body.boxed());

        if self.compression_enabled {
            resp = compression::maybe_compress_response(accept_encoding, resp).await;
        }

        resp
    }

    fn record_metrics(&self, method: &Method, status: &str, start: std::time::Instant) {
        if let Some(ref m) = self.metrics {
            crate::middleware::record_metrics(m, method, status, start);
        }
    }

    async fn maybe_cache_response(
        &self,
        method: &Method,
        path: &str,
        request_headers: &HeaderMap,
        resp: &mut Response<BoxBody<Bytes, hyper::Error>>,
    ) {
        if let Some(ref cache) = self.cache {
            if matches!(*method, Method::GET | Method::HEAD) {
                let cache_key =
                    ResponseCache::cache_key(method, path, resp.headers(), request_headers);

                // Collect body for caching, then reconstruct.
                let (parts, body) = std::mem::replace(
                    resp,
                    Response::new(
                        Full::new(Bytes::new())
                            .map_err(|never| match never {})
                            .boxed(),
                    ),
                )
                .into_parts();

                let body_bytes = match body.collect().await {
                    Ok(collected) => collected.to_bytes(),
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to collect response body for cache write");
                        *resp = error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
                        return;
                    }
                };

                cache.put(cache_key, parts.status, &parts.headers, &body_bytes);

                *resp = Response::from_parts(
                    parts,
                    Full::new(body_bytes)
                        .map_err(|never| match never {})
                        .boxed(),
                );
            }
        }
    }
}

/// Handle a WebSocket upgrade request by establishing a bidirectional tunnel.
async fn handle_websocket_upgrade(
    mut req: Request<Incoming>,
    upstream: &str,
    peer_addr: Option<SocketAddr>,
    is_tls: bool,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // Extract the client's upgrade future before consuming the request.
    let client_on_upgrade = req.extensions_mut().remove::<OnUpgrade>();

    let (mut parts, body) = req.into_parts();

    let original_host = parts
        .headers
        .get(HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Preserve Connection and Upgrade headers for WebSocket.
    strip_hop_by_hop_headers_for_upgrade(&mut parts.headers);

    if let Some(peer_addr) = peer_addr {
        add_forwarding_headers(
            &mut parts.headers,
            peer_addr,
            original_host.as_deref(),
            is_tls,
        );
    }

    parts
        .headers
        .append("via", via_header_value(Version::HTTP_11));

    let pq = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");

    let target = match parse_websocket_upstream_target(upstream) {
        Ok(target) => target,
        Err(e) => {
            tracing::error!(upstream = %upstream, error = %e, "invalid websocket upstream");
            return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
        }
    };

    // Build the upstream request.
    let host_value: HeaderValue = match target.host_header.parse() {
        Ok(v) => v,
        Err(_) => {
            tracing::error!(upstream = %upstream, "malformed upstream host header");
            return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
        }
    };
    let mut upstream_req = Request::builder().method(parts.method).uri(pq);
    if let Some(headers) = upstream_req.headers_mut() {
        *headers = parts.headers;
        headers.insert(HOST, host_value);
    }
    let upstream_req = match upstream_req.body(body) {
        Ok(req) => req,
        Err(e) => {
            tracing::error!(upstream = %upstream, error = %e, "failed to build upstream request");
            return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
        }
    };

    let upstream_addr = upstream.to_string();
    let mut upstream_resp = if target.use_tls {
        let stream = match TcpStream::connect(&target.connect_addr).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    method = %method,
                    path = %path,
                    upstream = %upstream,
                    error = %e,
                    "upstream connect error"
                );
                return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
            }
        };
        let connector = TlsConnector::from(build_websocket_tls_config());
        let server_name = if let Ok(ip) = target.server_name.parse::<std::net::IpAddr>() {
            rustls::pki_types::ServerName::IpAddress(ip.into())
        } else {
            match rustls::pki_types::ServerName::try_from(target.server_name.clone()) {
                Ok(name) => name,
                Err(_) => {
                    tracing::error!(upstream = %upstream, "invalid TLS server name");
                    return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
                }
            }
        };
        let tls_stream = match connector.connect(server_name, stream).await {
            Ok(stream) => stream,
            Err(e) => {
                tracing::warn!(
                    method = %method,
                    path = %path,
                    upstream = %upstream,
                    error = %e,
                    "upstream TLS handshake error"
                );
                return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
            }
        };
        match send_websocket_upgrade_request(TokioIo::new(tls_stream), upstream_req).await {
            Ok(resp) => resp,
            Err(WebSocketUpstreamError::Handshake(e)) => {
                tracing::warn!(
                    method = %method,
                    path = %path,
                    upstream = %upstream_addr,
                    error = %e,
                    "upstream handshake error"
                );
                return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
            }
            Err(WebSocketUpstreamError::Request(e)) => {
                tracing::warn!(
                    method = %method,
                    path = %path,
                    upstream = %upstream_addr,
                    error = %e,
                    "upstream error"
                );
                return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
            }
            Err(WebSocketUpstreamError::Timeout) => {
                tracing::warn!(
                    method = %method,
                    path = %path,
                    upstream = %upstream_addr,
                    "upstream timeout"
                );
                return error_response(StatusCode::GATEWAY_TIMEOUT, "504 Gateway Timeout\n");
            }
        }
    } else {
        let stream = match TcpStream::connect(&target.connect_addr).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    method = %method,
                    path = %path,
                    upstream = %upstream,
                    error = %e,
                    "upstream connect error"
                );
                return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
            }
        };
        match send_websocket_upgrade_request(TokioIo::new(stream), upstream_req).await {
            Ok(resp) => resp,
            Err(WebSocketUpstreamError::Handshake(e)) => {
                tracing::warn!(
                    method = %method,
                    path = %path,
                    upstream = %upstream_addr,
                    error = %e,
                    "upstream handshake error"
                );
                return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
            }
            Err(WebSocketUpstreamError::Request(e)) => {
                tracing::warn!(
                    method = %method,
                    path = %path,
                    upstream = %upstream_addr,
                    error = %e,
                    "upstream error"
                );
                return error_response(StatusCode::BAD_GATEWAY, "502 Bad Gateway\n");
            }
            Err(WebSocketUpstreamError::Timeout) => {
                tracing::warn!(
                    method = %method,
                    path = %path,
                    upstream = %upstream_addr,
                    "upstream timeout"
                );
                return error_response(StatusCode::GATEWAY_TIMEOUT, "504 Gateway Timeout\n");
            }
        }
    };

    // If upstream rejected the upgrade, forward the response as-is.
    if upstream_resp.status() != StatusCode::SWITCHING_PROTOCOLS {
        let (mut resp_parts, resp_body) = upstream_resp.into_parts();
        resp_parts
            .headers
            .append("via", via_header_value(Version::HTTP_11));
        tracing::info!(
            method = %method,
            path = %path,
            upstream = %upstream_addr,
            status = resp_parts.status.as_u16(),
            "websocket upgrade rejected"
        );
        return Response::from_parts(resp_parts, resp_body.boxed());
    }

    // Upstream accepted — set up bidirectional tunnel.
    let upstream_on_upgrade = upstream_resp.extensions_mut().remove::<OnUpgrade>();

    tracing::info!(
        method = %method,
        path = %path,
        upstream = %upstream_addr,
        "websocket upgrade accepted"
    );

    if let (Some(client_upgrade), Some(upstream_upgrade)) = (client_on_upgrade, upstream_on_upgrade)
    {
        tokio::spawn(async move {
            let (client_upgraded, upstream_upgraded) =
                match tokio::try_join!(client_upgrade, upstream_upgrade) {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::warn!("websocket upgrade failed: {}", e);
                        return;
                    }
                };

            let mut client_io = TokioIo::new(client_upgraded);
            let mut upstream_io = TokioIo::new(upstream_upgraded);

            if let Err(e) = tokio::io::copy_bidirectional(&mut client_io, &mut upstream_io).await {
                tracing::debug!("websocket tunnel closed: {}", e);
            }
        });
    }

    // Build the 101 response to send to the client.
    let (mut resp_parts, resp_body) = upstream_resp.into_parts();
    resp_parts
        .headers
        .append("via", via_header_value(Version::HTTP_11));
    Response::from_parts(resp_parts, resp_body.boxed())
}

fn error_response(status: StatusCode, msg: &'static str) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(status)
        .body(
            Full::new(Bytes::from(msg))
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_breaker::CircuitState;
    use crate::config::{LbStrategy, RouteConfig};
    use crate::router::PathResolution;
    use crate::tls;
    use hyper::body::Incoming;
    use hyper::service::service_fn;
    use hyper::upgrade::OnUpgrade;
    use hyper_util::rt::TokioIo;
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    struct CatchAllRouter {
        route_config: RouteConfig,
    }

    impl PathResolution for CatchAllRouter {
        fn get_servers(&self, _path: &str) -> Option<&Vec<String>> {
            Some(&self.route_config.servers)
        }

        fn get_route_config(&self, _path: &str) -> Option<&RouteConfig> {
            Some(&self.route_config)
        }
    }

    async fn start_upstream(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };

                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(
                            io,
                            service_fn(move |_req: Request<Incoming>| async move {
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(StatusCode::OK)
                                        .body(Full::new(Bytes::from_static(body.as_bytes())))
                                        .unwrap(),
                                )
                            }),
                        )
                        .await;
                });
            }
        });

        addr
    }

    #[derive(Debug)]
    struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

    impl SkipServerVerification {
        fn new() -> Arc<Self> {
            Arc::new(Self(Arc::new(rustls::crypto::ring::default_provider())))
        }
    }

    impl ServerCertVerifier for SkipServerVerification {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &rustls::DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            rustls::crypto::verify_tls12_signature(
                message,
                cert,
                dss,
                &self.0.signature_verification_algorithms,
            )
        }

        fn verify_tls13_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &rustls::DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            rustls::crypto::verify_tls13_signature(
                message,
                cert,
                dss,
                &self.0.signature_verification_algorithms,
            )
        }

        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            self.0.signature_verification_algorithms.supported_schemes()
        }
    }

    #[tokio::test]
    async fn test_collect_body_with_limit_accepts_within_limit() {
        let body = Full::new(Bytes::from_static(b"hello"))
            .map_err(|never| match never {})
            .boxed();
        let collected = collect_body_with_limit(body, 5).await.unwrap();
        assert_eq!(collected, Bytes::from_static(b"hello"));
    }

    #[tokio::test]
    async fn test_collect_body_with_limit_rejects_oversize() {
        let body = Full::new(Bytes::from_static(b"0123456789ab"))
            .map_err(|never| match never {})
            .boxed();
        let result = collect_body_with_limit(body, 10).await;
        assert!(matches!(result, Err(CollectBodyError::TooLarge)));
    }

    #[test]
    fn test_strip_hop_by_hop_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("connection", HeaderValue::from_static("keep-alive"));
        headers.insert("keep-alive", HeaderValue::from_static("timeout=5"));
        headers.insert("te", HeaderValue::from_static("trailers"));
        headers.insert("transfer-encoding", HeaderValue::from_static("chunked"));
        headers.insert("upgrade", HeaderValue::from_static("h2c"));
        headers.insert("proxy-authorization", HeaderValue::from_static("Basic xxx"));
        headers.insert("proxy-connection", HeaderValue::from_static("keep-alive"));
        headers.insert("x-custom", HeaderValue::from_static("should-stay"));

        strip_hop_by_hop_headers(&mut headers);

        assert!(headers.get("connection").is_none());
        assert!(headers.get("keep-alive").is_none());
        assert!(headers.get("te").is_none());
        assert!(headers.get("transfer-encoding").is_none());
        assert!(headers.get("upgrade").is_none());
        assert!(headers.get("proxy-authorization").is_none());
        assert!(headers.get("proxy-connection").is_none());
        assert_eq!(headers.get("x-custom").unwrap(), "should-stay");
    }

    #[test]
    fn test_strip_custom_connection_tokens() {
        let mut headers = HeaderMap::new();
        headers.insert("connection", HeaderValue::from_static("x-foo"));
        headers.insert("x-foo", HeaderValue::from_static("bar"));
        headers.insert("x-keep", HeaderValue::from_static("yes"));

        strip_hop_by_hop_headers(&mut headers);

        assert!(headers.get("connection").is_none());
        assert!(headers.get("x-foo").is_none());
        assert_eq!(headers.get("x-keep").unwrap(), "yes");
    }

    #[test]
    fn test_strip_hop_by_hop_h2_preserves_te_trailers() {
        let mut headers = HeaderMap::new();
        headers.insert("te", HeaderValue::from_static("trailers"));
        headers.insert("connection", HeaderValue::from_static("keep-alive"));
        headers.insert("keep-alive", HeaderValue::from_static("timeout=5"));

        strip_hop_by_hop_if_needed(&mut headers, Version::HTTP_2);

        // TE: trailers is preserved for H2 (needed for gRPC)
        assert_eq!(headers.get("te").unwrap(), "trailers");
        assert!(headers.get("connection").is_none());
        assert!(headers.get("keep-alive").is_none());
    }

    #[test]
    fn test_strip_hop_by_hop_h2_removes_te_non_trailers() {
        let mut headers = HeaderMap::new();
        headers.insert("te", HeaderValue::from_static("chunked"));

        strip_hop_by_hop_if_needed(&mut headers, Version::HTTP_2);

        assert!(headers.get("te").is_none());
    }

    #[test]
    fn test_add_forwarding_headers_new() {
        let mut headers = HeaderMap::new();
        let peer_addr: SocketAddr = "10.0.0.1:12345".parse().unwrap();

        add_forwarding_headers(&mut headers, peer_addr, Some("example.com"), false);

        assert_eq!(headers.get("x-forwarded-for").unwrap(), "10.0.0.1");
        assert_eq!(headers.get("x-forwarded-proto").unwrap(), "http");
        assert_eq!(headers.get("x-forwarded-host").unwrap(), "example.com");
    }

    #[test]
    fn test_add_forwarding_headers_append() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", HeaderValue::from_static("1.2.3.4"));
        let peer_addr: SocketAddr = "10.0.0.1:12345".parse().unwrap();

        add_forwarding_headers(&mut headers, peer_addr, None, false);

        assert_eq!(headers.get("x-forwarded-for").unwrap(), "1.2.3.4, 10.0.0.1");
    }

    #[test]
    fn test_add_forwarding_headers_tls() {
        let mut headers = HeaderMap::new();
        let peer_addr: SocketAddr = "10.0.0.1:12345".parse().unwrap();

        add_forwarding_headers(&mut headers, peer_addr, Some("example.com"), true);

        assert_eq!(headers.get("x-forwarded-for").unwrap(), "10.0.0.1");
        assert_eq!(headers.get("x-forwarded-proto").unwrap(), "https");
        assert_eq!(headers.get("x-forwarded-host").unwrap(), "example.com");
    }

    #[test]
    fn test_via_header_value_h1() {
        let via = via_header_value(Version::HTTP_11);
        let s = via.to_str().unwrap();
        assert!(s.contains("1.1"));
        assert!(s.contains("tiny-reverse-proxy"));
    }

    #[test]
    fn test_via_header_value_h2() {
        let via = via_header_value(Version::HTTP_2);
        let s = via.to_str().unwrap();
        assert!(s.contains("2.0"));
        assert!(s.contains("tiny-reverse-proxy"));
    }

    #[test]
    fn test_is_websocket_upgrade() {
        let req = Request::builder()
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .body(Full::new(Bytes::new()))
            .unwrap();
        assert!(is_websocket_upgrade(&req));

        let req = Request::builder()
            .header("connection", "upgrade")
            .header("upgrade", "WebSocket")
            .body(Full::new(Bytes::new()))
            .unwrap();
        assert!(is_websocket_upgrade(&req));

        let req = Request::builder()
            .header("Connection", "keep-alive, Upgrade")
            .header("Upgrade", "websocket")
            .body(Full::new(Bytes::new()))
            .unwrap();
        assert!(is_websocket_upgrade(&req));

        let req = Request::builder()
            .header("Connection", "Upgrade")
            .body(Full::new(Bytes::new()))
            .unwrap();
        assert!(!is_websocket_upgrade(&req));

        let req = Request::builder()
            .header("Upgrade", "websocket")
            .body(Full::new(Bytes::new()))
            .unwrap();
        assert!(!is_websocket_upgrade(&req));

        let req = Request::builder()
            .header("Connection", "Upgrade")
            .header("Upgrade", "h2c")
            .body(Full::new(Bytes::new()))
            .unwrap();
        assert!(!is_websocket_upgrade(&req));
    }

    #[test]
    fn test_strip_hop_by_hop_headers_for_upgrade() {
        let mut headers = HeaderMap::new();
        headers.insert("connection", HeaderValue::from_static("Upgrade"));
        headers.insert("upgrade", HeaderValue::from_static("websocket"));
        headers.insert("keep-alive", HeaderValue::from_static("timeout=5"));
        headers.insert("te", HeaderValue::from_static("trailers"));
        headers.insert(
            "sec-websocket-key",
            HeaderValue::from_static("dGhlIHNhbXBsZSBub25jZQ=="),
        );
        headers.insert("x-custom", HeaderValue::from_static("should-stay"));

        strip_hop_by_hop_headers_for_upgrade(&mut headers);

        assert_eq!(headers.get("connection").unwrap(), "Upgrade");
        assert_eq!(headers.get("upgrade").unwrap(), "websocket");
        assert!(headers.get("keep-alive").is_none());
        assert!(headers.get("te").is_none());
        assert_eq!(
            headers.get("sec-websocket-key").unwrap(),
            "dGhlIHNhbXBsZSBub25jZQ=="
        );
        assert_eq!(headers.get("x-custom").unwrap(), "should-stay");
    }

    #[test]
    fn test_is_idempotent() {
        assert!(is_idempotent(&Method::GET));
        assert!(is_idempotent(&Method::HEAD));
        assert!(is_idempotent(&Method::OPTIONS));
        assert!(is_idempotent(&Method::PUT));
        assert!(is_idempotent(&Method::DELETE));
        assert!(!is_idempotent(&Method::POST));
        assert!(!is_idempotent(&Method::PATCH));
    }

    #[test]
    fn test_is_retryable_status() {
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY));
        assert!(is_retryable_status(StatusCode::SERVICE_UNAVAILABLE));
        assert!(!is_retryable_status(StatusCode::OK));
        assert!(!is_retryable_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(!is_retryable_status(StatusCode::GATEWAY_TIMEOUT));
        assert!(!is_retryable_status(StatusCode::NOT_FOUND));
    }

    #[test]
    fn test_is_provider_retryable_status() {
        assert!(is_provider_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_provider_retryable_status(
            StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(is_provider_retryable_status(StatusCode::BAD_GATEWAY));
        assert!(!is_provider_retryable_status(StatusCode::OK));
        assert!(!is_provider_retryable_status(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn test_upstream_uri_with_scheme() {
        assert_eq!(
            upstream_uri("https://backend.example.com", "/api/data"),
            "https://backend.example.com/api/data"
        );
    }

    #[test]
    fn test_upstream_uri_without_scheme() {
        assert_eq!(
            upstream_uri("backend:8080", "/api/data"),
            "http://backend:8080/api/data"
        );
    }

    #[test]
    fn test_upstream_uri_port_443_uses_https() {
        assert_eq!(
            upstream_uri("backend.example.com:443", "/api/data"),
            "https://backend.example.com:443/api/data"
        );
    }

    #[test]
    fn test_upstream_host_strips_scheme() {
        assert_eq!(
            upstream_host("https://example.com:1012"),
            "example.com:1012"
        );
        assert_eq!(upstream_host("http://example.com:80"), "example.com:80");
        assert_eq!(upstream_host("example.com:443"), "example.com:443");
    }

    async fn start_proxy(router: Arc<CatchAllRouter>) -> String {
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

                let svc =
                    ProxyService::new(Arc::clone(&router), client.clone(), Arc::clone(&counter))
                        .with_peer_addr(peer_addr);

                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(
                            io,
                            service_fn(move |req: Request<Incoming>| {
                                let svc = svc.clone();
                                async move { svc.handle(req).await }
                            }),
                        )
                        .with_upgrades()
                        .await;
                });
            }
        });

        addr
    }

    async fn start_tls_websocket_echo_upstream() -> (String, Arc<rustls::ClientConfig>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let (certs, key) = tls::generate_self_signed_cert(&["localhost"]).unwrap();
        let acceptor = tls::build_tls_acceptor(certs, key).unwrap();

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    let tls_stream = match acceptor.accept(stream).await {
                        Ok(stream) => stream,
                        Err(_) => return,
                    };
                    let io = TokioIo::new(tls_stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(
                            io,
                            service_fn(|mut req: Request<Incoming>| async move {
                                let is_ws = req
                                    .headers()
                                    .get("upgrade")
                                    .and_then(|v| v.to_str().ok())
                                    .map(|v| v.eq_ignore_ascii_case("websocket"))
                                    .unwrap_or(false);

                                if !is_ws {
                                    return Ok::<_, hyper::Error>(
                                        Response::builder()
                                            .status(StatusCode::BAD_REQUEST)
                                            .body(Full::new(Bytes::from("not a websocket")))
                                            .unwrap(),
                                    );
                                }

                                let on_upgrade = req.extensions_mut().remove::<OnUpgrade>();
                                if let Some(on_upgrade) = on_upgrade {
                                    tokio::spawn(async move {
                                        let Ok(upgraded) = on_upgrade.await else {
                                            return;
                                        };
                                        let io = TokioIo::new(upgraded);
                                        let (mut rd, mut wr) = tokio::io::split(io);
                                        let _ = tokio::io::copy(&mut rd, &mut wr).await;
                                    });
                                }

                                Ok(Response::builder()
                                    .status(StatusCode::SWITCHING_PROTOCOLS)
                                    .header("Upgrade", "websocket")
                                    .header("Connection", "Upgrade")
                                    .body(Full::new(Bytes::new()))
                                    .unwrap())
                            }),
                        )
                        .with_upgrades()
                        .await;
                });
            }
        });

        let mut client_config = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_protocol_versions(&[&rustls::version::TLS13, &rustls::version::TLS12])
        .unwrap()
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
        client_config.alpn_protocols = vec![b"http/1.1".to_vec()];
        let client_config = Arc::new(client_config);

        (addr, client_config)
    }

    struct WebSocketTlsTestGuard;

    impl WebSocketTlsTestGuard {
        fn set(config: Arc<rustls::ClientConfig>) -> Self {
            *websocket_test_tls_config_slot().lock().unwrap() = Some(config);
            Self
        }
    }

    impl Drop for WebSocketTlsTestGuard {
        fn drop(&mut self) {
            *websocket_test_tls_config_slot().lock().unwrap() = None;
        }
    }

    #[tokio::test]
    async fn websocket_upgrade_supports_tls_upstreams() {
        let (upstream_addr, client_config) = start_tls_websocket_echo_upstream().await;
        let upstream_port = upstream_addr.rsplit(':').next().unwrap();
        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![format!("https://localhost:{upstream_port}")],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy(router).await;
        let _tls_guard = WebSocketTlsTestGuard::set(client_config);

        let stream = TcpStream::connect(&proxy_addr).await.unwrap();
        let io = TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();

        tokio::spawn(async move {
            conn.with_upgrades().await.ok();
        });

        let req = Request::builder()
            .uri("/test")
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
            .header("Sec-WebSocket-Version", "13")
            .body(Full::new(Bytes::new()))
            .unwrap();

        let resp = sender.send_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SWITCHING_PROTOCOLS);

        let upgraded = hyper::upgrade::on(resp).await.unwrap();
        let mut io = TokioIo::new(upgraded);
        io.write_all(b"tls websocket").await.unwrap();

        let mut buf = vec![0u8; 13];
        io.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"tls websocket");
    }

    #[tokio::test]
    async fn cache_hit_survives_unhealthy_origin() {
        let upstream = start_upstream("cached body").await;
        let health_state = HealthState::new(&[upstream.clone()]);
        let cache = ResponseCache::new(1, 300);
        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream.clone()],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let client = build_client();
        let counter = Arc::new(AtomicUsize::new(0));
        let service = ProxyService::new(router, client, counter)
            .with_cache(cache)
            .with_health_state(health_state.clone());

        let req = Request::builder()
            .uri("/cache")
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();
        let first = service.clone().handle_boxed(req).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(
            first.headers().get("x-cache").and_then(|v| v.to_str().ok()),
            Some("MISS")
        );
        let _ = first.into_body().collect().await.unwrap();

        health_state.mark_unhealthy(&upstream);

        let req = Request::builder()
            .uri("/cache")
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();
        let second = service.handle_boxed(req).await.unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(
            second
                .headers()
                .get("x-cache")
                .and_then(|v| v.to_str().ok()),
            Some("HIT")
        );
        let body = second.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(body, Bytes::from_static(b"cached body"));
    }

    #[tokio::test]
    async fn cache_hit_does_not_transition_circuit_breaker_half_open() {
        let upstream = start_upstream("cached body").await;
        let cache = ResponseCache::new(1, 300);
        let circuit_breaker = CircuitBreaker::new(1, 0, 60);
        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream.clone()],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let client = build_client();
        let counter = Arc::new(AtomicUsize::new(0));
        let service = ProxyService::new(router, client, counter)
            .with_cache(cache)
            .with_circuit_breaker(circuit_breaker.clone());

        let req = Request::builder()
            .uri("/cache")
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();
        let first = service.clone().handle_boxed(req).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let _ = first.into_body().collect().await.unwrap();

        circuit_breaker.record_failure(&upstream);
        assert_eq!(circuit_breaker.state(&upstream), CircuitState::Open);

        let req = Request::builder()
            .uri("/cache")
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();
        let second = service.handle_boxed(req).await.unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(
            second
                .headers()
                .get("x-cache")
                .and_then(|v| v.to_str().ok()),
            Some("HIT")
        );
        assert_eq!(
            circuit_breaker.state(&upstream),
            CircuitState::Open,
            "cache hits must not consume or strand half-open probes"
        );
    }
}

#[cfg(test)]
mod retry_tests {
    use super::*;
    use crate::config::{LbStrategy, RouteConfig};
    use crate::router::PathResolution;
    use hyper::service::service_fn;
    use hyper_util::rt::TokioIo;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;

    struct MockRouterWithConfig {
        config: RouteConfig,
    }

    impl PathResolution for MockRouterWithConfig {
        fn get_servers(&self, _path: &str) -> Option<&Vec<String>> {
            Some(&self.config.servers)
        }

        fn get_route_config(&self, _path: &str) -> Option<&RouteConfig> {
            Some(&self.config)
        }
    }

    async fn start_upstream(status: StatusCode) -> (SocketAddr, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hit_count = Arc::new(AtomicUsize::new(0));
        let hit_count_inner = Arc::clone(&hit_count);

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let hit_count = Arc::clone(&hit_count_inner);
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(
                            io,
                            service_fn(move |_req: hyper::Request<hyper::body::Incoming>| {
                                hit_count.fetch_add(1, Ordering::Relaxed);
                                async move {
                                    Ok::<_, Infallible>(
                                        hyper::Response::builder()
                                            .status(status)
                                            .body(Full::new(Bytes::from("ok")))
                                            .unwrap(),
                                    )
                                }
                            }),
                        )
                        .await;
                });
            }
        });

        (addr, hit_count)
    }

    async fn start_proxy(servers: Vec<String>) -> SocketAddr {
        let route_config = RouteConfig {
            servers,
            lb: LbStrategy::RoundRobin,
            weights: None,
        };
        let router = Arc::new(MockRouterWithConfig {
            config: route_config,
        });
        let client = build_client();
        let counter = Arc::new(AtomicUsize::new(0));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            loop {
                let Ok((stream, peer_addr)) = listener.accept().await else {
                    break;
                };
                let svc =
                    ProxyService::new(Arc::clone(&router), client.clone(), Arc::clone(&counter))
                        .with_peer_addr(peer_addr);

                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(
                            io,
                            service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
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

    type TestClient = Client<
        hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>,
        BoxBody<Bytes, hyper::Error>,
    >;

    fn test_client() -> TestClient {
        build_client()
    }

    async fn send_request(
        client: &TestClient,
        proxy_addr: SocketAddr,
        method: Method,
        path: &str,
    ) -> hyper::Response<hyper::body::Incoming> {
        let uri = format!("http://{}{}", proxy_addr, path);
        let req = hyper::Request::builder()
            .method(method)
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();
        client.request(req).await.unwrap()
    }

    fn unused_addr() -> SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    }

    #[tokio::test]
    async fn test_retry_on_502() {
        let (bad_addr, bad_hits) = start_upstream(StatusCode::BAD_GATEWAY).await;
        let (good_addr, good_hits) = start_upstream(StatusCode::OK).await;

        let proxy_addr = start_proxy(vec![bad_addr.to_string(), good_addr.to_string()]).await;

        let client = test_client();
        let resp = send_request(&client, proxy_addr, Method::GET, "/test").await;

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(bad_hits.load(Ordering::Relaxed), 1);
        assert_eq!(good_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_retry_on_503() {
        let (bad_addr, bad_hits) = start_upstream(StatusCode::SERVICE_UNAVAILABLE).await;
        let (good_addr, good_hits) = start_upstream(StatusCode::OK).await;

        let proxy_addr = start_proxy(vec![bad_addr.to_string(), good_addr.to_string()]).await;

        let client = test_client();
        let resp = send_request(&client, proxy_addr, Method::GET, "/test").await;

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(bad_hits.load(Ordering::Relaxed), 1);
        assert_eq!(good_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_retry_on_connection_error() {
        let dead_addr = unused_addr();
        let (good_addr, good_hits) = start_upstream(StatusCode::OK).await;

        let proxy_addr = start_proxy(vec![dead_addr.to_string(), good_addr.to_string()]).await;

        let client = test_client();
        let resp = send_request(&client, proxy_addr, Method::GET, "/test").await;

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(good_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_no_retry_on_post() {
        let (bad_addr, bad_hits) = start_upstream(StatusCode::BAD_GATEWAY).await;
        let (good_addr, good_hits) = start_upstream(StatusCode::OK).await;

        let proxy_addr = start_proxy(vec![bad_addr.to_string(), good_addr.to_string()]).await;

        let client = test_client();
        let resp = send_request(&client, proxy_addr, Method::POST, "/test").await;

        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(bad_hits.load(Ordering::Relaxed), 1);
        assert_eq!(good_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_all_upstreams_fail() {
        let (bad1_addr, bad1_hits) = start_upstream(StatusCode::BAD_GATEWAY).await;
        let (bad2_addr, bad2_hits) = start_upstream(StatusCode::BAD_GATEWAY).await;

        let proxy_addr = start_proxy(vec![bad1_addr.to_string(), bad2_addr.to_string()]).await;

        let client = test_client();
        let resp = send_request(&client, proxy_addr, Method::GET, "/test").await;

        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(bad1_hits.load(Ordering::Relaxed), 1);
        assert_eq!(bad2_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_max_three_attempts() {
        let (bad1_addr, bad1_hits) = start_upstream(StatusCode::SERVICE_UNAVAILABLE).await;
        let (bad2_addr, bad2_hits) = start_upstream(StatusCode::SERVICE_UNAVAILABLE).await;
        let (bad3_addr, bad3_hits) = start_upstream(StatusCode::SERVICE_UNAVAILABLE).await;
        let (good_addr, good_hits) = start_upstream(StatusCode::OK).await;

        let proxy_addr = start_proxy(vec![
            bad1_addr.to_string(),
            bad2_addr.to_string(),
            bad3_addr.to_string(),
            good_addr.to_string(),
        ])
        .await;

        let client = test_client();
        let resp = send_request(&client, proxy_addr, Method::GET, "/test").await;

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(bad1_hits.load(Ordering::Relaxed), 1);
        assert_eq!(bad2_hits.load(Ordering::Relaxed), 1);
        assert_eq!(bad3_hits.load(Ordering::Relaxed), 1);
        assert_eq!(good_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_retry_idempotent_methods() {
        for method in [Method::HEAD, Method::PUT, Method::DELETE, Method::OPTIONS] {
            let (bad_addr, _) = start_upstream(StatusCode::BAD_GATEWAY).await;
            let (good_addr, good_hits) = start_upstream(StatusCode::OK).await;

            let proxy_addr = start_proxy(vec![bad_addr.to_string(), good_addr.to_string()]).await;

            let client = test_client();
            let resp = send_request(&client, proxy_addr, method.clone(), "/test").await;

            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "{} should be retried",
                method
            );
            assert_eq!(good_hits.load(Ordering::Relaxed), 1);
        }
    }

    #[tokio::test]
    async fn test_single_upstream_no_retry() {
        let (bad_addr, bad_hits) = start_upstream(StatusCode::BAD_GATEWAY).await;

        let proxy_addr = start_proxy(vec![bad_addr.to_string()]).await;

        let client = test_client();
        let resp = send_request(&client, proxy_addr, Method::GET, "/test").await;

        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(bad_hits.load(Ordering::Relaxed), 1);
    }
}
