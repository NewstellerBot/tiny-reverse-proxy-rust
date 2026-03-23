use std::net::SocketAddr;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use arc_swap::ArcSwap;
use bytes::{Buf, Bytes};
use h3_quinn::quinn;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::http::Extensions;
use hyper::{Request, Response, StatusCode};
use tokio_rustls::rustls;

use crate::cache::ResponseCache;
use crate::circuit_breaker::CircuitBreaker;
use crate::handlers::proxy::{HttpClient, ProxyService};
use crate::health::HealthState;
use crate::metrics::Metrics;
use crate::plugin::{self, PluginChain};
use crate::rate_limit::RateLimiter;
use crate::router::Router;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum H3BodyLimitError {
    TooLarge,
}

fn append_chunk_with_limit(
    body: &mut Vec<u8>,
    chunk: &[u8],
    max_request_body_bytes: u64,
) -> Result<(), H3BodyLimitError> {
    let max = max_request_body_bytes.min(usize::MAX as u64) as usize;
    if body.len().saturating_add(chunk.len()) > max {
        return Err(H3BodyLimitError::TooLarge);
    }
    body.extend_from_slice(chunk);
    Ok(())
}

/// Build a Quinn server endpoint for HTTP/3.
pub fn build_quic_endpoint(
    addr: SocketAddr,
    tls_config: Arc<rustls::ServerConfig>,
) -> Result<quinn::Endpoint, Box<dyn std::error::Error>> {
    let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(tls_config)?,
    ));
    server_config.transport_config(Arc::new({
        let mut transport = quinn::TransportConfig::default();
        transport.keep_alive_interval(Some(std::time::Duration::from_secs(15)));
        transport
    }));

    let endpoint = quinn::Endpoint::server(server_config, addr)?;
    Ok(endpoint)
}

/// Dependencies for the HTTP/3 accept loop.
pub struct H3Deps {
    pub router: Arc<ArcSwap<Router>>,
    pub client: HttpClient,
    pub counter: Arc<AtomicUsize>,
    pub health_state: Option<HealthState>,
    pub upstream_timeout_secs: u64,
    pub rate_limiter: Option<RateLimiter>,
    pub metrics: Option<Metrics>,
    pub circuit_breaker: Option<CircuitBreaker>,
    pub cache: Option<ResponseCache>,
    pub plugins: Option<Arc<PluginChain>>,
    pub compression_enabled: bool,
    pub max_request_body_bytes: u64,
}

/// Run the HTTP/3 accept loop. Each incoming QUIC connection is handled with h3.
pub async fn accept_h3_loop(endpoint: quinn::Endpoint, deps: Arc<H3Deps>) {
    while let Some(incoming) = endpoint.accept().await {
        let deps = Arc::clone(&deps);
        tokio::spawn(async move {
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    tracing::debug!("QUIC connection failed: {}", e);
                    return;
                }
            };
            let peer_addr = conn.remote_address();
            let connection_extensions = if let Some(ref plugins) = deps.plugins {
                let mut conn_ctx = plugin::ConnectionContext {
                    peer_addr,
                    tls_client_hello: None,
                    extensions: Extensions::new(),
                };
                if let plugin::Action::Respond(_) = plugins.run_on_accept(&mut conn_ctx).await {
                    return;
                }
                Arc::new(conn_ctx.extensions)
            } else {
                Arc::new(Extensions::new())
            };
            let mut h3_conn: h3::server::Connection<h3_quinn::Connection, Bytes> =
                match h3::server::Connection::new(h3_quinn::Connection::new(conn)).await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::debug!("H3 handshake failed: {}", e);
                        return;
                    }
                };

            loop {
                match h3_conn.accept().await {
                    Ok(Some(resolver)) => {
                        let deps = Arc::clone(&deps);
                        let connection_extensions = Arc::clone(&connection_extensions);
                        tokio::spawn(async move {
                            if let Err(e) =
                                handle_h3_request(resolver, &deps, peer_addr, connection_extensions)
                                    .await
                            {
                                tracing::debug!("H3 request error: {}", e);
                            }
                        });
                    }
                    Ok(None) => break,
                    Err(e) => {
                        tracing::debug!("H3 accept error: {}", e);
                        break;
                    }
                }
            }
        });
    }
}

/// Handle a single HTTP/3 request by forwarding through ProxyService.
async fn handle_h3_request(
    resolver: h3::server::RequestResolver<h3_quinn::Connection, Bytes>,
    deps: &H3Deps,
    peer_addr: SocketAddr,
    connection_extensions: Arc<Extensions>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (req, mut stream) = resolver.resolve_request().await?;

    // Collect the request body from the H3 stream.
    let mut body_bytes = Vec::new();
    while let Some(chunk) = stream.recv_data().await? {
        if append_chunk_with_limit(&mut body_bytes, chunk.chunk(), deps.max_request_body_bytes)
            .is_err()
        {
            let too_large_resp = Response::builder()
                .status(StatusCode::PAYLOAD_TOO_LARGE)
                .body(())
                .unwrap();
            stream.send_response(too_large_resp).await?;
            stream.finish().await?;
            return Ok(());
        }
    }

    // Build a BoxBody request for handle_boxed.
    let (parts, _) = req.into_parts();
    let body: BoxBody<Bytes, hyper::Error> = Full::new(Bytes::from(body_bytes))
        .map_err(|never| match never {})
        .boxed();
    let proxy_req = Request::from_parts(parts, body);

    // Build a ProxyService with the current router snapshot.
    let router = deps.router.load();
    let mut svc = ProxyService::new(
        Arc::clone(&*router),
        deps.client.clone(),
        Arc::clone(&deps.counter),
    )
    .with_peer_addr(peer_addr)
    .with_tls(true)
    .with_compression(deps.compression_enabled)
    .with_max_request_body(deps.max_request_body_bytes)
    .with_upstream_timeout(deps.upstream_timeout_secs)
    .with_connection_extensions(connection_extensions);

    if let Some(ref rl) = deps.rate_limiter {
        svc = svc.with_rate_limiter(rl.clone());
    }
    if let Some(ref m) = deps.metrics {
        svc = svc.with_metrics(m.clone());
    }
    if let Some(ref cb) = deps.circuit_breaker {
        svc = svc.with_circuit_breaker(cb.clone());
    }
    if let Some(ref c) = deps.cache {
        svc = svc.with_cache(c.clone());
    }
    if let Some(ref hs) = deps.health_state {
        svc = svc.with_health_state(hs.clone());
    }
    if let Some(ref plugins) = deps.plugins {
        svc = svc.with_plugins(Arc::clone(plugins));
    }

    // Forward through handle_boxed.
    let resp = svc.handle_boxed(proxy_req).await.unwrap();

    // Send response headers via H3.
    let (resp_parts, resp_body) = resp.into_parts();
    let h3_resp = Response::builder()
        .status(resp_parts.status)
        .body(())
        .unwrap();
    // Copy response headers.
    let (mut h3_parts, _) = h3_resp.into_parts();
    h3_parts.headers = resp_parts.headers;
    let h3_resp = Response::from_parts(h3_parts, ());

    stream.send_response(h3_resp).await?;

    // Stream response body to the H3 stream without buffering it all in memory.
    let mut resp_body = resp_body;
    while let Some(frame) = resp_body.frame().await {
        let frame = frame?;
        if let Some(data) = frame.data_ref() {
            if !data.is_empty() {
                stream.send_data(data.clone()).await?;
            }
        }
    }

    stream.finish().await?;

    Ok(())
}

/// Add Alt-Svc header to HTTP/1.1 and HTTP/2 responses to advertise HTTP/3.
pub fn add_alt_svc_header(headers: &mut hyper::header::HeaderMap, port: u16) {
    let alt_svc = format!("h3=\":{}\"; ma=86400", port);
    if let Ok(val) = hyper::header::HeaderValue::from_str(&alt_svc) {
        headers.insert("alt-svc", val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::poll_fn;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use glob::Pattern;
    use http_body_util::Full;
    use hyper::body::Incoming;
    use hyper::header::HeaderValue;
    use hyper::service::service_fn;
    use hyper_util::rt::TokioIo;
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use tokio::net::TcpListener;

    use crate::config::{LbStrategy, RouteConfig};
    use crate::handlers::proxy::build_client;
    use crate::plugin::{Action, ConnectionContext, Plugin, RequestContext};
    use crate::router::Router;
    use crate::tls;

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

    #[test]
    fn append_chunk_with_limit_allows_exact_limit() {
        let mut body = Vec::new();
        append_chunk_with_limit(&mut body, b"hello", 5).unwrap();
        assert_eq!(body, b"hello");
    }

    #[test]
    fn append_chunk_with_limit_rejects_over_limit() {
        let mut body = Vec::new();
        append_chunk_with_limit(&mut body, b"abc", 5).unwrap();
        assert_eq!(
            append_chunk_with_limit(&mut body, b"def", 5),
            Err(H3BodyLimitError::TooLarge)
        );
        assert_eq!(body, b"abc");
    }

    async fn start_upstream<F>(handler: F) -> String
    where
        F: Fn(Request<Incoming>) -> Response<Full<Bytes>> + Send + Sync + Clone + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let handler = handler.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
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

    async fn start_delayed_upstream(delay: Duration) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(
                            io,
                            service_fn(move |_req: Request<Incoming>| async move {
                                tokio::time::sleep(delay).await;
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(StatusCode::OK)
                                        .body(Full::new(Bytes::from("slow-ok")))
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

    async fn start_h3_proxy(
        upstream: String,
        plugins: Option<Arc<PluginChain>>,
        upstream_timeout_secs: u64,
    ) -> (SocketAddr, rustls::pki_types::CertificateDer<'static>) {
        let router = Router::new(vec![(
            Pattern::new("/**").unwrap(),
            RouteConfig {
                servers: vec![upstream],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        )]);
        let router = Arc::new(ArcSwap::from_pointee(router));
        let counter = Arc::new(AtomicUsize::new(0));
        let client = build_client();
        let (certs, key) = tls::generate_self_signed_cert(&["localhost"]).unwrap();
        let cert = certs[0].clone();
        let quic_config = tls::build_quic_server_config(certs, key).unwrap();
        let endpoint = build_quic_endpoint("127.0.0.1:0".parse().unwrap(), quic_config).unwrap();
        let addr = endpoint.local_addr().unwrap();
        let deps = Arc::new(H3Deps {
            router,
            client,
            counter,
            health_state: None,
            upstream_timeout_secs,
            rate_limiter: None,
            metrics: None,
            circuit_breaker: None,
            cache: None,
            plugins,
            compression_enabled: false,
            max_request_body_bytes: 1024 * 1024,
        });
        tokio::spawn(accept_h3_loop(endpoint, deps));
        tokio::task::yield_now().await;
        (addr, cert)
    }

    fn build_h3_client_config(
        _cert: rustls::pki_types::CertificateDer<'static>,
    ) -> quinn::ClientConfig {
        let mut tls = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .unwrap()
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
        tls.alpn_protocols = vec![b"h3".to_vec()];

        let mut client_config = quinn::ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(tls).unwrap(),
        ));
        let mut transport = quinn::TransportConfig::default();
        transport.keep_alive_interval(Some(Duration::from_secs(1)));
        client_config.transport_config(Arc::new(transport));
        client_config
    }

    async fn send_h3_request(
        addr: SocketAddr,
        cert: rustls::pki_types::CertificateDer<'static>,
        path: &str,
    ) -> (u16, Bytes) {
        let mut endpoint = quinn::Endpoint::client("127.0.0.1:0".parse().unwrap()).unwrap();
        endpoint.set_default_client_config(build_h3_client_config(cert));

        let conn = endpoint.connect(addr, "localhost").unwrap().await.unwrap();
        let (driver, mut sender) = h3::client::builder()
            .build::<_, _, Bytes>(h3_quinn::Connection::new(conn))
            .await
            .unwrap();
        tokio::spawn(async move {
            let mut driver = driver;
            let _ = poll_fn(|cx| driver.poll_close(cx)).await;
        });

        let req = Request::builder()
            .uri(format!("https://localhost:{}{}", addr.port(), path))
            .body(())
            .unwrap();
        let mut stream = sender.send_request(req).await.unwrap();
        stream.finish().await.unwrap();

        let resp = stream.recv_response().await.unwrap();
        let status = resp.status().as_u16();
        let mut body = Vec::new();
        while let Some(chunk) = stream.recv_data().await.unwrap() {
            let mut chunk = chunk;
            while chunk.has_remaining() {
                let data = chunk.chunk();
                body.extend_from_slice(data);
                let remaining = data.len();
                chunk.advance(remaining);
            }
        }

        (status, Bytes::from(body))
    }

    #[derive(Clone, Debug)]
    struct AcceptMarker;

    struct H3ParityPlugin;

    #[async_trait]
    impl Plugin for H3ParityPlugin {
        fn name(&self) -> &str {
            "h3_parity_test"
        }

        async fn on_accept(&self, ctx: &mut ConnectionContext) -> Action {
            ctx.extensions.insert(AcceptMarker);
            Action::Continue
        }

        async fn on_request(&self, ctx: &mut RequestContext) -> Action {
            ctx.headers
                .insert("x-plugin-ran", HeaderValue::from_static("yes"));
            ctx.headers.insert(
                "x-conn-marker",
                HeaderValue::from_static(if ctx.connection.get::<AcceptMarker>().is_some() {
                    "yes"
                } else {
                    "no"
                }),
            );
            Action::Continue
        }
    }

    #[tokio::test]
    async fn http3_runs_plugins_and_marks_forwarded_proto_as_https() {
        let upstream = start_upstream(|req: Request<Incoming>| {
            let proto = req
                .headers()
                .get("x-forwarded-proto")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("missing");
            let plugin = req
                .headers()
                .get("x-plugin-ran")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("no");
            let conn = req
                .headers()
                .get("x-conn-marker")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("no");
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from(format!(
                    "proto={proto};plugin={plugin};conn={conn}"
                ))))
                .unwrap()
        })
        .await;
        let plugins = Arc::new(PluginChain::new(vec![Box::new(H3ParityPlugin)]));
        let (addr, cert) = start_h3_proxy(upstream, Some(plugins), 5).await;

        let (status, body) = send_h3_request(addr, cert, "/parity").await;

        assert_eq!(status, 200);
        assert_eq!(
            std::str::from_utf8(&body).unwrap(),
            "proto=https;plugin=yes;conn=yes"
        );
    }

    #[tokio::test]
    async fn http3_respects_configured_upstream_timeout() {
        let upstream = start_delayed_upstream(Duration::from_secs(2)).await;
        let (addr, cert) = start_h3_proxy(upstream, None, 1).await;

        let start = Instant::now();
        let (status, body) = send_h3_request(addr, cert, "/timeout").await;

        assert_eq!(status, 504);
        assert_eq!(body, Bytes::from("504 Gateway Timeout\n"));
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "HTTP/3 request should time out before upstream responds"
        );
    }
}
