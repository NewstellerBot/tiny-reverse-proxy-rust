use std::env;
use std::net::SocketAddr;
use std::process::exit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::Request;
use hyper_util::rt::{TokioIo, TokioTimer};
use socket2::{Domain, Protocol, Socket, Type};
use tokio::net::TcpListener;
use tokio::sync::{watch, Notify};
use tokio_rustls::TlsAcceptor;

use proxy_core::cache::ResponseCache;
use proxy_core::circuit_breaker::CircuitBreaker;
use proxy_core::config::Config;
use proxy_core::handlers::proxy::{build_client, HttpClient, ProxyService};
use proxy_core::health::{spawn_health_checker_with_targets, HealthCheckTargets, HealthState};
use proxy_core::metrics::{self, Metrics};
#[cfg(not(feature = "plugin-llm-gateway"))]
use proxy_core::plugin::PluginRegistry;
use proxy_core::plugin::{self, PluginChain};
use proxy_core::rate_limit::{self, RateLimiter};
use proxy_core::router::{ReloadableRouter, RouteResolver, Router};
use proxy_core::{http3, proxy_protocol, tls};

/// Tracks active connections and notifies when all have completed.
struct ConnectionTracker {
    active: AtomicUsize,
    all_done: Notify,
}

impl ConnectionTracker {
    fn new() -> Self {
        Self {
            active: AtomicUsize::new(0),
            all_done: Notify::new(),
        }
    }

    fn increment(&self) {
        self.active.fetch_add(1, Ordering::SeqCst);
    }

    fn decrement(&self) {
        if self.active.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.all_done.notify_waiters();
        }
    }

    #[cfg(test)]
    fn active_count(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    async fn wait_for_zero(&self) {
        loop {
            let notified = self.all_done.notified();
            if self.active.load(Ordering::SeqCst) == 0 {
                return;
            }
            notified.await;
        }
    }
}

#[cfg(not(tarpaulin_include))]
async fn serve_hyper_connection<I, R>(
    io: I,
    svc: ProxyService<R>,
    mut shutdown_rx: watch::Receiver<bool>,
    header_read_timeout: Duration,
) where
    R: RouteResolver + 'static,
    I: hyper::rt::Read + hyper::rt::Write + Unpin + Send + 'static,
{
    // Use auto::Builder to negotiate HTTP/1.1 or HTTP/2.
    let mut builder =
        hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new());
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(Some(header_read_timeout));
    let conn = builder.serve_connection_with_upgrades(
        io,
        service_fn(move |req: Request<Incoming>| {
            let svc = svc.clone();
            async move { svc.handle(req).await }
        }),
    );

    tokio::pin!(conn);

    tokio::select! {
        result = conn.as_mut() => {
            if let Err(e) = result {
                let msg = e.to_string();
                if !msg.contains("connection closed") {
                    tracing::debug!("Connection error: {}", msg);
                }
            }
        }
        _ = shutdown_rx.changed() => {
            conn.as_mut().graceful_shutdown();
            if let Err(e) = conn.await {
                let msg = e.to_string();
                if !msg.contains("connection closed") {
                    tracing::debug!("Connection error: {}", msg);
                }
            }
        }
    }
}

#[cfg(not(tarpaulin_include))]
fn bind_reuseport(addr: SocketAddr) -> std::io::Result<TcpListener> {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_reuse_address(true)?;
    socket.set_reuse_port(true)?;
    socket.set_nonblocking(true)?;
    socket.bind(&addr.into())?;
    socket.listen(1024)?;
    TcpListener::from_std(socket.into())
}

async fn parse_proxy_peer_addr_with_timeout(
    stream: &mut tokio::net::TcpStream,
    header_read_timeout: Duration,
) -> Result<SocketAddr, Box<dyn std::error::Error + Send + Sync>> {
    match tokio::time::timeout(
        header_read_timeout,
        proxy_protocol::parse_proxy_protocol(stream),
    )
    .await
    {
        Ok(Ok((real_addr, remaining))) => {
            if !remaining.is_empty() {
                return Err("PROXY protocol parser consumed extra bytes".into());
            }
            Ok(real_addr)
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err("PROXY protocol parse timeout".into()),
    }
}

#[cfg(not(tarpaulin_include))]
#[allow(clippy::too_many_arguments)]
async fn accept_loop(
    listener: TcpListener,
    router: Arc<ArcSwap<Router>>,
    client: HttpClient,
    counter: Arc<AtomicUsize>,
    mut shutdown_rx: watch::Receiver<bool>,
    tracker: Arc<ConnectionTracker>,
    health_state: Option<HealthState>,
    tls_acceptor: Option<TlsAcceptor>,
    rate_limiter: Option<RateLimiter>,
    metrics: Option<Metrics>,
    circuit_breaker: Option<CircuitBreaker>,
    cache: Option<ResponseCache>,
    compression_enabled: bool,
    max_request_body_bytes: u64,
    upstream_timeout_secs: u64,
    header_read_timeout: Duration,
    proxy_protocol_enabled: bool,
    h3_port: Option<u16>,
    plugin_chain: Option<Arc<PluginChain>>,
) {
    // Mark initial value as seen so changed() waits for actual shutdown signal.
    shutdown_rx.borrow_and_update();
    let is_tls = tls_acceptor.is_some();
    let live_router = Arc::new(ReloadableRouter::new(Arc::clone(&router)));

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((mut stream, mut peer_addr)) => {
                        // Parse PROXY protocol header if enabled.
                        if proxy_protocol_enabled {
                            match parse_proxy_peer_addr_with_timeout(&mut stream, header_read_timeout).await {
                                Ok(real_addr) => {
                                    peer_addr = real_addr;
                                }
                                Err(e) => {
                                    tracing::debug!("PROXY protocol parse error: {}", e);
                                    continue;
                                }
                            }
                        }

                        // Resolve routes from the live ArcSwap so keep-alive and
                        // HTTP/2 connections observe reloads on subsequent requests.
                        let current_router = Arc::clone(&live_router);
                        let mut svc = ProxyService::new(
                            current_router,
                            client.clone(),
                            Arc::clone(&counter),
                        )
                        .with_peer_addr(peer_addr)
                        .with_tls(is_tls)
                        .with_compression(compression_enabled)
                        .with_max_request_body(max_request_body_bytes)
                        .with_upstream_timeout(upstream_timeout_secs);

                        if let Some(ref hs) = health_state {
                            svc = svc.with_health_state(hs.clone());
                        }
                        if let Some(ref rl) = rate_limiter {
                            svc = svc.with_rate_limiter(rl.clone());
                        }
                        if let Some(ref m) = metrics {
                            svc = svc.with_metrics(m.clone());
                        }
                        if let Some(ref cb) = circuit_breaker {
                            svc = svc.with_circuit_breaker(cb.clone());
                        }
                        if let Some(ref c) = cache {
                            svc = svc.with_cache(c.clone());
                        }
                        if let Some(port) = h3_port {
                            svc = svc.with_h3_port(port);
                        }
                        if let Some(ref pc) = plugin_chain {
                            svc = svc.with_plugins(Arc::clone(pc));
                        }

                        let mut conn_shutdown_rx = shutdown_rx.clone();
                        conn_shutdown_rx.borrow_and_update();
                        let conn_tracker = Arc::clone(&tracker);
                        conn_tracker.increment();

                        if let Some(ref m) = metrics {
                            m.active_connections.inc();
                        }
                        let conn_metrics = metrics.clone();

                        let acceptor = tls_acceptor.clone();
                        let conn_plugins = plugin_chain.clone();
                        let conn_header_read_timeout = header_read_timeout;

                        tokio::spawn(async move {
                            // Plugin: on_accept hook.
                            if let Some(ref plugins) = conn_plugins {
                                let mut conn_ctx = plugin::ConnectionContext {
                                    peer_addr,
                                    tls_client_hello: None,
                                    extensions: hyper::http::Extensions::new(),
                                };
                                if let plugin::Action::Respond(_) = plugins.run_on_accept(&mut conn_ctx).await {
                                    // Plugin short-circuited; drop the connection.
                                    if let Some(ref m) = conn_metrics {
                                        m.active_connections.dec();
                                    }
                                    conn_tracker.decrement();
                                    return;
                                }
                                // Pass connection-level extensions to the service.
                                svc = svc.with_connection_extensions(std::sync::Arc::new(conn_ctx.extensions));
                            }

                            if let Some(acceptor) = acceptor {
                                match acceptor.accept(stream).await {
                                    Ok(tls_stream) => {
                                        serve_hyper_connection(
                                            TokioIo::new(tls_stream),
                                            svc,
                                            conn_shutdown_rx,
                                            conn_header_read_timeout,
                                        )
                                        .await;
                                    }
                                    Err(e) => {
                                        tracing::debug!("TLS handshake failed: {}", e);
                                    }
                                }
                            } else {
                                serve_hyper_connection(
                                    TokioIo::new(stream),
                                    svc,
                                    conn_shutdown_rx,
                                    conn_header_read_timeout,
                                )
                                .await;
                            }

                            if let Some(ref m) = conn_metrics {
                                m.active_connections.dec();
                            }
                            conn_tracker.decrement();
                        });
                    }
                    Err(e) => tracing::error!("Failed to accept connection: {}", e),
                }
            }
            _ = shutdown_rx.changed() => {
                tracing::info!("Accept loop shutting down");
                break;
            }
        }
    }
}

/// Build router from config and set up health checking.
fn build_router(config: &Config) -> Router {
    let routes = config.routes.clone();
    Router::new(routes)
}

/// Build plugin chain from config.
///
/// When the `plugin-llm-gateway` feature is enabled, uses `create_plugins()` to
/// get both the boxed plugins and an `LlmGatewayApi` handle for management.
#[cfg(feature = "plugin-llm-gateway")]
async fn build_plugin_chain(
    config: &Config,
    registry: &prometheus::Registry,
) -> (
    Option<Arc<PluginChain>>,
    Option<plugin_llm_gateway::api::LlmGatewayApi>,
) {
    let enabled: Vec<_> = config.plugins.iter().filter(|p| p.enabled).collect();
    if enabled.is_empty() {
        return (None, None);
    }

    let store_url = config.store_url.as_deref();
    match plugin_llm_gateway::create_plugins_with_options(
        &config.plugins,
        store_url,
        &config.providers,
        &config.model_aliases,
        plugin_llm_gateway::CreatePluginsOptions {
            bootstrap_admin_token: config.management_api_token.clone(),
            allow_direct_provider_keys: config.allow_direct_provider_keys,
        },
        Some(registry),
    )
    .await
    {
        Ok((plugins, api)) => {
            if plugins.is_empty() {
                (None, None)
            } else {
                for p in &plugins {
                    tracing::info!("Plugin loaded: {}", p.name());
                }
                (Some(Arc::new(PluginChain::new(plugins))), Some(api))
            }
        }
        Err(e) => {
            tracing::error!("Failed to create LLM gateway plugins: {}", e);
            (None, None)
        }
    }
}

/// Build plugin chain from config via the plugin registry (no LLM gateway feature).
#[cfg(not(feature = "plugin-llm-gateway"))]
async fn build_plugin_chain(
    config: &Config,
    _registry: &prometheus::Registry,
) -> (Option<Arc<PluginChain>>, ()) {
    let enabled: Vec<_> = config.plugins.iter().filter(|p| p.enabled).collect();
    if enabled.is_empty() {
        return (None, ());
    }

    let registry = PluginRegistry::new();

    let mut chain_plugins: Vec<Box<dyn plugin::Plugin>> = Vec::new();
    for pc in &enabled {
        match registry.create(&pc.name, &pc.config) {
            Ok(p) => {
                tracing::info!("Plugin loaded: {}", pc.name);
                chain_plugins.push(p);
            }
            Err(e) => {
                tracing::error!("Failed to load plugin '{}': {}", pc.name, e);
            }
        }
    }

    if chain_plugins.is_empty() {
        (None, ())
    } else {
        (Some(Arc::new(PluginChain::new(chain_plugins))), ())
    }
}

/// Collect all upstream addresses from config routes.
fn collect_upstreams(config: &Config) -> Vec<String> {
    config
        .routes
        .iter()
        .flat_map(|(_, rc)| rc.servers.iter().cloned())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect()
}

fn non_reloadable_changes(old: &Config, new: &Config) -> Vec<&'static str> {
    let mut changed = Vec::new();
    if old.port != new.port {
        changed.push("port");
    }
    if old.health_check != new.health_check {
        changed.push("health_check");
    }
    if old.tls_cert != new.tls_cert || old.tls_key != new.tls_key || old.tls_auto != new.tls_auto {
        changed.push("tls");
    }
    if old.metrics_port != new.metrics_port {
        changed.push("metrics_port");
    }
    if old.management_api_port != new.management_api_port
        || old.management_api_token != new.management_api_token
    {
        changed.push("management_api");
    }
    if old.store_url != new.store_url {
        changed.push("store_url");
    }
    if old.max_request_body_bytes != new.max_request_body_bytes {
        changed.push("max_request_body_bytes");
    }
    if old.header_read_timeout_secs != new.header_read_timeout_secs {
        changed.push("header_read_timeout_secs");
    }
    if old.upstream_timeout_secs != new.upstream_timeout_secs {
        changed.push("upstream_timeout_secs");
    }
    if old.rate_limit != new.rate_limit {
        changed.push("rate_limit");
    }
    if old.compression_enabled != new.compression_enabled {
        changed.push("compression_enabled");
    }
    if old.circuit_breaker != new.circuit_breaker {
        changed.push("circuit_breaker");
    }
    if old.cache != new.cache {
        changed.push("cache");
    }
    if old.proxy_protocol != new.proxy_protocol {
        changed.push("proxy_protocol");
    }
    if old.plugins != new.plugins {
        changed.push("plugins");
    }
    if old.providers != new.providers {
        changed.push("providers");
    }
    if old.model_aliases != new.model_aliases {
        changed.push("model_aliases");
    }
    changed
}

fn apply_route_reload(
    router: &Arc<ArcSwap<Router>>,
    new_router: Router,
    new_upstreams: &[String],
    health_targets: Option<&HealthCheckTargets>,
    health_state: Option<&HealthState>,
) {
    if let Some(targets) = health_targets {
        targets.replace(new_upstreams.to_vec());
    }
    if let Some(state) = health_state {
        state.sync_upstreams(new_upstreams);
    }
    router.store(Arc::new(new_router));
}

#[cfg(not(tarpaulin_include))]
async fn start_server(config_file: &str) -> std::io::Result<()> {
    // Parse config first so we know whether OTEL is requested.
    let mut config = Config::load_from_file(config_file).unwrap_or_else(|err| {
        eprintln!("Failed to load config: {}", err);
        exit(1);
    });

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        "proxy_core=info,plugin_llm_gateway=info,tiny_reverse_proxy=info,warn"
            .parse()
            .unwrap()
    });

    // Set up tracing subscriber, optionally with OpenTelemetry layer.
    //
    // The OTEL layer uses `Option<Layer>` so both branches produce the same
    // subscriber type, avoiding monomorphisation issues.
    #[cfg(feature = "opentelemetry")]
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        let otel_layer = if config.opentelemetry_enabled {
            Some(proxy_core::otel::init_tracer())
        } else {
            None
        };

        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer())
            .with(otel_layer)
            .init();

        if config.opentelemetry_enabled {
            tracing::info!("OpenTelemetry tracing enabled (OTLP exporter)");
        }
    }
    #[cfg(not(feature = "opentelemetry"))]
    {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    }

    if let Err(errors) = config.validate() {
        for e in &errors {
            tracing::error!("Config validation error: {}", e);
        }
        exit(1);
    }

    let addr: SocketAddr = format!("0.0.0.0:{}", config.port).parse().unwrap();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let router = Arc::new(ArcSwap::from_pointee(build_router(&config)));
    let request_counter = Arc::new(AtomicUsize::new(0));
    let client = build_client();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let tracker = Arc::new(ConnectionTracker::new());

    let all_upstreams = collect_upstreams(&config);

    // Set up TLS (auto-generate or load from files).
    // Retain DER bytes for QUIC config if needed.
    let (tls_acceptor, tls_certs_for_quic) = if config.tls_auto
        && config.tls_cert.is_none()
        && config.tls_key.is_none()
    {
        // Auto-TLS: generate self-signed certificate.
        let hostnames: Vec<&str> = vec!["localhost", "127.0.0.1", "0.0.0.0"];
        match tls::generate_self_signed_cert(&hostnames) {
            Ok((certs, key)) => {
                let fingerprint = tls::cert_fingerprint(&certs[0]);
                tracing::info!("TLS auto-enabled (fingerprint: SHA256:{})", fingerprint);

                // Persist cert so the user can trust it.
                let cert_dir = std::env::current_dir()
                    .unwrap_or_else(|_| std::path::PathBuf::from("."))
                    .join(".tls");
                if std::fs::create_dir_all(&cert_dir).is_ok() {
                    let key_pem = tls::private_key_to_pem(&key);
                    if tls::persist_certs(&cert_dir, &certs, &key_pem).is_ok() {
                        let cert_path = cert_dir.join("cert.pem");
                        tracing::info!("Certificate saved to {}", cert_path.display());
                        #[cfg(target_os = "macos")]
                        tracing::info!(
                            "Trust it with: sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain {}",
                            cert_path.display()
                        );
                        #[cfg(target_os = "linux")]
                        tracing::info!(
                            "Trust it with: sudo cp {} /usr/local/share/ca-certificates/tiny-proxy.crt && sudo update-ca-certificates",
                            cert_path.display()
                        );
                    }
                }

                // Clone for QUIC before consuming.
                let quic_certs = certs.clone();
                let quic_key = key.clone_key();
                match tls::build_tls_acceptor(certs, key) {
                    Ok(acceptor) => (Some(acceptor), Some((quic_certs, quic_key))),
                    Err(e) => {
                        tracing::error!("Failed to build TLS acceptor: {}", e);
                        exit(1);
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to generate self-signed cert: {}", e);
                exit(1);
            }
        }
    } else {
        match (&config.tls_cert, &config.tls_key) {
            (Some(cert_path), Some(key_path)) => {
                // Load certs from PEM files for both TLS and QUIC.
                let (certs, key) =
                    tls::load_tls_material(cert_path, key_path).unwrap_or_else(|e| {
                        tracing::error!("Failed to load TLS material: {}", e);
                        exit(1);
                    });

                let quic_certs = certs.clone();
                let quic_key = key.clone_key();
                let acceptor = tls::build_tls_acceptor(certs, key).unwrap_or_else(|e| {
                    tracing::error!("Failed to build TLS acceptor: {}", e);
                    exit(1);
                });
                tracing::info!("TLS enabled");
                (Some(acceptor), Some((quic_certs, quic_key)))
            }
            (None, None) => (None, None),
            _ => {
                tracing::error!("Both tls_cert and tls_key must be specified for TLS");
                exit(1);
            }
        }
    };

    // Set up Prometheus metrics.
    // Use a shared Registry so both core proxy and LLM metrics are exposed on /metrics.
    let registry = prometheus::Registry::new();
    let metrics = config.metrics_port.map(|port| {
        let m = Metrics::new_with_registry(registry.clone());
        let m_clone = m.clone();
        tokio::spawn(async move {
            metrics::start_metrics_server(port, m_clone).await;
        });
        tracing::info!("Metrics server listening on 0.0.0.0:{}", port);
        m
    });

    // Set up health checking if configured.
    let (health_targets, health_state) = if let Some(hc_config) = config.health_check.as_ref() {
        let targets = HealthCheckTargets::new(all_upstreams.clone());
        let state = match metrics.clone() {
            Some(m) => HealthState::new(&all_upstreams).with_metrics(m),
            None => HealthState::new(&all_upstreams),
        };
        spawn_health_checker_with_targets(targets.clone(), state.clone(), hc_config);
        (Some(targets), Some(state))
    } else {
        (None, None)
    };

    // Set up rate limiting.
    let rate_limiter = config.rate_limit.as_ref().map(|rl_config| {
        let limiter = RateLimiter::new(rl_config.requests_per_second, rl_config.burst);
        rate_limit::spawn_cleanup_task(limiter.clone());
        tracing::info!(
            "Rate limiting enabled: {}/s, burst {}",
            rl_config.requests_per_second,
            rl_config.burst
        );
        limiter
    });

    // Set up circuit breaker.
    let circuit_breaker = config.circuit_breaker.as_ref().map(|cb_config| {
        tracing::info!(
            "Circuit breaker enabled: threshold={}, cooldown={}s, window={}s",
            cb_config.failure_threshold,
            cb_config.cooldown_secs,
            cb_config.window_secs
        );
        CircuitBreaker::new(
            cb_config.failure_threshold,
            cb_config.cooldown_secs,
            cb_config.window_secs,
        )
    });

    // Set up response cache.
    let cache = config.cache.as_ref().map(|cache_config| {
        tracing::info!(
            "Response cache enabled: max_size={}MB, default_ttl={}s",
            cache_config.max_size_mb,
            cache_config.default_ttl_secs
        );
        ResponseCache::new(cache_config.max_size_mb, cache_config.default_ttl_secs)
    });

    // Build plugin chain from config.
    let (_plugin_chain_result, _api_handle) = build_plugin_chain(&config, &registry).await;

    #[cfg(feature = "plugin-llm-gateway")]
    let plugin_chain = _plugin_chain_result;
    #[cfg(not(feature = "plugin-llm-gateway"))]
    let plugin_chain = _plugin_chain_result;

    // Start management API server if configured and LLM gateway is enabled.
    #[cfg(feature = "plugin-llm-gateway")]
    if let (Some(port), Some(ref api)) = (config.management_api_port, &_api_handle) {
        if !api.has_admin_access_path() {
            tracing::error!(
                "management API requires a bootstrap admin token or a stored instance_admin token"
            );
            exit(1);
        }
        let api_clone = api.clone();
        tokio::spawn(async move {
            plugin_llm_gateway::management_server::start_management_server_with_auth(
                port, api_clone, None,
            )
            .await;
        });
        tracing::info!("Management API server listening on 127.0.0.1:{}", port);
    }

    let compression_enabled = config.compression_enabled;
    let max_request_body_bytes = config.max_request_body_bytes;
    let upstream_timeout_secs = config.upstream_timeout_secs;
    let header_read_timeout = Duration::from_secs(config.header_read_timeout_secs);
    let proxy_protocol_enabled = config.proxy_protocol;

    // Determine H3 port (same as main port if TLS is enabled).
    let h3_port = if tls_acceptor.is_some() {
        Some(config.port)
    } else {
        None
    };

    tracing::info!("Listening on {} with {} accept workers...", addr, workers);

    for _ in 0..workers {
        let listener = bind_reuseport(addr)?;
        let router = Arc::clone(&router);
        let client = client.clone();
        let counter = Arc::clone(&request_counter);
        let rx = shutdown_rx.clone();
        let tracker = Arc::clone(&tracker);
        let hs = health_state.clone();
        let tls = tls_acceptor.clone();
        let rl = rate_limiter.clone();
        let m = metrics.clone();
        let cb = circuit_breaker.clone();
        let c = cache.clone();
        let pc = plugin_chain.clone();
        tokio::spawn(accept_loop(
            listener,
            router,
            client,
            counter,
            rx,
            tracker,
            hs,
            tls,
            rl,
            m,
            cb,
            c,
            compression_enabled,
            max_request_body_bytes,
            upstream_timeout_secs,
            header_read_timeout,
            proxy_protocol_enabled,
            h3_port,
            pc,
        ));
    }

    // Launch HTTP/3 endpoint if TLS is configured.
    if let Some((quic_certs, quic_key)) = tls_certs_for_quic {
        match tls::build_quic_server_config(quic_certs, quic_key) {
            Ok(quic_config) => match http3::build_quic_endpoint(addr, quic_config) {
                Ok(endpoint) => {
                    let h3_deps = Arc::new(http3::H3Deps {
                        router: Arc::clone(&router),
                        client: client.clone(),
                        counter: Arc::clone(&request_counter),
                        health_state: health_state.clone(),
                        upstream_timeout_secs,
                        rate_limiter: rate_limiter.clone(),
                        metrics: metrics.clone(),
                        circuit_breaker: circuit_breaker.clone(),
                        cache: cache.clone(),
                        plugins: plugin_chain.clone(),
                        compression_enabled,
                        max_request_body_bytes,
                    });
                    tokio::spawn(http3::accept_h3_loop(endpoint, h3_deps));
                    tracing::info!("HTTP/3 (QUIC) enabled on {}", addr);
                }
                Err(e) => {
                    tracing::warn!("Failed to start HTTP/3 endpoint: {}", e);
                }
            },
            Err(e) => {
                tracing::warn!("Failed to build QUIC TLS config: {}", e);
            }
        }
    }

    // Drop our copy so only workers hold receivers.
    drop(shutdown_rx);

    // Set up signal handlers: SIGHUP for config reload, SIGINT/SIGTERM for shutdown.
    let mut sighup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
        .expect("failed to install SIGHUP handler");
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to install SIGTERM handler");

    let config_file_owned = config_file.to_string();

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Received SIGINT, initiating graceful shutdown...");
                break;
            }
            _ = sigterm.recv() => {
                tracing::info!("Received SIGTERM, initiating graceful shutdown...");
                break;
            }
            _ = sighup.recv() => {
                tracing::info!("Received SIGHUP, reloading configuration...");
                match Config::load_from_file(&config_file_owned) {
                    Ok(new_config) => {
                        if let Err(errors) = new_config.validate() {
                            for e in &errors {
                                tracing::error!("Config reload validation error: {}", e);
                            }
                            tracing::error!("Reload aborted due to invalid config");
                            continue;
                        }

                        let non_reloadable = non_reloadable_changes(&config, &new_config);
                        if !non_reloadable.is_empty() {
                            tracing::warn!(
                                "SIGHUP only reloads routes. Restart required for changed settings: {}",
                                non_reloadable.join(", ")
                            );
                        }

                        let new_upstreams = collect_upstreams(&new_config);
                        let new_router = build_router(&new_config);
                        apply_route_reload(
                            &router,
                            new_router,
                            &new_upstreams,
                            health_targets.as_ref(),
                            health_state.as_ref(),
                        );
                        config = new_config;
                        tracing::info!("Configuration reloaded successfully");
                    }
                    Err(e) => {
                        tracing::error!("Failed to reload config: {}, keeping current config", e);
                    }
                }
            }
        }
    }

    // Signal shutdown to all accept loops and connections.
    let _ = shutdown_tx.send(true);

    // Final flush of LLM gateway state before shutdown.
    #[cfg(feature = "plugin-llm-gateway")]
    if let Some(ref api) = _api_handle {
        tracing::info!("Flushing LLM gateway state to store...");
        api.flush().await;
    }

    // Wait for in-flight connections to drain, with a deadline.
    tracing::info!("Draining in-flight connections (30s deadline)...");
    match tokio::time::timeout(Duration::from_secs(30), tracker.wait_for_zero()).await {
        Ok(()) => tracing::info!("All connections drained, shutdown complete"),
        Err(_) => tracing::warn!("Drain deadline exceeded, forcing shutdown"),
    }

    // Flush and shut down OpenTelemetry tracer provider.
    #[cfg(feature = "opentelemetry")]
    if config.opentelemetry_enabled {
        proxy_core::otel::shutdown_tracer();
    }

    Ok(())
}

#[cfg(not(tarpaulin_include))]
#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <config file>", args[0]);
        exit(0);
    }

    if let Err(e) = start_server(&args[1]).await {
        eprintln!("Server error: {}", e);
        exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures_util::{SinkExt, StreamExt};
    use glob::Pattern;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::header::{AUTHORIZATION, CONTENT_TYPE};
    use hyper::service::service_fn;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use proxy_core::config::{LbStrategy, RouteConfig};
    use proxy_core::router::PathResolution;
    use serde_json::{json, Value};
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpStream;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::{http::StatusCode as WsStatusCode, Message};
    use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
    use trp_test_support::{
        openai_api_key, openai_organization, openai_project, openai_realtime_model,
        openai_realtime_timeout, openai_responses_model, openai_responses_timeout,
    };

    type LiveWs = WebSocketStream<MaybeTlsStream<TcpStream>>;

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

    fn request_id(prefix: &str) -> String {
        format!(
            "{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        )
    }

    async fn start_openai_proxy() -> (SocketAddr, watch::Sender<bool>) {
        let router = Arc::new(ArcSwap::from_pointee(Router::new(vec![(
            Pattern::new("/v1/*").unwrap(),
            RouteConfig {
                servers: vec!["https://api.openai.com".to_string()],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        )])));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        let client = build_client();
        let counter = Arc::new(AtomicUsize::new(0));
        let tracker = Arc::new(ConnectionTracker::new());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(accept_loop(
            listener,
            router,
            client,
            counter,
            shutdown_rx,
            tracker,
            None,
            None,
            None,
            None,
            None,
            None,
            true,
            1024 * 1024,
            30,
            Duration::from_secs(10),
            false,
            None,
            None,
        ));

        tokio::task::yield_now().await;
        (proxy_addr, shutdown_tx)
    }

    async fn connect_openai_realtime_proxy(proxy_addr: SocketAddr) -> LiveWs {
        let api_key = openai_api_key("run the live OpenAI smoke tests");
        let model = openai_realtime_model();
        let url = format!("ws://{proxy_addr}/v1/realtime?model={model}");
        let mut request = url.into_client_request().unwrap();
        request.headers_mut().insert(
            "Authorization",
            format!("Bearer {api_key}").parse().unwrap(),
        );

        if let Some(org) = openai_organization() {
            request
                .headers_mut()
                .insert("OpenAI-Organization", org.parse().unwrap());
        }
        if let Some(project) = openai_project() {
            request
                .headers_mut()
                .insert("OpenAI-Project", project.parse().unwrap());
        }

        request.headers_mut().insert(
            "X-Client-Request-Id",
            request_id("trp-realtime").parse().unwrap(),
        );

        let (ws, response) = tokio_tungstenite::connect_async(request).await.unwrap();
        assert_eq!(response.status(), WsStatusCode::SWITCHING_PROTOCOLS);
        ws
    }

    async fn post_openai_proxy_json(
        proxy_addr: SocketAddr,
        path: &str,
        body_json: Value,
    ) -> Response<Incoming> {
        let api_key = openai_api_key("run the live OpenAI smoke tests");
        let client = build_client();
        let mut builder = Request::builder()
            .method("POST")
            .uri(format!("http://{proxy_addr}{path}"))
            .header(AUTHORIZATION, format!("Bearer {api_key}"))
            .header(CONTENT_TYPE, "application/json")
            .header("X-Client-Request-Id", request_id("trp-http"));

        if let Some(org) = openai_organization() {
            builder = builder.header("OpenAI-Organization", org);
        }
        if let Some(project) = openai_project() {
            builder = builder.header("OpenAI-Project", project);
        }

        let request = builder
            .body(
                Full::new(Bytes::from(body_json.to_string()))
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        client.request(request).await.unwrap()
    }

    async fn next_realtime_event(ws: &mut LiveWs, timeout: Duration) -> Value {
        loop {
            let next = tokio::time::timeout(timeout, ws.next())
                .await
                .expect("timed out waiting for Realtime event");
            let message = next
                .expect("Realtime socket closed unexpectedly")
                .expect("Realtime socket returned an error");

            match message {
                Message::Text(text) => {
                    return serde_json::from_str(text.as_ref())
                        .unwrap_or_else(|e| panic!("invalid Realtime JSON event: {e}"));
                }
                Message::Binary(_) => continue,
                Message::Ping(payload) => {
                    ws.send(Message::Pong(payload)).await.unwrap();
                }
                Message::Pong(_) => continue,
                Message::Close(frame) => {
                    panic!("Realtime socket closed before test completed: {frame:?}")
                }
                Message::Frame(_) => continue,
            }
        }
    }

    fn response_output_text(event: &Value) -> Option<String> {
        let mut text = String::new();
        for item in event.pointer("/response/output")?.as_array()? {
            for part in item.get("content")?.as_array()? {
                if let Some(fragment) = part.get("text").and_then(Value::as_str) {
                    text.push_str(fragment);
                }
            }
        }

        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    fn response_items_text(items: &[Value]) -> Option<String> {
        let mut text = String::new();

        for item in items {
            for part in item
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(fragment) = part.get("text").and_then(Value::as_str) {
                    text.push_str(fragment);
                }
            }
        }

        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    fn responses_output_text(value: &Value) -> Option<String> {
        if let Some(text) = value.get("output_text").and_then(Value::as_str) {
            if !text.trim().is_empty() {
                return Some(text.to_string());
            }
        }

        if let Some(items) = value.get("output").and_then(Value::as_array) {
            if let Some(text) = response_items_text(items) {
                return Some(text);
            }
        }

        value.get("response").and_then(responses_output_text)
    }

    fn parse_sse_events(body: &str) -> Vec<Value> {
        body.split("\n\n")
            .filter_map(|chunk| {
                let mut data_lines = Vec::new();
                for line in chunk.lines() {
                    let trimmed = line.trim_end();
                    if let Some(data) = trimmed.strip_prefix("data:") {
                        data_lines.push(data.trim().to_string());
                    }
                }

                if data_lines.is_empty() {
                    return None;
                }

                let payload = data_lines.join("\n");
                if payload == "[DONE]" {
                    return None;
                }

                Some(
                    serde_json::from_str::<Value>(&payload).unwrap_or_else(|e| {
                        panic!("invalid SSE JSON payload: {e}; payload={payload}")
                    }),
                )
            })
            .collect()
    }

    fn responses_sse_output_text(events: &[Value]) -> Option<String> {
        let mut text = String::new();

        for event in events {
            let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");
            match event_type {
                "response.output_text.delta" | "response.text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                        text.push_str(delta);
                    }
                }
                "response.output_text.done" | "response.text.done" => {
                    if text.trim().is_empty() {
                        if let Some(done_text) = event.get("text").and_then(Value::as_str) {
                            text.push_str(done_text);
                        }
                    }
                }
                "response.completed" | "response.done" => {
                    if text.trim().is_empty() {
                        if let Some(done_text) = responses_output_text(event) {
                            text.push_str(&done_text);
                        }
                    }
                }
                "error" => panic!("Responses API returned error event: {}", event),
                _ => {}
            }
        }

        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    async fn wait_for_event_type(ws: &mut LiveWs, expected: &str, timeout: Duration) -> Value {
        loop {
            let event = next_realtime_event(ws, timeout).await;
            let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");

            if event_type == "error" {
                panic!("Realtime API returned an error event: {}", event);
            }

            if event_type == expected {
                return event;
            }
        }
    }

    async fn wait_for_any_event_type(
        ws: &mut LiveWs,
        expected: &[&str],
        timeout: Duration,
    ) -> Value {
        loop {
            let event = next_realtime_event(ws, timeout).await;
            let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");

            if event_type == "error" {
                panic!("Realtime API returned an error event: {}", event);
            }

            if expected.contains(&event_type) {
                return event;
            }
        }
    }

    async fn wait_for_response_text(ws: &mut LiveWs, timeout: Duration) -> String {
        let mut text = String::new();

        loop {
            let event = next_realtime_event(ws, timeout).await;
            let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");

            match event_type {
                "error" => panic!("Realtime API returned an error event: {}", event),
                "response.output_text.delta" | "response.text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                        text.push_str(delta);
                    }
                }
                "response.output_text.done" | "response.text.done" => {
                    if text.is_empty() {
                        if let Some(done_text) = event.get("text").and_then(Value::as_str) {
                            text.push_str(done_text);
                        }
                    }
                }
                "response.done" => {
                    let status = event
                        .pointer("/response/status")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    assert_eq!(
                        status, "completed",
                        "unexpected response.done payload: {}",
                        event
                    );

                    if text.is_empty() {
                        if let Some(done_text) = response_output_text(&event) {
                            text = done_text;
                        }
                    }

                    assert!(
                        !text.trim().is_empty(),
                        "Realtime response completed without any text output: {}",
                        event
                    );
                    return text;
                }
                _ => {}
            }
        }
    }

    #[tokio::test]
    async fn test_connection_tracker_increment_decrement() {
        let tracker = ConnectionTracker::new();
        assert_eq!(tracker.active_count(), 0);

        tracker.increment();
        assert_eq!(tracker.active_count(), 1);

        tracker.increment();
        assert_eq!(tracker.active_count(), 2);

        tracker.decrement();
        assert_eq!(tracker.active_count(), 1);

        tracker.decrement();
        assert_eq!(tracker.active_count(), 0);
    }

    #[tokio::test]
    async fn test_connection_tracker_wait_for_zero_immediate() {
        let tracker = ConnectionTracker::new();
        // Should return immediately when count is already zero.
        tracker.wait_for_zero().await;
    }

    #[tokio::test]
    async fn test_connection_tracker_wait_for_zero_with_drain() {
        let tracker = Arc::new(ConnectionTracker::new());
        tracker.increment();
        tracker.increment();

        let t = Arc::clone(&tracker);
        let handle = tokio::spawn(async move {
            t.wait_for_zero().await;
        });

        // Give the spawned task time to start waiting.
        tokio::task::yield_now().await;

        tracker.decrement();
        tracker.decrement();

        handle.await.unwrap();
        assert_eq!(tracker.active_count(), 0);
    }

    #[tokio::test]
    async fn test_shutdown_watch_channel_propagation() {
        let (tx, rx) = watch::channel(false);
        let mut rx1 = rx.clone();
        let mut rx2 = rx.clone();
        drop(rx);

        rx1.borrow_and_update();
        rx2.borrow_and_update();

        let h1 = tokio::spawn(async move {
            let _ = rx1.changed().await;
            *rx1.borrow()
        });

        let h2 = tokio::spawn(async move {
            let _ = rx2.changed().await;
            *rx2.borrow()
        });

        tokio::task::yield_now().await;

        // Signal shutdown.
        tx.send(true).unwrap();

        assert!(h1.await.unwrap());
        assert!(h2.await.unwrap());
    }

    #[tokio::test]
    async fn test_parse_proxy_peer_addr_with_timeout_success() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            stream
                .write_all(b"PROXY TCP4 203.0.113.1 198.51.100.2 42300 443\r\n")
                .await
                .unwrap();
        });

        let (mut server_stream, _) = listener.accept().await.unwrap();
        let parsed =
            parse_proxy_peer_addr_with_timeout(&mut server_stream, Duration::from_millis(200))
                .await
                .unwrap();
        assert_eq!(parsed.ip().to_string(), "203.0.113.1");
        assert_eq!(parsed.port(), 42300);

        client.await.unwrap();
    }

    #[tokio::test]
    async fn test_parse_proxy_peer_addr_with_timeout_rejects_slow_header() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            stream.write_all(b"PROXY TCP4 203.0.113.1").await.unwrap();
            tokio::time::sleep(Duration::from_millis(250)).await;
        });

        let (mut server_stream, _) = listener.accept().await.unwrap();
        let err = parse_proxy_peer_addr_with_timeout(&mut server_stream, Duration::from_millis(50))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("timeout"),
            "expected timeout parse error, got: {}",
            err
        );

        client.await.unwrap();
    }

    #[test]
    fn test_apply_route_reload_updates_router_and_health_tracking() {
        let initial_router = Router::new(vec![(
            Pattern::new("/**").unwrap(),
            RouteConfig {
                servers: vec!["old:80".to_string()],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        )]);
        let router = Arc::new(ArcSwap::from_pointee(initial_router));
        let health_targets = HealthCheckTargets::new(vec!["old:80".to_string()]);
        let health_state = HealthState::new(&["old:80".to_string()]);
        health_state.mark_unhealthy("old:80");

        let new_upstreams = vec!["new:80".to_string()];
        let new_router = Router::new(vec![(
            Pattern::new("/**").unwrap(),
            RouteConfig {
                servers: new_upstreams.clone(),
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        )]);

        apply_route_reload(
            &router,
            new_router,
            &new_upstreams,
            Some(&health_targets),
            Some(&health_state),
        );

        let route = router.load();
        assert_eq!(
            route.get_route_config("/test").unwrap().servers,
            new_upstreams
        );
        assert_eq!(health_targets.snapshot(), vec!["new:80".to_string()]);
        assert!(health_state.is_healthy("new:80"));
        assert!(!health_state.is_healthy("old:80"));
    }

    #[tokio::test]
    async fn test_route_reload_applies_on_existing_keep_alive_connection() {
        let old_upstream = start_upstream("old-route").await;
        let new_upstream = start_upstream("new-route").await;

        let router = Arc::new(ArcSwap::from_pointee(Router::new(vec![(
            Pattern::new("/**").unwrap(),
            RouteConfig {
                servers: vec![old_upstream],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        )])));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        let client = build_client();
        let counter = Arc::new(AtomicUsize::new(0));
        let tracker = Arc::new(ConnectionTracker::new());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(accept_loop(
            listener,
            Arc::clone(&router),
            client,
            counter,
            shutdown_rx,
            tracker,
            None,
            None,
            None,
            None,
            None,
            None,
            true,
            1024 * 1024,
            5,
            Duration::from_secs(5),
            false,
            None,
            None,
        ));

        let stream = TcpStream::connect(proxy_addr).await.unwrap();
        let io = TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();
        tokio::spawn(async move {
            let _ = conn.await;
        });

        let req = Request::builder()
            .uri("/reload")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = sender.send_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.into_body().collect().await.unwrap().to_bytes(),
            Bytes::from_static(b"old-route")
        );

        router.store(Arc::new(Router::new(vec![(
            Pattern::new("/**").unwrap(),
            RouteConfig {
                servers: vec![new_upstream],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        )])));
        tokio::task::yield_now().await;

        let req = Request::builder()
            .uri("/reload")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = sender.send_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.into_body().collect().await.unwrap().to_bytes(),
            Bytes::from_static(b"new-route")
        );

        let _ = shutdown_tx.send(true);
    }

    #[tokio::test]
    async fn test_route_reload_applies_on_existing_http2_connection() {
        let old_upstream = start_upstream("old-h2-route").await;
        let new_upstream = start_upstream("new-h2-route").await;

        let router = Arc::new(ArcSwap::from_pointee(Router::new(vec![(
            Pattern::new("/**").unwrap(),
            RouteConfig {
                servers: vec![old_upstream],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        )])));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        let client = build_client();
        let counter = Arc::new(AtomicUsize::new(0));
        let tracker = Arc::new(ConnectionTracker::new());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(accept_loop(
            listener,
            Arc::clone(&router),
            client,
            counter,
            shutdown_rx,
            tracker,
            None,
            None,
            None,
            None,
            None,
            None,
            true,
            1024 * 1024,
            5,
            Duration::from_secs(5),
            false,
            None,
            None,
        ));

        let stream = TcpStream::connect(proxy_addr).await.unwrap();
        let io = TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http2::handshake(TokioExecutor::new(), io)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = conn.await;
        });

        let req = Request::builder()
            .uri("http://localhost/reload")
            .version(hyper::Version::HTTP_2)
            .header("host", "localhost")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = sender.send_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.into_body().collect().await.unwrap().to_bytes(),
            Bytes::from_static(b"old-h2-route")
        );

        router.store(Arc::new(Router::new(vec![(
            Pattern::new("/**").unwrap(),
            RouteConfig {
                servers: vec![new_upstream],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        )])));
        tokio::task::yield_now().await;

        let req = Request::builder()
            .uri("http://localhost/reload")
            .version(hyper::Version::HTTP_2)
            .header("host", "localhost")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = sender.send_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.into_body().collect().await.unwrap().to_bytes(),
            Bytes::from_static(b"new-h2-route")
        );

        let _ = shutdown_tx.send(true);
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and live OpenAI Realtime access"]
    async fn openai_realtime_proxy_connects_and_receives_session_created() {
        let (proxy_addr, shutdown_tx) = start_openai_proxy().await;
        let mut ws = connect_openai_realtime_proxy(proxy_addr).await;

        let session_created =
            wait_for_event_type(&mut ws, "session.created", openai_realtime_timeout()).await;
        assert!(
            session_created.get("session").is_some(),
            "session.created must include session payload: {}",
            session_created
        );

        ws.close(None).await.unwrap();
        let _ = shutdown_tx.send(true);
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and live OpenAI Realtime access"]
    async fn openai_realtime_proxy_can_generate_text_response() {
        let (proxy_addr, shutdown_tx) = start_openai_proxy().await;
        let mut ws = connect_openai_realtime_proxy(proxy_addr).await;
        let timeout = openai_realtime_timeout();

        let _ = wait_for_event_type(&mut ws, "session.created", timeout).await;

        ws.send(Message::Text(
            json!({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": "Reply with exactly PONG and nothing else.",
                    "output_modalities": ["text"]
                }
            })
            .to_string(),
        ))
        .await
        .unwrap();
        let _ = wait_for_event_type(&mut ws, "session.updated", timeout).await;

        ws.send(Message::Text(
            json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Reply with exactly PONG."
                        }
                    ]
                }
            })
            .to_string(),
        ))
        .await
        .unwrap();
        let _ = wait_for_any_event_type(
            &mut ws,
            &[
                "conversation.item.created",
                "conversation.item.added",
                "conversation.item.done",
            ],
            timeout,
        )
        .await;

        ws.send(Message::Text(
            json!({
                "type": "response.create",
                "response": {
                    "instructions": "Reply with exactly PONG and nothing else.",
                    "output_modalities": ["text"]
                }
            })
            .to_string(),
        ))
        .await
        .unwrap();

        let response_text = wait_for_response_text(&mut ws, timeout).await;
        eprintln!("Realtime text response: {response_text}");
        assert!(
            response_text.to_ascii_uppercase().contains("PONG"),
            "expected response to contain PONG, got: {response_text}"
        );

        ws.close(None).await.unwrap();
        let _ = shutdown_tx.send(true);
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and live OpenAI Responses API access"]
    async fn openai_responses_proxy_can_generate_text_response() {
        let (proxy_addr, shutdown_tx) = start_openai_proxy().await;
        let timeout = openai_responses_timeout();

        let response = tokio::time::timeout(
            timeout,
            post_openai_proxy_json(
                proxy_addr,
                "/v1/responses",
                json!({
                    "model": openai_responses_model(),
                    "input": "Reply with exactly PONG and nothing else."
                }),
            ),
        )
        .await
        .expect("timed out waiting for non-streaming Responses API request");

        let status = response.status();
        let body_bytes = tokio::time::timeout(timeout, response.into_body().collect())
            .await
            .expect("timed out collecting non-streaming Responses API body")
            .unwrap()
            .to_bytes();
        let body_text = String::from_utf8(body_bytes.to_vec()).unwrap();

        assert_eq!(
            status,
            StatusCode::OK,
            "unexpected status with body: {body_text}"
        );

        let response_json: Value = serde_json::from_str(&body_text).unwrap_or_else(|e| {
            panic!("invalid Responses JSON body: {e}; body={body_text}");
        });
        let output_text = responses_output_text(&response_json)
            .unwrap_or_else(|| panic!("Responses body missing output text: {response_json}"));
        eprintln!("Responses text response: {output_text}");
        assert!(
            output_text.to_ascii_uppercase().contains("PONG"),
            "expected response to contain PONG, got: {output_text}"
        );

        let _ = shutdown_tx.send(true);
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and live OpenAI Responses API access"]
    async fn openai_responses_proxy_can_stream_text_response() {
        let (proxy_addr, shutdown_tx) = start_openai_proxy().await;
        let timeout = openai_responses_timeout();

        let response = tokio::time::timeout(
            timeout,
            post_openai_proxy_json(
                proxy_addr,
                "/v1/responses",
                json!({
                    "model": openai_responses_model(),
                    "input": "Reply with exactly PONG and nothing else.",
                    "stream": true
                }),
            ),
        )
        .await
        .expect("timed out waiting for streaming Responses API request");

        let status = response.status();
        let body_bytes = tokio::time::timeout(timeout, response.into_body().collect())
            .await
            .expect("timed out collecting streaming Responses API body")
            .unwrap()
            .to_bytes();
        let body_text = String::from_utf8(body_bytes.to_vec()).unwrap();

        assert_eq!(
            status,
            StatusCode::OK,
            "unexpected status with body: {body_text}"
        );
        assert!(
            body_text.contains("response.output_text.delta")
                || body_text.contains("response.completed")
                || body_text.contains("response.done"),
            "expected streaming SSE events, got: {body_text}"
        );

        let events = parse_sse_events(&body_text);
        assert!(
            !events.is_empty(),
            "expected at least one SSE event in body: {body_text}"
        );
        let output_text = responses_sse_output_text(&events)
            .unwrap_or_else(|| panic!("Responses stream missing output text: {body_text}"));
        eprintln!("Responses streaming text response: {output_text}");
        assert!(
            output_text.to_ascii_uppercase().contains("PONG"),
            "expected stream to contain PONG, got: {output_text}"
        );

        let _ = shutdown_tx.send(true);
    }
}
