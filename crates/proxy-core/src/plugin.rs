use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use hyper::header::HeaderMap;
use hyper::http::Extensions;
use hyper::{Method, Response, StatusCode, Uri, Version};
use proxy_auth::AuthContext;

/// Per-provider upstream timeout override, injected via `ctx.extensions`.
#[derive(Clone, Debug)]
pub struct UpstreamTimeout(pub Duration);

/// A single provider candidate for cross-provider retry.
#[derive(Clone, Debug)]
pub struct ProviderCandidate {
    pub upstream: String,
    pub headers: HeaderMap,
}

/// List of provider candidates, injected by VirtualKeys for cross-provider retry.
#[derive(Clone, Debug, Default)]
pub struct ProviderCandidates(pub Vec<ProviderCandidate>);

/// Error type passed to the `on_error` hook.
#[derive(Debug)]
pub enum ProxyError {
    Timeout,
    ConnectError(String),
    UpstreamError(String),
    ServerError(StatusCode),
    UpstreamStatus(StatusCode),
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProxyError::Timeout => write!(f, "upstream timeout"),
            ProxyError::ConnectError(e) => write!(f, "connect error: {}", e),
            ProxyError::UpstreamError(e) => write!(f, "upstream error: {}", e),
            ProxyError::ServerError(s) => write!(f, "server error: {}", s),
            ProxyError::UpstreamStatus(s) => write!(f, "upstream status: {}", s),
        }
    }
}

/// Action returned by plugin hooks to control request flow.
pub enum Action {
    /// Continue to the next plugin / normal processing.
    Continue,
    /// Short-circuit: return this response immediately to the client.
    Respond(Response<BoxBody<Bytes, hyper::Error>>),
}

/// Context available during the `on_accept` hook (connection level).
pub struct ConnectionContext {
    pub peer_addr: SocketAddr,
    pub tls_client_hello: Option<Vec<u8>>,
    pub extensions: Extensions,
}

/// Context available during request-level hooks (`on_request`, `on_upstream_select`,
/// `on_response`, `on_error`).
pub struct RequestContext {
    pub peer_addr: Option<SocketAddr>,
    pub method: Method,
    pub uri: Uri,
    pub version: Version,
    pub headers: HeaderMap,
    pub body: Option<Bytes>,
    pub route: Option<String>,
    pub selected_upstream: Option<String>,
    pub auth: Option<AuthContext>,
    pub connection: Arc<Extensions>,
    pub extensions: Extensions,
}

/// Context available during the `on_response` hook.
pub struct ResponseContext {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub upstream: String,
    pub duration: Duration,
}

/// Plugin trait with default no-op async methods.
/// Plugins only implement the hooks they need.
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Human-readable plugin name.
    fn name(&self) -> &str;

    /// After TCP accept + TLS handshake, before HTTP parsing.
    async fn on_accept(&self, _ctx: &mut ConnectionContext) -> Action {
        Action::Continue
    }

    /// After HTTP headers parsed, before any processing.
    async fn on_request(&self, _ctx: &mut RequestContext) -> Action {
        Action::Continue
    }

    /// After health/CB filtering, before LB picks upstream.
    async fn on_upstream_select(
        &self,
        _ctx: &mut RequestContext,
        _servers: &mut Vec<&String>,
    ) -> Action {
        Action::Continue
    }

    /// After upstream response received, before sending to client.
    async fn on_response(&self, _ctx: &mut RequestContext, _resp: &mut ResponseContext) -> Action {
        Action::Continue
    }

    /// Transform the upstream response body before downstream response hooks run.
    async fn transform_response(
        &self,
        _ctx: &mut RequestContext,
        resp: Response<BoxBody<Bytes, hyper::Error>>,
    ) -> Response<BoxBody<Bytes, hyper::Error>> {
        resp
    }

    /// On upstream error (timeout, connection refused, 5xx).
    async fn on_error(&self, _ctx: &mut RequestContext, _error: &ProxyError) -> Action {
        Action::Continue
    }

    /// Wrap the response body before sending to the client.
    /// Non-async, default pass-through. Useful for SSE stream inspection.
    fn wrap_response_body(
        &self,
        _ctx: &RequestContext,
        _resp: &ResponseContext,
        body: BoxBody<Bytes, hyper::Error>,
    ) -> BoxBody<Bytes, hyper::Error> {
        body
    }
}

/// Ordered chain of plugins. Iterates in registration order, short-circuits on `Action::Respond`.
pub struct PluginChain {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginChain {
    pub fn new(plugins: Vec<Box<dyn Plugin>>) -> Self {
        Self { plugins }
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    pub async fn run_on_accept(&self, ctx: &mut ConnectionContext) -> Action {
        for plugin in &self.plugins {
            match plugin.on_accept(ctx).await {
                Action::Continue => {}
                action => return action,
            }
        }
        Action::Continue
    }

    pub async fn run_on_request(&self, ctx: &mut RequestContext) -> Action {
        for plugin in &self.plugins {
            match plugin.on_request(ctx).await {
                Action::Continue => {}
                action => return action,
            }
        }
        Action::Continue
    }

    pub async fn run_on_upstream_select(
        &self,
        ctx: &mut RequestContext,
        servers: &mut Vec<&String>,
    ) -> Action {
        for plugin in &self.plugins {
            match plugin.on_upstream_select(ctx, servers).await {
                Action::Continue => {}
                action => return action,
            }
        }
        Action::Continue
    }

    pub async fn run_on_response(
        &self,
        ctx: &mut RequestContext,
        resp: &mut ResponseContext,
    ) -> Action {
        for plugin in &self.plugins {
            match plugin.on_response(ctx, resp).await {
                Action::Continue => {}
                action => return action,
            }
        }
        Action::Continue
    }

    pub async fn run_transform_response(
        &self,
        ctx: &mut RequestContext,
        mut resp: Response<BoxBody<Bytes, hyper::Error>>,
    ) -> Response<BoxBody<Bytes, hyper::Error>> {
        for plugin in &self.plugins {
            resp = plugin.transform_response(ctx, resp).await;
        }
        resp
    }

    pub async fn run_on_error(&self, ctx: &mut RequestContext, error: &ProxyError) -> Action {
        for plugin in &self.plugins {
            match plugin.on_error(ctx, error).await {
                Action::Continue => {}
                action => return action,
            }
        }
        Action::Continue
    }

    pub fn run_wrap_response_body(
        &self,
        ctx: &RequestContext,
        resp: &ResponseContext,
        mut body: BoxBody<Bytes, hyper::Error>,
    ) -> BoxBody<Bytes, hyper::Error> {
        for plugin in &self.plugins {
            body = plugin.wrap_response_body(ctx, resp, body);
        }
        body
    }
}

/// Factory function type for creating plugins from TOML config.
pub type PluginFactory = fn(&toml::Value) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>>;

/// Registry mapping plugin names to their factory functions.
pub struct PluginRegistry {
    factories: Vec<(&'static str, PluginFactory)>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            factories: Vec::new(),
        }
    }

    pub fn register(&mut self, name: &'static str, factory: PluginFactory) {
        self.factories.push((name, factory));
    }

    pub fn create(
        &self,
        name: &str,
        config: &toml::Value,
    ) -> Result<Box<dyn Plugin>, Box<dyn std::error::Error>> {
        for (n, factory) in &self.factories {
            if *n == name {
                return factory(config);
            }
        }
        Err(format!("unknown plugin: {}", name).into())
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}
