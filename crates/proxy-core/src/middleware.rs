use std::net::SocketAddr;

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::header::{HeaderMap, HeaderValue, CONTENT_LENGTH};
use hyper::{Method, Response, StatusCode, Version};

use crate::cache::ResponseCache;
use crate::circuit_breaker::CircuitBreaker;
use crate::metrics::Metrics;
use crate::rate_limit::RateLimiter;

pub type ProxyBody = BoxBody<Bytes, hyper::Error>;

pub fn error_response(status: StatusCode, msg: &'static str) -> Response<ProxyBody> {
    Response::builder()
        .status(status)
        .body(
            Full::new(Bytes::from(msg))
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap()
}

/// Check rate limit. Returns Some(429 response) if the request should be rejected.
pub fn check_rate_limit(
    limiter: &RateLimiter,
    peer_addr: Option<SocketAddr>,
    metrics: Option<&Metrics>,
    method: &Method,
) -> Option<Response<ProxyBody>> {
    let addr = peer_addr?;
    if limiter.check(addr.ip()) {
        return None;
    }
    if let Some(m) = metrics {
        m.requests_total
            .with_label_values(&["429", method.as_str()])
            .inc();
    }
    Some(error_response(
        StatusCode::TOO_MANY_REQUESTS,
        "429 Too Many Requests\n",
    ))
}

/// Check body size limit via Content-Length. Returns Some(413) if oversized.
pub fn check_body_limit(headers: &HeaderMap, max_bytes: u64) -> Option<Response<ProxyBody>> {
    let cl = headers.get(CONTENT_LENGTH)?;
    let size: u64 = cl.to_str().unwrap_or("0").parse().ok()?;
    if size > max_bytes {
        Some(error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "413 Payload Too Large\n",
        ))
    } else {
        None
    }
}

/// Check cache for a HIT. Returns Some(response) on cache hit.
pub fn check_cache(
    cache: &ResponseCache,
    method: &Method,
    path: &str,
    headers: &HeaderMap,
    version: Version,
) -> Option<Response<ProxyBody>> {
    if !ResponseCache::is_cacheable_request(method, headers) {
        return None;
    }
    let (status, mut resp_headers, body) = cache.get_for_request(method, path, headers)?;
    resp_headers.insert("x-cache", HeaderValue::from_static("HIT"));
    resp_headers.append("via", super::handlers::proxy::via_header_value(version));
    let resp = Response::builder()
        .status(status)
        .body(Full::new(body).map_err(|never| match never {}).boxed())
        .unwrap();
    let (parts, body) = resp.into_parts();
    let mut resp = Response::from_parts(parts, body);
    *resp.headers_mut() = resp_headers;
    Some(resp)
}

/// Filter servers by circuit breaker state. Returns the subset of allowed servers.
pub fn filter_by_circuit_breaker<'a>(
    cb: &CircuitBreaker,
    servers: Vec<&'a String>,
) -> Vec<&'a String> {
    servers.into_iter().filter(|s| cb.is_allowed(s)).collect()
}

/// Record request metrics (counter + duration histogram).
pub fn record_metrics(metrics: &Metrics, method: &Method, status: &str, start: std::time::Instant) {
    metrics
        .requests_total
        .with_label_values(&[status, method.as_str()])
        .inc();
    metrics
        .request_duration_seconds
        .with_label_values(&[method.as_str()])
        .observe(start.elapsed().as_secs_f64());
}

/// Add Alt-Svc header for HTTP/3 if configured.
pub fn add_alt_svc(headers: &mut HeaderMap, h3_port: Option<u16>) {
    if let Some(port) = h3_port {
        crate::http3::add_alt_svc_header(headers, port);
    }
}
