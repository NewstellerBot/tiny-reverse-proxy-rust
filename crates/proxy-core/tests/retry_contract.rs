use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, StatusCode};
use tokio::time::sleep;

use proxy_core::config::ReliabilityConfig;
use proxy_core::metrics::Metrics;
use proxy_core::plugin::{
    Action, Plugin, PluginChain, ProviderCandidate, ProviderCandidates, RequestContext,
};
use trp_test_support::{
    catch_all_router, send_request, start_proxy_with_config, start_upstream, start_upstream_async,
    TestProxyConfig,
};

#[derive(Clone)]
struct CandidateModePlugin {
    candidate_upstream: Option<String>,
}

#[async_trait]
impl Plugin for CandidateModePlugin {
    fn name(&self) -> &str {
        "candidate-mode-test"
    }

    async fn on_upstream_select(
        &self,
        ctx: &mut RequestContext,
        servers: &mut Vec<&String>,
    ) -> Action {
        servers.clear();
        let candidates = self
            .candidate_upstream
            .as_ref()
            .map(|upstream| ProviderCandidate {
                upstream: upstream.clone(),
                headers: ctx.headers.clone(),
            })
            .into_iter()
            .collect();
        ctx.extensions.insert(ProviderCandidates(candidates));
        Action::Continue
    }
}

fn get_request(path: &str) -> Request<Full<Bytes>> {
    Request::builder()
        .method("GET")
        .uri(path)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

fn post_request(path: &str) -> Request<Full<Bytes>> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from_static(br#"{"hello":"world"}"#)))
        .unwrap()
}

#[tokio::test]
async fn core_timeout_is_classified_without_retry() {
    let timeout_hits = Arc::new(AtomicUsize::new(0));
    let timeout_addr = start_upstream_async({
        let timeout_hits = Arc::clone(&timeout_hits);
        move |_req| {
            let timeout_hits = Arc::clone(&timeout_hits);
            async move {
                timeout_hits.fetch_add(1, Ordering::Relaxed);
                sleep(Duration::from_millis(150)).await;
                hyper::Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from_static(b"late-ok")))
                    .unwrap()
            }
        }
    })
    .await;
    let backup_hits = Arc::new(AtomicUsize::new(0));
    let backup_addr = start_upstream({
        let backup_hits = Arc::clone(&backup_hits);
        move |_req| {
            backup_hits.fetch_add(1, Ordering::Relaxed);
            hyper::Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from_static(b"backup-ok")))
                .unwrap()
        }
    })
    .await;

    let metrics = Metrics::new();
    let proxy_addr = start_proxy_with_config(
        catch_all_router(vec![timeout_addr, backup_addr]),
        TestProxyConfig {
            metrics: Some(metrics.clone()),
            upstream_timeout_secs: 0,
            ..Default::default()
        },
    )
    .await;

    let response = send_request(&proxy_addr, get_request("/timeout")).await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(timeout_hits.load(Ordering::Relaxed), 1);
    assert_eq!(backup_hits.load(Ordering::Relaxed), 0);
    assert_eq!(
        metrics
            .retry_attempts_total
            .with_label_values(&["core", "timeout"])
            .get(),
        0
    );
    assert_eq!(
        metrics
            .retry_exhaustions_total
            .with_label_values(&["core", "timeout"])
            .get(),
        1
    );
}

#[tokio::test]
async fn core_connection_error_retries_next_upstream() {
    let success_hits = Arc::new(AtomicUsize::new(0));
    let success_addr = start_upstream({
        let success_hits = Arc::clone(&success_hits);
        move |_req| {
            success_hits.fetch_add(1, Ordering::Relaxed);
            hyper::Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from_static(b"ok")))
                .unwrap()
        }
    })
    .await;

    let metrics = Metrics::new();
    let proxy_addr = start_proxy_with_config(
        catch_all_router(vec!["127.0.0.1:9".to_string(), success_addr]),
        TestProxyConfig {
            metrics: Some(metrics.clone()),
            ..Default::default()
        },
    )
    .await;

    let response = send_request(&proxy_addr, get_request("/connect-error")).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(success_hits.load(Ordering::Relaxed), 1);
    assert_eq!(
        metrics
            .retry_attempts_total
            .with_label_values(&["core", "connection_error"])
            .get(),
        1
    );
    assert_eq!(
        metrics
            .retry_exhaustions_total
            .with_label_values(&["core", "connection_error"])
            .get(),
        0
    );
}

#[tokio::test]
async fn core_502_retry_is_counted() {
    let failing_hits = Arc::new(AtomicUsize::new(0));
    let failing_addr = start_upstream({
        let failing_hits = Arc::clone(&failing_hits);
        move |_req| {
            failing_hits.fetch_add(1, Ordering::Relaxed);
            hyper::Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Full::new(Bytes::from_static(b"bad gateway")))
                .unwrap()
        }
    })
    .await;
    let success_addr = start_upstream(|_req| {
        hyper::Response::builder()
            .status(StatusCode::OK)
            .body(Full::new(Bytes::from_static(b"ok")))
            .unwrap()
    })
    .await;

    let metrics = Metrics::new();
    let proxy_addr = start_proxy_with_config(
        catch_all_router(vec![failing_addr, success_addr]),
        TestProxyConfig {
            metrics: Some(metrics.clone()),
            ..Default::default()
        },
    )
    .await;

    let response = send_request(&proxy_addr, get_request("/retry-502")).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(failing_hits.load(Ordering::Relaxed), 1);
    assert_eq!(
        metrics
            .retry_attempts_total
            .with_label_values(&["core", "5xx"])
            .get(),
        1
    );
}

#[tokio::test]
async fn post_without_provider_candidates_does_not_retry() {
    let failing_hits = Arc::new(AtomicUsize::new(0));
    let failing_addr = start_upstream({
        let failing_hits = Arc::clone(&failing_hits);
        move |_req| {
            failing_hits.fetch_add(1, Ordering::Relaxed);
            hyper::Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Full::new(Bytes::from_static(b"bad gateway")))
                .unwrap()
        }
    })
    .await;
    let backup_hits = Arc::new(AtomicUsize::new(0));
    let backup_addr = start_upstream({
        let backup_hits = Arc::clone(&backup_hits);
        move |_req| {
            backup_hits.fetch_add(1, Ordering::Relaxed);
            hyper::Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from_static(b"ok")))
                .unwrap()
        }
    })
    .await;

    let metrics = Metrics::new();
    let proxy_addr = start_proxy_with_config(
        catch_all_router(vec![failing_addr, backup_addr]),
        TestProxyConfig {
            metrics: Some(metrics.clone()),
            ..Default::default()
        },
    )
    .await;

    let response = send_request(&proxy_addr, post_request("/post-no-retry")).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(failing_hits.load(Ordering::Relaxed), 1);
    assert_eq!(backup_hits.load(Ordering::Relaxed), 0);
    assert_eq!(
        metrics
            .retry_attempts_total
            .with_label_values(&["core", "5xx"])
            .get(),
        0
    );
    assert_eq!(
        metrics
            .retry_exhaustions_total
            .with_label_values(&["core", "5xx"])
            .get(),
        1
    );
}

#[tokio::test]
async fn retry_budget_exhaustion_is_reported() {
    let first_hits = Arc::new(AtomicUsize::new(0));
    let first_addr = start_upstream({
        let first_hits = Arc::clone(&first_hits);
        move |_req| {
            first_hits.fetch_add(1, Ordering::Relaxed);
            hyper::Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Full::new(Bytes::from_static(b"first-down")))
                .unwrap()
        }
    })
    .await;
    let second_hits = Arc::new(AtomicUsize::new(0));
    let second_addr = start_upstream({
        let second_hits = Arc::clone(&second_hits);
        move |_req| {
            second_hits.fetch_add(1, Ordering::Relaxed);
            hyper::Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Full::new(Bytes::from_static(b"second-down")))
                .unwrap()
        }
    })
    .await;
    let third_hits = Arc::new(AtomicUsize::new(0));
    let third_addr = start_upstream({
        let third_hits = Arc::clone(&third_hits);
        move |_req| {
            third_hits.fetch_add(1, Ordering::Relaxed);
            hyper::Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from_static(b"third-ok")))
                .unwrap()
        }
    })
    .await;

    let metrics = Metrics::new();
    let proxy_addr = start_proxy_with_config(
        catch_all_router(vec![first_addr, second_addr, third_addr]),
        TestProxyConfig {
            metrics: Some(metrics.clone()),
            reliability: ReliabilityConfig {
                max_inflight_requests: None,
                brownout_inflight_requests: None,
                retry_budget_per_request: 1,
            },
            ..Default::default()
        },
    )
    .await;

    let response = send_request(&proxy_addr, get_request("/retry-budget")).await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response.collect().await.unwrap().to_bytes();
    assert_eq!(body.as_ref(), b"second-down");
    assert_eq!(first_hits.load(Ordering::Relaxed), 1);
    assert_eq!(second_hits.load(Ordering::Relaxed), 1);
    assert_eq!(third_hits.load(Ordering::Relaxed), 0);
    assert_eq!(
        metrics
            .retry_attempts_total
            .with_label_values(&["core", "5xx"])
            .get(),
        1
    );
    assert_eq!(
        metrics
            .retry_exhaustions_total
            .with_label_values(&["core", "5xx"])
            .get(),
        1
    );
}

#[tokio::test]
async fn provider_candidates_take_over_when_plugin_filters_route_servers() {
    let candidate_hits = Arc::new(AtomicUsize::new(0));
    let candidate_addr = start_upstream({
        let candidate_hits = Arc::clone(&candidate_hits);
        move |_req| {
            candidate_hits.fetch_add(1, Ordering::Relaxed);
            hyper::Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from_static(b"candidate-ok")))
                .unwrap()
        }
    })
    .await;

    let proxy_addr = start_proxy_with_config(
        catch_all_router(vec!["127.0.0.1:9".to_string()]),
        TestProxyConfig {
            plugins: Some(Arc::new(PluginChain::new(vec![Box::new(
                CandidateModePlugin {
                    candidate_upstream: Some(format!("http://{candidate_addr}")),
                },
            )]))),
            ..Default::default()
        },
    )
    .await;

    let response = send_request(&proxy_addr, get_request("/candidate-mode")).await;
    let status = response.status();
    let body = response.collect().await.unwrap().to_bytes();

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_ref(), b"candidate-ok");
    assert_eq!(candidate_hits.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn empty_filtered_route_and_candidates_returns_503() {
    let proxy_addr = start_proxy_with_config(
        catch_all_router(vec!["127.0.0.1:9".to_string()]),
        TestProxyConfig {
            plugins: Some(Arc::new(PluginChain::new(vec![Box::new(
                CandidateModePlugin {
                    candidate_upstream: None,
                },
            )]))),
            ..Default::default()
        },
    )
    .await;

    let response = send_request(&proxy_addr, get_request("/no-upstreams-left")).await;
    let status = response.status();
    let body = response.collect().await.unwrap().to_bytes();

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(std::str::from_utf8(&body)
        .unwrap()
        .contains("503 Service Unavailable"));
}
