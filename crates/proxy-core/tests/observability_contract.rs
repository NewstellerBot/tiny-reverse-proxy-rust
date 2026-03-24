use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::Full;
use hyper::{Request, StatusCode};

use proxy_core::config::ReliabilityConfig;
use proxy_core::metrics::Metrics;
use proxy_core::runtime::ProbeState;
use trp_test_support::{
    catch_all_router, ok_upstream, send_request, start_proxy_with_config, TestProxyConfig,
};

fn get_request(path: &str) -> Request<Full<Bytes>> {
    Request::builder()
        .method("GET")
        .uri(path)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

#[tokio::test]
async fn admission_reject_increments_counter() {
    let upstream_addr = ok_upstream().await;
    let router = catch_all_router(vec![upstream_addr]);
    let metrics = Metrics::new();
    let inflight_requests = Arc::new(AtomicUsize::new(1));

    let proxy_addr = start_proxy_with_config(
        router,
        TestProxyConfig {
            metrics: Some(metrics.clone()),
            reliability: ReliabilityConfig {
                max_inflight_requests: Some(1),
                brownout_inflight_requests: None,
                retry_budget_per_request: 2,
            },
            inflight_requests: Some(Arc::clone(&inflight_requests)),
            ..Default::default()
        },
    )
    .await;

    let response = send_request(&proxy_addr, get_request("/reject")).await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(metrics.admission_rejections_total.get(), 1);
}

#[tokio::test]
async fn brownout_activation_updates_metrics() {
    let upstream_addr = ok_upstream().await;
    let router = catch_all_router(vec![upstream_addr]);
    let metrics = Metrics::new();
    let inflight_requests = Arc::new(AtomicUsize::new(0));

    let proxy_addr = start_proxy_with_config(
        router,
        TestProxyConfig {
            metrics: Some(metrics.clone()),
            reliability: ReliabilityConfig {
                max_inflight_requests: None,
                brownout_inflight_requests: Some(1),
                retry_budget_per_request: 2,
            },
            inflight_requests: Some(Arc::clone(&inflight_requests)),
            ..Default::default()
        },
    )
    .await;

    let response = send_request(&proxy_addr, get_request("/brownout")).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(metrics.brownout_activations_total.get(), 1);
    assert_eq!(metrics.brownout_active.get(), 0);
    assert_eq!(inflight_requests.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn readiness_and_draining_endpoints_follow_probe_state() {
    let upstream_addr = ok_upstream().await;
    let router = catch_all_router(vec![upstream_addr]);
    let probe_state = ProbeState::new();

    let proxy_addr = start_proxy_with_config(
        router,
        TestProxyConfig {
            probe_state: Some(probe_state.clone()),
            ..Default::default()
        },
    )
    .await;

    let starting = send_request(&proxy_addr, get_request("/_trp/readyz")).await;
    assert_eq!(starting.status(), StatusCode::SERVICE_UNAVAILABLE);

    probe_state.mark_ready();
    let ready = send_request(&proxy_addr, get_request("/_trp/readyz")).await;
    assert_eq!(ready.status(), StatusCode::OK);

    probe_state.mark_draining("draining for deploy");
    let draining = send_request(&proxy_addr, get_request("/_trp/readyz")).await;
    assert_eq!(draining.status(), StatusCode::SERVICE_UNAVAILABLE);
}
