#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};

    use proxy_core::circuit_breaker::CircuitBreaker;

    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream, TestProxyConfig,
    };

    #[tokio::test]
    async fn circuit_opens_after_failures() {
        // Upstream always returns 500.
        let upstream_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::new(Bytes::from("error")))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);

        // threshold=2, so after 2 failures the circuit opens.
        let cb = CircuitBreaker::new(2, 60, 60);
        let config = TestProxyConfig {
            circuit_breaker: Some(cb),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // First 2 requests go through (upstream returns 500, circuit records failures).
        for i in 0..2 {
            let req = Request::builder()
                .uri(format!("/test/{}", i))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(
                resp.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "request {} should reach upstream",
                i
            );
            let _ = resp.collect().await;
        }

        // 3rd request: circuit should be open, returning 503.
        let req = Request::builder()
            .uri("/test/2")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "circuit should be open, returning 503"
        );
    }

    #[tokio::test]
    async fn circuit_recovers_on_success() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let should_fail = Arc::new(AtomicBool::new(true));
        let sf = Arc::clone(&should_fail);

        let upstream_addr = start_upstream(move |_req: Request<Incoming>| {
            if sf.load(Ordering::Relaxed) {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::new(Bytes::from("error")))
                    .unwrap()
            } else {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from("ok")))
                    .unwrap()
            }
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);

        // threshold=2, cooldown=1s so circuit stays open long enough to verify 503.
        let cb = CircuitBreaker::new(2, 1, 60);
        let config = TestProxyConfig {
            circuit_breaker: Some(cb),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // Cause 2 failures to open the circuit.
        for i in 0..2 {
            let req = Request::builder()
                .uri(format!("/fail/{}", i))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
            let _ = resp.collect().await;
        }

        // Circuit is open, cooldown hasn't elapsed -> 503.
        let req = Request::builder()
            .uri("/test/open")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let _ = resp.collect().await;

        // Switch upstream to success, then wait for cooldown to elapse.
        should_fail.store(false, Ordering::Relaxed);
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

        // After cooldown, circuit transitions to half-open and allows a probe.
        let req = Request::builder()
            .uri("/test/recover")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "circuit should recover on success after cooldown"
        );
    }
}
