#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use http_body_util::Full;
    use hyper::{Request, StatusCode};

    use proxy_core::rate_limit::RateLimiter;

    use trp_test_support::{
        catch_all_router, ok_upstream, send_request, start_proxy_with_config, TestProxyConfig,
    };

    #[tokio::test]
    async fn returns_429_after_burst_exceeded() {
        let upstream_addr = ok_upstream().await;
        let router = catch_all_router(vec![upstream_addr]);

        let config = TestProxyConfig {
            rate_limiter: Some(RateLimiter::new(1.0, 3)),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // First 3 requests should succeed (burst=3).
        for i in 0..3 {
            let req = Request::builder()
                .uri(format!("/test/{}", i))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "request {} should succeed",
                i
            );
        }

        // 4th request should be rate-limited.
        let req = Request::builder()
            .uri("/test/4")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn allows_after_token_refill() {
        let upstream_addr = ok_upstream().await;
        let router = catch_all_router(vec![upstream_addr]);

        // rate=100/s, burst=2: exhaust quickly, then refill is fast enough to test.
        let config = TestProxyConfig {
            rate_limiter: Some(RateLimiter::new(100.0, 2)),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // Exhaust both burst tokens.
        for i in 0..2 {
            let req = Request::builder()
                .uri(format!("/test/{}", i))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "burst request {} should succeed",
                i
            );
        }

        // Should be rejected now.
        let req = Request::builder()
            .uri("/test/exhausted")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        // Wait for token refill (at 100/s, 50ms refills ~5 tokens).
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let req = Request::builder()
            .uri("/test/refilled")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK, "should succeed after refill");
    }
}
