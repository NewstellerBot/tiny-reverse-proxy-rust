#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::{Request, StatusCode};

    use proxy_core::metrics::Metrics;

    use trp_test_support::{
        catch_all_router, ok_upstream, send_request, start_proxy_with_config, TestProxyConfig,
    };

    #[tokio::test]
    async fn metrics_endpoint_returns_counters() {
        let upstream_addr = ok_upstream().await;
        let router = catch_all_router(vec![upstream_addr]);

        let metrics = Metrics::new();
        let config = TestProxyConfig {
            metrics: Some(metrics.clone()),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // Send a few requests through the proxy.
        for i in 0..3 {
            let req = Request::builder()
                .uri(format!("/test/{}", i))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let _ = resp.collect().await;
        }

        // Scrape metrics directly from the Metrics instance.
        let output = metrics.encode();
        assert!(
            output.contains("proxy_requests_total"),
            "metrics should contain proxy_requests_total"
        );
        assert!(
            output.contains("proxy_request_duration_seconds"),
            "metrics should contain proxy_request_duration_seconds"
        );
    }
}
