#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::AsyncReadExt;

    use proxy_core::cache::ResponseCache;

    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream, TestProxyConfig,
    };

    #[tokio::test]
    async fn cache_hit_on_second_get() {
        let upstream_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from("cached body")))
                .unwrap()
        })
        .await;

        // Use 2 upstreams so the idempotent retry path (which includes caching) is used.
        let router = catch_all_router(vec![upstream_addr.clone(), upstream_addr]);

        let config = TestProxyConfig {
            cache: Some(ResponseCache::new(1, 300)),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // First request: MISS.
        let req = Request::builder()
            .uri("/test")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let x_cache = resp
            .headers()
            .get("x-cache")
            .map(|v| v.to_str().unwrap().to_string());
        assert_eq!(x_cache.as_deref(), Some("MISS"));
        let _ = resp.collect().await;

        // Second request: HIT.
        let req = Request::builder()
            .uri("/test")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let x_cache = resp
            .headers()
            .get("x-cache")
            .map(|v| v.to_str().unwrap().to_string());
        assert_eq!(x_cache.as_deref(), Some("HIT"));

        let body = resp.collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"cached body");
    }

    #[tokio::test]
    async fn post_bypasses_cache() {
        let upstream_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from("post response")))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr.clone(), upstream_addr]);

        let config = TestProxyConfig {
            cache: Some(ResponseCache::new(1, 300)),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // POST requests should never be cached.
        for _ in 0..2 {
            let req = Request::builder()
                .method("POST")
                .uri("/test")
                .body(Full::new(Bytes::from("body")))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            // POST should always go to upstream (no cache HIT).
            let x_cache = resp
                .headers()
                .get("x-cache")
                .map(|v| v.to_str().unwrap().to_string());
            assert_eq!(x_cache.as_deref(), Some("MISS"));
            let _ = resp.collect().await;
        }
    }

    #[tokio::test]
    async fn no_store_skips_cache() {
        let upstream_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from("no-store body")))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr.clone(), upstream_addr]);

        let config = TestProxyConfig {
            cache: Some(ResponseCache::new(1, 300)),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // Request with Cache-Control: no-store should always be a MISS.
        for _ in 0..2 {
            let req = Request::builder()
                .uri("/test")
                .header("Cache-Control", "no-store")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let x_cache = resp
                .headers()
                .get("x-cache")
                .map(|v| v.to_str().unwrap().to_string());
            assert_eq!(x_cache.as_deref(), Some("MISS"));
            let _ = resp.collect().await;
        }
    }

    #[tokio::test]
    async fn single_upstream_route_still_caches() {
        let upstream_calls = Arc::new(AtomicUsize::new(0));
        let upstream_calls_clone = Arc::clone(&upstream_calls);

        let upstream_addr = start_upstream(move |_req: Request<Incoming>| {
            upstream_calls_clone.fetch_add(1, Ordering::SeqCst);
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from("single upstream body")))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let config = TestProxyConfig {
            cache: Some(ResponseCache::new(1, 300)),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        let req1 = Request::builder()
            .uri("/single")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp1 = send_request(&proxy_addr, req1).await;
        assert_eq!(resp1.status(), StatusCode::OK);
        assert_eq!(
            resp1.headers().get("x-cache").and_then(|v| v.to_str().ok()),
            Some("MISS")
        );
        let _ = resp1.collect().await.unwrap();

        let req2 = Request::builder()
            .uri("/single")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp2 = send_request(&proxy_addr, req2).await;
        assert_eq!(resp2.status(), StatusCode::OK);
        assert_eq!(
            resp2.headers().get("x-cache").and_then(|v| v.to_str().ok()),
            Some("HIT")
        );
        let _ = resp2.collect().await.unwrap();

        assert_eq!(upstream_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cache_key_includes_query_string() {
        let upstream_addr = start_upstream(|req: Request<Incoming>| {
            let query = req.uri().query().unwrap_or("");
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from(format!("query={query}"))))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let config = TestProxyConfig {
            cache: Some(ResponseCache::new(1, 300)),
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        let req_a1 = Request::builder()
            .uri("/search?q=a")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp_a1 = send_request(&proxy_addr, req_a1).await;
        assert_eq!(
            resp_a1
                .headers()
                .get("x-cache")
                .and_then(|v| v.to_str().ok()),
            Some("MISS")
        );
        assert_eq!(
            resp_a1.collect().await.unwrap().to_bytes(),
            Bytes::from("query=q=a")
        );

        let req_a2 = Request::builder()
            .uri("/search?q=a")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp_a2 = send_request(&proxy_addr, req_a2).await;
        assert_eq!(
            resp_a2
                .headers()
                .get("x-cache")
                .and_then(|v| v.to_str().ok()),
            Some("HIT")
        );
        assert_eq!(
            resp_a2.collect().await.unwrap().to_bytes(),
            Bytes::from("query=q=a")
        );

        let req_b = Request::builder()
            .uri("/search?q=b")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp_b = send_request(&proxy_addr, req_b).await;
        assert_eq!(
            resp_b
                .headers()
                .get("x-cache")
                .and_then(|v| v.to_str().ok()),
            Some("MISS")
        );
        assert_eq!(
            resp_b.collect().await.unwrap().to_bytes(),
            Bytes::from("query=q=b")
        );
    }

    #[tokio::test]
    async fn cache_respects_accept_encoding_variants() {
        let upstream_calls = Arc::new(AtomicUsize::new(0));
        let upstream_calls_clone = Arc::clone(&upstream_calls);
        let upstream_addr = start_upstream(move |_req: Request<Incoming>| {
            upstream_calls_clone.fetch_add(1, Ordering::SeqCst);
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/plain")
                .body(Full::new(Bytes::from("x".repeat(1024))))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let config = TestProxyConfig {
            cache: Some(ResponseCache::new(1, 300)),
            compression_enabled: true,
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // First: gzip-capable client.
        let req_gzip = Request::builder()
            .uri("/encoding")
            .header("Accept-Encoding", "gzip")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp_gzip = send_request(&proxy_addr, req_gzip).await;
        assert_eq!(resp_gzip.status(), StatusCode::OK);
        assert_eq!(
            resp_gzip
                .headers()
                .get("x-cache")
                .and_then(|v| v.to_str().ok()),
            Some("MISS")
        );
        assert_eq!(
            resp_gzip
                .headers()
                .get("content-encoding")
                .and_then(|v| v.to_str().ok()),
            Some("gzip")
        );
        let vary = resp_gzip
            .headers()
            .get("vary")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            vary.split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("accept-encoding")),
            "compressed responses must vary on Accept-Encoding"
        );
        let compressed_body = resp_gzip.collect().await.unwrap().to_bytes();
        let mut gzip_decoder =
            async_compression::tokio::bufread::GzipDecoder::new(&compressed_body[..]);
        let mut decoded = Vec::new();
        gzip_decoder.read_to_end(&mut decoded).await.unwrap();
        assert_eq!(decoded, vec![b'x'; 1024]);

        // Second gzip request should hit the compressed variant, not miss due to Vary lookup.
        let req_gzip_again = Request::builder()
            .uri("/encoding")
            .header("Accept-Encoding", "gzip")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp_gzip_again = send_request(&proxy_addr, req_gzip_again).await;
        assert_eq!(resp_gzip_again.status(), StatusCode::OK);
        assert_eq!(
            resp_gzip_again
                .headers()
                .get("x-cache")
                .and_then(|v| v.to_str().ok()),
            Some("HIT")
        );
        assert_eq!(
            resp_gzip_again
                .headers()
                .get("content-encoding")
                .and_then(|v| v.to_str().ok()),
            Some("gzip")
        );
        let compressed_body_again = resp_gzip_again.collect().await.unwrap().to_bytes();
        let mut gzip_decoder_again =
            async_compression::tokio::bufread::GzipDecoder::new(&compressed_body_again[..]);
        let mut decoded_again = Vec::new();
        gzip_decoder_again
            .read_to_end(&mut decoded_again)
            .await
            .unwrap();
        assert_eq!(decoded_again, vec![b'x'; 1024]);

        // Second: client with no Accept-Encoding should not receive the compressed variant.
        let req_plain = Request::builder()
            .uri("/encoding")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp_plain = send_request(&proxy_addr, req_plain).await;
        assert_eq!(resp_plain.status(), StatusCode::OK);
        assert_eq!(
            resp_plain
                .headers()
                .get("x-cache")
                .and_then(|v| v.to_str().ok()),
            Some("MISS")
        );
        assert!(
            resp_plain.headers().get("content-encoding").is_none(),
            "identity clients should not receive cached compressed bytes"
        );
        assert_eq!(
            resp_plain.collect().await.unwrap().to_bytes(),
            Bytes::from("x".repeat(1024))
        );

        // Third plain request should hit cache for the identity variant.
        let req_plain_again = Request::builder()
            .uri("/encoding")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp_plain_again = send_request(&proxy_addr, req_plain_again).await;
        assert_eq!(resp_plain_again.status(), StatusCode::OK);
        assert_eq!(
            resp_plain_again
                .headers()
                .get("x-cache")
                .and_then(|v| v.to_str().ok()),
            Some("HIT")
        );
        assert!(resp_plain_again.headers().get("content-encoding").is_none());
        assert_eq!(
            resp_plain_again.collect().await.unwrap().to_bytes(),
            Bytes::from("x".repeat(1024))
        );

        assert_eq!(upstream_calls.load(Ordering::SeqCst), 2);
    }
}
