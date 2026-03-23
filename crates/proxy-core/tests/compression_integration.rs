#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};
    use tokio::io::AsyncReadExt;

    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream, TestProxyConfig,
    };

    /// Generate a 1KB text response.
    fn large_text_handler(_req: Request<Incoming>) -> Response<Full<Bytes>> {
        let body = "x".repeat(1024);
        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/plain")
            .body(Full::new(Bytes::from(body)))
            .unwrap()
    }

    /// Generate a small (< 256 bytes) response.
    fn small_text_handler(_req: Request<Incoming>) -> Response<Full<Bytes>> {
        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/plain")
            .body(Full::new(Bytes::from("hello")))
            .unwrap()
    }

    #[tokio::test]
    async fn gzip_compresses_large_response() {
        let upstream_addr = start_upstream(large_text_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let config = TestProxyConfig {
            compression_enabled: true,
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        let req = Request::builder()
            .uri("/test")
            .header("Accept-Encoding", "gzip")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let encoding = resp
            .headers()
            .get("content-encoding")
            .map(|v| v.to_str().unwrap().to_string());
        assert_eq!(
            encoding.as_deref(),
            Some("gzip"),
            "response should be gzip-compressed"
        );
        let vary = resp
            .headers()
            .get("vary")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            vary.split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("accept-encoding")),
            "compressed responses must set Vary: Accept-Encoding"
        );

        // Decompress and verify content.
        let compressed_body = resp.collect().await.unwrap().to_bytes();
        let mut decoder = async_compression::tokio::bufread::GzipDecoder::new(&compressed_body[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).await.unwrap();
        assert_eq!(decompressed.len(), 1024);
        assert!(decompressed.iter().all(|&b| b == b'x'));
    }

    #[tokio::test]
    async fn skips_small_responses() {
        let upstream_addr = start_upstream(small_text_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let config = TestProxyConfig {
            compression_enabled: true,
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        let req = Request::builder()
            .uri("/test")
            .header("Accept-Encoding", "gzip")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Small response should NOT be compressed.
        assert!(
            resp.headers().get("content-encoding").is_none(),
            "small responses should not be compressed"
        );
    }

    #[tokio::test]
    async fn no_compression_without_accept_encoding() {
        let upstream_addr = start_upstream(large_text_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let config = TestProxyConfig {
            compression_enabled: true,
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        // No Accept-Encoding header.
        let req = Request::builder()
            .uri("/test")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        assert!(
            resp.headers().get("content-encoding").is_none(),
            "should not compress without Accept-Encoding"
        );
    }

    #[tokio::test]
    async fn respects_disabled_flag() {
        let upstream_addr = start_upstream(large_text_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let config = TestProxyConfig {
            compression_enabled: false,
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        let req = Request::builder()
            .uri("/test")
            .header("Accept-Encoding", "gzip")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        assert!(
            resp.headers().get("content-encoding").is_none(),
            "should not compress when disabled"
        );
    }
}
