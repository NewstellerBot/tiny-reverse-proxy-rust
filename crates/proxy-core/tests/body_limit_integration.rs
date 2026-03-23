#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use http_body_util::Full;
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream, TestProxyConfig,
    };

    async fn send_raw_chunked_request(proxy_addr: &str, path: &str, chunks: &[&[u8]]) -> String {
        let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
        let request_head = format!(
            "POST {path} HTTP/1.1\r\nHost: {proxy_addr}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        );
        stream.write_all(request_head.as_bytes()).await.unwrap();

        for chunk in chunks {
            stream
                .write_all(format!("{:X}\r\n", chunk.len()).as_bytes())
                .await
                .unwrap();
            stream.write_all(chunk).await.unwrap();
            stream.write_all(b"\r\n").await.unwrap();
        }
        stream.write_all(b"0\r\n\r\n").await.unwrap();
        let _ = stream.shutdown().await;

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await.unwrap();
        String::from_utf8_lossy(&raw).into_owned()
    }

    #[tokio::test]
    async fn returns_413_for_oversized_body() {
        let upstream_calls = Arc::new(AtomicUsize::new(0));
        let upstream_calls_clone = Arc::clone(&upstream_calls);
        let upstream_addr = start_upstream(move |_req: Request<Incoming>| {
            upstream_calls_clone.fetch_add(1, Ordering::SeqCst);
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from("ok")))
                .unwrap()
        })
        .await;
        let router = catch_all_router(vec![upstream_addr]);

        let config = TestProxyConfig {
            max_request_body_bytes: 100,
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .header("Content-Length", "200")
            .body(Full::new(Bytes::from(vec![0u8; 200])))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(upstream_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn allows_body_within_limit() {
        let upstream_calls = Arc::new(AtomicUsize::new(0));
        let upstream_calls_clone = Arc::clone(&upstream_calls);
        let upstream_addr = start_upstream(move |_req: Request<Incoming>| {
            upstream_calls_clone.fetch_add(1, Ordering::SeqCst);
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from("ok")))
                .unwrap()
        })
        .await;
        let router = catch_all_router(vec![upstream_addr]);

        let config = TestProxyConfig {
            max_request_body_bytes: 1000,
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .header("Content-Length", "500")
            .body(Full::new(Bytes::from(vec![0u8; 500])))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(upstream_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn returns_413_for_chunked_body_without_content_length() {
        let upstream_calls = Arc::new(AtomicUsize::new(0));
        let upstream_calls_clone = Arc::clone(&upstream_calls);
        let upstream_addr = start_upstream(move |_req: Request<Incoming>| {
            upstream_calls_clone.fetch_add(1, Ordering::SeqCst);
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from("ok")))
                .unwrap()
        })
        .await;
        let router = catch_all_router(vec![upstream_addr]);

        let config = TestProxyConfig {
            max_request_body_bytes: 100,
            ..Default::default()
        };
        let proxy_addr = start_proxy_with_config(router, config).await;

        let chunk_a = vec![b'a'; 60];
        let chunk_b = vec![b'b'; 60];
        let response = send_raw_chunked_request(
            &proxy_addr,
            "/test",
            &[chunk_a.as_slice(), chunk_b.as_slice()],
        )
        .await;

        assert!(
            response.starts_with("HTTP/1.1 413"),
            "expected 413 response, got: {}",
            response.lines().next().unwrap_or("<empty>")
        );
        assert_eq!(upstream_calls.load(Ordering::SeqCst), 0);
    }
}
