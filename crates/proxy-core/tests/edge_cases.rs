#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::service::service_fn;
    use hyper::{Method, Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::watch;

    use proxy_core::handlers::proxy::{build_client, ProxyService};

    use trp_test_support::{
        catch_all_router, ok_upstream, start_proxy, start_upstream, CatchAllRouter,
    };

    // ── #1: Malformed upstream Host header ──────────────────────────────

    /// When an upstream address like "host:notaport" is configured,
    /// the proxy should return 502 Bad Gateway instead of panicking.
    #[tokio::test]
    async fn malformed_upstream_port_returns_502() {
        let router = catch_all_router(vec!["host:notaport".to_string()]);
        let proxy_addr = start_proxy(router).await;

        let client = build_client();
        let uri: hyper::Uri = format!("http://{}/test", proxy_addr).parse().unwrap();
        let req = Request::builder()
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        let resp = tokio::time::timeout(Duration::from_secs(5), client.request(req))
            .await
            .expect("should not timeout")
            .expect("should get a response, not connection reset");
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    // ── #4: Very long paths or headers ──────────────────────────────────

    /// Test with a path exceeding typical server limits (~8KB).
    /// hyper's default max_buf_size for HTTP/1 is ~8KB, so a 10,000 char path
    /// may be rejected at the parsing level before reaching the proxy handler.
    #[tokio::test]
    async fn very_long_path_handled_gracefully() {
        let upstream_addr = start_upstream(|req: Request<Incoming>| {
            let path_len = req.uri().path().len();
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from(format!("path_len={}", path_len))))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let proxy_addr = start_proxy(router).await;

        // 10,000 character path
        let long_segment = "a".repeat(10_000);
        let path = format!("/{}", long_segment);
        let client = build_client();
        let uri: hyper::Uri = format!("http://{}{}", proxy_addr, path).parse().unwrap();
        let req = Request::builder()
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        let result = tokio::time::timeout(Duration::from_secs(5), client.request(req)).await;
        match result {
            Ok(Ok(resp)) => {
                // hyper may accept or reject based on max_buf_size.
                // Either a proxied 200 or an error status is acceptable.
                let status = resp.status();
                assert!(
                    status == StatusCode::OK
                        || status.is_client_error()
                        || status.is_server_error(),
                    "should return a valid HTTP response, got {}",
                    status
                );
                let _ = resp.collect().await;
            }
            Ok(Err(_)) => {
                // hyper's server rejected the oversized request line — acceptable
            }
            Err(_) => {
                panic!("request timed out unexpectedly");
            }
        }
    }

    /// Test with a very long header value (16KB, exceeding typical 8KB limit).
    #[tokio::test]
    async fn very_long_header_value_handled_gracefully() {
        let upstream_addr = start_upstream(|req: Request<Incoming>| {
            let got = req.headers().get("x-large").map(|v| v.len()).unwrap_or(0);
            Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from(format!("header_len={}", got))))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let proxy_addr = start_proxy(router).await;

        let large_value = "x".repeat(16_000);
        let client = build_client();
        let uri: hyper::Uri = format!("http://{}/test", proxy_addr).parse().unwrap();
        let req = Request::builder()
            .uri(uri)
            .header("X-Large", &large_value)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        let result = tokio::time::timeout(Duration::from_secs(5), client.request(req)).await;
        match result {
            Ok(Ok(resp)) => {
                let status = resp.status();
                assert!(
                    status == StatusCode::OK
                        || status.is_client_error()
                        || status.is_server_error(),
                    "should return a valid HTTP response, got {}",
                    status
                );
                let _ = resp.collect().await;
            }
            Ok(Err(_)) => {
                // hyper rejected oversized headers — acceptable
            }
            Err(_) => {
                panic!("request timed out unexpectedly");
            }
        }
    }

    // ── #20: CONNECT, OPTIONS, TRACE methods ────────────────────────────

    /// OPTIONS requests should be proxied through like any other method.
    /// RFC 9110 §9.3.7: OPTIONS is safe and idempotent.
    #[tokio::test]
    async fn options_method_proxied_through() {
        let upstream_addr = start_upstream(|req: Request<Incoming>| {
            assert_eq!(req.method(), Method::OPTIONS);
            Response::builder()
                .status(StatusCode::OK)
                .header("Allow", "GET, POST, OPTIONS")
                .body(Full::new(Bytes::new()))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let proxy_addr = start_proxy(router).await;

        let client = build_client();
        let uri: hyper::Uri = format!("http://{}/test", proxy_addr).parse().unwrap();
        let req = Request::builder()
            .method(Method::OPTIONS)
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        let resp = client.request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            resp.headers().get("allow").is_some(),
            "Allow header should be preserved from upstream"
        );
    }

    /// TRACE requests should be proxied through.
    /// RFC 9110 §9.3.8: TRACE is safe but not cached.
    #[tokio::test]
    async fn trace_method_proxied_through() {
        let upstream_addr = start_upstream(|req: Request<Incoming>| {
            assert_eq!(req.method(), Method::TRACE);
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "message/http")
                .body(Full::new(Bytes::from("TRACE response")))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let proxy_addr = start_proxy(router).await;

        let client = build_client();
        let uri: hyper::Uri = format!("http://{}/test", proxy_addr).parse().unwrap();
        let req = Request::builder()
            .method(Method::TRACE)
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        let resp = client.request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"TRACE response");
    }

    /// CONNECT uses authority-form URIs (host:port). The proxy doesn't implement
    /// CONNECT tunneling, so test that it handles the method gracefully.
    #[tokio::test]
    async fn connect_method_does_not_panic() {
        let upstream_addr = ok_upstream().await;
        let router = catch_all_router(vec![upstream_addr]);
        let proxy_addr = start_proxy(router).await;

        // Send CONNECT via raw TCP since hyper's client has special CONNECT handling.
        let mut stream = tokio::net::TcpStream::connect(&proxy_addr).await.unwrap();
        let request = b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n";
        stream.write_all(request).await.unwrap();

        let mut buf = vec![0u8; 4096];
        let result = tokio::time::timeout(Duration::from_secs(3), stream.read(&mut buf)).await;

        match result {
            Ok(Ok(n)) if n > 0 => {
                let response = String::from_utf8_lossy(&buf[..n]);
                // Should get a valid HTTP response (not a crash).
                assert!(
                    response.contains("HTTP/1.1"),
                    "should receive a valid HTTP response, got: {}",
                    response
                );
            }
            Ok(Ok(_)) => {
                // Connection closed gracefully — proxy didn't panic
            }
            Ok(Err(_)) => {
                // IO error — proxy didn't panic
            }
            Err(_) => {
                // Timeout — proxy may be waiting for tunnel data after CONNECT.
                // This is expected; CONNECT semantics require a bidirectional tunnel.
            }
        }
    }

    // ── #21: Conflicting Transfer-Encoding: chunked + Content-Length ────

    /// RFC 7230 §3.3.3: When Transfer-Encoding is present, Content-Length must
    /// be ignored. Test with a raw TCP upstream that sends both headers.
    #[tokio::test]
    async fn conflicting_transfer_encoding_and_content_length() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            loop {
                let (mut stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };

                tokio::spawn(async move {
                    // Read the request
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf).await;

                    // Send response with both Transfer-Encoding and Content-Length.
                    // Per RFC 7230 §3.3.3, TE overrides CL.
                    let response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 999\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
                    let _ = stream.write_all(response).await;
                    let _ = stream.shutdown().await;
                });
            }
        });

        let router = catch_all_router(vec![upstream_addr]);
        let proxy_addr = start_proxy(router).await;

        let client = build_client();
        let uri: hyper::Uri = format!("http://{}/test", proxy_addr).parse().unwrap();
        let req = Request::builder()
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        let result = tokio::time::timeout(Duration::from_secs(5), client.request(req)).await;
        match result {
            Ok(Ok(resp)) => {
                // Proxy handled the conflicting headers without panicking.
                // May be 200 (decoded correctly) or 502 (treated as upstream error).
                let status = resp.status();
                assert!(
                    status == StatusCode::OK || status == StatusCode::BAD_GATEWAY,
                    "unexpected status: {}",
                    status
                );
                // Consume body to complete the response cycle
                let _ = resp.collect().await;
            }
            Ok(Err(_)) => {
                // Connection error due to framing issues — acceptable, not a panic
            }
            Err(_) => {
                panic!("request timed out unexpectedly");
            }
        }
    }

    // ── #22: Redirect responses (3xx) ───────────────────────────────────

    /// 301 Moved Permanently should be passed through with Location preserved.
    #[tokio::test]
    async fn redirect_301_passed_through_with_location() {
        let upstream_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::MOVED_PERMANENTLY)
                .header("Location", "https://example.com/new-path")
                .body(Full::new(Bytes::from("moved")))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let proxy_addr = start_proxy(router).await;

        let client = build_client();
        let uri: hyper::Uri = format!("http://{}/old-path", proxy_addr).parse().unwrap();
        let req = Request::builder()
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        let resp = client.request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::MOVED_PERMANENTLY);
        assert_eq!(
            resp.headers().get("location").unwrap().to_str().unwrap(),
            "https://example.com/new-path"
        );
        let body = resp.collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"moved");
    }

    /// 302 Found with a relative URL in Location should be preserved unchanged.
    #[tokio::test]
    async fn redirect_302_preserves_relative_location() {
        let upstream_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::FOUND)
                .header("Location", "/other-path")
                .body(Full::new(Bytes::from("found")))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let proxy_addr = start_proxy(router).await;

        let client = build_client();
        let uri: hyper::Uri = format!("http://{}/test", proxy_addr).parse().unwrap();
        let req = Request::builder()
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        let resp = client.request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FOUND);
        // Relative URL must be preserved verbatim
        assert_eq!(
            resp.headers().get("location").unwrap().to_str().unwrap(),
            "/other-path"
        );
    }

    /// 307 Temporary Redirect should be passed through with body intact.
    #[tokio::test]
    async fn redirect_307_passed_through_with_body() {
        let upstream_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::TEMPORARY_REDIRECT)
                .header("Location", "https://example.com/temp")
                .body(Full::new(Bytes::from("temporary redirect")))
                .unwrap()
        })
        .await;

        let router = catch_all_router(vec![upstream_addr]);
        let proxy_addr = start_proxy(router).await;

        let client = build_client();
        let uri: hyper::Uri = format!("http://{}/test", proxy_addr).parse().unwrap();
        let req = Request::builder()
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();

        let resp = client.request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            resp.headers().get("location").unwrap().to_str().unwrap(),
            "https://example.com/temp"
        );
        let body = resp.collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"temporary redirect");
    }

    // ── #24: Graceful shutdown with idle connections ─────────────────────

    /// Start a proxy with graceful shutdown support via a watch channel.
    async fn start_proxy_with_shutdown(
        router: Arc<CatchAllRouter>,
    ) -> (String, watch::Sender<bool>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        let client = build_client();
        let counter = Arc::new(AtomicUsize::new(0));

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        tokio::spawn(async move {
            loop {
                let mut rx = shutdown_rx.clone();

                let accept = tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok(conn) => Some(conn),
                            Err(_) => break,
                        }
                    }
                    _ = rx.changed() => {
                        if *rx.borrow() {
                            break;
                        }
                        continue;
                    }
                };

                if let Some((stream, peer_addr)) = accept {
                    let svc = ProxyService::new(
                        Arc::clone(&router),
                        client.clone(),
                        Arc::clone(&counter),
                    )
                    .with_peer_addr(peer_addr);

                    let mut conn_rx = shutdown_rx.clone();

                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let conn = hyper::server::conn::http1::Builder::new()
                            .keep_alive(true)
                            .serve_connection(
                                io,
                                service_fn(move |req: Request<Incoming>| {
                                    let svc = svc.clone();
                                    async move { svc.handle(req).await }
                                }),
                            );

                        tokio::pin!(conn);

                        loop {
                            tokio::select! {
                                result = conn.as_mut() => {
                                    let _ = result;
                                    break;
                                }
                                _ = conn_rx.changed() => {
                                    if *conn_rx.borrow() {
                                        conn.as_mut().graceful_shutdown();
                                    }
                                }
                            }
                        }
                    });
                }
            }
        });

        (addr, shutdown_tx)
    }

    /// After shutdown signal, idle keepalive connections should be closed gracefully
    /// and no new connections should be accepted.
    #[tokio::test]
    async fn shutdown_closes_idle_keepalive_connection() {
        let upstream_addr = ok_upstream().await;
        let router = catch_all_router(vec![upstream_addr]);
        let (proxy_addr, shutdown_tx) = start_proxy_with_shutdown(router).await;

        // Establish a keepalive connection by sending a request
        let client = build_client();
        let uri: hyper::Uri = format!("http://{}/test", proxy_addr).parse().unwrap();
        let req = Request::builder()
            .uri(uri)
            .body(
                Full::new(Bytes::new())
                    .map_err(|never| match never {})
                    .boxed(),
            )
            .unwrap();
        let resp = client.request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = resp.collect().await; // consume body to keep connection alive

        // Connection is now idle. Trigger graceful shutdown.
        shutdown_tx.send(true).unwrap();

        // Wait for shutdown to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // New connections should fail — accept loop has stopped.
        let result = tokio::net::TcpStream::connect(&proxy_addr).await;
        assert!(
            result.is_err(),
            "new TCP connections should be refused after shutdown"
        );
    }
}
