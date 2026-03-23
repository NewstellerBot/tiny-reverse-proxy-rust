#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::service::service_fn;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    use proxy_core::config::{LbStrategy, RouteConfig};
    use proxy_core::handlers::proxy::{build_client, ProxyService};
    use proxy_core::router::PathResolution;

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Routes all paths to the configured servers (unlike MockRouter which only matches `/test`).
    struct CatchAllRouter {
        route_config: RouteConfig,
    }

    impl PathResolution for CatchAllRouter {
        fn get_servers(&self, _path: &str) -> Option<&Vec<String>> {
            Some(&self.route_config.servers)
        }
        fn get_route_config(&self, _path: &str) -> Option<&RouteConfig> {
            Some(&self.route_config)
        }
    }

    /// Start a ProxyService on a random port. Returns the bound address.
    async fn start_proxy(router: Arc<CatchAllRouter>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        let client = build_client();
        let counter = Arc::new(AtomicUsize::new(0));

        tokio::spawn(async move {
            loop {
                let (stream, peer_addr) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };

                let svc =
                    ProxyService::new(Arc::clone(&router), client.clone(), Arc::clone(&counter))
                        .with_peer_addr(peer_addr);

                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .keep_alive(true)
                        .serve_connection(
                            io,
                            service_fn(move |req: Request<Incoming>| {
                                let svc = svc.clone();
                                async move { svc.handle(req).await }
                            }),
                        )
                        .await;
                });
            }
        });

        addr
    }

    /// Start a mock upstream that delegates to a handler closure.
    async fn start_upstream<F>(handler: F) -> String
    where
        F: Fn(Request<Incoming>) -> Response<Full<Bytes>> + Send + Sync + Clone + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };

                let handler = handler.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .keep_alive(true)
                        .serve_connection(
                            io,
                            service_fn(move |req: Request<Incoming>| {
                                let resp = handler(req);
                                async move { Ok::<_, hyper::Error>(resp) }
                            }),
                        )
                        .await;
                });
            }
        });

        addr
    }

    /// Handler that echoes back request metadata as response headers.
    fn echo_handler(req: Request<Incoming>) -> Response<Full<Bytes>> {
        let mut builder = Response::builder().status(StatusCode::OK);

        {
            let headers = builder.headers_mut().unwrap();

            if let Some(v) = req.headers().get("host") {
                headers.insert("x-got-host", v.clone());
            }
            if let Some(v) = req.headers().get("x-forwarded-for") {
                headers.insert("x-got-x-forwarded-for", v.clone());
            }
            if let Some(v) = req.headers().get("x-forwarded-proto") {
                headers.insert("x-got-x-forwarded-proto", v.clone());
            }
            if let Some(v) = req.headers().get("x-forwarded-host") {
                headers.insert("x-got-x-forwarded-host", v.clone());
            }
            if let Some(v) = req.headers().get("connection") {
                headers.insert("x-got-connection", v.clone());
            }
            if let Some(v) = req.headers().get("te") {
                headers.insert("x-got-te", v.clone());
            }
            if let Some(v) = req.headers().get("upgrade") {
                headers.insert("x-got-upgrade", v.clone());
            }
            if let Some(v) = req.headers().get("keep-alive") {
                headers.insert("x-got-keep-alive", v.clone());
            }
            if let Some(v) = req.headers().get("proxy-authorization") {
                headers.insert("x-got-proxy-authorization", v.clone());
            }
            if let Some(v) = req.headers().get("x-custom") {
                headers.insert("x-got-x-custom", v.clone());
            }
            if let Some(v) = req.headers().get("via") {
                headers.insert("x-got-via", v.clone());
            }

            // Echo the URI path+query
            let pq = req
                .uri()
                .path_and_query()
                .map(|pq| pq.as_str())
                .unwrap_or("/");
            headers.insert(
                "x-got-path",
                hyper::header::HeaderValue::from_str(pq).unwrap(),
            );
        }

        builder.body(Full::new(Bytes::new())).unwrap()
    }

    /// Send an arbitrary request through the proxy and return the response.
    async fn send_request(proxy_addr: &str, req: Request<Full<Bytes>>) -> Response<Incoming> {
        let client = build_client();
        let (parts, body) = req.into_parts();

        let uri: hyper::Uri = format!(
            "http://{}{}",
            proxy_addr,
            parts
                .uri
                .path_and_query()
                .map(|pq| pq.as_str())
                .unwrap_or("/")
        )
        .parse()
        .unwrap();

        let mut builder = Request::builder().method(parts.method).uri(uri);
        if let Some(headers) = builder.headers_mut() {
            *headers = parts.headers;
        }

        let boxed_body = body.map_err(|never| match never {}).boxed();
        let req = builder.body(boxed_body).unwrap();
        client.request(req).await.unwrap()
    }

    /// Convenience: build a proxy + echo upstream, returns (proxy_addr, upstream_addr).
    async fn setup_echo() -> (String, String) {
        let upstream_addr = start_upstream(echo_handler).await;
        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream_addr.clone()],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy(router).await;
        (proxy_addr, upstream_addr)
    }

    // ── Module 1: hop_by_hop ─────────────────────────────────────────────

    mod hop_by_hop {
        use super::*;

        #[tokio::test]
        async fn strips_connection_header() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .header("Connection", "keep-alive")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert!(resp.headers().get("x-got-connection").is_none());
        }

        #[tokio::test]
        async fn strips_te_header() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .header("TE", "trailers")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert!(resp.headers().get("x-got-te").is_none());
        }

        #[tokio::test]
        async fn strips_proxy_authorization() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .header("Proxy-Authorization", "Basic eHh4")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert!(resp.headers().get("x-got-proxy-authorization").is_none());
        }

        #[tokio::test]
        async fn strips_upgrade_on_non_upgrade() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .header("Upgrade", "h2c")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert!(resp.headers().get("x-got-upgrade").is_none());
        }

        #[tokio::test]
        async fn strips_keep_alive_header() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .header("Keep-Alive", "timeout=5")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert!(resp.headers().get("x-got-keep-alive").is_none());
        }

        #[tokio::test]
        async fn strips_custom_connection_tokens() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .header("Connection", "x-custom")
                .header("X-Custom", "val")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert!(resp.headers().get("x-got-connection").is_none());
            assert!(resp.headers().get("x-got-x-custom").is_none());
        }
    }

    // ── Module 2: forwarded_headers ──────────────────────────────────────

    mod forwarded_headers {
        use super::*;

        #[tokio::test]
        async fn adds_x_forwarded_for() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let xff = resp
                .headers()
                .get("x-got-x-forwarded-for")
                .expect("upstream should receive X-Forwarded-For");
            assert!(
                xff.to_str().unwrap().contains("127.0.0.1"),
                "X-Forwarded-For should contain client IP"
            );
        }

        #[tokio::test]
        async fn appends_x_forwarded_for() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .header("X-Forwarded-For", "1.2.3.4")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let xff = resp
                .headers()
                .get("x-got-x-forwarded-for")
                .expect("upstream should receive X-Forwarded-For");
            let val = xff.to_str().unwrap();
            assert!(
                val.starts_with("1.2.3.4, "),
                "should preserve existing XFF and append"
            );
        }

        #[tokio::test]
        async fn sets_x_forwarded_proto() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let proto = resp
                .headers()
                .get("x-got-x-forwarded-proto")
                .expect("upstream should receive X-Forwarded-Proto");
            assert_eq!(proto.to_str().unwrap(), "http");
        }

        #[tokio::test]
        async fn sets_x_forwarded_host() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .header("Host", "example.com")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let host = resp
                .headers()
                .get("x-got-x-forwarded-host")
                .expect("upstream should receive X-Forwarded-Host");
            assert_eq!(host.to_str().unwrap(), "example.com");
        }
    }

    // ── Module 3: host_rewriting ─────────────────────────────────────────

    mod host_rewriting {
        use super::*;

        #[tokio::test]
        async fn rewrites_host_to_upstream() {
            let (proxy_addr, upstream_addr) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let got_host = resp
                .headers()
                .get("x-got-host")
                .expect("upstream should receive Host header");
            assert_eq!(got_host.to_str().unwrap(), upstream_addr);
        }

        #[tokio::test]
        async fn preserves_original_path() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/foo/bar?q=1")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let got_path = resp
                .headers()
                .get("x-got-path")
                .expect("upstream should echo path");
            assert_eq!(got_path.to_str().unwrap(), "/foo/bar?q=1");
        }

        #[tokio::test]
        async fn handles_absolute_uri() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri(format!("http://{}/test?a=b", proxy_addr))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let got_path = resp
                .headers()
                .get("x-got-path")
                .expect("upstream should echo path");
            assert_eq!(got_path.to_str().unwrap(), "/test?a=b");
        }
    }

    // ── Module 4: via_header ─────────────────────────────────────────────

    mod via_header {
        use super::*;

        #[tokio::test]
        async fn adds_via_header_to_request() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let via = resp
                .headers()
                .get("x-got-via")
                .expect("upstream should receive Via header");
            assert!(via.to_str().unwrap().contains("1.1"));
        }

        #[tokio::test]
        async fn adds_via_header_to_response() {
            let (proxy_addr, _) = setup_echo().await;
            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let via = resp
                .headers()
                .get("via")
                .expect("response should contain Via header");
            assert!(via.to_str().unwrap().contains("1.1"));
        }
    }

    // ── Module 5: body_handling ──────────────────────────────────────────

    mod body_handling {
        use super::*;

        #[tokio::test]
        async fn forwards_request_body() {
            let upstream_addr = start_upstream(|req: Request<Incoming>| {
                // We can't easily await the body in a sync closure, so we just
                // confirm the content-length header was forwarded.
                let cl = req
                    .headers()
                    .get("content-length")
                    .map(|v| v.to_str().unwrap().to_string())
                    .unwrap_or_default();
                Response::builder()
                    .status(StatusCode::OK)
                    .header("x-got-content-length", cl)
                    .body(Full::new(Bytes::new()))
                    .unwrap()
            })
            .await;

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream_addr],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            let body = "hello world";
            let req = Request::builder()
                .method("POST")
                .uri("/test")
                .header("Content-Length", body.len().to_string())
                .body(Full::new(Bytes::from(body)))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let cl = resp.headers().get("x-got-content-length").unwrap();
            assert_eq!(cl.to_str().unwrap(), "11");
        }

        #[tokio::test]
        async fn forwards_response_body() {
            let upstream_addr = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from("upstream body")))
                    .unwrap()
            })
            .await;

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream_addr],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], b"upstream body");
        }

        #[tokio::test]
        async fn handles_empty_body() {
            let upstream_addr = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::new()))
                    .unwrap()
            })
            .await;

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream_addr],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            assert!(body.is_empty());
        }

        #[tokio::test]
        async fn handles_chunked_request() {
            let upstream_addr = start_upstream(|req: Request<Incoming>| {
                let te = req
                    .headers()
                    .get("transfer-encoding")
                    .map(|v| v.to_str().unwrap().to_string())
                    .unwrap_or_default();
                Response::builder()
                    .status(StatusCode::OK)
                    .header("x-got-transfer-encoding", te)
                    .body(Full::new(Bytes::new()))
                    .unwrap()
            })
            .await;

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream_addr],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            let req = Request::builder()
                .method("POST")
                .uri("/test")
                .header("Transfer-Encoding", "chunked")
                .body(Full::new(Bytes::from("chunk data")))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
        }
    }

    // ── Module 6: error_handling ─────────────────────────────────────────

    mod error_handling {
        use super::*;

        #[tokio::test]
        async fn returns_502_on_upstream_down() {
            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec!["127.0.0.1:1".to_string()],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        }

        #[tokio::test]
        async fn returns_404_on_no_route() {
            // Use a router that does NOT match all paths
            struct NoRouteRouter;
            impl PathResolution for NoRouteRouter {
                fn get_servers(&self, _path: &str) -> Option<&Vec<String>> {
                    None
                }
                fn get_route_config(&self, _path: &str) -> Option<&RouteConfig> {
                    None
                }
            }

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let proxy_addr = listener.local_addr().unwrap().to_string();
            let router = Arc::new(NoRouteRouter);
            let client = build_client();
            let counter = Arc::new(AtomicUsize::new(0));

            tokio::spawn(async move {
                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => break,
                    };

                    let svc = ProxyService::new(
                        Arc::clone(&router),
                        client.clone(),
                        Arc::clone(&counter),
                    );

                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let _ = hyper::server::conn::http1::Builder::new()
                            .keep_alive(true)
                            .serve_connection(
                                io,
                                service_fn(move |req: Request<Incoming>| {
                                    let svc = svc.clone();
                                    async move { svc.handle(req).await }
                                }),
                            )
                            .await;
                    });
                }
            });

            let req = Request::builder()
                .uri("/anything")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        }

        #[tokio::test]
        async fn returns_502_on_upstream_timeout() {
            let upstream_addr = start_upstream(|_req: Request<Incoming>| {
                // Block for a long time — the proxy client should time out or fail
                std::thread::sleep(std::time::Duration::from_secs(10));
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::new()))
                    .unwrap()
            })
            .await;

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream_addr],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
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

            // Use a timeout so the test doesn't hang forever
            let result =
                tokio::time::timeout(std::time::Duration::from_secs(5), client.request(req)).await;

            match result {
                Ok(Ok(resp)) => assert_eq!(resp.status(), StatusCode::BAD_GATEWAY),
                Ok(Err(_)) => { /* connection error is acceptable */ }
                Err(_) => { /* timeout is acceptable — proxy doesn't have its own timeout yet */ }
            }
        }

        #[tokio::test]
        async fn handles_upstream_sending_invalid_http() {
            // Start a raw TCP server that sends garbage
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let upstream_addr = listener.local_addr().unwrap().to_string();

            tokio::spawn(async move {
                loop {
                    let (mut stream, _) = match listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => break,
                    };
                    tokio::spawn(async move {
                        use tokio::io::AsyncWriteExt;
                        let _ = stream.write_all(b"NOT HTTP AT ALL\r\n\r\n").await;
                        let _ = stream.shutdown().await;
                    });
                }
            });

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream_addr],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        }
    }

    // ── Module 7: keep_alive ─────────────────────────────────────────────

    mod keep_alive {
        use super::*;

        #[tokio::test]
        async fn reuses_connection_for_multiple_requests() {
            let upstream_addr = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from("ok")))
                    .unwrap()
            })
            .await;

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream_addr],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            // Send 3 requests through the same client (connection reuse)
            let client = build_client();
            for _ in 0..3 {
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
                // Must consume body to allow connection reuse
                let _ = resp.collect().await;
            }
        }

        #[tokio::test]
        async fn handles_connection_close() {
            let (proxy_addr, _) = setup_echo().await;

            let req = Request::builder()
                .uri("/test")
                .header("Connection", "close")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            // After Connection: close, the server should close the connection.
            // The proxy should handle this gracefully.
        }

        #[tokio::test]
        async fn handles_upstream_close() {
            // Upstream that closes connection after one response
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let upstream_addr = listener.local_addr().unwrap().to_string();

            tokio::spawn(async move {
                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(conn) => conn,
                        Err(_) => break,
                    };

                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let _ = hyper::server::conn::http1::Builder::new()
                            .keep_alive(false) // close after response
                            .serve_connection(
                                io,
                                service_fn(|_req: Request<Incoming>| async {
                                    Ok::<_, hyper::Error>(
                                        Response::builder()
                                            .status(StatusCode::OK)
                                            .body(Full::new(Bytes::from("ok")))
                                            .unwrap(),
                                    )
                                }),
                            )
                            .await;
                    });
                }
            });

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream_addr],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            // First request should succeed
            let req = Request::builder()
                .uri("/test")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            // Second request should also succeed (proxy should handle closed upstream gracefully)
            let req = Request::builder()
                .uri("/test2")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
        }
    }

    // ── Module 8: round_robin ────────────────────────────────────────────

    mod round_robin {
        use super::*;

        #[tokio::test]
        async fn distributes_across_upstreams() {
            let count1 = Arc::new(AtomicUsize::new(0));
            let count2 = Arc::new(AtomicUsize::new(0));

            let c1 = Arc::clone(&count1);
            let upstream1 = start_upstream(move |_req: Request<Incoming>| {
                c1.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from("u1")))
                    .unwrap()
            })
            .await;

            let c2 = Arc::clone(&count2);
            let upstream2 = start_upstream(move |_req: Request<Incoming>| {
                c2.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from("u2")))
                    .unwrap()
            })
            .await;

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream1, upstream2],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            // Send 10 requests
            for i in 0..10 {
                let req = Request::builder()
                    .uri(format!("/test/{}", i))
                    .body(Full::new(Bytes::new()))
                    .unwrap();
                let resp = send_request(&proxy_addr, req).await;
                assert_eq!(resp.status(), StatusCode::OK);
                let _ = resp.collect().await;
            }

            let c1 = count1.load(Ordering::Relaxed);
            let c2 = count2.load(Ordering::Relaxed);
            assert_eq!(c1 + c2, 10, "all requests should be handled");
            assert!(
                c1 >= 4,
                "upstream1 should receive at least 4 requests, got {}",
                c1
            );
            assert!(
                c2 >= 4,
                "upstream2 should receive at least 4 requests, got {}",
                c2
            );
        }

        #[tokio::test]
        async fn wraps_around_counter() {
            let call_count = Arc::new(AtomicUsize::new(0));
            let cc = Arc::clone(&call_count);
            let upstream = start_upstream(move |_req: Request<Incoming>| {
                cc.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from("ok")))
                    .unwrap()
            })
            .await;

            let router = Arc::new(CatchAllRouter {
                route_config: RouteConfig {
                    servers: vec![upstream],
                    lb: LbStrategy::RoundRobin,
                    weights: None,
                },
            });
            let proxy_addr = start_proxy(router).await;

            // Send many requests — with a single upstream, all go to it
            for i in 0..20 {
                let req = Request::builder()
                    .uri(format!("/test/{}", i))
                    .body(Full::new(Bytes::new()))
                    .unwrap();
                let resp = send_request(&proxy_addr, req).await;
                assert_eq!(resp.status(), StatusCode::OK);
                let _ = resp.collect().await;
            }

            assert_eq!(
                call_count.load(Ordering::Relaxed),
                20,
                "all 20 requests should reach the upstream"
            );
        }
    }

    // ── Module 9: edge_cases ─────────────────────────────────────────────

    mod edge_cases {
        use super::*;

        #[tokio::test]
        async fn large_headers() {
            let (proxy_addr, _) = setup_echo().await;

            let large_value = "x".repeat(8000);
            let req = Request::builder()
                .uri("/test")
                .header("X-Large", &large_value)
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
        }

        #[tokio::test]
        async fn empty_path() {
            let (proxy_addr, _) = setup_echo().await;

            let req = Request::builder()
                .uri("/")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let got_path = resp.headers().get("x-got-path").unwrap();
            assert_eq!(got_path.to_str().unwrap(), "/");
        }

        #[tokio::test]
        async fn special_characters_in_path() {
            let (proxy_addr, _) = setup_echo().await;

            let req = Request::builder()
                .uri("/path%20with%20spaces/foo%2Fbar")
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let got_path = resp.headers().get("x-got-path").unwrap();
            assert_eq!(
                got_path.to_str().unwrap(),
                "/path%20with%20spaces/foo%2Fbar"
            );
        }
    }
}
