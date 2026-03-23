#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
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

    struct MockRouter {
        route_config: RouteConfig,
    }

    impl PathResolution for MockRouter {
        fn get_servers(&self, path: &str) -> Option<&Vec<String>> {
            if path == "/test" {
                Some(&self.route_config.servers)
            } else {
                None
            }
        }
        fn get_route_config(&self, path: &str) -> Option<&RouteConfig> {
            if path == "/test" {
                Some(&self.route_config)
            } else {
                None
            }
        }
    }

    /// Start a ProxyService on a random port. Returns the bound address.
    async fn start_proxy(router: Arc<MockRouter>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        let client = build_client();
        let counter = Arc::new(AtomicUsize::new(0));

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };

                let svc =
                    ProxyService::new(Arc::clone(&router), client.clone(), Arc::clone(&counter));

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

    /// Start a mock upstream HTTP server that responds with a given body.
    async fn start_mock_upstream(response_body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };

                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .keep_alive(true)
                        .serve_connection(
                            io,
                            service_fn(move |_req: Request<Incoming>| async move {
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(StatusCode::OK)
                                        .body(Full::new(Bytes::from(response_body)))
                                        .unwrap(),
                                )
                            }),
                        )
                        .await;
                });
            }
        });

        addr
    }

    /// Helper: send a GET request to the proxy and return (status, body).
    async fn get(addr: &str, path: &str) -> (StatusCode, String) {
        let client = build_client();
        let uri = format!("http://{}{}", addr, path);
        let resp = client
            .request(
                Request::builder()
                    .uri(&uri)
                    .body(
                        Full::new(Bytes::new())
                            .map_err(|never| match never {})
                            .boxed(),
                    )
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = resp.collect().await.unwrap().to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn test_proxy_forward_success() {
        let upstream_addr = start_mock_upstream("Hello from upstream").await;

        let router = Arc::new(MockRouter {
            route_config: RouteConfig {
                servers: vec![upstream_addr],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy(router).await;

        let (status, body) = get(&proxy_addr, "/test").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "Hello from upstream");
    }

    #[tokio::test]
    async fn test_proxy_no_route_404() {
        let router = Arc::new(MockRouter {
            route_config: RouteConfig {
                servers: vec!["127.0.0.1:1".to_string()],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy(router).await;

        let (status, body) = get(&proxy_addr, "/unknown").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("404 Not Found"));
    }

    #[tokio::test]
    async fn test_proxy_upstream_unreachable_502() {
        // Point at a port that nothing is listening on
        let router = Arc::new(MockRouter {
            route_config: RouteConfig {
                servers: vec!["127.0.0.1:1".to_string()],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy(router).await;

        let (status, body) = get(&proxy_addr, "/test").await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert!(body.contains("502 Bad Gateway"));
    }
}
