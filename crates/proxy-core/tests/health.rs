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
    use proxy_core::health::HealthState;
    use proxy_core::router::PathResolution;

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

    /// Start a ProxyService with health state on a random port.
    async fn start_proxy_with_health(
        router: Arc<CatchAllRouter>,
        health_state: HealthState,
    ) -> String {
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
                        .with_peer_addr(peer_addr)
                        .with_health_state(health_state.clone());

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

    /// Start a mock upstream that responds with 200 OK.
    async fn start_healthy_upstream() -> String {
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

        addr
    }

    /// Send a GET request and return (status, body).
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
    async fn returns_503_when_all_upstreams_unhealthy() {
        let upstream_addr = start_healthy_upstream().await;
        let health_state = HealthState::new(std::slice::from_ref(&upstream_addr));

        // Mark the upstream as unhealthy.
        health_state.mark_unhealthy(&upstream_addr);

        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream_addr],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy_with_health(router, health_state).await;

        let (status, body) = get(&proxy_addr, "/test").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains("503"));
    }

    #[tokio::test]
    async fn routes_only_to_healthy_upstreams() {
        let healthy_upstream = start_healthy_upstream().await;
        let unhealthy_addr = "127.0.0.1:1".to_string(); // nothing listens here

        let all = vec![unhealthy_addr.clone(), healthy_upstream.clone()];
        let health_state = HealthState::new(&all);

        // Mark the unreachable upstream as unhealthy.
        health_state.mark_unhealthy(&unhealthy_addr);

        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: all,
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy_with_health(router, health_state).await;

        // All requests should succeed (routed to the healthy upstream only).
        for _ in 0..5 {
            let (status, _) = get(&proxy_addr, "/test").await;
            assert_eq!(status, StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn healthy_upstream_serves_traffic() {
        let upstream = start_healthy_upstream().await;
        let health_state = HealthState::new(std::slice::from_ref(&upstream));
        // Upstream starts healthy.

        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy_with_health(router, health_state).await;

        let (status, body) = get(&proxy_addr, "/test").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "ok");
    }

    #[tokio::test]
    async fn recovery_after_marking_healthy() {
        let upstream = start_healthy_upstream().await;
        let health_state = HealthState::new(std::slice::from_ref(&upstream));

        // Mark unhealthy, then recover.
        health_state.mark_unhealthy(&upstream);
        health_state.mark_healthy(&upstream);

        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy_with_health(router, health_state).await;

        let (status, _) = get(&proxy_addr, "/test").await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn round_robin_skips_unhealthy() {
        let count = Arc::new(AtomicUsize::new(0));

        let c = Arc::clone(&count);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let healthy_addr = listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };

                let c = Arc::clone(&c);
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .keep_alive(true)
                        .serve_connection(
                            io,
                            service_fn(move |_req: Request<Incoming>| {
                                c.fetch_add(1, Ordering::Relaxed);
                                async {
                                    Ok::<_, hyper::Error>(
                                        Response::builder()
                                            .status(StatusCode::OK)
                                            .body(Full::new(Bytes::from("ok")))
                                            .unwrap(),
                                    )
                                }
                            }),
                        )
                        .await;
                });
            }
        });

        let unhealthy_addr = "127.0.0.1:1".to_string();
        let all = vec![unhealthy_addr.clone(), healthy_addr.clone()];
        let health_state = HealthState::new(&all);
        health_state.mark_unhealthy(&unhealthy_addr);

        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: all,
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy_with_health(router, health_state).await;

        for i in 0..5 {
            let (status, _) = get(&proxy_addr, &format!("/test/{}", i)).await;
            assert_eq!(status, StatusCode::OK);
        }

        assert_eq!(
            count.load(Ordering::Relaxed),
            5,
            "all 5 requests should reach the healthy upstream"
        );
    }
}
