#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    use proxy_core::config::{LbStrategy, RouteConfig};
    use proxy_core::handlers::proxy::{build_client, ProxyService};
    use trp_test_support::{send_request, CatchAllRouter};

    /// Start a counting upstream, returning (addr, counter).
    async fn start_counting_upstream() -> (String, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&count);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

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
                            hyper::service::service_fn(move |_req: Request<Incoming>| {
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

        (addr, count)
    }

    /// Start a proxy with a custom route config.
    async fn start_proxy_with_route(route_config: RouteConfig) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        let router = Arc::new(CatchAllRouter { route_config });
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
                            hyper::service::service_fn(move |req: Request<Incoming>| {
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

    #[tokio::test]
    async fn consistent_hash_same_client_same_server() {
        let (addr1, count1) = start_counting_upstream().await;
        let (addr2, count2) = start_counting_upstream().await;

        let route_config = RouteConfig {
            servers: vec![addr1, addr2],
            lb: LbStrategy::ConsistentHash,
            weights: None,
        };
        let proxy_addr = start_proxy_with_route(route_config).await;

        // Same client IP (127.0.0.1) should always hash to the same server.
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

        // All 10 requests should go to the same server.
        assert!(
            c1 == 10 || c2 == 10,
            "consistent hash should route all requests to one server: c1={}, c2={}",
            c1,
            c2
        );
    }

    #[tokio::test]
    async fn round_robin_distributes_evenly() {
        let (addr1, count1) = start_counting_upstream().await;
        let (addr2, count2) = start_counting_upstream().await;

        let route_config = RouteConfig {
            servers: vec![addr1, addr2],
            lb: LbStrategy::RoundRobin,
            weights: None,
        };
        let proxy_addr = start_proxy_with_route(route_config).await;

        let total = 20usize;
        for i in 0..total {
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

        assert_eq!(c1 + c2, total, "all requests should be handled");

        // Round-robin should distribute evenly across 2 servers.
        assert_eq!(
            c1,
            total / 2,
            "server1 should get half: c1={}, c2={}",
            c1,
            c2
        );
        assert_eq!(
            c2,
            total / 2,
            "server2 should get half: c1={}, c2={}",
            c1,
            c2
        );
    }
}
