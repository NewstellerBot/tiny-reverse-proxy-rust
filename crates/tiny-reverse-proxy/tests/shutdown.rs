#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::service::service_fn;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;
    use tokio::sync::watch;

    use proxy_core::config::{LbStrategy, RouteConfig};
    use proxy_core::handlers::proxy::{build_client, ProxyService};

    use trp_test_support::CatchAllRouter;

    /// Start a proxy that can be shut down via a watch channel.
    /// Returns (bound_address, shutdown_sender).
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

                // Select between accepting new connections and shutdown signal.
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

                        // Drive connection, but graceful shutdown on signal.
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

    /// Start a slow upstream that delays response by the given duration.
    async fn start_slow_upstream(delay: Duration) -> String {
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
                                tokio::time::sleep(delay).await;
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(StatusCode::OK)
                                        .body(Full::new(Bytes::from("slow ok")))
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

    #[tokio::test]
    async fn in_flight_request_completes_after_shutdown() {
        let upstream_addr = start_slow_upstream(Duration::from_millis(500)).await;
        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream_addr],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });

        let (proxy_addr, shutdown_tx) = start_proxy_with_shutdown(router).await;

        // Send a request (it will take 500ms to complete).
        let client = build_client();
        let request_handle = tokio::spawn({
            let proxy_addr = proxy_addr.clone();
            async move {
                let uri = format!("http://{}/test", proxy_addr);
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
        });

        // Give the request time to start, then trigger shutdown.
        tokio::time::sleep(Duration::from_millis(50)).await;
        shutdown_tx.send(true).unwrap();

        // The in-flight request should still complete successfully.
        let (status, body) = request_handle.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "slow ok");
    }

    #[tokio::test]
    async fn new_connections_refused_after_shutdown() {
        let upstream_addr = start_slow_upstream(Duration::from_millis(10)).await;
        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream_addr],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });

        let (proxy_addr, shutdown_tx) = start_proxy_with_shutdown(router).await;

        // Trigger shutdown immediately.
        shutdown_tx.send(true).unwrap();

        // Give the accept loop time to process the shutdown signal.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // New connections should fail.
        let result = tokio::net::TcpStream::connect(&proxy_addr).await;
        assert!(
            result.is_err(),
            "new TCP connections should be refused after shutdown"
        );
    }
}
