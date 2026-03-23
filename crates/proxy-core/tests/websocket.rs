#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::service::service_fn;
    use hyper::upgrade::OnUpgrade;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    use proxy_core::config::{LbStrategy, RouteConfig};
    use proxy_core::handlers::proxy::{build_client, ProxyService};
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

    /// Start a proxy with HTTP upgrade support.
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
                        .with_upgrades()
                        .await;
                });
            }
        });

        addr
    }

    /// Mock upstream that accepts WebSocket upgrades and echoes data back.
    async fn start_websocket_echo_upstream() -> String {
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
                        .serve_connection(
                            io,
                            service_fn(|mut req: Request<Incoming>| async move {
                                let is_ws = req
                                    .headers()
                                    .get("upgrade")
                                    .and_then(|v| v.to_str().ok())
                                    .map(|v| v.eq_ignore_ascii_case("websocket"))
                                    .unwrap_or(false);

                                if !is_ws {
                                    return Ok::<_, hyper::Error>(
                                        Response::builder()
                                            .status(StatusCode::BAD_REQUEST)
                                            .body(Full::new(Bytes::from("not a websocket")))
                                            .unwrap(),
                                    );
                                }

                                let on_upgrade = req.extensions_mut().remove::<OnUpgrade>();

                                if let Some(on_upgrade) = on_upgrade {
                                    tokio::spawn(async move {
                                        match on_upgrade.await {
                                            Ok(upgraded) => {
                                                let io = TokioIo::new(upgraded);
                                                let (mut rd, mut wr) = tokio::io::split(io);
                                                let _ = tokio::io::copy(&mut rd, &mut wr).await;
                                            }
                                            Err(e) => {
                                                eprintln!("upstream upgrade error: {}", e);
                                            }
                                        }
                                    });
                                }

                                Ok(Response::builder()
                                    .status(StatusCode::SWITCHING_PROTOCOLS)
                                    .header("Upgrade", "websocket")
                                    .header("Connection", "Upgrade")
                                    .body(Full::new(Bytes::new()))
                                    .unwrap())
                            }),
                        )
                        .with_upgrades()
                        .await;
                });
            }
        });

        addr
    }

    /// Mock upstream that rejects WebSocket upgrades with 403.
    async fn start_rejecting_upstream() -> String {
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
                        .serve_connection(
                            io,
                            service_fn(|_req: Request<Incoming>| async move {
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(StatusCode::FORBIDDEN)
                                        .body(Full::new(Bytes::from("upgrade rejected")))
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
    async fn websocket_upgrade_success() {
        let upstream_addr = start_websocket_echo_upstream().await;
        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream_addr],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy(router).await;

        // Use low-level hyper client API for upgrade support.
        let stream = TcpStream::connect(&proxy_addr).await.unwrap();
        let io = TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();

        tokio::spawn(async move {
            conn.with_upgrades().await.ok();
        });

        let req = Request::builder()
            .uri("/test")
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
            .header("Sec-WebSocket-Version", "13")
            .body(Full::new(Bytes::new()))
            .unwrap();

        let resp = sender.send_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SWITCHING_PROTOCOLS);
        assert_eq!(
            resp.headers().get("upgrade").unwrap().to_str().unwrap(),
            "websocket"
        );

        // Get the upgraded connection and verify bidirectional data flow.
        let upgraded = hyper::upgrade::on(resp).await.unwrap();
        let mut io = TokioIo::new(upgraded);

        io.write_all(b"hello websocket").await.unwrap();

        let mut buf = vec![0u8; 15];
        io.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"hello websocket");

        // Send a second message to verify continued tunnel operation.
        io.write_all(b"second").await.unwrap();

        let mut buf2 = vec![0u8; 6];
        io.read_exact(&mut buf2).await.unwrap();
        assert_eq!(&buf2, b"second");
    }

    #[tokio::test]
    async fn websocket_upgrade_rejected() {
        let upstream_addr = start_rejecting_upstream().await;
        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream_addr],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_proxy(router).await;

        let stream = TcpStream::connect(&proxy_addr).await.unwrap();
        let io = TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();

        tokio::spawn(async move {
            conn.with_upgrades().await.ok();
        });

        let req = Request::builder()
            .uri("/test")
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
            .header("Sec-WebSocket-Version", "13")
            .body(Full::new(Bytes::new()))
            .unwrap();

        let resp = sender.send_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        let body = resp.collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"upgrade rejected");
    }
}
