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
    use tokio::net::{TcpListener, TcpStream};
    use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
    use tokio_rustls::rustls::{ClientConfig, RootCertStore, ServerConfig};
    use tokio_rustls::{TlsAcceptor, TlsConnector};

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

    /// Generate a self-signed certificate for testing using rcgen.
    fn generate_test_cert() -> (CertificateDer<'static>, PrivateKeyDer<'static>) {
        let rcgen::CertifiedKey { cert, signing_key } =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert_der = cert.der().clone();
        let key_der = PrivateKeyDer::Pkcs8(signing_key.serialize_der().into());
        (cert_der, key_der)
    }

    /// Start a TLS-enabled proxy on a random port. Returns the bound address.
    async fn start_tls_proxy(router: Arc<CatchAllRouter>, acceptor: TlsAcceptor) -> String {
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

                let acceptor = acceptor.clone();
                let svc =
                    ProxyService::new(Arc::clone(&router), client.clone(), Arc::clone(&counter))
                        .with_peer_addr(peer_addr)
                        .with_tls(true);

                tokio::spawn(async move {
                    let tls_stream = match acceptor.accept(stream).await {
                        Ok(s) => s,
                        Err(_) => return,
                    };

                    let io = TokioIo::new(tls_stream);
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

    /// Start a mock upstream that echoes the X-Forwarded-Proto header back.
    async fn start_echo_upstream() -> String {
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
                            service_fn(|req: Request<Incoming>| async move {
                                let proto = req
                                    .headers()
                                    .get("x-forwarded-proto")
                                    .map(|v| v.to_str().unwrap().to_string())
                                    .unwrap_or_default();
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(StatusCode::OK)
                                        .header("x-got-proto", proto)
                                        .body(Full::new(Bytes::from("tls ok")))
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
    async fn tls_proxy_sets_forwarded_proto_https() {
        let (cert_der, key_der) = generate_test_cert();

        // Build server TLS config
        let server_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert_der.clone()], key_der)
            .unwrap();
        let acceptor = TlsAcceptor::from(Arc::new(server_config));

        // Start upstream and TLS proxy
        let upstream_addr = start_echo_upstream().await;
        let router = Arc::new(CatchAllRouter {
            route_config: RouteConfig {
                servers: vec![upstream_addr],
                lb: LbStrategy::RoundRobin,
                weights: None,
            },
        });
        let proxy_addr = start_tls_proxy(router, acceptor).await;

        // Build TLS client that trusts our self-signed cert
        let mut root_store = RootCertStore::empty();
        root_store.add(cert_der).unwrap();
        let client_config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();
        let connector = TlsConnector::from(Arc::new(client_config));

        // Connect to proxy via TLS
        let tcp = TcpStream::connect(&proxy_addr).await.unwrap();
        let server_name = ServerName::try_from("localhost").unwrap();
        let tls_stream = connector.connect(server_name, tcp).await.unwrap();

        // Send HTTP request over TLS
        let io = TokioIo::new(tls_stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();
        tokio::spawn(conn);

        let req = Request::builder()
            .uri("/test")
            .header("Host", "localhost")
            .body({
                let b: http_body_util::combinators::BoxBody<Bytes, hyper::Error> =
                    Full::new(Bytes::new())
                        .map_err(|never| match never {})
                        .boxed();
                b
            })
            .unwrap();

        let resp = sender.send_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify upstream received X-Forwarded-Proto: https
        let proto = resp
            .headers()
            .get("x-got-proto")
            .expect("upstream should echo X-Forwarded-Proto");
        assert_eq!(proto.to_str().unwrap(), "https");

        // Verify body
        let body = resp.collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"tls ok");
    }

    #[tokio::test]
    async fn tls_proxy_forwards_request_body() {
        let (cert_der, key_der) = generate_test_cert();

        let server_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert_der.clone()], key_der)
            .unwrap();
        let acceptor = TlsAcceptor::from(Arc::new(server_config));

        // Upstream echoes content-length
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_addr = upstream_listener.local_addr().unwrap().to_string();

        tokio::spawn(async move {
            loop {
                let (stream, _) = match upstream_listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let _ = hyper::server::conn::http1::Builder::new()
                        .keep_alive(true)
                        .serve_connection(
                            io,
                            service_fn(|req: Request<Incoming>| async move {
                                let cl = req
                                    .headers()
                                    .get("content-length")
                                    .map(|v| v.to_str().unwrap().to_string())
                                    .unwrap_or_default();
                                Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(StatusCode::OK)
                                        .header("x-got-content-length", cl)
                                        .body(Full::new(Bytes::new()))
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
        let proxy_addr = start_tls_proxy(router, acceptor).await;

        // Build TLS client
        let mut root_store = RootCertStore::empty();
        root_store.add(cert_der).unwrap();
        let client_config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();
        let connector = TlsConnector::from(Arc::new(client_config));

        let tcp = TcpStream::connect(&proxy_addr).await.unwrap();
        let server_name = ServerName::try_from("localhost").unwrap();
        let tls_stream = connector.connect(server_name, tcp).await.unwrap();

        let io = TokioIo::new(tls_stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io).await.unwrap();
        tokio::spawn(conn);

        let body = "hello via tls";
        let req = Request::builder()
            .method("POST")
            .uri("/test")
            .header("Host", "localhost")
            .header("Content-Length", body.len().to_string())
            .body({
                let b: http_body_util::combinators::BoxBody<Bytes, hyper::Error> =
                    Full::new(Bytes::from(body))
                        .map_err(|never| match never {})
                        .boxed();
                b
            })
            .unwrap();

        let resp = sender.send_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let cl = resp.headers().get("x-got-content-length").unwrap();
        assert_eq!(cl.to_str().unwrap(), "13");
    }
}
