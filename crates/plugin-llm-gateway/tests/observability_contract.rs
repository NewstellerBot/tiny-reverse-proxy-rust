#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::http::Extensions;
    use hyper::{Request, Version};

    use plugin_llm_gateway::api::LlmGatewayApi;
    use plugin_llm_gateway::metrics::LlmMetrics;
    use plugin_llm_gateway::provider_failover::ProviderFailover;
    use proxy_core::plugin::{Plugin, ProxyError, RequestContext};
    use proxy_core::runtime::{ProbeState, RuntimeReliabilityState};

    async fn start_mgmt_server(api: LlmGatewayApi) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            use hyper::service::service_fn;
            use hyper_util::rt::TokioIo;

            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => continue,
                };
                let api = api.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let svc = service_fn(move |req: Request<Incoming>| {
                        let api = api.clone();
                        async move {
                            plugin_llm_gateway::management_server::handle_request(req, api).await
                        }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, svc)
                        .await;
                });
            }
        });

        port
    }

    async fn mgmt_get(port: u16, path: &str) -> (u16, serde_json::Value) {
        let client =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http::<Full<Bytes>>();
        let uri: hyper::Uri = format!("http://127.0.0.1:{port}{path}").parse().unwrap();
        let req = Request::builder()
            .method("GET")
            .uri(uri)
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = client.request(req).await.unwrap();
        let status = resp.status().as_u16();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn status_endpoint_exposes_runtime_reliability_fields() {
        let inflight_requests = Arc::new(AtomicUsize::new(2));
        let probe_state = ProbeState::new();
        probe_state.mark_draining("deploy");

        let api = LlmGatewayApi::new(None, None, None, None, None, None, None, None, None)
            .with_runtime_reliability(RuntimeReliabilityState::new(
                inflight_requests,
                Some(5),
                Some(2),
                probe_state,
            ));
        let port = start_mgmt_server(api).await;

        let (status, body) = mgmt_get(port, "/api/v1/status").await;
        assert_eq!(status, 200);
        assert_eq!(body["reliability"]["inflight_requests"].as_u64(), Some(2));
        assert_eq!(
            body["reliability"]["max_inflight_requests"].as_u64(),
            Some(5)
        );
        assert_eq!(body["reliability"]["brownout_threshold"].as_u64(), Some(2));
        assert_eq!(body["reliability"]["brownout_active"].as_bool(), Some(true));
        assert_eq!(body["reliability"]["draining"].as_bool(), Some(true));
    }

    #[tokio::test]
    async fn failed_provider_surfaces_expose_failover_state() {
        let registry = prometheus::Registry::new();
        let llm_metrics = LlmMetrics::new(&registry);
        let failover = ProviderFailover::new(
            vec![("provider-a".to_string(), "http://provider-a".to_string())],
            60,
        )
        .with_metrics(llm_metrics.clone());

        let mut ctx = RequestContext {
            peer_addr: None,
            method: hyper::Method::POST,
            uri: hyper::Uri::from_static("http://localhost/v1/chat/completions"),
            version: Version::HTTP_11,
            headers: hyper::HeaderMap::new(),
            body: None,
            route: None,
            selected_upstream: Some("http://provider-a/v1/chat/completions".to_string()),
            auth: None,
            connection: Arc::new(Extensions::new()),
            extensions: Extensions::new(),
        };
        let _ = failover
            .on_error(
                &mut ctx,
                &ProxyError::UpstreamStatus(hyper::StatusCode::TOO_MANY_REQUESTS),
            )
            .await;

        let api = LlmGatewayApi::new(
            None,
            None,
            Some(failover.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let port = start_mgmt_server(api).await;

        let (failed_status, failed_body) = mgmt_get(port, "/api/v1/providers/failed").await;
        assert_eq!(failed_status, 200);
        assert_eq!(
            failed_body["failed"][0]["name"].as_str(),
            Some("provider-a")
        );
        assert_eq!(
            failed_body["failed"][0]["reason"].as_str(),
            Some("rate_limited")
        );

        let (health_status, health_body) = mgmt_get(port, "/api/v1/providers/health").await;
        assert_eq!(health_status, 200);
        assert_eq!(
            health_body["providers"][0]["cooldown_reason"].as_str(),
            Some("rate_limited")
        );
        assert_eq!(
            llm_metrics
                .provider_cooldown_active
                .with_label_values(&["provider-a"])
                .get(),
            1
        );
    }
}
