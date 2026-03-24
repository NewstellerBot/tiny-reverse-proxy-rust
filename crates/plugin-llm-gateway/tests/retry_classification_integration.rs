#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};

    use plugin_llm_gateway::metrics::LlmMetrics;
    use plugin_llm_gateway::provider_failover::ProviderFailover;
    use plugin_llm_gateway::virtual_keys::VirtualKeys;
    use proxy_core::config::{
        EmbeddingSurface, EmbeddingSurfaceProtocol, ProviderCommonConfig, ProviderFamily,
        ProviderFamilyConfig, ProviderKeyConfig, ProviderSurfaceCatalog, ResponsesSurface,
        ToolSurface,
    };
    use proxy_core::metrics::Metrics;
    use proxy_core::plugin::PluginChain;
    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream, TestProxyConfig,
    };

    fn canonical_provider(
        name: &str,
        api_key: &str,
        base_url: impl Into<String>,
        models: Vec<String>,
        family: ProviderFamily,
        surfaces: ProviderSurfaceCatalog,
    ) -> ProviderKeyConfig {
        ProviderKeyConfig::new(
            ProviderCommonConfig {
                name: name.to_string(),
                api_key: api_key.to_string(),
                base_url: base_url.into(),
                models,
                api_key_header: "authorization".to_string(),
                timeout_secs: None,
                routing_metadata: Default::default(),
            },
            ProviderFamilyConfig::from_parts(family, surfaces).unwrap(),
        )
    }

    fn openai_chat_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            ..Default::default()
        }
    }

    fn openai_responses_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            responses: Some(ResponsesSurface::OpenAiCompatible),
            ..Default::default()
        }
    }

    fn gemini_embedding_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            embeddings: Some(EmbeddingSurface {
                protocol: EmbeddingSurfaceProtocol::GeminiEmbedContent,
            }),
            ..Default::default()
        }
    }

    fn openai_embedding_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            embeddings: Some(EmbeddingSurface {
                protocol: EmbeddingSurfaceProtocol::OpenAiEmbeddings,
            }),
            ..Default::default()
        }
    }

    fn request_body(model: &str) -> Bytes {
        Bytes::from(
            serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "hello"}]
            })
            .to_string(),
        )
    }

    fn responses_request_body(model: &str) -> Bytes {
        Bytes::from(
            serde_json::json!({
                "model": model,
                "input": "hello"
            })
            .to_string(),
        )
    }

    async fn start_gateway_with_failover(
        primary_addr: &str,
        fallback_addr: &str,
        primary_handler_name: &str,
    ) -> (String, String, ProviderFailover, Metrics, LlmMetrics) {
        let registry = prometheus::Registry::new();
        let core_metrics = Metrics::new_with_registry(registry.clone());
        let llm_metrics = LlmMetrics::new(&registry);

        let providers = vec![
            canonical_provider(
                "provider-a",
                "sk-a",
                format!("http://{primary_addr}"),
                vec!["gpt-4o".to_string()],
                ProviderFamily::OpenAi,
                openai_chat_surfaces(),
            ),
            canonical_provider(
                "provider-b",
                "sk-b",
                format!("http://{fallback_addr}"),
                vec!["gpt-4o".to_string()],
                ProviderFamily::OpenAi,
                openai_chat_surfaces(),
            ),
        ];

        let mut virtual_keys = VirtualKeys::new(&providers, &[], None);
        virtual_keys = virtual_keys.with_metrics(llm_metrics.clone());
        let (plaintext_key, _) = virtual_keys
            .create_key_for_project(
                Some("project-a"),
                primary_handler_name,
                "provider-a",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        let failover = ProviderFailover::new(
            vec![
                ("provider-a".to_string(), format!("http://{primary_addr}")),
                ("provider-b".to_string(), format!("http://{fallback_addr}")),
            ],
            60,
        )
        .with_metrics(llm_metrics.clone());

        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![primary_addr.to_string()]),
            TestProxyConfig {
                metrics: Some(core_metrics.clone()),
                plugins: Some(Arc::new(PluginChain::new(vec![
                    Box::new(virtual_keys),
                    Box::new(failover.clone()),
                ]))),
                ..Default::default()
            },
        )
        .await;

        (
            proxy_addr,
            plaintext_key,
            failover,
            core_metrics,
            llm_metrics,
        )
    }

    async fn start_responses_gateway_with_failover(
        primary_addr: &str,
        fallback_addr: &str,
        key_name: &str,
    ) -> (String, String, ProviderFailover, Metrics, LlmMetrics) {
        let registry = prometheus::Registry::new();
        let core_metrics = Metrics::new_with_registry(registry.clone());
        let llm_metrics = LlmMetrics::new(&registry);

        let providers = vec![
            canonical_provider(
                "provider-a",
                "sk-a",
                format!("http://{primary_addr}"),
                vec!["gpt-4.1-mini".to_string()],
                ProviderFamily::OpenAi,
                openai_responses_surfaces(),
            ),
            canonical_provider(
                "provider-b",
                "sk-b",
                format!("http://{fallback_addr}"),
                vec!["gpt-4.1-mini".to_string()],
                ProviderFamily::OpenAi,
                openai_responses_surfaces(),
            ),
        ];

        let mut virtual_keys = VirtualKeys::new(&providers, &[], None);
        virtual_keys = virtual_keys.with_metrics(llm_metrics.clone());
        let (plaintext_key, _) = virtual_keys
            .create_key_for_project(
                Some("project-a"),
                key_name,
                "provider-a",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        let failover = ProviderFailover::new(
            vec![
                ("provider-a".to_string(), format!("http://{primary_addr}")),
                ("provider-b".to_string(), format!("http://{fallback_addr}")),
            ],
            60,
        )
        .with_metrics(llm_metrics.clone());

        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![primary_addr.to_string()]),
            TestProxyConfig {
                metrics: Some(core_metrics.clone()),
                plugins: Some(Arc::new(PluginChain::new(vec![
                    Box::new(virtual_keys),
                    Box::new(failover.clone()),
                ]))),
                ..Default::default()
            },
        )
        .await;

        (
            proxy_addr,
            plaintext_key,
            failover,
            core_metrics,
            llm_metrics,
        )
    }

    #[tokio::test]
    async fn provider_429_failover_records_retry_and_cooldown() {
        let primary_hits = Arc::new(AtomicUsize::new(0));
        let primary_addr = start_upstream({
            let primary_hits = Arc::clone(&primary_hits);
            move |_req: Request<Incoming>| {
                primary_hits.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::TOO_MANY_REQUESTS)
                    .body(Full::new(Bytes::from("rate limited")))
                    .unwrap()
            }
        })
        .await;

        let fallback_hits = Arc::new(AtomicUsize::new(0));
        let fallback_addr = start_upstream({
            let fallback_hits = Arc::clone(&fallback_hits);
            move |_req: Request<Incoming>| {
                fallback_hits.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(r#"{"ok":true}"#)))
                    .unwrap()
            }
        })
        .await;

        let (proxy_addr, plaintext_key, failover, core_metrics, llm_metrics) =
            start_gateway_with_failover(&primary_addr, &fallback_addr, "retry-429").await;

        let response = send_request(
            &proxy_addr,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {plaintext_key}"))
                .header("content-type", "application/json")
                .body(Full::new(request_body("gpt-4o")))
                .unwrap(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(primary_hits.load(Ordering::Relaxed), 1);
        assert_eq!(fallback_hits.load(Ordering::Relaxed), 1);
        let failed = failover.get_failed_providers();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, "provider-a");
        assert_eq!(
            core_metrics
                .retry_attempts_total
                .with_label_values(&["provider", "429"])
                .get(),
            1
        );
        assert_eq!(
            llm_metrics
                .provider_cooldowns_total
                .with_label_values(&["provider-a", "rate_limited"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn second_responses_request_uses_fallback_without_disconnect_when_primary_is_in_cooldown()
    {
        let primary_hits = Arc::new(AtomicUsize::new(0));
        let primary_addr = start_upstream({
            let primary_hits = Arc::clone(&primary_hits);
            move |_req: Request<Incoming>| {
                primary_hits.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::TOO_MANY_REQUESTS)
                    .body(Full::new(Bytes::from("rate limited")))
                    .unwrap()
            }
        })
        .await;

        let fallback_hits = Arc::new(AtomicUsize::new(0));
        let fallback_addr = start_upstream({
            let fallback_hits = Arc::clone(&fallback_hits);
            move |_req: Request<Incoming>| {
                fallback_hits.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "id": "resp-fallback",
                            "object": "response",
                            "model": "gpt-4.1-mini",
                            "output": [{
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": "ok"
                                }]
                            }]
                        })
                        .to_string(),
                    )))
                    .unwrap()
            }
        })
        .await;

        let (proxy_addr, plaintext_key, failover, core_metrics, llm_metrics) =
            start_responses_gateway_with_failover(&primary_addr, &fallback_addr, "retry-responses")
                .await;

        let make_request = || {
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", format!("Bearer {plaintext_key}"))
                .header("content-type", "application/json")
                .body(Full::new(responses_request_body("gpt-4.1-mini")))
                .unwrap()
        };

        let first = send_request(&proxy_addr, make_request()).await;
        assert_eq!(first.status(), StatusCode::OK);

        let second = send_request(&proxy_addr, make_request()).await;
        assert_eq!(second.status(), StatusCode::OK);

        assert_eq!(primary_hits.load(Ordering::Relaxed), 1);
        assert_eq!(fallback_hits.load(Ordering::Relaxed), 2);
        assert_eq!(failover.get_failed_providers().len(), 1);
        assert_eq!(
            core_metrics
                .retry_attempts_total
                .with_label_values(&["provider", "429"])
                .get(),
            1
        );
        assert_eq!(
            llm_metrics
                .provider_cooldowns_total
                .with_label_values(&["provider-a", "rate_limited"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn provider_5xx_failover_records_retry_and_cooldown() {
        let primary_hits = Arc::new(AtomicUsize::new(0));
        let primary_addr = start_upstream({
            let primary_hits = Arc::clone(&primary_hits);
            move |_req: Request<Incoming>| {
                primary_hits.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::new(Bytes::from("provider error")))
                    .unwrap()
            }
        })
        .await;

        let fallback_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(r#"{"ok":true}"#)))
                .unwrap()
        })
        .await;

        let (proxy_addr, plaintext_key, failover, core_metrics, llm_metrics) =
            start_gateway_with_failover(&primary_addr, &fallback_addr, "retry-5xx").await;

        let response = send_request(
            &proxy_addr,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {plaintext_key}"))
                .header("content-type", "application/json")
                .body(Full::new(request_body("gpt-4o")))
                .unwrap(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(failover.get_failed_providers().len(), 1);
        assert_eq!(
            core_metrics
                .retry_attempts_total
                .with_label_values(&["provider", "5xx"])
                .get(),
            1
        );
        assert_eq!(
            llm_metrics
                .provider_cooldowns_total
                .with_label_values(&["provider-a", "upstream_5xx"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn auth_failure_is_non_retryable_and_does_not_failover() {
        let primary_hits = Arc::new(AtomicUsize::new(0));
        let primary_addr = start_upstream({
            let primary_hits = Arc::clone(&primary_hits);
            move |_req: Request<Incoming>| {
                primary_hits.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::UNAUTHORIZED)
                    .body(Full::new(Bytes::from("bad auth")))
                    .unwrap()
            }
        })
        .await;

        let fallback_hits = Arc::new(AtomicUsize::new(0));
        let fallback_addr = start_upstream({
            let fallback_hits = Arc::clone(&fallback_hits);
            move |_req: Request<Incoming>| {
                fallback_hits.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(r#"{"ok":true}"#)))
                    .unwrap()
            }
        })
        .await;

        let (proxy_addr, plaintext_key, failover, core_metrics, _llm_metrics) =
            start_gateway_with_failover(&primary_addr, &fallback_addr, "no-retry-401").await;

        let response = send_request(
            &proxy_addr,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {plaintext_key}"))
                .header("content-type", "application/json")
                .body(Full::new(request_body("gpt-4o")))
                .unwrap(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(primary_hits.load(Ordering::Relaxed), 1);
        assert_eq!(fallback_hits.load(Ordering::Relaxed), 0);
        assert!(failover.get_failed_providers().is_empty());
        assert_eq!(
            core_metrics
                .retry_attempts_total
                .with_label_values(&["provider", "non_retryable"])
                .get(),
            0
        );
        assert_eq!(
            core_metrics
                .retry_exhaustions_total
                .with_label_values(&["provider", "non_retryable"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn all_provider_candidates_failing_reports_exhaustion() {
        let primary_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::new(Bytes::from("a-down")))
                .unwrap()
        })
        .await;

        let fallback_addr = start_upstream(|_req: Request<Incoming>| {
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Full::new(Bytes::from("b-down")))
                .unwrap()
        })
        .await;

        let (proxy_addr, plaintext_key, _failover, core_metrics, _llm_metrics) =
            start_gateway_with_failover(&primary_addr, &fallback_addr, "all-fail").await;

        let response = send_request(
            &proxy_addr,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {plaintext_key}"))
                .header("content-type", "application/json")
                .body(Full::new(request_body("gpt-4o")))
                .unwrap(),
        )
        .await;

        assert!(response.status().is_server_error());
        assert_eq!(
            core_metrics
                .retry_attempts_total
                .with_label_values(&["provider", "5xx"])
                .get(),
            1
        );
        assert_eq!(
            core_metrics
                .retry_exhaustions_total
                .with_label_values(&["provider", "5xx"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn malformed_translated_embeddings_response_fails_closed_without_fallback() {
        let translated_hits = Arc::new(AtomicUsize::new(0));
        let translated_addr = start_upstream({
            let translated_hits = Arc::clone(&translated_hits);
            move |_req: Request<Incoming>| {
                translated_hits.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from("{not-json")))
                    .unwrap()
            }
        })
        .await;
        let fallback_hits = Arc::new(AtomicUsize::new(0));
        let fallback_addr = start_upstream({
            let fallback_hits = Arc::clone(&fallback_hits);
            move |_req: Request<Incoming>| {
                fallback_hits.fetch_add(1, Ordering::Relaxed);
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "object": "list",
                            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                            "model": "text-embedding-3-large"
                        })
                        .to_string(),
                    )))
                    .unwrap()
            }
        })
        .await;

        let providers = vec![
            canonical_provider(
                "gemini-embeddings",
                "sk-gemini",
                format!("http://{translated_addr}"),
                vec!["text-embedding-3-large".to_string()],
                ProviderFamily::Gemini,
                gemini_embedding_surfaces(),
            ),
            canonical_provider(
                "openai-embeddings",
                "sk-openai",
                format!("http://{fallback_addr}"),
                vec!["text-embedding-3-large".to_string()],
                ProviderFamily::OpenAi,
                openai_embedding_surfaces(),
            ),
        ];

        let vk = VirtualKeys::new(&providers, &[], None);
        let (plaintext_key, _) = vk
            .create_key_for_project(
                Some("project-a"),
                "malformed-embedding-upstream",
                "gemini-embeddings",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![translated_addr.clone()]),
            TestProxyConfig {
                plugins: Some(Arc::new(PluginChain::new(vec![Box::new(vk)]))),
                ..Default::default()
            },
        )
        .await;

        let response = send_request(
            &proxy_addr,
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("authorization", format!("Bearer {plaintext_key}"))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "text-embedding-3-large",
                        "input": "hello world"
                    })
                    .to_string(),
                )))
                .unwrap(),
        )
        .await;

        let status = response.status();
        let body = response.collect().await.unwrap().to_bytes();
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert!(!body.is_empty());
        assert_eq!(translated_hits.load(Ordering::Relaxed), 1);
        assert_eq!(fallback_hits.load(Ordering::Relaxed), 0);
    }
}
