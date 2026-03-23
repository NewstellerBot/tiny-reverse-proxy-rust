#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};

    use proxy_core::plugin::PluginChain;

    use proxy_core::config::{
        AudioSurface, AudioSurfaceProtocol, BatchSurface, EmbeddingSurface,
        EmbeddingSurfaceProtocol, FileSurface, ImageSurface, ImageSurfaceProtocol,
        PromptCacheProtocol, PromptCacheSurface, ProviderCommonConfig, ProviderFamily,
        ProviderFamilyConfig, ProviderKeyConfig, ProviderRoutingMetadataConfig,
        ProviderSurfaceCatalog, RealtimeSurface, ToolSurface,
    };
    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream,
        start_upstream_async, TestProxyConfig,
    };

    /// Mock upstream that returns a JSON response mimicking an LLM chat completion.
    /// The response body is ~200 bytes so Content-Length is set for cost estimation.
    fn llm_chat_handler(_req: Request<Incoming>) -> Response<Full<Bytes>> {
        let body = r#"{"id":"chatcmpl-abc","object":"chat.completion","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I help you today?"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}"#;
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .header("content-length", body.len().to_string())
            .body(Full::new(Bytes::from(body)))
            .unwrap()
    }

    /// Build a request body that looks like an LLM chat completion request.
    fn chat_request_body() -> Vec<u8> {
        r#"{"model":"gpt-4","messages":[{"role":"user","content":"Say hello"}]}"#
            .as_bytes()
            .to_vec()
    }

    fn chat_request(path: &str, api_key: &str) -> Request<Full<Bytes>> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header("authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(chat_request_body())))
            .unwrap()
    }

    fn canonical_provider(
        name: &str,
        api_key: &str,
        base_url: String,
        models: Vec<String>,
        api_key_header: &str,
        family: ProviderFamily,
        surfaces: ProviderSurfaceCatalog,
    ) -> ProviderKeyConfig {
        ProviderKeyConfig::new(
            ProviderCommonConfig {
                name: name.to_string(),
                api_key: api_key.to_string(),
                base_url,
                models,
                api_key_header: api_key_header.to_string(),
                timeout_secs: None,
                routing_metadata: ProviderRoutingMetadataConfig::default(),
            },
            ProviderFamilyConfig::from_parts(family, surfaces).unwrap(),
        )
    }

    fn openai_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            ..ProviderSurfaceCatalog::default()
        }
    }

    fn openai_file_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            files: Some(FileSurface::OpenAiCompatible),
            ..ProviderSurfaceCatalog::default()
        }
    }

    fn realtime_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            realtime: Some(RealtimeSurface::OpenAiCompatible),
            ..ProviderSurfaceCatalog::default()
        }
    }

    // -----------------------------------------------------------------------
    // rate_limiter integration tests
    // -----------------------------------------------------------------------

    mod rate_limiter {
        use super::*;
        use plugin_llm_gateway::rate_limiter as trl;

        fn make_limiter(tokens_per_minute: f64, burst: f64) -> Arc<PluginChain> {
            let config = toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert(
                    "tokens_per_minute".into(),
                    toml::Value::Float(tokens_per_minute),
                );
                t.insert("burst_tokens".into(), toml::Value::Float(burst));
                t
            });
            let plugin = trl::create(&config).unwrap();
            Arc::new(PluginChain::new(vec![plugin]))
        }

        #[tokio::test]
        async fn allows_requests_within_burst() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr]);

            // burst = 1000 tokens, body ~66 bytes => ~16 tokens per request
            let plugins = make_limiter(60_000.0, 1000.0);
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            for i in 0..5 {
                let req = chat_request(&format!("/v1/chat/completions?n={}", i), "sk-test-key");
                let resp = send_request(&proxy_addr, req).await;
                assert_eq!(
                    resp.status(),
                    StatusCode::OK,
                    "request {} should succeed",
                    i
                );
            }
        }

        #[tokio::test]
        async fn returns_429_when_token_budget_exhausted() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr]);

            // burst = 20 tokens; body ~66 bytes => 16 tokens first request exhausts most
            let plugins = make_limiter(60.0, 20.0);
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            // First request: 16 tokens — should succeed (20 burst)
            let req = chat_request("/v1/chat/completions", "sk-limited");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            // Second request: 16 tokens — only ~4 left, should be rejected
            let req = chat_request("/v1/chat/completions", "sk-limited");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

            // Verify JSON error body
            let body = resp.collect().await.unwrap().to_bytes();
            let body_str = String::from_utf8(body.to_vec()).unwrap();
            assert!(body_str.contains("rate_limit_error"));
        }

        #[tokio::test]
        async fn different_api_keys_have_separate_budgets() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr]);

            let plugins = make_limiter(60.0, 20.0);
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            // Exhaust key A
            let req = chat_request("/v1/chat/completions", "sk-key-a");
            send_request(&proxy_addr, req).await;
            let req = chat_request("/v1/chat/completions", "sk-key-a");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

            // Key B should still work
            let req = chat_request("/v1/chat/completions", "sk-key-b");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
        }

        #[tokio::test]
        async fn no_auth_header_passes_through() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr]);

            // Very tight limit — but no auth means rate limiter is skipped
            let plugins = make_limiter(60.0, 1.0);
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(chat_request_body())))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
        }
    }

    // -----------------------------------------------------------------------
    // cost_tracker integration tests
    // -----------------------------------------------------------------------

    mod cost_tracker {
        use super::*;
        use plugin_llm_gateway::cost_tracker as ct;

        fn make_tracker(budget: f64) -> Arc<PluginChain> {
            let config = toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(budget));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t.insert("default_cost_per_1k_input".into(), toml::Value::Float(0.01));
                t.insert(
                    "default_cost_per_1k_output".into(),
                    toml::Value::Float(0.02),
                );
                t
            });
            let plugin = ct::create(&config).unwrap();
            Arc::new(PluginChain::new(vec![plugin]))
        }

        #[tokio::test]
        async fn tracks_cost_across_requests() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr]);

            // No budget limit — just track
            let plugins = make_tracker(0.0);
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            // Send several requests — all should succeed
            for i in 0..5 {
                let req = chat_request(&format!("/v1/chat/completions?n={}", i), "sk-track");
                let resp = send_request(&proxy_addr, req).await;
                assert_eq!(
                    resp.status(),
                    StatusCode::OK,
                    "request {} should succeed",
                    i
                );
            }
        }

        #[tokio::test]
        async fn returns_402_when_budget_exceeded() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr]);

            // Very tiny budget: $0.001. Each request costs roughly:
            // input: ~16 tokens => (16/1000)*0.01 = 0.00016
            // output: response ~290 bytes => ~72 tokens => (72/1000)*0.02 = 0.00144
            // total per request ~ 0.0016
            // So budget of 0.001 should be exceeded after first request's cost is recorded
            let plugins = make_tracker(0.001);
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            // First request succeeds (budget check is pre-accumulation)
            let req = chat_request("/v1/chat/completions", "sk-expensive");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            // Second request should be rejected — cost from first request exceeded budget
            let req = chat_request("/v1/chat/completions", "sk-expensive");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);

            let body = resp.collect().await.unwrap().to_bytes();
            let body_str = String::from_utf8(body.to_vec()).unwrap();
            assert!(body_str.contains("budget_exceeded_error"));
        }
    }

    // -----------------------------------------------------------------------
    // provider_failover integration tests
    // -----------------------------------------------------------------------

    mod provider_failover {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use plugin_llm_gateway::provider_failover as pf;

        #[tokio::test]
        async fn fails_over_to_healthy_provider() {
            let failing_hits = Arc::new(AtomicUsize::new(0));
            let failing_addr = start_upstream({
                let failing_hits = Arc::clone(&failing_hits);
                move |_req: Request<Incoming>| {
                    failing_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Full::new(Bytes::from("provider error")))
                        .unwrap()
                }
            })
            .await;

            let healthy_hits = Arc::new(AtomicUsize::new(0));
            let healthy_addr = start_upstream({
                let healthy_hits = Arc::clone(&healthy_hits);
                move |req: Request<Incoming>| {
                    healthy_hits.fetch_add(1, Ordering::Relaxed);
                    llm_chat_handler(req)
                }
            })
            .await;

            let router = catch_all_router(vec![failing_addr.clone(), healthy_addr.clone()]);

            let config_toml = toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("cooldown_secs".into(), toml::Value::Integer(60));
                t.insert(
                    "providers".into(),
                    toml::Value::Array(vec![
                        {
                            let mut p = toml::value::Map::new();
                            p.insert("name".into(), toml::Value::String("failing".into()));
                            p.insert("pattern".into(), toml::Value::String(failing_addr.clone()));
                            toml::Value::Table(p)
                        },
                        {
                            let mut p = toml::value::Map::new();
                            p.insert("name".into(), toml::Value::String("healthy".into()));
                            p.insert("pattern".into(), toml::Value::String(healthy_addr.clone()));
                            toml::Value::Table(p)
                        },
                    ]),
                );
                t
            });
            let plugin = pf::create(&config_toml).unwrap();
            let plugins = Arc::new(PluginChain::new(vec![plugin]));

            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            let mut ok_responses = 0;
            let mut error_responses = 0;
            for _ in 0..4 {
                let req = chat_request("/v1/chat/completions", "sk-test");
                let resp = send_request(&proxy_addr, req).await;
                match resp.status() {
                    StatusCode::OK => ok_responses += 1,
                    StatusCode::INTERNAL_SERVER_ERROR => error_responses += 1,
                    status => panic!("unexpected status from provider failover test: {status}"),
                }
            }

            assert_eq!(
                failing_hits.load(Ordering::Relaxed),
                1,
                "failing provider should only be hit once before quarantine",
            );
            assert_eq!(
                healthy_hits.load(Ordering::Relaxed),
                3,
                "healthy provider should serve all remaining requests after quarantine",
            );
            assert_eq!(
                ok_responses, 3,
                "three requests should reach the healthy provider"
            );
            assert_eq!(
                error_responses, 1,
                "one request should observe the initial provider failure",
            );
        }
    }

    // -----------------------------------------------------------------------
    // virtual_keys integration tests
    // -----------------------------------------------------------------------

    mod virtual_keys {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use plugin_llm_gateway::virtual_keys::VirtualKeys;
        use proxy_core::config::{
            FileSurface, ProviderCapabilityConfig, ProviderSurfaceCatalog, ResponsesSurface,
        };

        fn surfaces_from_capabilities(
            capability_config: &ProviderCapabilityConfig,
        ) -> ProviderSurfaceCatalog {
            let prompt_cache = if capability_config.prompt_cache_openai
                || capability_config.prompt_cache_anthropic
                || capability_config.prompt_cache_request_controls
            {
                Some(PromptCacheSurface {
                    protocol: if capability_config.prompt_cache_anthropic {
                        PromptCacheProtocol::Anthropic
                    } else {
                        PromptCacheProtocol::OpenAi
                    },
                    request_controls: capability_config.prompt_cache_request_controls,
                })
            } else {
                None
            };

            ProviderSurfaceCatalog {
                tools: Some(ToolSurface::OpenAi),
                responses: capability_config
                    .responses_api
                    .then_some(ResponsesSurface::OpenAiCompatible),
                reasoning: capability_config.reasoning,
                structured_output_json_mode: capability_config.structured_output_json_mode,
                structured_output_json_schema: capability_config.structured_output_json_schema,
                files: capability_config
                    .files
                    .then_some(FileSurface::OpenAiCompatible),
                batches: capability_config
                    .batches
                    .then_some(BatchSurface::OpenAiCompatible),
                images: (capability_config.image_input
                    || capability_config.images_generations
                    || capability_config.images_edits
                    || capability_config.images_variations)
                    .then_some(ImageSurface {
                        protocol: ImageSurfaceProtocol::OpenAiImages,
                        input: capability_config.image_input,
                        generations: capability_config.images_generations,
                        edits: capability_config.images_edits,
                        variations: capability_config.images_variations,
                    }),
                audio: (capability_config.audio_input
                    || capability_config.audio_output
                    || capability_config.audio_transcription
                    || capability_config.audio_translation)
                    .then_some(AudioSurface {
                        protocol: AudioSurfaceProtocol::OpenAiAudio,
                        input: capability_config.audio_input,
                        output: capability_config.audio_output,
                        transcription: capability_config.audio_transcription,
                        translation: capability_config.audio_translation,
                    }),
                embeddings: capability_config.embeddings.then_some(EmbeddingSurface {
                    protocol: EmbeddingSurfaceProtocol::OpenAiEmbeddings,
                }),
                realtime: capability_config
                    .realtime
                    .then_some(RealtimeSurface::OpenAiCompatible),
                prompt_cache,
            }
        }

        fn multipart_form(parts: &[(&str, &str)]) -> (String, Vec<u8>) {
            let boundary = "trp-boundary";
            let mut body = Vec::new();
            for (name, value) in parts {
                body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
                body.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                );
                body.extend_from_slice(value.as_bytes());
                body.extend_from_slice(b"\r\n");
            }
            body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
            (boundary.to_string(), body)
        }

        async fn assert_request_routes_to_capability_provider(
            path: &str,
            model: &str,
            request_body: serde_json::Value,
            capability_config: ProviderCapabilityConfig,
            expected_body: &'static [u8],
        ) {
            let standard_hits = Arc::new(AtomicUsize::new(0));
            let standard_addr = start_upstream({
                let standard_hits = Arc::clone(&standard_hits);
                move |_req: Request<Incoming>| {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"standard-upstream")))
                        .unwrap()
                }
            })
            .await;

            let capable_hits = Arc::new(AtomicUsize::new(0));
            let capable_addr = start_upstream({
                let capable_hits = Arc::clone(&capable_hits);
                move |_req: Request<Incoming>| {
                    capable_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from(expected_body)))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![standard_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "standard",
                    "sk-openai-standard",
                    format!("http://{}", standard_addr),
                    vec![model.to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "capable",
                    "sk-openai-capable",
                    format!("http://{}", capable_addr),
                    vec![model.to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    surfaces_from_capabilities(&capability_config),
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "capability-route-key",
                    "standard",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri(path)
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(request_body.to_string())))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], expected_body);
            assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
            assert_eq!(capable_hits.load(Ordering::Relaxed), 1);
        }

        async fn assert_request_fails_without_capability_provider(
            path: &str,
            model: &str,
            request_body: serde_json::Value,
        ) {
            let upstream_hits = Arc::new(AtomicUsize::new(0));
            let upstream_addr = start_upstream({
                let upstream_hits = Arc::clone(&upstream_hits);
                move |_req: Request<Incoming>| {
                    upstream_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![upstream_addr.clone()]);
            let providers = vec![canonical_provider(
                "default",
                "sk-openai-real",
                format!("http://{}", upstream_addr),
                vec![model.to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "capability-fail-key",
                    "default",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri(path)
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(request_body.to_string())))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_capability_unsupported")
            );
            assert_eq!(upstream_hits.load(Ordering::Relaxed), 0);
        }

        #[tokio::test]
        async fn request_provider_only_routes_to_named_provider_and_strips_policy() {
            let openai_hits = Arc::new(AtomicUsize::new(0));
            let openai_addr = start_upstream({
                let openai_hits = Arc::clone(&openai_hits);
                move |_req: Request<Incoming>| {
                    openai_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"openai-upstream")))
                        .unwrap()
                }
            })
            .await;

            let anthropic_hits = Arc::new(AtomicUsize::new(0));
            let anthropic_body = Arc::new(tokio::sync::Mutex::new(None::<serde_json::Value>));
            let anthropic_addr = start_upstream_async({
                let anthropic_hits = Arc::clone(&anthropic_hits);
                let anthropic_body = Arc::clone(&anthropic_body);
                move |req: Request<Incoming>| {
                    let anthropic_hits = Arc::clone(&anthropic_hits);
                    let anthropic_body = Arc::clone(&anthropic_body);
                    async move {
                        anthropic_hits.fetch_add(1, Ordering::Relaxed);
                        let body = req.collect().await.unwrap().to_bytes();
                        let json = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
                        anthropic_body.lock().await.replace(json);
                        Response::builder()
                            .status(StatusCode::OK)
                            .body(Full::new(Bytes::from_static(b"anthropic-upstream")))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![openai_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "openai",
                    "sk-openai-real",
                    format!("http://{}", openai_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "anthropic",
                    "sk-anthropic-real",
                    format!("http://{}", anthropic_addr),
                    vec!["gpt-4o".to_string()],
                    "x-api-key",
                    ProviderFamily::Anthropic,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::Anthropic),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "provider-only-route-key",
                    "openai",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .header("x-trp-routing-debug", "1")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hello"}],
                        "provider": {
                            "only": ["anthropic"],
                            "allow_fallbacks": false,
                        }
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert_eq!(
                resp.headers()
                    .get("x-trp-provider-policy-applied")
                    .and_then(|value| value.to_str().ok()),
                Some("1")
            );
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], b"anthropic-upstream");
            assert_eq!(openai_hits.load(Ordering::Relaxed), 0);
            assert_eq!(anthropic_hits.load(Ordering::Relaxed), 1);
            let upstream_body = anthropic_body.lock().await.clone().expect("upstream body");
            assert!(
                upstream_body.get("provider").is_none(),
                "provider field should be stripped"
            );
        }

        #[tokio::test]
        async fn request_provider_policy_unknown_provider_fails_fast() {
            let upstream_hits = Arc::new(AtomicUsize::new(0));
            let upstream_addr = start_upstream({
                let upstream_hits = Arc::clone(&upstream_hits);
                move |_req: Request<Incoming>| {
                    upstream_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![upstream_addr.clone()]);
            let providers = vec![canonical_provider(
                "openai",
                "sk-openai-real",
                format!("http://{}", upstream_addr),
                vec!["gpt-4o".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "provider-unknown-key",
                    "openai",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hello"}],
                        "provider": {
                            "only": ["missing-provider"]
                        }
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("invalid_provider_policy")
            );
            assert_eq!(upstream_hits.load(Ordering::Relaxed), 0);
        }

        #[tokio::test]
        async fn selected_upstream_override_is_honored() {
            let openai_addr = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from("openai-upstream")))
                    .unwrap()
            })
            .await;

            let anthropic_addr = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Full::new(Bytes::from("anthropic-upstream")))
                    .unwrap()
            })
            .await;

            // Route table only contains openai upstream. The virtual key plugin should
            // still force routing to anthropic based on model mapping.
            let router = catch_all_router(vec![openai_addr.clone()]);

            let providers = vec![
                canonical_provider(
                    "openai",
                    "sk-openai-real",
                    format!("http://{}", openai_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "anthropic",
                    "sk-anthropic-real",
                    format!("http://{}", anthropic_addr),
                    vec!["claude-sonnet-4-20250514".to_string()],
                    "x-api-key",
                    ProviderFamily::Anthropic,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::Anthropic),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "integration-key",
                    "openai",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    r#"{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}]}"#,
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], b"anthropic-upstream");
        }

        #[tokio::test]
        async fn responses_requests_route_to_a_provider_with_responses_api_capability() {
            let standard_hits = Arc::new(AtomicUsize::new(0));
            let standard_addr = start_upstream({
                let standard_hits = Arc::clone(&standard_hits);
                move |_req: Request<Incoming>| {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("standard-upstream")))
                        .unwrap()
                }
            })
            .await;

            let responses_hits = Arc::new(AtomicUsize::new(0));
            let responses_addr = start_upstream({
                let responses_hits = Arc::clone(&responses_hits);
                move |_req: Request<Incoming>| {
                    responses_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("responses-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![standard_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "openai-chat",
                    "sk-openai-real",
                    format!("http://{}", standard_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "openai-responses",
                    "sk-openai-responses",
                    format!("http://{}", responses_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        responses: Some(ResponsesSurface::OpenAiCompatible),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "responses-key",
                    "openai-chat",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-4o",
                        "input": "hello"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], b"responses-upstream");
            assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
            assert_eq!(responses_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn responses_requests_route_with_explicit_responses_surface() {
            let standard_hits = Arc::new(AtomicUsize::new(0));
            let standard_addr = start_upstream({
                let standard_hits = Arc::clone(&standard_hits);
                move |_req: Request<Incoming>| {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("standard-upstream")))
                        .unwrap()
                }
            })
            .await;

            let responses_hits = Arc::new(AtomicUsize::new(0));
            let responses_addr = start_upstream({
                let responses_hits = Arc::clone(&responses_hits);
                move |_req: Request<Incoming>| {
                    responses_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("responses-surface-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![standard_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "openai-chat",
                    "sk-openai-real",
                    format!("http://{}", standard_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    openai_surfaces(),
                ),
                canonical_provider(
                    "openai-responses-surface",
                    "sk-openai-responses",
                    format!("http://{}", responses_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        responses: Some(ResponsesSurface::OpenAiCompatible),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "responses-surface-key",
                    "openai-chat",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    r#"{"model":"gpt-4o","input":"hi via responses surface"}"#,
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], b"responses-surface-upstream");
            assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
            assert_eq!(responses_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn chat_json_mode_requests_route_to_provider_with_structured_output_json_mode() {
            assert_request_routes_to_capability_provider(
                "/v1/chat/completions",
                "gpt-4o",
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "return json"}],
                    "response_format": { "type": "json_object" }
                }),
                ProviderCapabilityConfig {
                    structured_output_json_mode: true,
                    ..ProviderCapabilityConfig::default()
                },
                b"json-mode-upstream",
            )
            .await;
        }

        #[tokio::test]
        async fn chat_json_schema_requests_route_to_provider_with_structured_output_json_schema() {
            assert_request_routes_to_capability_provider(
                "/v1/chat/completions",
                "gpt-4o",
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "return typed json"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "answer": { "type": "string" }
                                },
                                "required": ["answer"]
                            }
                        }
                    }
                }),
                ProviderCapabilityConfig {
                    structured_output_json_schema: true,
                    ..ProviderCapabilityConfig::default()
                },
                b"json-schema-upstream",
            )
            .await;
        }

        #[tokio::test]
        async fn responses_json_mode_requests_route_to_provider_with_structured_output_json_mode() {
            assert_request_routes_to_capability_provider(
                "/v1/responses",
                "gpt-4o",
                serde_json::json!({
                    "model": "gpt-4o",
                    "input": "hello",
                    "text": {
                        "format": { "type": "json_object" }
                    }
                }),
                ProviderCapabilityConfig {
                    responses_api: true,
                    structured_output_json_mode: true,
                    ..ProviderCapabilityConfig::default()
                },
                b"responses-json-mode-upstream",
            )
            .await;
        }

        #[tokio::test]
        async fn responses_json_schema_requests_route_to_provider_with_structured_output_json_schema(
        ) {
            assert_request_routes_to_capability_provider(
                "/v1/responses",
                "gpt-4o",
                serde_json::json!({
                    "model": "gpt-4o",
                    "input": "hello",
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "answer",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "answer": { "type": "string" }
                                },
                                "required": ["answer"]
                            }
                        }
                    }
                }),
                ProviderCapabilityConfig {
                    responses_api: true,
                    structured_output_json_schema: true,
                    ..ProviderCapabilityConfig::default()
                },
                b"responses-json-schema-upstream",
            )
            .await;
        }

        #[tokio::test]
        async fn structured_output_requests_fail_when_no_provider_declares_support() {
            assert_request_fails_without_capability_provider(
                "/v1/chat/completions",
                "gpt-4o",
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "return json"}],
                    "response_format": { "type": "json_schema" }
                }),
            )
            .await;
        }

        #[tokio::test]
        async fn responses_json_schema_requests_fail_without_schema_capability() {
            let upstream_hits = Arc::new(AtomicUsize::new(0));
            let upstream_addr = start_upstream({
                let upstream_hits = Arc::clone(&upstream_hits);
                move |_req: Request<Incoming>| {
                    upstream_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![upstream_addr.clone()]);
            let providers = vec![canonical_provider(
                "responses-only",
                "sk-openai-responses",
                format!("http://{}", upstream_addr),
                vec!["gpt-4o".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    responses: Some(ResponsesSurface::OpenAiCompatible),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "responses-schema-fail-key",
                    "responses-only",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-4o",
                        "input": "hello",
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "answer",
                                "schema": {
                                    "type": "object"
                                }
                            }
                        }
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_capability_unsupported")
            );
            assert_eq!(upstream_hits.load(Ordering::Relaxed), 0);
        }

        #[tokio::test]
        async fn image_input_requests_fail_when_no_provider_declares_support() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr.clone()]);

            let providers = vec![canonical_provider(
                "openai",
                "sk-openai-real",
                format!("http://{}", upstream_addr),
                vec!["gpt-4o".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "vision-key",
                    "openai",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-4o",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe this image"},
                                {
                                    "type": "image_url",
                                    "image_url": { "url": "https://example.com/cat.png" }
                                }
                            ]
                        }]
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_capability_unsupported")
            );
        }

        #[tokio::test]
        async fn batch_requests_fail_when_no_provider_declares_support() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr.clone()]);

            let providers = vec![canonical_provider(
                "openai",
                "sk-openai-real",
                format!("http://{}", upstream_addr),
                vec!["gpt-4o".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "batch-key",
                    "openai",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/batches")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "input_file_id": "file-123",
                        "endpoint": "/v1/responses",
                        "completion_window": "24h"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_capability_unsupported")
            );
        }

        #[tokio::test]
        async fn batch_create_routes_to_provider_with_batches_capability_and_strips_policy() {
            let standard_hits = Arc::new(AtomicUsize::new(0));
            let standard_addr = start_upstream({
                let standard_hits = Arc::clone(&standard_hits);
                move |_req: Request<Incoming>| {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-standard")))
                        .unwrap()
                }
            })
            .await;

            let batch_hits = Arc::new(AtomicUsize::new(0));
            let batch_addr = start_upstream_async({
                let batch_hits = Arc::clone(&batch_hits);
                move |req: Request<Incoming>| {
                    let batch_hits = Arc::clone(&batch_hits);
                    async move {
                        batch_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/batches");
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        assert!(body_json.get("provider").is_none());
                        assert_eq!(body_json["input_file_id"].as_str(), Some("file-123"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "batch_123",
                                    "object": "batch",
                                    "status": "validating"
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![standard_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "standard",
                    "sk-openai-standard",
                    format!("http://{}", standard_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "batch-native",
                    "sk-openai-batches",
                    format!("http://{}", batch_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        responses: Some(ResponsesSurface::OpenAiCompatible),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "batch-create-key",
                    "standard",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/batches")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "input_file_id": "file-123",
                        "endpoint": "/v1/responses",
                        "completion_window": "24h",
                        "provider": {
                            "only": ["batch-native"],
                            "allow_fallbacks": false
                        }
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["id"].as_str(), Some("batch_123"));
            assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
            assert_eq!(batch_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn batch_create_routes_to_provider_that_supports_target_endpoint_surface() {
            let plain_batch_hits = Arc::new(AtomicUsize::new(0));
            let plain_batch_addr = start_upstream({
                let plain_batch_hits = Arc::clone(&plain_batch_hits);
                move |_req: Request<Incoming>| {
                    plain_batch_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-plain-batch")))
                        .unwrap()
                }
            })
            .await;

            let responses_batch_hits = Arc::new(AtomicUsize::new(0));
            let responses_batch_addr = start_upstream_async({
                let responses_batch_hits = Arc::clone(&responses_batch_hits);
                move |req: Request<Incoming>| {
                    let responses_batch_hits = Arc::clone(&responses_batch_hits);
                    async move {
                        responses_batch_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/batches");
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        assert_eq!(body_json["endpoint"].as_str(), Some("/v1/responses"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "batch_surface",
                                    "object": "batch",
                                    "status": "validating"
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![plain_batch_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "plain-batches",
                    "sk-plain-batches",
                    format!("http://{}", plain_batch_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "responses-batches",
                    "sk-responses-batches",
                    format!("http://{}", responses_batch_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        responses: Some(ResponsesSurface::OpenAiCompatible),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "batch-surface-key",
                    "plain-batches",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/batches")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "input_file_id": "file-123",
                        "endpoint": "/v1/responses",
                        "completion_window": "24h"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["id"].as_str(), Some("batch_surface"));
            assert_eq!(plain_batch_hits.load(Ordering::Relaxed), 0);
            assert_eq!(responses_batch_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn batch_create_image_generations_require_native_image_surface() {
            let translated_hits = Arc::new(AtomicUsize::new(0));
            let translated_addr = start_upstream({
                let translated_hits = Arc::clone(&translated_hits);
                move |_req: Request<Incoming>| {
                    translated_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(
                            b"unexpected-translated-image-batch",
                        )))
                        .unwrap()
                }
            })
            .await;

            let native_hits = Arc::new(AtomicUsize::new(0));
            let native_addr = start_upstream_async({
                let native_hits = Arc::clone(&native_hits);
                move |req: Request<Incoming>| {
                    let native_hits = Arc::clone(&native_hits);
                    async move {
                        native_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/batches");
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        assert_eq!(
                            body_json["endpoint"].as_str(),
                            Some("/v1/images/generations")
                        );
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "batch_native_image",
                                    "object": "batch",
                                    "status": "validating"
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![translated_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "translated-image-batch",
                    "sk-translated-image-batch",
                    format!("http://{}", translated_addr),
                    vec!["gpt-image-1".to_string()],
                    "authorization",
                    ProviderFamily::OpenRouter,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        images: Some(ImageSurface {
                            protocol: ImageSurfaceProtocol::OpenRouterChatImages,
                            input: false,
                            generations: true,
                            edits: false,
                            variations: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "native-image-batch",
                    "sk-native-image-batch",
                    format!("http://{}", native_addr),
                    vec!["gpt-image-1".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        images: Some(ImageSurface {
                            protocol: ImageSurfaceProtocol::OpenAiImages,
                            input: false,
                            generations: true,
                            edits: false,
                            variations: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "batch-native-image-surface-key",
                    "translated-image-batch",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/batches")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "input_file_id": "file-123",
                        "endpoint": "/v1/images/generations",
                        "completion_window": "24h"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["id"].as_str(), Some("batch_native_image"));
            assert_eq!(translated_hits.load(Ordering::Relaxed), 0);
            assert_eq!(native_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn batch_create_audio_speech_requires_native_audio_surface() {
            let translated_hits = Arc::new(AtomicUsize::new(0));
            let translated_addr = start_upstream({
                let translated_hits = Arc::clone(&translated_hits);
                move |_req: Request<Incoming>| {
                    translated_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(
                            b"unexpected-translated-audio-batch",
                        )))
                        .unwrap()
                }
            })
            .await;

            let native_hits = Arc::new(AtomicUsize::new(0));
            let native_addr = start_upstream_async({
                let native_hits = Arc::clone(&native_hits);
                move |req: Request<Incoming>| {
                    let native_hits = Arc::clone(&native_hits);
                    async move {
                        native_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/batches");
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        assert_eq!(body_json["endpoint"].as_str(), Some("/v1/audio/speech"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "batch_native_audio",
                                    "object": "batch",
                                    "status": "validating"
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![translated_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "translated-audio-batch",
                    "sk-translated-audio-batch",
                    format!("http://{}", translated_addr),
                    vec!["gpt-4o-mini-tts".to_string()],
                    "authorization",
                    ProviderFamily::OpenRouter,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        audio: Some(AudioSurface {
                            protocol: AudioSurfaceProtocol::OpenRouterChatAudio,
                            input: false,
                            output: true,
                            transcription: false,
                            translation: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "native-audio-batch",
                    "sk-native-audio-batch",
                    format!("http://{}", native_addr),
                    vec!["gpt-4o-mini-tts".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        audio: Some(AudioSurface {
                            protocol: AudioSurfaceProtocol::OpenAiAudio,
                            input: false,
                            output: true,
                            transcription: false,
                            translation: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "batch-native-audio-surface-key",
                    "translated-audio-batch",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/batches")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "input_file_id": "file-123",
                        "endpoint": "/v1/audio/speech",
                        "completion_window": "24h"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["id"].as_str(), Some("batch_native_audio"));
            assert_eq!(translated_hits.load(Ordering::Relaxed), 0);
            assert_eq!(native_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn batch_create_embeddings_require_native_embedding_surface() {
            let translated_hits = Arc::new(AtomicUsize::new(0));
            let translated_addr = start_upstream({
                let translated_hits = Arc::clone(&translated_hits);
                move |_req: Request<Incoming>| {
                    translated_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(
                            b"unexpected-translated-embeddings-batch",
                        )))
                        .unwrap()
                }
            })
            .await;

            let native_hits = Arc::new(AtomicUsize::new(0));
            let native_addr = start_upstream_async({
                let native_hits = Arc::clone(&native_hits);
                move |req: Request<Incoming>| {
                    let native_hits = Arc::clone(&native_hits);
                    async move {
                        native_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/batches");
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        assert_eq!(body_json["endpoint"].as_str(), Some("/v1/embeddings"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "batch_native_embeddings",
                                    "object": "batch",
                                    "status": "validating"
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![translated_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "translated-embeddings-batch",
                    "sk-translated-embeddings-batch",
                    format!("http://{}", translated_addr),
                    vec!["text-embedding-3-large".to_string()],
                    "authorization",
                    ProviderFamily::Gemini,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        embeddings: Some(EmbeddingSurface {
                            protocol: EmbeddingSurfaceProtocol::GeminiEmbedContent,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "native-embeddings-batch",
                    "sk-native-embeddings-batch",
                    format!("http://{}", native_addr),
                    vec!["text-embedding-3-large".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        embeddings: Some(EmbeddingSurface {
                            protocol: EmbeddingSurfaceProtocol::OpenAiEmbeddings,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "batch-native-embeddings-surface-key",
                    "translated-embeddings-batch",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/batches")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "input_file_id": "file-123",
                        "endpoint": "/v1/embeddings",
                        "completion_window": "24h"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["id"].as_str(), Some("batch_native_embeddings"));
            assert_eq!(translated_hits.load(Ordering::Relaxed), 0);
            assert_eq!(native_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn batch_create_fails_when_no_batch_provider_supports_target_endpoint_surface() {
            let upstream_hits = Arc::new(AtomicUsize::new(0));
            let upstream_addr = start_upstream({
                let upstream_hits = Arc::clone(&upstream_hits);
                move |_req: Request<Incoming>| {
                    upstream_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-batch-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![upstream_addr.clone()]);
            let providers = vec![canonical_provider(
                "batch-only",
                "sk-batch-only",
                format!("http://{}", upstream_addr),
                vec!["gpt-4o".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    batches: Some(BatchSurface::OpenAiCompatible),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "batch-unsupported-surface-key",
                    "batch-only",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/batches")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "input_file_id": "file-123",
                        "endpoint": "/v1/images/generations",
                        "completion_window": "24h"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_surface_unsupported")
            );
            assert_eq!(upstream_hits.load(Ordering::Relaxed), 0);
        }

        #[tokio::test]
        async fn batch_follow_up_requests_route_to_provider_that_created_batch() {
            let alpha_hits = Arc::new(AtomicUsize::new(0));
            let alpha_addr = start_upstream_async({
                let alpha_hits = Arc::clone(&alpha_hits);
                move |req: Request<Incoming>| {
                    let alpha_hits = Arc::clone(&alpha_hits);
                    async move {
                        alpha_hits.fetch_add(1, Ordering::Relaxed);
                        match (req.method().clone(), req.uri().path()) {
                            (hyper::Method::POST, "/v1/batches") => Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from(
                                    serde_json::json!({
                                        "id": "batch_alpha",
                                        "object": "batch",
                                        "status": "validating"
                                    })
                                    .to_string(),
                                )))
                                .unwrap(),
                            (hyper::Method::GET, "/v1/batches/batch_alpha") => Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from(
                                    serde_json::json!({
                                        "id": "batch_alpha",
                                        "object": "batch",
                                        "status": "in_progress"
                                    })
                                    .to_string(),
                                )))
                                .unwrap(),
                            (hyper::Method::POST, "/v1/batches/batch_alpha/cancel") => {
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(
                                        serde_json::json!({
                                            "id": "batch_alpha",
                                            "object": "batch",
                                            "status": "cancelling"
                                        })
                                        .to_string(),
                                    )))
                                    .unwrap()
                            }
                            other => Response::builder()
                                .status(StatusCode::INTERNAL_SERVER_ERROR)
                                .body(Full::new(Bytes::from(format!(
                                    "unexpected-alpha-{other:?}"
                                ))))
                                .unwrap(),
                        }
                    }
                }
            })
            .await;

            let beta_hits = Arc::new(AtomicUsize::new(0));
            let beta_addr = start_upstream({
                let beta_hits = Arc::clone(&beta_hits);
                move |_req: Request<Incoming>| {
                    beta_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-beta")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![beta_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "alpha-batches",
                    "sk-alpha",
                    format!("http://{}", alpha_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        responses: Some(ResponsesSurface::OpenAiCompatible),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "beta-batches",
                    "sk-beta",
                    format!("http://{}", beta_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        responses: Some(ResponsesSurface::OpenAiCompatible),
                        batches: Some(BatchSurface::OpenAiCompatible),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "batch-follow-up-key",
                    "beta-batches",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let create_req = Request::builder()
                .method("POST")
                .uri("/v1/batches")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "input_file_id": "file-123",
                        "endpoint": "/v1/responses",
                        "completion_window": "24h",
                        "provider": { "only": ["alpha-batches"] }
                    })
                    .to_string(),
                )))
                .unwrap();
            let create_resp = send_request(&proxy_addr, create_req).await;
            assert_eq!(create_resp.status(), StatusCode::OK);
            let create_body = create_resp.collect().await.unwrap().to_bytes();
            let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
            assert_eq!(create_json["id"].as_str(), Some("batch_alpha"));

            let retrieve_req = Request::builder()
                .method("GET")
                .uri("/v1/batches/batch_alpha")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let retrieve_resp = send_request(&proxy_addr, retrieve_req).await;
            assert_eq!(retrieve_resp.status(), StatusCode::OK);
            let retrieve_body = retrieve_resp.collect().await.unwrap().to_bytes();
            let retrieve_json: serde_json::Value = serde_json::from_slice(&retrieve_body).unwrap();
            assert_eq!(retrieve_json["status"].as_str(), Some("in_progress"));

            let cancel_req = Request::builder()
                .method("POST")
                .uri("/v1/batches/batch_alpha/cancel")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let cancel_resp = send_request(&proxy_addr, cancel_req).await;
            assert_eq!(cancel_resp.status(), StatusCode::OK);
            let cancel_body = cancel_resp.collect().await.unwrap().to_bytes();
            let cancel_json: serde_json::Value = serde_json::from_slice(&cancel_body).unwrap();
            assert_eq!(cancel_json["status"].as_str(), Some("cancelling"));

            assert_eq!(alpha_hits.load(Ordering::Relaxed), 3);
            assert_eq!(beta_hits.load(Ordering::Relaxed), 0);
        }

        #[tokio::test]
        async fn file_requests_fail_when_no_provider_declares_support() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr.clone()]);

            let providers = vec![canonical_provider(
                "openai",
                "sk-openai-real",
                format!("http://{}", upstream_addr),
                vec!["gpt-4o".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "file-key",
                    "openai",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let (boundary, body) = multipart_form(&[("purpose", "batch"), ("file", "contents")]);
            let req = Request::builder()
                .method("POST")
                .uri("/v1/files")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Full::new(Bytes::from(body)))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_capability_unsupported")
            );
        }

        #[tokio::test]
        async fn file_create_routes_to_provider_with_files_capability() {
            let standard_hits = Arc::new(AtomicUsize::new(0));
            let standard_addr = start_upstream({
                let standard_hits = Arc::clone(&standard_hits);
                move |_req: Request<Incoming>| {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-standard")))
                        .unwrap()
                }
            })
            .await;

            let file_hits = Arc::new(AtomicUsize::new(0));
            let file_addr = start_upstream_async({
                let file_hits = Arc::clone(&file_hits);
                move |req: Request<Incoming>| {
                    let file_hits = Arc::clone(&file_hits);
                    async move {
                        file_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/files");
                        assert!(req
                            .headers()
                            .get("content-type")
                            .and_then(|value| value.to_str().ok())
                            .is_some_and(|value| value.contains("multipart/form-data")));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "file_123",
                                    "object": "file",
                                    "purpose": "batch"
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![standard_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "standard",
                    "sk-openai-standard",
                    format!("http://{}", standard_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "file-native",
                    "sk-openai-files",
                    format!("http://{}", file_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    openai_file_surfaces(),
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "file-create-key",
                    "standard",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let (boundary, body) = multipart_form(&[("purpose", "batch"), ("file", "contents")]);
            let req = Request::builder()
                .method("POST")
                .uri("/v1/files")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Full::new(Bytes::from(body)))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["id"].as_str(), Some("file_123"));
            assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
            assert_eq!(file_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn file_create_routes_with_explicit_file_surface() {
            let standard_hits = Arc::new(AtomicUsize::new(0));
            let standard_addr = start_upstream({
                let standard_hits = Arc::clone(&standard_hits);
                move |_req: Request<Incoming>| {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-standard")))
                        .unwrap()
                }
            })
            .await;

            let file_hits = Arc::new(AtomicUsize::new(0));
            let file_addr = start_upstream_async({
                let file_hits = Arc::clone(&file_hits);
                move |req: Request<Incoming>| {
                    let file_hits = Arc::clone(&file_hits);
                    async move {
                        file_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/files");
                        assert!(req
                            .headers()
                            .get("content-type")
                            .and_then(|value| value.to_str().ok())
                            .is_some_and(|value| value.contains("multipart/form-data")));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "id": "file_surface",
                                    "object": "file",
                                    "purpose": "batch"
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![standard_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "standard",
                    "sk-openai-standard",
                    format!("http://{}", standard_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    openai_surfaces(),
                ),
                canonical_provider(
                    "file-surface",
                    "sk-openai-files",
                    format!("http://{}", file_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        files: Some(FileSurface::OpenAiCompatible),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "file-surface-key",
                    "standard",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let (boundary, body) = multipart_form(&[("purpose", "batch"), ("file", "contents")]);
            let req = Request::builder()
                .method("POST")
                .uri("/v1/files")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Full::new(Bytes::from(body)))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["id"].as_str(), Some("file_surface"));
            assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
            assert_eq!(file_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn file_follow_up_requests_route_to_provider_that_created_file() {
            let alpha_hits = Arc::new(AtomicUsize::new(0));
            let alpha_addr = start_upstream_async({
                let alpha_hits = Arc::clone(&alpha_hits);
                move |req: Request<Incoming>| {
                    let alpha_hits = Arc::clone(&alpha_hits);
                    async move {
                        alpha_hits.fetch_add(1, Ordering::Relaxed);
                        match (req.method().clone(), req.uri().path()) {
                            (hyper::Method::POST, "/v1/files") => Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from(
                                    serde_json::json!({
                                        "id": "file_alpha",
                                        "object": "file",
                                        "purpose": "batch"
                                    })
                                    .to_string(),
                                )))
                                .unwrap(),
                            (hyper::Method::GET, "/v1/files/file_alpha") => Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from(
                                    serde_json::json!({
                                        "id": "file_alpha",
                                        "object": "file",
                                        "status": "processed"
                                    })
                                    .to_string(),
                                )))
                                .unwrap(),
                            (hyper::Method::GET, "/v1/files/file_alpha/content") => {
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header("content-type", "text/plain")
                                    .body(Full::new(Bytes::from_static(b"file-body")))
                                    .unwrap()
                            }
                            (hyper::Method::DELETE, "/v1/files/file_alpha") => Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from(
                                    serde_json::json!({
                                        "id": "file_alpha",
                                        "object": "file",
                                        "deleted": true
                                    })
                                    .to_string(),
                                )))
                                .unwrap(),
                            other => Response::builder()
                                .status(StatusCode::INTERNAL_SERVER_ERROR)
                                .body(Full::new(Bytes::from(format!(
                                    "unexpected-alpha-{other:?}"
                                ))))
                                .unwrap(),
                        }
                    }
                }
            })
            .await;

            let beta_hits = Arc::new(AtomicUsize::new(0));
            let beta_addr = start_upstream({
                let beta_hits = Arc::clone(&beta_hits);
                move |_req: Request<Incoming>| {
                    beta_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-beta")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![beta_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "alpha-files",
                    "sk-alpha",
                    format!("http://{}", alpha_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    openai_file_surfaces(),
                ),
                canonical_provider(
                    "beta-files",
                    "sk-beta",
                    format!("http://{}", beta_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    openai_file_surfaces(),
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "file-follow-up-key",
                    "beta-files",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let (boundary, body) = multipart_form(&[("purpose", "batch"), ("file", "contents")]);
            let create_req = Request::builder()
                .method("POST")
                .uri("/v1/files")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Full::new(Bytes::from(body)))
                .unwrap();
            let create_resp = send_request(&proxy_addr, create_req).await;
            assert_eq!(create_resp.status(), StatusCode::OK);

            let retrieve_req = Request::builder()
                .method("GET")
                .uri("/v1/files/file_alpha")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let retrieve_resp = send_request(&proxy_addr, retrieve_req).await;
            assert_eq!(retrieve_resp.status(), StatusCode::OK);
            let retrieve_body = retrieve_resp.collect().await.unwrap().to_bytes();
            let retrieve_json: serde_json::Value = serde_json::from_slice(&retrieve_body).unwrap();
            assert_eq!(retrieve_json["status"].as_str(), Some("processed"));

            let content_req = Request::builder()
                .method("GET")
                .uri("/v1/files/file_alpha/content")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let content_resp = send_request(&proxy_addr, content_req).await;
            assert_eq!(content_resp.status(), StatusCode::OK);
            let content_body = content_resp.collect().await.unwrap().to_bytes();
            assert_eq!(&content_body[..], b"file-body");

            let delete_req = Request::builder()
                .method("DELETE")
                .uri("/v1/files/file_alpha")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .body(Full::new(Bytes::new()))
                .unwrap();
            let delete_resp = send_request(&proxy_addr, delete_req).await;
            assert_eq!(delete_resp.status(), StatusCode::OK);
            let delete_body = delete_resp.collect().await.unwrap().to_bytes();
            let delete_json: serde_json::Value = serde_json::from_slice(&delete_body).unwrap();
            assert_eq!(delete_json["deleted"].as_bool(), Some(true));

            assert_eq!(alpha_hits.load(Ordering::Relaxed), 4);
            assert_eq!(beta_hits.load(Ordering::Relaxed), 0);
        }

        #[tokio::test]
        async fn file_create_provider_policy_fails_fast_for_multipart_requests() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr.clone()]);
            let providers = vec![canonical_provider(
                "openai",
                "sk-openai-real",
                format!("http://{}", upstream_addr),
                vec!["gpt-4o".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "file-policy-key",
                    "openai",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let (boundary, body) = multipart_form(&[
                ("purpose", "batch"),
                ("file", "contents"),
                ("provider", "{\"only\":[\"openai\"]}"),
            ]);
            let req = Request::builder()
                .method("POST")
                .uri("/v1/files")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Full::new(Bytes::from(body)))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_policy_unsupported")
            );
        }

        #[tokio::test]
        async fn image_generations_route_to_openai_images_provider_and_strip_provider_policy() {
            let standard_hits = Arc::new(AtomicUsize::new(0));
            let standard_addr = start_upstream({
                let standard_hits = Arc::clone(&standard_hits);
                move |_req: Request<Incoming>| {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-standard")))
                        .unwrap()
                }
            })
            .await;

            let image_hits = Arc::new(AtomicUsize::new(0));
            let image_addr = start_upstream_async({
                let image_hits = Arc::clone(&image_hits);
                move |req: Request<Incoming>| {
                    let image_hits = Arc::clone(&image_hits);
                    async move {
                        image_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/images/generations");
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        assert!(body_json.get("provider").is_none());
                        assert_eq!(body_json["prompt"].as_str(), Some("a cat"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "created": 123,
                                    "data": [{ "b64_json": "Zm9v" }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![standard_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "standard",
                    "sk-openai-standard",
                    format!("http://{}", standard_addr),
                    vec!["gpt-image-1".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "images-native",
                    "sk-openai-images",
                    format!("http://{}", image_addr),
                    vec!["gpt-image-1".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        images: Some(ImageSurface {
                            protocol: ImageSurfaceProtocol::OpenAiImages,
                            input: false,
                            generations: true,
                            edits: false,
                            variations: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "image-generation-native-key",
                    "standard",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("x-trp-routing-debug", "1")
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-image-1",
                        "prompt": "a cat",
                        "provider": {
                            "only": ["images-native"],
                            "allow_fallbacks": false
                        }
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["data"][0]["b64_json"].as_str(), Some("Zm9v"));
            assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
            assert_eq!(image_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn image_generations_translate_to_openrouter_chat_images() {
            let image_addr = start_upstream_async(move |req: Request<Incoming>| async move {
                assert_eq!(req.uri().path(), "/v1/chat/completions");
                let body = req.into_body().collect().await.unwrap().to_bytes();
                let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(body_json["model"].as_str(), Some("openai/dall-e-3"));
                assert_eq!(
                    body_json["messages"][0]["content"].as_str(),
                    Some("paint a fox")
                );
                assert!(body_json["modalities"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .any(|value| value.as_str() == Some("image")));
                assert_eq!(
                    body_json["image_config"]["aspect_ratio"].as_str(),
                    Some("3:2")
                );
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "id": "chatcmpl-img",
                            "created": 456,
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "done",
                                    "images": [{
                                        "image_url": {
                                            "url": "data:image/png;base64,Zm9vYmFy"
                                        }
                                    }]
                                }
                            }]
                        })
                        .to_string(),
                    )))
                    .unwrap()
            })
            .await;

            let router = catch_all_router(vec![image_addr.clone()]);
            let providers = vec![canonical_provider(
                "openrouter-images",
                "sk-openrouter",
                format!("http://{}", image_addr),
                vec!["openai/dall-e-3".to_string()],
                "authorization",
                ProviderFamily::OpenRouter,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    images: Some(ImageSurface {
                        protocol: ImageSurfaceProtocol::OpenRouterChatImages,
                        input: false,
                        generations: true,
                        edits: false,
                        variations: false,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "image-generation-openrouter-key",
                    "openrouter-images",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "openai/dall-e-3",
                        "prompt": "paint a fox",
                        "size": "1536x1024"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert_eq!(
                resp.headers()
                    .get("content-type")
                    .and_then(|value| value.to_str().ok()),
                Some("application/json")
            );
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["created"].as_i64(), Some(456));
            assert_eq!(body_json["data"][0]["b64_json"].as_str(), Some("Zm9vYmFy"));
        }

        #[tokio::test]
        async fn image_generations_fall_back_to_native_images_when_translation_is_incompatible() {
            let openrouter_hits = Arc::new(AtomicUsize::new(0));
            let openrouter_addr = start_upstream({
                let openrouter_hits = Arc::clone(&openrouter_hits);
                move |_req: Request<Incoming>| {
                    openrouter_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(
                            b"unexpected-openrouter-image",
                        )))
                        .unwrap()
                }
            })
            .await;

            let native_hits = Arc::new(AtomicUsize::new(0));
            let native_addr = start_upstream_async({
                let native_hits = Arc::clone(&native_hits);
                move |req: Request<Incoming>| {
                    let native_hits = Arc::clone(&native_hits);
                    async move {
                        native_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/images/generations");
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        assert_eq!(body_json["model"].as_str(), Some("gpt-image-1"));
                        assert_eq!(body_json["prompt"].as_str(), Some("paint a storm"));
                        assert_eq!(body_json["quality"].as_str(), Some("high"));
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "created": 111,
                                    "data": [{ "b64_json": "bmF0aXZl" }]
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![openrouter_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "openrouter-images",
                    "sk-openrouter",
                    format!("http://{}", openrouter_addr),
                    vec!["gpt-image-1".to_string()],
                    "authorization",
                    ProviderFamily::OpenRouter,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        images: Some(ImageSurface {
                            protocol: ImageSurfaceProtocol::OpenRouterChatImages,
                            input: false,
                            generations: true,
                            edits: false,
                            variations: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "native-images",
                    "sk-native-images",
                    format!("http://{}", native_addr),
                    vec!["gpt-image-1".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        images: Some(ImageSurface {
                            protocol: ImageSurfaceProtocol::OpenAiImages,
                            input: false,
                            generations: true,
                            edits: false,
                            variations: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "image-native-fallback-key",
                    "openrouter-images",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-image-1",
                        "prompt": "paint a storm",
                        "quality": "high"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["data"][0]["b64_json"].as_str(), Some("bmF0aXZl"));
            assert_eq!(openrouter_hits.load(Ordering::Relaxed), 0);
            assert_eq!(native_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn image_edits_preserve_multipart_for_native_image_providers() {
            let image_addr = start_upstream_async(move |req: Request<Incoming>| async move {
                assert_eq!(req.uri().path(), "/v1/images/edits");
                assert!(req
                    .headers()
                    .get("content-type")
                    .and_then(|value| value.to_str().ok())
                    .is_some_and(|value| value.contains("multipart/form-data")));
                let body = req.into_body().collect().await.unwrap().to_bytes();
                let body_text = String::from_utf8_lossy(&body);
                assert!(body_text.contains("name=\"model\""));
                assert!(body_text.contains("name=\"prompt\""));
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "created": 789,
                            "data": [{ "b64_json": "cGFzcw==" }]
                        })
                        .to_string(),
                    )))
                    .unwrap()
            })
            .await;

            let router = catch_all_router(vec![image_addr.clone()]);
            let providers = vec![canonical_provider(
                "openai-images",
                "sk-openai-images",
                format!("http://{}", image_addr),
                vec!["gpt-image-1".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    images: Some(ImageSurface {
                        protocol: ImageSurfaceProtocol::OpenAiImages,
                        input: false,
                        generations: false,
                        edits: true,
                        variations: false,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "image-edit-key",
                    "openai-images",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let (boundary, body) =
                multipart_form(&[("model", "gpt-image-1"), ("prompt", "make it brighter")]);
            let req = Request::builder()
                .method("POST")
                .uri("/v1/images/edits")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Full::new(Bytes::from(body)))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["data"][0]["b64_json"].as_str(), Some("cGFzcw=="));
        }

        #[tokio::test]
        async fn image_edit_provider_policy_fails_fast_for_multipart_requests() {
            let upstream_hits = Arc::new(AtomicUsize::new(0));
            let upstream_addr = start_upstream({
                let upstream_hits = Arc::clone(&upstream_hits);
                move |_req: Request<Incoming>| {
                    upstream_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![upstream_addr.clone()]);
            let providers = vec![canonical_provider(
                "openai-images",
                "sk-openai-images",
                format!("http://{}", upstream_addr),
                vec!["gpt-image-1".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    images: Some(ImageSurface {
                        protocol: ImageSurfaceProtocol::OpenAiImages,
                        input: false,
                        generations: false,
                        edits: true,
                        variations: false,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "image-edit-provider-policy-key",
                    "openai-images",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let (boundary, body) = multipart_form(&[
                ("model", "gpt-image-1"),
                ("provider", r#"{"only":["openai-images"]}"#),
            ]);
            let req = Request::builder()
                .method("POST")
                .uri("/v1/images/edits")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Full::new(Bytes::from(body)))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_policy_unsupported")
            );
            assert_eq!(upstream_hits.load(Ordering::Relaxed), 0);
        }

        #[tokio::test]
        async fn image_variations_require_native_image_protocol() {
            let upstream_hits = Arc::new(AtomicUsize::new(0));
            let upstream_addr = start_upstream({
                let upstream_hits = Arc::clone(&upstream_hits);
                move |_req: Request<Incoming>| {
                    upstream_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(b"unexpected-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![upstream_addr.clone()]);
            let providers = vec![canonical_provider(
                "openrouter-images",
                "sk-openrouter",
                format!("http://{}", upstream_addr),
                vec!["openai/dall-e-3".to_string()],
                "authorization",
                ProviderFamily::OpenRouter,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    images: Some(ImageSurface {
                        protocol: ImageSurfaceProtocol::OpenRouterChatImages,
                        input: false,
                        generations: false,
                        edits: false,
                        variations: true,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "image-variation-key",
                    "openrouter-images",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let (boundary, body) = multipart_form(&[("model", "openai/dall-e-3")]);
            let req = Request::builder()
                .method("POST")
                .uri("/v1/images/variations")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Full::new(Bytes::from(body)))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_surface_unsupported")
            );
            assert_eq!(upstream_hits.load(Ordering::Relaxed), 0);
        }

        #[tokio::test]
        async fn audio_speech_requests_route_to_a_provider_with_audio_output_capability() {
            let standard_hits = Arc::new(AtomicUsize::new(0));
            let standard_addr = start_upstream({
                let standard_hits = Arc::clone(&standard_hits);
                move |_req: Request<Incoming>| {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("standard-audio-upstream")))
                        .unwrap()
                }
            })
            .await;

            let audio_hits = Arc::new(AtomicUsize::new(0));
            let audio_addr = start_upstream({
                let audio_hits = Arc::clone(&audio_hits);
                move |_req: Request<Incoming>| {
                    audio_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("audio-output-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![standard_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "openai-chat",
                    "sk-openai-real",
                    format!("http://{}", standard_addr),
                    vec!["gpt-4o-mini-tts".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "openai-audio",
                    "sk-openai-audio",
                    format!("http://{}", audio_addr),
                    vec!["gpt-4o-mini-tts".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        audio: Some(AudioSurface {
                            protocol: AudioSurfaceProtocol::OpenAiAudio,
                            input: false,
                            output: true,
                            transcription: false,
                            translation: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "audio-output-key",
                    "openai-chat",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-4o-mini-tts",
                        "input": "say hello",
                        "voice": "alloy"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], b"audio-output-upstream");
            assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
            assert_eq!(audio_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn audio_speech_translates_to_openrouter_chat_audio() {
            let audio_addr = start_upstream_async(move |req: Request<Incoming>| async move {
                assert_eq!(req.uri().path(), "/v1/chat/completions");
                let body = req.into_body().collect().await.unwrap().to_bytes();
                let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(
                    body_json["model"].as_str(),
                    Some("openai/gpt-4o-audio-preview")
                );
                assert_eq!(body_json["messages"][0]["role"].as_str(), Some("system"));
                assert_eq!(
                    body_json["messages"][0]["content"].as_str(),
                    Some("Speak warmly")
                );
                assert_eq!(
                    body_json["messages"][1]["content"].as_str(),
                    Some("Hello from the gateway")
                );
                assert!(body_json["modalities"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .any(|value| value.as_str() == Some("audio")));
                assert_eq!(body_json["audio"]["voice"].as_str(), Some("alloy"));
                assert_eq!(body_json["audio"]["format"].as_str(), Some("wav"));
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "id": "chatcmpl-audio",
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "done",
                                    "audio": {
                                        "format": "wav",
                                        "data": "UklGRg=="
                                    }
                                }
                            }]
                        })
                        .to_string(),
                    )))
                    .unwrap()
            })
            .await;

            let router = catch_all_router(vec![audio_addr.clone()]);
            let providers = vec![canonical_provider(
                "openrouter-audio",
                "sk-openrouter-audio",
                format!("http://{}", audio_addr),
                vec!["openai/gpt-4o-audio-preview".to_string()],
                "authorization",
                ProviderFamily::OpenRouter,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    audio: Some(AudioSurface {
                        protocol: AudioSurfaceProtocol::OpenRouterChatAudio,
                        input: false,
                        output: true,
                        transcription: false,
                        translation: false,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "audio-openrouter-key",
                    "openrouter-audio",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "openai/gpt-4o-audio-preview",
                        "input": "Hello from the gateway",
                        "instructions": "Speak warmly",
                        "voice": "alloy",
                        "response_format": "wav"
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert_eq!(
                resp.headers()
                    .get("content-type")
                    .and_then(|value| value.to_str().ok()),
                Some("audio/wav")
            );
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(body.as_ref(), b"RIFF");
        }

        #[tokio::test]
        async fn audio_speech_falls_back_to_native_audio_when_translation_is_incompatible() {
            let openrouter_hits = Arc::new(AtomicUsize::new(0));
            let openrouter_addr = start_upstream({
                let openrouter_hits = Arc::clone(&openrouter_hits);
                move |_req: Request<Incoming>| {
                    openrouter_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(
                            b"unexpected-openrouter-audio",
                        )))
                        .unwrap()
                }
            })
            .await;

            let native_hits = Arc::new(AtomicUsize::new(0));
            let native_addr = start_upstream_async({
                let native_hits = Arc::clone(&native_hits);
                move |req: Request<Incoming>| {
                    let native_hits = Arc::clone(&native_hits);
                    async move {
                        native_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/audio/speech");
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        assert_eq!(body_json["input"].as_str(), Some("Hello natively"));
                        assert_eq!(body_json["speed"].as_f64(), Some(1.1));
                        Response::builder()
                            .status(StatusCode::OK)
                            .body(Full::new(Bytes::from_static(b"native-audio")))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![openrouter_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "openrouter-audio",
                    "sk-openrouter-audio",
                    format!("http://{}", openrouter_addr),
                    vec!["gpt-4o-mini-tts".to_string()],
                    "authorization",
                    ProviderFamily::OpenRouter,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        audio: Some(AudioSurface {
                            protocol: AudioSurfaceProtocol::OpenRouterChatAudio,
                            input: false,
                            output: true,
                            transcription: false,
                            translation: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "native-audio",
                    "sk-native-audio",
                    format!("http://{}", native_addr),
                    vec!["gpt-4o-mini-tts".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        audio: Some(AudioSurface {
                            protocol: AudioSurfaceProtocol::OpenAiAudio,
                            input: false,
                            output: true,
                            transcription: false,
                            translation: false,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "audio-native-fallback-key",
                    "openrouter-audio",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-4o-mini-tts",
                        "input": "Hello natively",
                        "voice": "alloy",
                        "speed": 1.1
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(body.as_ref(), b"native-audio");
            assert_eq!(openrouter_hits.load(Ordering::Relaxed), 0);
            assert_eq!(native_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn audio_transcription_requests_route_to_provider_with_audio_transcription_capability(
        ) {
            assert_request_routes_to_capability_provider(
                "/v1/audio/transcriptions",
                "gpt-4o-mini-transcribe",
                serde_json::json!({
                    "model": "gpt-4o-mini-transcribe",
                    "language": "en"
                }),
                ProviderCapabilityConfig {
                    audio_transcription: true,
                    ..ProviderCapabilityConfig::default()
                },
                b"audio-transcription-upstream",
            )
            .await;
        }

        #[tokio::test]
        async fn audio_translation_requests_route_to_provider_with_audio_translation_capability() {
            assert_request_routes_to_capability_provider(
                "/v1/audio/translations",
                "gpt-4o-mini-transcribe",
                serde_json::json!({
                    "model": "gpt-4o-mini-transcribe"
                }),
                ProviderCapabilityConfig {
                    audio_translation: true,
                    ..ProviderCapabilityConfig::default()
                },
                b"audio-translation-upstream",
            )
            .await;
        }

        #[tokio::test]
        async fn audio_transcription_requests_fail_when_no_provider_declares_support() {
            assert_request_fails_without_capability_provider(
                "/v1/audio/transcriptions",
                "gpt-4o-mini-transcribe",
                serde_json::json!({
                    "model": "gpt-4o-mini-transcribe"
                }),
            )
            .await;
        }

        #[tokio::test]
        async fn audio_translation_requests_fail_when_no_provider_declares_support() {
            assert_request_fails_without_capability_provider(
                "/v1/audio/translations",
                "gpt-4o-mini-transcribe",
                serde_json::json!({
                    "model": "gpt-4o-mini-transcribe"
                }),
            )
            .await;
        }

        #[tokio::test]
        async fn embeddings_requests_route_to_provider_with_embeddings_capability() {
            assert_request_routes_to_capability_provider(
                "/v1/embeddings",
                "text-embedding-3-large",
                serde_json::json!({
                    "model": "text-embedding-3-large",
                    "input": "hello"
                }),
                ProviderCapabilityConfig {
                    embeddings: true,
                    ..ProviderCapabilityConfig::default()
                },
                b"embeddings-upstream",
            )
            .await;
        }

        #[tokio::test]
        async fn embeddings_requests_translate_batch_inputs_to_gemini_embed_content() {
            let embeddings_addr = start_upstream_async(move |req: Request<Incoming>| async move {
                assert_eq!(
                    req.uri().path(),
                    "/v1beta/models/text-embedding-004:batchEmbedContents"
                );
                let body = req.into_body().collect().await.unwrap().to_bytes();
                let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                let requests = body_json["requests"].as_array().expect("requests");
                assert_eq!(requests.len(), 2);
                assert_eq!(
                    requests[0]["model"].as_str(),
                    Some("models/text-embedding-004")
                );
                assert_eq!(
                    requests[0]["content"]["parts"][0]["text"].as_str(),
                    Some("hello")
                );
                assert_eq!(
                    requests[1]["content"]["parts"][0]["text"].as_str(),
                    Some("world")
                );
                assert_eq!(requests[0]["outputDimensionality"].as_u64(), Some(2));
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "embeddings": [
                                { "values": [0.1, 0.2] },
                                { "values": [0.3, 0.4] }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 9
                            }
                        })
                        .to_string(),
                    )))
                    .unwrap()
            })
            .await;

            let router = catch_all_router(vec![embeddings_addr.clone()]);
            let providers = vec![canonical_provider(
                "gemini-embeddings",
                "sk-gemini-embed",
                format!("http://{}", embeddings_addr),
                vec!["text-embedding-004".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    embeddings: Some(EmbeddingSurface {
                        protocol: proxy_core::config::EmbeddingSurfaceProtocol::GeminiEmbedContent,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "gemini-embeddings-key",
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

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "text-embedding-004",
                        "input": ["hello", "world"],
                        "dimensions": 2
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert_eq!(
                resp.headers()
                    .get("content-type")
                    .and_then(|value| value.to_str().ok()),
                Some("application/json")
            );
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["object"].as_str(), Some("list"));
            assert_eq!(body_json["model"].as_str(), Some("text-embedding-004"));
            assert_eq!(body_json["data"].as_array().map(Vec::len), Some(2));
            assert_eq!(body_json["data"][0]["index"].as_u64(), Some(0));
            assert_eq!(body_json["data"][1]["index"].as_u64(), Some(1));
            assert_eq!(
                body_json["data"][0]["embedding"].as_array().map(Vec::len),
                Some(2)
            );
            assert_eq!(body_json["usage"]["prompt_tokens"].as_u64(), Some(9));
            assert_eq!(body_json["usage"]["total_tokens"].as_u64(), Some(9));
        }

        #[tokio::test]
        async fn embeddings_requests_fall_back_to_native_openai_when_gemini_is_incompatible() {
            let gemini_hits = Arc::new(AtomicUsize::new(0));
            let gemini_addr = start_upstream({
                let gemini_hits = Arc::clone(&gemini_hits);
                move |_req: Request<Incoming>| {
                    gemini_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from_static(
                            b"unexpected-gemini-embeddings",
                        )))
                        .unwrap()
                }
            })
            .await;

            let openai_hits = Arc::new(AtomicUsize::new(0));
            let openai_addr = start_upstream_async({
                let openai_hits = Arc::clone(&openai_hits);
                move |req: Request<Incoming>| {
                    let openai_hits = Arc::clone(&openai_hits);
                    async move {
                        openai_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(req.uri().path(), "/v1/embeddings");
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        assert_eq!(body_json["model"].as_str(), Some("text-embedding-3-large"));
                        assert_eq!(
                            body_json["input"],
                            serde_json::json!([[1, 2, 3], [4, 5, 6]])
                        );
                        Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "object": "list",
                                    "data": [
                                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                                        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
                                    ],
                                    "model": "text-embedding-3-large",
                                    "usage": { "prompt_tokens": 6, "total_tokens": 6 }
                                })
                                .to_string(),
                            )))
                            .unwrap()
                    }
                }
            })
            .await;

            let router = catch_all_router(vec![gemini_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "gemini-embeddings",
                    "sk-gemini-embed",
                    format!("http://{}", gemini_addr),
                    vec!["text-embedding-3-large".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        embeddings: Some(EmbeddingSurface {
                            protocol:
                                proxy_core::config::EmbeddingSurfaceProtocol::GeminiEmbedContent,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "openai-embeddings",
                    "sk-openai-embed",
                    format!("http://{}", openai_addr),
                    vec!["text-embedding-3-large".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        embeddings: Some(EmbeddingSurface {
                            protocol: EmbeddingSurfaceProtocol::OpenAiEmbeddings,
                        }),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "embeddings-native-fallback-key",
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

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "text-embedding-3-large",
                        "input": [[1, 2, 3], [4, 5, 6]]
                    })
                    .to_string(),
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["model"].as_str(), Some("text-embedding-3-large"));
            assert_eq!(body_json["data"].as_array().map(Vec::len), Some(2));
            assert_eq!(gemini_hits.load(Ordering::Relaxed), 0);
            assert_eq!(openai_hits.load(Ordering::Relaxed), 1);
        }

        #[tokio::test]
        async fn embeddings_requests_fail_when_no_provider_declares_support() {
            assert_request_fails_without_capability_provider(
                "/v1/embeddings",
                "text-embedding-3-large",
                serde_json::json!({
                    "model": "text-embedding-3-large",
                    "input": "hello"
                }),
            )
            .await;
        }

        #[tokio::test]
        async fn realtime_requests_fail_when_no_provider_declares_support() {
            let upstream_hits = Arc::new(AtomicUsize::new(0));
            let upstream_addr = start_upstream({
                let upstream_hits = Arc::clone(&upstream_hits);
                move |_req: Request<Incoming>| {
                    upstream_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("realtime-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![upstream_addr.clone()]);
            let providers = vec![canonical_provider(
                "openai",
                "sk-openai-real",
                format!("http://{}", upstream_addr),
                vec!["gpt-realtime".to_string()],
                "authorization",
                ProviderFamily::OpenAi,
                ProviderSurfaceCatalog {
                    tools: Some(ToolSurface::OpenAi),
                    ..ProviderSurfaceCatalog::default()
                },
            )];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "realtime-key",
                    "openai",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("GET")
                .uri("/v1/realtime?model=gpt-realtime")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .body(Full::new(Bytes::new()))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body = resp.collect().await.unwrap().to_bytes();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json["error"]["code"].as_str(),
                Some("provider_capability_unsupported")
            );
            assert_eq!(upstream_hits.load(Ordering::Relaxed), 0);
        }

        #[tokio::test]
        async fn realtime_requests_route_to_provider_with_realtime_capability() {
            let standard_hits = Arc::new(AtomicUsize::new(0));
            let standard_addr = start_upstream({
                let standard_hits = Arc::clone(&standard_hits);
                move |_req: Request<Incoming>| {
                    standard_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("standard-realtime-upstream")))
                        .unwrap()
                }
            })
            .await;

            let realtime_hits = Arc::new(AtomicUsize::new(0));
            let realtime_addr = start_upstream({
                let realtime_hits = Arc::clone(&realtime_hits);
                move |_req: Request<Incoming>| {
                    realtime_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .body(Full::new(Bytes::from("realtime-capable-upstream")))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![standard_addr.clone()]);
            let providers = vec![
                canonical_provider(
                    "openai-chat",
                    "sk-openai-real",
                    format!("http://{}", standard_addr),
                    vec!["gpt-realtime".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "openai-realtime",
                    "sk-openai-realtime",
                    format!("http://{}", realtime_addr),
                    vec!["gpt-realtime".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    realtime_surfaces(),
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "realtime-route-key",
                    "openai-chat",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let proxy_addr = start_proxy_with_config(
                router,
                TestProxyConfig {
                    plugins: Some(plugins),
                    ..Default::default()
                },
            )
            .await;

            let req = Request::builder()
                .method("GET")
                .uri("/v1/realtime?model=gpt-realtime")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .body(Full::new(Bytes::new()))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            let body = resp.collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], b"realtime-capable-upstream");
            assert_eq!(standard_hits.load(Ordering::Relaxed), 0);
            assert_eq!(realtime_hits.load(Ordering::Relaxed), 1);
        }
    }

    // -----------------------------------------------------------------------
    // Combined plugin stack tests
    // -----------------------------------------------------------------------

    mod combined {
        use super::*;
        use plugin_llm_gateway::{cost_tracker as ct, rate_limiter as trl};

        #[tokio::test]
        async fn rate_limiter_and_cost_tracker_together() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr]);

            // Rate limiter: generous burst
            let rl_config = toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("tokens_per_minute".into(), toml::Value::Float(600_000.0));
                t.insert("burst_tokens".into(), toml::Value::Float(10_000.0));
                t
            });
            let rl = trl::create(&rl_config).unwrap();

            // Cost tracker: no budget limit
            let ct_config = toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(0.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t
            });
            let ct = ct::create(&ct_config).unwrap();

            let plugins = Arc::new(PluginChain::new(vec![rl, ct]));
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            // Multiple requests through the full stack
            for i in 0..5 {
                let req = chat_request(&format!("/v1/chat/completions?n={}", i), "sk-combo");
                let resp = send_request(&proxy_addr, req).await;
                assert_eq!(
                    resp.status(),
                    StatusCode::OK,
                    "request {} should succeed",
                    i
                );
            }
        }

        #[tokio::test]
        async fn rate_limiter_fires_before_cost_tracker() {
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr]);

            // Tight rate limiter — will reject after first request
            let rl_config = toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("tokens_per_minute".into(), toml::Value::Float(60.0));
                t.insert("burst_tokens".into(), toml::Value::Float(20.0));
                t
            });
            let rl = trl::create(&rl_config).unwrap();

            // Cost tracker with budget — but rate limiter should fire first
            let ct_config = toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(100.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t
            });
            let ct = ct::create(&ct_config).unwrap();

            // Rate limiter first in chain
            let plugins = Arc::new(PluginChain::new(vec![rl, ct]));
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            let req = chat_request("/v1/chat/completions", "sk-order");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            // Second request: rate limiter should reject with 429 (not 402)
            let req = chat_request("/v1/chat/completions", "sk-order");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        }
    }

    // -----------------------------------------------------------------------
    // Cross-provider retry tests
    // -----------------------------------------------------------------------

    mod cross_provider_retry {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use plugin_llm_gateway::provider_failover as pf;
        use plugin_llm_gateway::virtual_keys::VirtualKeys;

        #[tokio::test]
        async fn proxy_retries_post_on_next_provider_after_500() {
            // Provider A returns 500, Provider B returns 200. Client should get 200.
            let failing_addr = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::new(Bytes::from("error")))
                    .unwrap()
            })
            .await;

            let success_addr = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(r#"{"ok":true}"#)))
                    .unwrap()
            })
            .await;

            let router = catch_all_router(vec![failing_addr.clone()]);

            let providers = vec![
                canonical_provider(
                    "provider-a",
                    "sk-a",
                    format!("http://{}", failing_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "provider-b",
                    "sk-b",
                    format!("http://{}", success_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "retry-test",
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

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "should retry on provider B and succeed"
            );
        }

        #[tokio::test]
        async fn proxy_retries_post_on_next_provider_after_429() {
            // Provider A returns 429, Provider B returns 200. Client should get 200.
            let rate_limited_addr = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::TOO_MANY_REQUESTS)
                    .body(Full::new(Bytes::from("rate limited")))
                    .unwrap()
            })
            .await;

            let success_addr = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(r#"{"ok":true}"#)))
                    .unwrap()
            })
            .await;

            let router = catch_all_router(vec![rate_limited_addr.clone()]);

            let providers = vec![
                canonical_provider(
                    "provider-a",
                    "sk-a",
                    format!("http://{}", rate_limited_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "provider-b",
                    "sk-b",
                    format!("http://{}", success_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "retry-429-test",
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

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "should retry on provider B after provider A returns 429"
            );
        }

        #[tokio::test]
        async fn provider_failover_skips_500ing_provider_on_next_request() {
            let failing_hits = Arc::new(AtomicUsize::new(0));
            let failing_addr = start_upstream({
                let failing_hits = Arc::clone(&failing_hits);
                move |_req: Request<Incoming>| {
                    failing_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Full::new(Bytes::from("provider error")))
                        .unwrap()
                }
            })
            .await;

            let healthy_hits = Arc::new(AtomicUsize::new(0));
            let healthy_addr = start_upstream({
                let healthy_hits = Arc::clone(&healthy_hits);
                move |_req: Request<Incoming>| {
                    healthy_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"ok":true}"#)))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![failing_addr.clone()]);

            let providers = vec![
                canonical_provider(
                    "provider-a",
                    "sk-a",
                    format!("http://{}", failing_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "provider-b",
                    "sk-b",
                    format!("http://{}", healthy_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "500-failover-test",
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

            let failover_config = toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("cooldown_secs".into(), toml::Value::Integer(60));
                t.insert(
                    "providers".into(),
                    toml::Value::Array(vec![
                        {
                            let mut p = toml::value::Map::new();
                            p.insert("name".into(), toml::Value::String("provider-a".into()));
                            p.insert(
                                "pattern".into(),
                                toml::Value::String(format!("http://{}", failing_addr)),
                            );
                            toml::Value::Table(p)
                        },
                        {
                            let mut p = toml::value::Map::new();
                            p.insert("name".into(), toml::Value::String("provider-b".into()));
                            p.insert(
                                "pattern".into(),
                                toml::Value::String(format!("http://{}", healthy_addr)),
                            );
                            toml::Value::Table(p)
                        },
                    ]),
                );
                t
            });
            let failover = pf::create(&failover_config).unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk), failover]));
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            for _ in 0..2 {
                let req = Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", format!("Bearer {}", plaintext_key))
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
                    )))
                    .unwrap();

                let resp = send_request(&proxy_addr, req).await;
                assert_eq!(resp.status(), StatusCode::OK);
            }

            assert_eq!(
                failing_hits.load(Ordering::Relaxed),
                1,
                "500ing provider should only be hit on the first request",
            );
            assert_eq!(
                healthy_hits.load(Ordering::Relaxed),
                2,
                "healthy provider should handle both client-visible requests",
            );
        }

        #[tokio::test]
        async fn all_candidates_failed_returns_error() {
            // Both providers return 500 → client gets 500.
            let failing_addr_a = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::new(Bytes::from("error-a")))
                    .unwrap()
            })
            .await;

            let failing_addr_b = start_upstream(|_req: Request<Incoming>| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::new(Bytes::from("error-b")))
                    .unwrap()
            })
            .await;

            let router = catch_all_router(vec![failing_addr_a.clone()]);

            let providers = vec![
                canonical_provider(
                    "provider-a",
                    "sk-a",
                    format!("http://{}", failing_addr_a),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "provider-b",
                    "sk-b",
                    format!("http://{}", failing_addr_b),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "fail-test",
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

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk)]));
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert!(
                resp.status().is_server_error(),
                "all providers failed → should return 5xx, got {}",
                resp.status()
            );
        }

        #[tokio::test]
        async fn no_candidates_falls_through_to_existing_path() {
            // Non-virtual-key request should use existing retry logic.
            let upstream_addr = start_upstream(llm_chat_handler).await;
            let router = catch_all_router(vec![upstream_addr]);

            let config = TestProxyConfig::default();
            let proxy_addr = start_proxy_with_config(router, config).await;

            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-regular-key")
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#,
                )))
                .unwrap();

            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
        }

        #[tokio::test]
        async fn provider_failover_skips_rate_limited_provider_on_next_request() {
            let rate_limited_hits = Arc::new(AtomicUsize::new(0));
            let rate_limited_addr = start_upstream({
                let rate_limited_hits = Arc::clone(&rate_limited_hits);
                move |_req: Request<Incoming>| {
                    rate_limited_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::TOO_MANY_REQUESTS)
                        .body(Full::new(Bytes::from("rate limited")))
                        .unwrap()
                }
            })
            .await;

            let healthy_hits = Arc::new(AtomicUsize::new(0));
            let healthy_addr = start_upstream({
                let healthy_hits = Arc::clone(&healthy_hits);
                move |_req: Request<Incoming>| {
                    healthy_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(r#"{"ok":true}"#)))
                        .unwrap()
                }
            })
            .await;

            let router = catch_all_router(vec![rate_limited_addr.clone()]);

            let providers = vec![
                canonical_provider(
                    "provider-a",
                    "sk-a",
                    format!("http://{}", rate_limited_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
                canonical_provider(
                    "provider-b",
                    "sk-b",
                    format!("http://{}", healthy_addr),
                    vec!["gpt-4o".to_string()],
                    "authorization",
                    ProviderFamily::OpenAi,
                    ProviderSurfaceCatalog {
                        tools: Some(ToolSurface::OpenAi),
                        ..ProviderSurfaceCatalog::default()
                    },
                ),
            ];

            let vk = VirtualKeys::new(&providers, &[], None);
            let (plaintext_key, _) = vk
                .create_key_for_project(
                    Some("project-a"),
                    "rate-limit-failover-test",
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

            let failover_config = toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("cooldown_secs".into(), toml::Value::Integer(60));
                t.insert(
                    "providers".into(),
                    toml::Value::Array(vec![
                        {
                            let mut p = toml::value::Map::new();
                            p.insert("name".into(), toml::Value::String("provider-a".into()));
                            p.insert(
                                "pattern".into(),
                                toml::Value::String(format!("http://{}", rate_limited_addr)),
                            );
                            toml::Value::Table(p)
                        },
                        {
                            let mut p = toml::value::Map::new();
                            p.insert("name".into(), toml::Value::String("provider-b".into()));
                            p.insert(
                                "pattern".into(),
                                toml::Value::String(format!("http://{}", healthy_addr)),
                            );
                            toml::Value::Table(p)
                        },
                    ]),
                );
                t
            });
            let failover = pf::create(&failover_config).unwrap();

            let plugins = Arc::new(PluginChain::new(vec![Box::new(vk), failover]));
            let config = TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            };
            let proxy_addr = start_proxy_with_config(router, config).await;

            for _ in 0..2 {
                let req = Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", format!("Bearer {}", plaintext_key))
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
                    )))
                    .unwrap();

                let resp = send_request(&proxy_addr, req).await;
                assert_eq!(resp.status(), StatusCode::OK);
            }

            assert_eq!(
                rate_limited_hits.load(Ordering::Relaxed),
                1,
                "rate-limited provider should only be hit on the first request",
            );
            assert_eq!(
                healthy_hits.load(Ordering::Relaxed),
                2,
                "healthy provider should handle both client-visible requests",
            );
        }
    }
}
