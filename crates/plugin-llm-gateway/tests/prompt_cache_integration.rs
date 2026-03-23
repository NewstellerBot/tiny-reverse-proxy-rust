#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};
    use tempfile::NamedTempFile;
    use tokio::sync::Mutex;
    use tokio::time::{sleep, Duration, Instant};

    use plugin_llm_gateway::store::ProjectPolicyRecord;
    use plugin_llm_gateway::CreatePluginsOptions;
    use proxy_core::config::{
        PluginConfig, PromptCacheProtocol, PromptCacheSurface, ProviderFamilyConfig,
        ProviderKeyConfig, ProviderSurfaceCatalog,
    };
    use proxy_core::plugin::PluginChain;

    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream_async,
        TestProxyConfig,
    };

    fn prompt_cache_config_with_options(
        anthropic_scope: Option<&str>,
        persist_routing_hints: Option<bool>,
        routing_flush_interval_ms: Option<u64>,
        routing_prune_interval_secs: Option<u64>,
    ) -> Vec<PluginConfig> {
        vec![PluginConfig {
            name: "prompt_cache".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut table = toml::value::Map::new();
                if let Some(scope) = anthropic_scope {
                    table.insert(
                        "anthropic_default_scope".into(),
                        toml::Value::String(scope.to_string()),
                    );
                }
                if let Some(enabled) = persist_routing_hints {
                    table.insert(
                        "persist_routing_hints".into(),
                        toml::Value::Boolean(enabled),
                    );
                }
                if let Some(interval_ms) = routing_flush_interval_ms {
                    table.insert(
                        "routing_flush_interval_ms".into(),
                        toml::Value::Integer(interval_ms as i64),
                    );
                }
                if let Some(interval_secs) = routing_prune_interval_secs {
                    table.insert(
                        "routing_prune_interval_secs".into(),
                        toml::Value::Integer(interval_secs as i64),
                    );
                }
                table
            }),
        }]
    }

    fn prompt_cache_config() -> Vec<PluginConfig> {
        prompt_cache_config_with_options(None, None, None, None)
    }

    fn prompt_cache_config_with_anthropic_scope(scope: &str) -> Vec<PluginConfig> {
        prompt_cache_config_with_options(Some(scope), None, None, None)
    }

    fn provider(
        name: &str,
        api_key: &str,
        base_url: String,
        models: Vec<String>,
        api_key_header: &str,
        family: ProviderFamilyConfig,
    ) -> ProviderKeyConfig {
        let surfaces = family.surfaces().clone();
        ProviderKeyConfig {
            name: name.to_string(),
            api_key: api_key.to_string(),
            base_url,
            models,
            api_key_header: api_key_header.to_string(),
            timeout_secs: None,
            family,
            tool_protocol: surfaces.derived_tool_protocol(),
            image_protocol: surfaces.derived_image_protocol(),
            audio_protocol: surfaces.derived_audio_protocol(),
            embedding_protocol: surfaces.derived_embedding_protocol(),
            routing_metadata: Default::default(),
            capabilities: surfaces.derived_capabilities(),
        }
    }

    fn openai_provider(base_url: String) -> ProviderKeyConfig {
        provider(
            "openai",
            "sk-openai-real",
            base_url,
            vec!["gpt-4o".to_string()],
            "authorization",
            ProviderFamilyConfig::OpenAi {
                surfaces: ProviderSurfaceCatalog::default(),
            },
        )
    }

    fn anthropic_provider(base_url: String) -> ProviderKeyConfig {
        provider(
            "anthropic",
            "sk-anthropic-real",
            base_url,
            vec!["claude-sonnet-4-20250514".to_string()],
            "x-api-key",
            ProviderFamilyConfig::Anthropic {
                surfaces: ProviderSurfaceCatalog::default(),
            },
        )
    }

    fn generic_provider(base_url: String) -> ProviderKeyConfig {
        provider(
            "generic",
            "sk-generic-real",
            base_url,
            vec!["gpt-4o".to_string()],
            "authorization",
            ProviderFamilyConfig::Custom {
                surfaces: ProviderSurfaceCatalog::default(),
            },
        )
    }

    fn generic_prompt_cache_provider(base_url: String) -> ProviderKeyConfig {
        provider(
            "generic-cache",
            "sk-generic-cache-real",
            base_url,
            vec!["gpt-4o".to_string()],
            "authorization",
            ProviderFamilyConfig::Custom {
                surfaces: ProviderSurfaceCatalog {
                    prompt_cache: Some(PromptCacheSurface {
                        protocol: PromptCacheProtocol::OpenAi,
                        request_controls: true,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            },
        )
    }

    fn openai_prompt_cache_provider(base_url: String, request_controls: bool) -> ProviderKeyConfig {
        provider(
            "openai",
            "sk-openai-real",
            base_url,
            vec!["gpt-4o".to_string()],
            "authorization",
            ProviderFamilyConfig::OpenAi {
                surfaces: ProviderSurfaceCatalog {
                    prompt_cache: Some(PromptCacheSurface {
                        protocol: PromptCacheProtocol::OpenAi,
                        request_controls,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            },
        )
    }

    fn reporting_only_prompt_cache_provider(base_url: String) -> ProviderKeyConfig {
        provider(
            "generic-cache-report",
            "sk-generic-cache-report-real",
            base_url,
            vec!["gpt-4o".to_string()],
            "authorization",
            ProviderFamilyConfig::Custom {
                surfaces: ProviderSurfaceCatalog {
                    prompt_cache: Some(PromptCacheSurface {
                        protocol: PromptCacheProtocol::OpenAi,
                        request_controls: false,
                    }),
                    ..ProviderSurfaceCatalog::default()
                },
            },
        )
    }

    async fn setup_gateway(
        configs: Vec<PluginConfig>,
        providers: &[ProviderKeyConfig],
    ) -> (Arc<PluginChain>, plugin_llm_gateway::api::LlmGatewayApi) {
        setup_gateway_with_store(configs, providers, "sqlite::memory:").await
    }

    async fn setup_gateway_with_store(
        configs: Vec<PluginConfig>,
        providers: &[ProviderKeyConfig],
        store_url: &str,
    ) -> (Arc<PluginChain>, plugin_llm_gateway::api::LlmGatewayApi) {
        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some(store_url),
            providers,
            &[],
            CreatePluginsOptions::default(),
            None,
        )
        .await
        .expect("create plugins");
        (Arc::new(PluginChain::new(plugins)), api)
    }

    async fn wait_for_prompt_cache_status<F>(
        api: &plugin_llm_gateway::api::LlmGatewayApi,
        predicate: F,
    ) -> plugin_llm_gateway::prompt_cache::PromptCacheStatusSnapshot
    where
        F: Fn(&plugin_llm_gateway::prompt_cache::PromptCacheStatusSnapshot) -> bool,
    {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let status = api.prompt_cache_status().expect("prompt cache enabled");
            if predicate(&status) {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for prompt cache status change"
            );
            sleep(Duration::from_millis(25)).await;
        }
    }

    #[tokio::test]
    async fn openai_prompt_cache_injects_controls_and_exposes_usage_headers() {
        let provider_requests = Arc::new(Mutex::new(Vec::new()));
        let provider_addr = start_upstream_async({
            let provider_requests = Arc::clone(&provider_requests);
            move |req: Request<Incoming>| {
                let provider_requests = Arc::clone(&provider_requests);
                async move {
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");
                    provider_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-cache-1",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "cached response"
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 32,
                                    "completion_tokens": 8,
                                    "total_tokens": 40,
                                    "prompt_tokens_details": {
                                        "cached_tokens": 128,
                                        "cache_write_tokens": 32
                                    }
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_provider(format!("http://{}", provider_addr))];
        let (plugins, api) = setup_gateway(
            prompt_cache_config_with_options(None, Some(true), Some(60_000), Some(60)),
            &providers,
        )
        .await;
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-cache-openai"),
                "cache-key",
                "openai",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![format!("http://{}", provider_addr)]);
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
                    "messages": [{"role": "user", "content": "cache this prompt"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:123"
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-provider")
                .and_then(|value| value.to_str().ok()),
            Some("openai")
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-protocol")
                .and_then(|value| value.to_str().ok()),
            Some("openai")
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-status")
                .and_then(|value| value.to_str().ok()),
            Some("hit_write")
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-read-tokens")
                .and_then(|value| value.to_str().ok()),
            Some("128")
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-write-tokens")
                .and_then(|value| value.to_str().ok()),
            Some("32")
        );

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("response json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("cached response")
        );

        let provider_requests = provider_requests.lock().await;
        assert_eq!(provider_requests.len(), 1);
        assert!(provider_requests[0].get("trp_prompt_cache").is_none());
        assert_eq!(
            provider_requests[0]["prompt_cache_key"].as_str(),
            Some("tenant:123")
        );
        assert_eq!(
            provider_requests[0]["prompt_cache_retention"].as_str(),
            Some("24h")
        );
    }

    #[tokio::test]
    async fn anthropic_prompt_cache_inserts_cache_control_and_exposes_usage_headers() {
        let provider_requests = Arc::new(Mutex::new(Vec::new()));
        let provider_addr = start_upstream_async({
            let provider_requests = Arc::clone(&provider_requests);
            move |req: Request<Incoming>| {
                let provider_requests = Arc::clone(&provider_requests);
                async move {
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");
                    provider_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "msg-cache-1",
                                "type": "message",
                                "role": "assistant",
                                "model": "claude-sonnet-4-20250514",
                                "content": [{"type": "text", "text": "cached answer"}],
                                "stop_reason": "end_turn",
                                "usage": {
                                    "input_tokens": 20,
                                    "output_tokens": 5,
                                    "cache_read_input_tokens": 64,
                                    "cache_creation_input_tokens": 8
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![anthropic_provider(format!("http://{}", provider_addr))];
        let (plugins, api) = setup_gateway(
            prompt_cache_config_with_options(None, Some(true), Some(60_000), Some(60)),
            &providers,
        )
        .await;
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-cache-anthropic"),
                "cache-key",
                "anthropic",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![format!("http://{}", provider_addr)]);
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
            .uri("/v1/messages")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "claude-sonnet-4-20250514",
                    "system": "You are concise",
                    "messages": [{"role": "user", "content": "hi"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "1h"
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-provider")
                .and_then(|value| value.to_str().ok()),
            Some("anthropic")
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-protocol")
                .and_then(|value| value.to_str().ok()),
            Some("anthropic")
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-status")
                .and_then(|value| value.to_str().ok()),
            Some("hit_write")
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-read-tokens")
                .and_then(|value| value.to_str().ok()),
            Some("64")
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-write-tokens")
                .and_then(|value| value.to_str().ok()),
            Some("8")
        );

        let provider_requests = provider_requests.lock().await;
        assert_eq!(provider_requests.len(), 1);
        assert!(provider_requests[0].get("trp_prompt_cache").is_none());
        assert_eq!(
            provider_requests[0]["system"][0]["cache_control"]["ttl"].as_str(),
            Some("1h")
        );
    }

    #[tokio::test]
    async fn openai_prompt_cache_rejects_invalid_ttl_without_forwarding_upstream() {
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |_req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from("{}")))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![openai_provider(format!("http://{}", provider_addr))];
        let (plugins, api) = setup_gateway(
            prompt_cache_config_with_options(None, Some(true), Some(60_000), Some(60)),
            &providers,
        )
        .await;
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-cache-invalid"),
                "cache-key",
                "openai",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![format!("http://{}", provider_addr)]);
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
                    "messages": [{"role": "user", "content": "cache this prompt"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "7d"
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("response json");
        assert!(body_json["error"]
            .as_str()
            .unwrap_or_default()
            .contains("OpenAI prompt cache ttl must be one of"));
        assert_eq!(provider_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn prompt_cache_request_prefers_supported_provider_for_same_model() {
        let generic_hits = Arc::new(AtomicUsize::new(0));
        let generic_addr = start_upstream_async({
            let generic_hits = Arc::clone(&generic_hits);
            move |_req: Request<Incoming>| {
                let generic_hits = Arc::clone(&generic_hits);
                async move {
                    generic_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "generic-hit",
                                "choices": [{
                                    "message": {
                                        "role": "assistant",
                                        "content": "generic"
                                    }
                                }]
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let openai_requests = Arc::new(Mutex::new(Vec::new()));
        let openai_addr = start_upstream_async({
            let openai_requests = Arc::clone(&openai_requests);
            move |req: Request<Incoming>| {
                let openai_requests = Arc::clone(&openai_requests);
                async move {
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");
                    openai_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-cache-routed",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "routed to cache-capable provider"
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 6,
                                    "total_tokens": 26,
                                    "prompt_tokens_details": {
                                        "cached_tokens": 64,
                                        "cache_write_tokens": 16
                                    }
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![
            generic_provider(format!("http://{}", generic_addr)),
            openai_provider(format!("http://{}", openai_addr)),
        ];
        let (plugins, api) = setup_gateway(prompt_cache_config(), &providers).await;
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-cache-routing"),
                "cache-key",
                "generic",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![
            format!("http://{}", generic_addr),
            format!("http://{}", openai_addr),
        ]);
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
                    "messages": [{"role": "user", "content": "cache this prompt"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:routing"
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-provider")
                .and_then(|value| value.to_str().ok()),
            Some("openai")
        );

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("response json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("routed to cache-capable provider")
        );
        assert_eq!(generic_hits.load(Ordering::Relaxed), 0);

        let openai_requests = openai_requests.lock().await;
        assert_eq!(openai_requests.len(), 1);
        assert!(openai_requests[0].get("trp_prompt_cache").is_none());
        assert_eq!(
            openai_requests[0]["prompt_cache_key"].as_str(),
            Some("tenant:routing")
        );
        assert_eq!(
            openai_requests[0]["prompt_cache_retention"].as_str(),
            Some("24h")
        );
    }

    #[tokio::test]
    async fn prompt_cache_request_prefers_provider_with_required_request_controls() {
        let reporting_hits = Arc::new(AtomicUsize::new(0));
        let reporting_addr = start_upstream_async({
            let reporting_hits = Arc::clone(&reporting_hits);
            move |_req: Request<Incoming>| {
                let reporting_hits = Arc::clone(&reporting_hits);
                async move {
                    reporting_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from("{}")))
                        .unwrap()
                }
            }
        })
        .await;

        let openai_requests = Arc::new(Mutex::new(Vec::new()));
        let openai_addr = start_upstream_async({
            let openai_requests = Arc::clone(&openai_requests);
            move |req: Request<Incoming>| {
                let openai_requests = Arc::clone(&openai_requests);
                async move {
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");
                    openai_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-cache-controls",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "used request-controls-capable provider"
                                    },
                                    "finish_reason": "stop"
                                }]
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![
            reporting_only_prompt_cache_provider(format!("http://{}", reporting_addr)),
            openai_provider(format!("http://{}", openai_addr)),
        ];
        let (plugins, api) = setup_gateway(prompt_cache_config(), &providers).await;
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-cache-controls-routing"),
                "cache-key",
                "generic-cache-report",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![
            format!("http://{}", reporting_addr),
            format!("http://{}", openai_addr),
        ]);
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
                    "messages": [{"role": "user", "content": "cache this prompt"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:controls"
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(reporting_hits.load(Ordering::Relaxed), 0);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("response json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("used request-controls-capable provider")
        );

        let openai_requests = openai_requests.lock().await;
        assert_eq!(openai_requests.len(), 1);
        assert_eq!(
            openai_requests[0]["prompt_cache_key"].as_str(),
            Some("tenant:controls")
        );
        assert_eq!(
            openai_requests[0]["prompt_cache_retention"].as_str(),
            Some("24h")
        );
    }

    #[tokio::test]
    async fn prompt_cache_key_prefers_warmed_provider_over_project_fallback_order() {
        let alpha_requests = Arc::new(Mutex::new(Vec::new()));
        let alpha_addr = start_upstream_async({
            let alpha_requests = Arc::clone(&alpha_requests);
            move |req: Request<Incoming>| {
                let alpha_requests = Arc::clone(&alpha_requests);
                async move {
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect alpha body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("alpha request json");
                    alpha_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-alpha-cache",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "alpha kept the warm cache"
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 18,
                                    "completion_tokens": 4,
                                    "total_tokens": 22,
                                    "prompt_tokens_details": {
                                        "cached_tokens": 96,
                                        "cache_write_tokens": 24
                                    }
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let beta_hits = Arc::new(AtomicUsize::new(0));
        let beta_addr = start_upstream_async({
            let beta_hits = Arc::clone(&beta_hits);
            move |_req: Request<Incoming>| {
                let beta_hits = Arc::clone(&beta_hits);
                async move {
                    beta_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-beta-cache",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "beta should not be chosen after alpha warms the cache"
                                    },
                                    "finish_reason": "stop"
                                }]
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![
            openai_prompt_cache_provider(format!("http://{}", alpha_addr), true),
            openai_prompt_cache_provider(format!("http://{}", beta_addr), true),
        ];
        let (plugins, api) = setup_gateway(
            prompt_cache_config_with_options(None, Some(true), Some(60_000), Some(60)),
            &providers,
        )
        .await;
        let project_id = "project-cache-warm-routing";
        api.upsert_project_policy(ProjectPolicyRecord {
            project_id: project_id.to_string(),
            budget_limit: None,
            budget_duration: None,
            rpm_limit: None,
            tpm_limit: None,
            fallback_order: Some(r#"["alpha","beta"]"#.to_string()),
            adaptive_enabled: false,
            timeout_secs: None,
            provider_rpm_limits: None,
            provider_tpm_limits: None,
            provider_timeouts: None,
            provider_input_costs: None,
            provider_output_costs: None,
            semantic_cache_enabled: None,
            semantic_cache_ttl_secs: None,
            semantic_cache_similarity_threshold: None,
            tool_approval_mode: None,
            allowed_tools: None,
            updated_at: "1".to_string(),
        })
        .await
        .expect("project policy API enabled")
        .expect("store project policy");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "cache-key",
                "alpha",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![
            format!("http://{}", alpha_addr),
            format!("http://{}", beta_addr),
        ]);
        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            },
        )
        .await;

        let first_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "warm alpha's prompt cache"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:warm"
                    }
                })
                .to_string(),
            )))
            .unwrap();
        let first_resp = send_request(&proxy_addr, first_req).await;
        assert_eq!(first_resp.status(), StatusCode::OK);
        let first_body = first_resp.into_body().collect().await.unwrap().to_bytes();
        let first_json: serde_json::Value =
            serde_json::from_slice(&first_body).expect("first response json");
        assert_eq!(
            first_json["choices"][0]["message"]["content"].as_str(),
            Some("alpha kept the warm cache")
        );
        let status_after_warm = api.prompt_cache_status().expect("prompt cache enabled");
        assert!(status_after_warm.store_backed);
        assert!(status_after_warm.routing_hint_persistence_enabled);
        assert_eq!(status_after_warm.routing_flush_interval_ms, 60_000);
        assert_eq!(status_after_warm.routing_prune_interval_secs, 60);
        assert_eq!(status_after_warm.pending_route_updates, 2);
        assert!(status_after_warm.last_route_flush_unix_ms.is_none());

        api.upsert_project_policy(ProjectPolicyRecord {
            project_id: project_id.to_string(),
            budget_limit: None,
            budget_duration: None,
            rpm_limit: None,
            tpm_limit: None,
            fallback_order: Some(r#"["beta","alpha"]"#.to_string()),
            adaptive_enabled: false,
            timeout_secs: None,
            provider_rpm_limits: None,
            provider_tpm_limits: None,
            provider_timeouts: None,
            provider_input_costs: None,
            provider_output_costs: None,
            semantic_cache_enabled: None,
            semantic_cache_ttl_secs: None,
            semantic_cache_similarity_threshold: None,
            tool_approval_mode: None,
            allowed_tools: None,
            updated_at: "2".to_string(),
        })
        .await
        .expect("project policy API enabled")
        .expect("update project policy");

        let second_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("x-trp-routing-debug", "1")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "use the warm cache again"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:warm"
                    }
                })
                .to_string(),
            )))
            .unwrap();
        let second_resp = send_request(&proxy_addr, second_req).await;
        assert_eq!(second_resp.status(), StatusCode::OK);
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-provider-selected")
                .and_then(|value| value.to_str().ok()),
            Some("alpha")
        );
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-prompt-cache-route")
                .and_then(|value| value.to_str().ok()),
            Some("alpha")
        );
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-prompt-cache-affinity")
                .and_then(|value| value.to_str().ok()),
            Some("applied")
        );
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-provider-order")
                .and_then(|value| value.to_str().ok()),
            Some("alpha,beta")
        );
        let second_body = second_resp.into_body().collect().await.unwrap().to_bytes();
        let second_json: serde_json::Value =
            serde_json::from_slice(&second_body).expect("second response json");
        assert_eq!(
            second_json["choices"][0]["message"]["content"].as_str(),
            Some("alpha kept the warm cache")
        );

        let alpha_requests = alpha_requests.lock().await;
        assert_eq!(alpha_requests.len(), 2);
        assert_eq!(beta_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn prompt_cache_miss_records_negative_affinity_and_avoids_stale_provider() {
        let alpha_hits = Arc::new(AtomicUsize::new(0));
        let alpha_addr = start_upstream_async({
            let alpha_hits = Arc::clone(&alpha_hits);
            move |_req: Request<Incoming>| {
                let alpha_hits = Arc::clone(&alpha_hits);
                async move {
                    let call_index = alpha_hits.fetch_add(1, Ordering::Relaxed);
                    let response_json = if call_index == 0 {
                        serde_json::json!({
                            "id": "chatcmpl-alpha-warm",
                            "object": "chat.completion",
                            "model": "gpt-4o",
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "alpha warmed the cache"
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": 18,
                                "completion_tokens": 4,
                                "total_tokens": 22,
                                "prompt_tokens_details": {
                                    "cached_tokens": 64,
                                    "cache_write_tokens": 16
                                }
                            }
                        })
                    } else {
                        serde_json::json!({
                            "id": "chatcmpl-alpha-miss",
                            "object": "chat.completion",
                            "model": "gpt-4o",
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "alpha missed the stale cache"
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": 18,
                                "completion_tokens": 4,
                                "total_tokens": 22
                            }
                        })
                    };
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(response_json.to_string())))
                        .unwrap()
                }
            }
        })
        .await;

        let beta_hits = Arc::new(AtomicUsize::new(0));
        let beta_addr = start_upstream_async({
            let beta_hits = Arc::clone(&beta_hits);
            move |_req: Request<Incoming>| {
                let beta_hits = Arc::clone(&beta_hits);
                async move {
                    beta_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-beta-after-miss",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "beta handled the request after alpha missed"
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 18,
                                    "completion_tokens": 4,
                                    "total_tokens": 22,
                                    "prompt_tokens_details": {
                                        "cached_tokens": 24,
                                        "cache_write_tokens": 8
                                    }
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![
            openai_prompt_cache_provider(format!("http://{}", alpha_addr), true),
            openai_prompt_cache_provider(format!("http://{}", beta_addr), true),
        ];
        let (plugins, api) = setup_gateway(prompt_cache_config(), &providers).await;
        let project_id = "project-cache-negative-routing";
        api.upsert_project_policy(ProjectPolicyRecord {
            project_id: project_id.to_string(),
            budget_limit: None,
            budget_duration: None,
            rpm_limit: None,
            tpm_limit: None,
            fallback_order: Some(r#"["alpha","beta"]"#.to_string()),
            adaptive_enabled: false,
            timeout_secs: None,
            provider_rpm_limits: None,
            provider_tpm_limits: None,
            provider_timeouts: None,
            provider_input_costs: None,
            provider_output_costs: None,
            semantic_cache_enabled: None,
            semantic_cache_ttl_secs: None,
            semantic_cache_similarity_threshold: None,
            tool_approval_mode: None,
            allowed_tools: None,
            updated_at: "1".to_string(),
        })
        .await
        .expect("project policy API enabled")
        .expect("store project policy");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "cache-key",
                "alpha",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![
            format!("http://{}", alpha_addr),
            format!("http://{}", beta_addr),
        ]);
        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(plugins),
                ..Default::default()
            },
        )
        .await;

        let request_body = |message: &str| {
            serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": message}],
                "trp_prompt_cache": {
                    "enabled": true,
                    "ttl": "24h",
                    "key": "tenant:negative"
                }
            })
            .to_string()
        };

        let first_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(request_body("warm alpha first"))))
            .unwrap();
        let first_resp = send_request(&proxy_addr, first_req).await;
        assert_eq!(first_resp.status(), StatusCode::OK);

        let second_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("x-trp-routing-debug", "1")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(request_body(
                "alpha should miss now",
            ))))
            .unwrap();
        let second_resp = send_request(&proxy_addr, second_req).await;
        assert_eq!(second_resp.status(), StatusCode::OK);
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-provider-selected")
                .and_then(|value| value.to_str().ok()),
            Some("alpha")
        );
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-prompt-cache-status")
                .and_then(|value| value.to_str().ok()),
            Some("miss")
        );

        let status_after_miss = api.prompt_cache_status().expect("prompt cache enabled");
        assert_eq!(status_after_miss.warmed_route_count, 0);
        assert_eq!(status_after_miss.negative_route_count, 1);

        let third_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("x-trp-routing-debug", "1")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(request_body(
                "avoid alpha after the miss",
            ))))
            .unwrap();
        let third_resp = send_request(&proxy_addr, third_req).await;
        assert_eq!(third_resp.status(), StatusCode::OK);
        assert_eq!(
            third_resp
                .headers()
                .get("x-trp-provider-selected")
                .and_then(|value| value.to_str().ok()),
            Some("beta")
        );
        assert_eq!(
            third_resp
                .headers()
                .get("x-trp-prompt-cache-negative")
                .and_then(|value| value.to_str().ok()),
            Some("alpha")
        );
        assert_eq!(
            third_resp
                .headers()
                .get("x-trp-prompt-cache-affinity")
                .and_then(|value| value.to_str().ok()),
            Some("applied")
        );
        assert!(
            third_resp
                .headers()
                .get("x-trp-prompt-cache-route")
                .is_none(),
            "negative-only routing should not advertise a preferred warm provider"
        );

        let third_body = third_resp.into_body().collect().await.unwrap().to_bytes();
        let third_json: serde_json::Value =
            serde_json::from_slice(&third_body).expect("third response json");
        assert_eq!(
            third_json["choices"][0]["message"]["content"].as_str(),
            Some("beta handled the request after alpha missed")
        );
        assert_eq!(alpha_hits.load(Ordering::Relaxed), 2);
        assert_eq!(beta_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn prompt_cache_request_uses_anthropic_default_scope_for_routing() {
        let openai_hits = Arc::new(AtomicUsize::new(0));
        let openai_addr = start_upstream_async({
            let openai_hits = Arc::clone(&openai_hits);
            move |_req: Request<Incoming>| {
                let openai_hits = Arc::clone(&openai_hits);
                async move {
                    openai_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from("{}")))
                        .unwrap()
                }
            }
        })
        .await;

        let anthropic_requests = Arc::new(Mutex::new(Vec::new()));
        let anthropic_addr = start_upstream_async({
            let anthropic_requests = Arc::clone(&anthropic_requests);
            move |req: Request<Incoming>| {
                let anthropic_requests = Arc::clone(&anthropic_requests);
                async move {
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");
                    anthropic_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "msg-cache-default-scope",
                                "type": "message",
                                "role": "assistant",
                                "model": "shared-cache-model",
                                "content": [{"type": "text", "text": "used anthropic default scope"}],
                                "stop_reason": "end_turn",
                                "usage": {
                                    "input_tokens": 18,
                                    "output_tokens": 4,
                                    "cache_read_input_tokens": 12,
                                    "cache_creation_input_tokens": 3
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![
            openai_provider(format!("http://{}", openai_addr)),
            anthropic_provider(format!("http://{}", anthropic_addr)),
        ];
        let (plugins, api) = setup_gateway(
            prompt_cache_config_with_anthropic_scope("tools"),
            &providers,
        )
        .await;
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-cache-default-scope"),
                "cache-key",
                "openai",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![
            format!("http://{}", openai_addr),
            format!("http://{}", anthropic_addr),
        ]);
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
            .uri("/v1/messages")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "shared-cache-model",
                    "system": "You are concise",
                    "tools": [{
                        "name": "search_docs",
                        "description": "Search docs",
                        "input_schema": {
                            "type": "object",
                            "properties": { "query": { "type": "string" } }
                        }
                    }],
                    "messages": [{"role": "user", "content": "hi"}],
                    "trp_prompt_cache": {
                        "enabled": true
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(openai_hits.load(Ordering::Relaxed), 0);
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-provider")
                .and_then(|value| value.to_str().ok()),
            Some("anthropic")
        );

        let anthropic_requests = anthropic_requests.lock().await;
        assert_eq!(anthropic_requests.len(), 1);
        assert_eq!(
            anthropic_requests[0]["tools"][0]["cache_control"]["type"].as_str(),
            Some("ephemeral")
        );
    }

    #[tokio::test]
    async fn prompt_cache_request_rejects_when_no_provider_supports_it() {
        let provider_hits = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_hits = Arc::clone(&provider_hits);
            move |_req: Request<Incoming>| {
                let provider_hits = Arc::clone(&provider_hits);
                async move {
                    provider_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from("{}")))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![generic_provider(format!("http://{}", provider_addr))];
        let (plugins, api) = setup_gateway(prompt_cache_config(), &providers).await;
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-cache-unsupported"),
                "cache-key",
                "generic",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![format!("http://{}", provider_addr)]);
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
                    "messages": [{"role": "user", "content": "cache this prompt"}],
                    "trp_prompt_cache": {
                        "enabled": true
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("response json");
        assert_eq!(
            body_json["error"]["code"].as_str(),
            Some("provider_prompt_cache_unsupported")
        );
        assert!(body_json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("supports gateway prompt cache controls"));
        assert_eq!(provider_hits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn explicit_prompt_cache_capabilities_enable_generic_provider_controls() {
        let provider_requests = Arc::new(Mutex::new(Vec::new()));
        let provider_addr = start_upstream_async({
            let provider_requests = Arc::clone(&provider_requests);
            move |req: Request<Incoming>| {
                let provider_requests = Arc::clone(&provider_requests);
                async move {
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect provider body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("provider request json");
                    provider_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-generic-cache-1",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "generic cache provider"
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 24,
                                    "completion_tokens": 7,
                                    "total_tokens": 31,
                                    "prompt_tokens_details": {
                                        "cached_tokens": 48,
                                        "cache_write_tokens": 12
                                    }
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![generic_prompt_cache_provider(format!(
            "http://{}",
            provider_addr
        ))];
        let (plugins, api) = setup_gateway(prompt_cache_config(), &providers).await;
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-cache-explicit"),
                "cache-key",
                "generic-cache",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![format!("http://{}", provider_addr)]);
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
                    "messages": [{"role": "user", "content": "cache this generic provider"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:explicit"
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-protocol")
                .and_then(|value| value.to_str().ok()),
            Some("openai")
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-prompt-cache-status")
                .and_then(|value| value.to_str().ok()),
            Some("hit_write")
        );

        let provider_requests = provider_requests.lock().await;
        assert_eq!(provider_requests.len(), 1);
        assert!(provider_requests[0].get("trp_prompt_cache").is_none());
        assert_eq!(
            provider_requests[0]["prompt_cache_key"].as_str(),
            Some("tenant:explicit")
        );
        assert_eq!(
            provider_requests[0]["prompt_cache_retention"].as_str(),
            Some("24h")
        );
    }

    #[tokio::test]
    async fn warmed_prompt_cache_route_survives_restart() {
        let temp_db = NamedTempFile::new().unwrap();
        let store_url = format!("sqlite://{}", temp_db.path().display());
        let alpha_requests = Arc::new(Mutex::new(Vec::new()));
        let alpha_addr = start_upstream_async({
            let alpha_requests = Arc::clone(&alpha_requests);
            move |req: Request<Incoming>| {
                let alpha_requests = Arc::clone(&alpha_requests);
                async move {
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect alpha body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("alpha request json");
                    alpha_requests.lock().await.push(body_json);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-alpha-restart",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "alpha kept the warm cache"
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 18,
                                    "completion_tokens": 4,
                                    "total_tokens": 22,
                                    "prompt_tokens_details": {
                                        "cached_tokens": 96,
                                        "cache_write_tokens": 24
                                    }
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let beta_hits = Arc::new(AtomicUsize::new(0));
        let beta_addr = start_upstream_async({
            let beta_hits = Arc::clone(&beta_hits);
            move |_req: Request<Incoming>| {
                let beta_hits = Arc::clone(&beta_hits);
                async move {
                    beta_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-beta-restart",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "beta should not be chosen after restart"
                                    },
                                    "finish_reason": "stop"
                                }]
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![
            openai_prompt_cache_provider(format!("http://{}", alpha_addr), true),
            openai_prompt_cache_provider(format!("http://{}", beta_addr), true),
        ];
        let (plugins_first, api_first) = setup_gateway_with_store(
            prompt_cache_config_with_options(None, Some(true), Some(20), Some(1)),
            &providers,
            &store_url,
        )
        .await;
        let project_id = "project-cache-restart";
        api_first
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: project_id.to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["alpha","beta"]"#.to_string()),
                adaptive_enabled: false,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: None,
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "1".to_string(),
            })
            .await
            .expect("project policy API enabled")
            .expect("store project policy");
        let (plaintext_key, _) = api_first
            .create_virtual_key(
                Some(project_id),
                "cache-key",
                "alpha",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![
            format!("http://{}", alpha_addr),
            format!("http://{}", beta_addr),
        ]);
        let proxy_first = start_proxy_with_config(
            router.clone(),
            TestProxyConfig {
                plugins: Some(plugins_first),
                ..Default::default()
            },
        )
        .await;

        let first_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "warm alpha's prompt cache"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:restart"
                    }
                })
                .to_string(),
            )))
            .unwrap();
        let first_resp = send_request(&proxy_first, first_req).await;
        assert_eq!(first_resp.status(), StatusCode::OK);
        let status_first = wait_for_prompt_cache_status(&api_first, |status| {
            status.pending_route_updates == 0 && status.last_route_flush_unix_ms.is_some()
        })
        .await;
        assert!(status_first.store_backed);
        assert!(status_first.routing_hint_persistence_enabled);
        assert_eq!(status_first.routing_flush_interval_ms, 20);
        assert_eq!(status_first.routing_prune_interval_secs, 1);
        assert_eq!(status_first.warmed_route_count, 1);
        assert_eq!(status_first.negative_route_count, 0);
        assert_eq!(status_first.pending_route_updates, 0);

        api_first
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: project_id.to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["beta","alpha"]"#.to_string()),
                adaptive_enabled: false,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: None,
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "2".to_string(),
            })
            .await
            .expect("project policy API enabled")
            .expect("update project policy");

        let (plugins_second, api_second) = setup_gateway_with_store(
            prompt_cache_config_with_options(None, Some(true), Some(20), Some(1)),
            &providers,
            &store_url,
        )
        .await;
        let proxy_second = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(plugins_second),
                ..Default::default()
            },
        )
        .await;

        let second_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("x-trp-routing-debug", "1")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "use the warm cache after restart"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:restart"
                    }
                })
                .to_string(),
            )))
            .unwrap();
        let second_resp = send_request(&proxy_second, second_req).await;
        assert_eq!(second_resp.status(), StatusCode::OK);
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-provider-selected")
                .and_then(|value| value.to_str().ok()),
            Some("alpha")
        );
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-prompt-cache-route")
                .and_then(|value| value.to_str().ok()),
            Some("alpha")
        );
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-prompt-cache-affinity")
                .and_then(|value| value.to_str().ok()),
            Some("applied")
        );
        assert_eq!(beta_hits.load(Ordering::Relaxed), 0);
        assert_eq!(alpha_requests.lock().await.len(), 2);
        let status_second = api_second
            .prompt_cache_status()
            .expect("prompt cache enabled");
        assert!(status_second.store_backed);
        assert!(status_second.routing_hint_persistence_enabled);
        assert_eq!(status_second.warmed_route_count, 1);
        assert_eq!(status_second.negative_route_count, 0);
        assert_eq!(status_second.pending_route_updates, 2);
        assert!(status_second.last_route_flush_unix_ms.is_none());
    }

    #[tokio::test]
    async fn warmed_prompt_cache_route_does_not_survive_restart_when_persistence_disabled() {
        let temp_db = NamedTempFile::new().unwrap();
        let store_url = format!("sqlite://{}", temp_db.path().display());
        let alpha_addr = start_upstream_async(move |_req: Request<Incoming>| async move {
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "id": "chatcmpl-alpha-no-persist",
                        "object": "chat.completion",
                        "model": "gpt-4o",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "alpha warmed in-memory only"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 18,
                            "completion_tokens": 4,
                            "total_tokens": 22,
                            "prompt_tokens_details": {
                                "cached_tokens": 96,
                                "cache_write_tokens": 24
                            }
                        }
                    })
                    .to_string(),
                )))
                .unwrap()
        })
        .await;

        let beta_hits = Arc::new(AtomicUsize::new(0));
        let beta_addr = start_upstream_async({
            let beta_hits = Arc::clone(&beta_hits);
            move |_req: Request<Incoming>| {
                let beta_hits = Arc::clone(&beta_hits);
                async move {
                    beta_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-beta-no-persist",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "beta won after restart because no route was persisted"
                                    },
                                    "finish_reason": "stop"
                                }]
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let providers = vec![
            openai_prompt_cache_provider(format!("http://{}", alpha_addr), true),
            openai_prompt_cache_provider(format!("http://{}", beta_addr), true),
        ];
        let (plugins_first, api_first) = setup_gateway_with_store(
            prompt_cache_config_with_options(None, Some(false), Some(20), Some(1)),
            &providers,
            &store_url,
        )
        .await;
        let project_id = "project-cache-no-persist";
        api_first
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: project_id.to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["alpha","beta"]"#.to_string()),
                adaptive_enabled: false,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: None,
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "1".to_string(),
            })
            .await
            .expect("project policy API enabled")
            .expect("store project policy");
        let (plaintext_key, _) = api_first
            .create_virtual_key(
                Some(project_id),
                "cache-key",
                "alpha",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("virtual keys enabled")
            .expect("create virtual key");

        let router = catch_all_router(vec![
            format!("http://{}", alpha_addr),
            format!("http://{}", beta_addr),
        ]);
        let proxy_first = start_proxy_with_config(
            router.clone(),
            TestProxyConfig {
                plugins: Some(plugins_first),
                ..Default::default()
            },
        )
        .await;

        let first_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "warm alpha's prompt cache without persistence"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:no-persist"
                    }
                })
                .to_string(),
            )))
            .unwrap();
        let first_resp = send_request(&proxy_first, first_req).await;
        assert_eq!(first_resp.status(), StatusCode::OK);
        let status_first = api_first
            .prompt_cache_status()
            .expect("prompt cache enabled");
        assert!(!status_first.store_backed);
        assert!(!status_first.routing_hint_persistence_enabled);
        assert_eq!(status_first.warmed_route_count, 1);
        assert_eq!(status_first.pending_route_updates, 0);

        api_first
            .upsert_project_policy(ProjectPolicyRecord {
                project_id: project_id.to_string(),
                budget_limit: None,
                budget_duration: None,
                rpm_limit: None,
                tpm_limit: None,
                fallback_order: Some(r#"["beta","alpha"]"#.to_string()),
                adaptive_enabled: false,
                timeout_secs: None,
                provider_rpm_limits: None,
                provider_tpm_limits: None,
                provider_timeouts: None,
                provider_input_costs: None,
                provider_output_costs: None,
                semantic_cache_enabled: None,
                semantic_cache_ttl_secs: None,
                semantic_cache_similarity_threshold: None,
                tool_approval_mode: None,
                allowed_tools: None,
                updated_at: "2".to_string(),
            })
            .await
            .expect("project policy API enabled")
            .expect("update project policy");

        let (plugins_second, _api_second) = setup_gateway_with_store(
            prompt_cache_config_with_options(None, Some(false), Some(20), Some(1)),
            &providers,
            &store_url,
        )
        .await;
        let proxy_second = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(plugins_second),
                ..Default::default()
            },
        )
        .await;

        let second_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("x-trp-routing-debug", "1")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "use the warm cache after restart"}],
                    "trp_prompt_cache": {
                        "enabled": true,
                        "ttl": "24h",
                        "key": "tenant:no-persist"
                    }
                })
                .to_string(),
            )))
            .unwrap();
        let second_resp = send_request(&proxy_second, second_req).await;
        assert_eq!(second_resp.status(), StatusCode::OK);
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-provider-selected")
                .and_then(|value| value.to_str().ok()),
            Some("beta")
        );
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-prompt-cache-route")
                .and_then(|value| value.to_str().ok()),
            None
        );
        assert_eq!(beta_hits.load(Ordering::Relaxed), 1);
    }
}
