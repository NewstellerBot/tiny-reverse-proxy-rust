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

    use plugin_llm_gateway::store::ProjectPolicyRecord;
    use plugin_llm_gateway::CreatePluginsOptions;
    use proxy_core::config::{
        PluginConfig, ProviderFamilyConfig, ProviderKeyConfig, ProviderSurfaceCatalog,
    };
    use proxy_core::plugin::PluginChain;

    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream_async,
        TestProxyConfig,
    };

    fn semantic_cache_config() -> Vec<PluginConfig> {
        vec![PluginConfig {
            name: "semantic_cache".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("default_ttl_secs".into(), toml::Value::Integer(600));
                t.insert(
                    "default_similarity_threshold".into(),
                    toml::Value::Float(0.8),
                );
                t.insert("max_entries".into(), toml::Value::Integer(32));
                t
            }),
        }]
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

    #[tokio::test]
    async fn openai_semantic_cache_reuses_similar_request_without_second_upstream_call() {
        let provider_requests = Arc::new(Mutex::new(Vec::new()));
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_requests = Arc::clone(&provider_requests);
            let provider_calls = Arc::clone(&provider_calls);
            move |req: Request<Incoming>| {
                let provider_requests = Arc::clone(&provider_requests);
                let provider_calls = Arc::clone(&provider_calls);
                async move {
                    provider_calls.fetch_add(1, Ordering::SeqCst);
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
                                "id": "chatcmpl-semantic-1",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Reset your password from the account settings page."
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 9,
                                    "total_tokens": 29
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
        let (plugins, api) = setup_gateway(semantic_cache_config(), &providers).await;
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-semantic-cache"),
                "semantic-cache-key",
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

        let first_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "reset password help"}],
                    "trp_semantic_cache": {
                        "enabled": true,
                        "ttl_secs": 600,
                        "similarity_threshold": 0.7
                    }
                })
                .to_string(),
            )))
            .unwrap();
        let first_resp = send_request(&proxy_addr, first_req).await;
        assert_eq!(first_resp.status(), StatusCode::OK);
        assert_eq!(
            first_resp
                .headers()
                .get("x-trp-semantic-cache")
                .and_then(|value| value.to_str().ok()),
            Some("miss")
        );
        let first_body = first_resp.into_body().collect().await.unwrap().to_bytes();
        let first_json: serde_json::Value = serde_json::from_slice(&first_body).unwrap();
        assert_eq!(
            first_json["choices"][0]["message"]["content"].as_str(),
            Some("Reset your password from the account settings page.")
        );

        let second_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "need password reset help"}],
                    "trp_semantic_cache": {
                        "enabled": true,
                        "ttl_secs": 600,
                        "similarity_threshold": 0.7
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
                .get("x-trp-semantic-cache")
                .and_then(|value| value.to_str().ok()),
            Some("hit")
        );
        let similarity = second_resp
            .headers()
            .get("x-trp-semantic-cache-similarity")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap();
        assert!(similarity >= 0.7, "unexpected similarity: {similarity}");
        let second_body = second_resp.into_body().collect().await.unwrap().to_bytes();
        let second_json: serde_json::Value = serde_json::from_slice(&second_body).unwrap();
        assert_eq!(
            second_json["choices"][0]["message"]["content"].as_str(),
            Some("Reset your password from the account settings page.")
        );

        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        let provider_requests = provider_requests.lock().await;
        assert_eq!(provider_requests.len(), 1);
        assert!(provider_requests[0].get("trp_semantic_cache").is_none());
        assert_eq!(
            provider_requests[0]["messages"][0]["content"].as_str(),
            Some("reset password help")
        );

        let status = api.semantic_cache_status().expect("semantic cache enabled");
        assert!(status.store_backed);
        assert_eq!(status.entry_count, 1);
        assert_eq!(status.hits, 1);
        assert_eq!(status.misses, 1);
        assert_eq!(status.stores, 1);
        assert!(status.saved_prompt_tokens > 0);
    }

    #[tokio::test]
    async fn semantic_cache_locality_can_override_project_fallback_order() {
        let alpha_calls = Arc::new(AtomicUsize::new(0));
        let alpha_addr = start_upstream_async({
            let alpha_calls = Arc::clone(&alpha_calls);
            move |_req: Request<Incoming>| {
                let alpha_calls = Arc::clone(&alpha_calls);
                async move {
                    alpha_calls.fetch_add(1, Ordering::SeqCst);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-semantic-alpha",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Reset your password from the account settings page."
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 9,
                                    "total_tokens": 29
                                }
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let beta_calls = Arc::new(AtomicUsize::new(0));
        let beta_addr = start_upstream_async({
            let beta_calls = Arc::clone(&beta_calls);
            move |_req: Request<Incoming>| {
                let beta_calls = Arc::clone(&beta_calls);
                async move {
                    beta_calls.fetch_add(1, Ordering::SeqCst);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-semantic-beta",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Beta should not answer once alpha has the semantic cache entry."
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 9,
                                    "total_tokens": 29
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
            openai_provider(format!("http://{}", alpha_addr)),
            openai_provider(format!("http://{}", beta_addr)),
        ];
        let (plugins, api) = setup_gateway(semantic_cache_config(), &providers).await;
        let project_id = "project-semantic-routing";
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
            semantic_cache_enabled: Some(true),
            semantic_cache_ttl_secs: Some(600),
            semantic_cache_similarity_threshold: Some(0.7),
            tool_approval_mode: None,
            allowed_tools: None,
            updated_at: "1".to_string(),
        })
        .await
        .expect("governance enabled")
        .expect("store project policy");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some(project_id),
                "semantic-routing-key",
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

        let request_body = |prompt: &str| {
            serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "trp_semantic_cache": {
                    "enabled": true,
                    "ttl_secs": 600,
                    "similarity_threshold": 0.7
                }
            })
            .to_string()
        };

        let first_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(request_body("reset password help"))))
            .unwrap();
        let first_resp = send_request(&proxy_addr, first_req).await;
        assert_eq!(first_resp.status(), StatusCode::OK);
        assert_eq!(
            first_resp
                .headers()
                .get("x-trp-semantic-cache")
                .and_then(|value| value.to_str().ok()),
            Some("miss")
        );

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
            semantic_cache_enabled: Some(true),
            semantic_cache_ttl_secs: Some(600),
            semantic_cache_similarity_threshold: Some(0.7),
            tool_approval_mode: None,
            allowed_tools: None,
            updated_at: "2".to_string(),
        })
        .await
        .expect("governance enabled")
        .expect("update project policy");

        let second_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("x-trp-routing-debug", "1")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(request_body(
                "need password reset help",
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
                .get("x-trp-semantic-cache-route")
                .and_then(|value| value.to_str().ok()),
            Some("alpha")
        );
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-provider-order")
                .and_then(|value| value.to_str().ok()),
            Some("alpha,beta")
        );
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-semantic-cache")
                .and_then(|value| value.to_str().ok()),
            Some("hit")
        );

        assert_eq!(alpha_calls.load(Ordering::SeqCst), 1);
        assert_eq!(beta_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn project_policy_can_enable_semantic_cache_by_default() {
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_calls = Arc::clone(&provider_calls);
            move |_req: Request<Incoming>| {
                let provider_calls = Arc::clone(&provider_calls);
                async move {
                    provider_calls.fetch_add(1, Ordering::SeqCst);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-semantic-default",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "You can update billing details from the billing tab."
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 12,
                                    "completion_tokens": 8,
                                    "total_tokens": 20
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
        let (plugins, api) = setup_gateway(semantic_cache_config(), &providers).await;
        api.upsert_project_policy(ProjectPolicyRecord {
            project_id: "project-semantic-default".to_string(),
            budget_limit: None,
            budget_duration: None,
            rpm_limit: None,
            tpm_limit: None,
            fallback_order: None,
            adaptive_enabled: true,
            timeout_secs: None,
            provider_rpm_limits: None,
            provider_tpm_limits: None,
            provider_timeouts: None,
            provider_input_costs: None,
            provider_output_costs: None,
            semantic_cache_enabled: Some(true),
            semantic_cache_ttl_secs: Some(900),
            semantic_cache_similarity_threshold: Some(0.5),
            tool_approval_mode: None,
            allowed_tools: None,
            updated_at: "0".to_string(),
        })
        .await
        .expect("governance enabled")
        .expect("upsert project policy");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-semantic-default"),
                "semantic-default-key",
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

        let first_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "change billing details help"}]
                })
                .to_string(),
            )))
            .unwrap();
        let first_resp = send_request(&proxy_addr, first_req).await;
        assert_eq!(
            first_resp
                .headers()
                .get("x-trp-semantic-cache")
                .and_then(|value| value.to_str().ok()),
            Some("miss")
        );

        let second_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "billing details update help"}]
                })
                .to_string(),
            )))
            .unwrap();
        let second_resp = send_request(&proxy_addr, second_req).await;
        assert_eq!(
            second_resp
                .headers()
                .get("x-trp-semantic-cache")
                .and_then(|value| value.to_str().ok()),
            Some("hit")
        );
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn project_policy_can_disable_semantic_cache_even_with_request_opt_in() {
        let provider_requests = Arc::new(Mutex::new(Vec::new()));
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_requests = Arc::clone(&provider_requests);
            let provider_calls = Arc::clone(&provider_calls);
            move |req: Request<Incoming>| {
                let provider_requests = Arc::clone(&provider_requests);
                let provider_calls = Arc::clone(&provider_calls);
                async move {
                    provider_calls.fetch_add(1, Ordering::SeqCst);
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
                                "id": "chatcmpl-semantic-disabled",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Live upstream answer."
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 5,
                                    "total_tokens": 15
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
        let (plugins, api) = setup_gateway(semantic_cache_config(), &providers).await;
        api.upsert_project_policy(ProjectPolicyRecord {
            project_id: "project-semantic-disabled".to_string(),
            budget_limit: None,
            budget_duration: None,
            rpm_limit: None,
            tpm_limit: None,
            fallback_order: None,
            adaptive_enabled: true,
            timeout_secs: None,
            provider_rpm_limits: None,
            provider_tpm_limits: None,
            provider_timeouts: None,
            provider_input_costs: None,
            provider_output_costs: None,
            semantic_cache_enabled: Some(false),
            semantic_cache_ttl_secs: Some(600),
            semantic_cache_similarity_threshold: Some(0.7),
            tool_approval_mode: None,
            allowed_tools: None,
            updated_at: "0".to_string(),
        })
        .await
        .expect("governance enabled")
        .expect("upsert project policy");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-semantic-disabled"),
                "semantic-disabled-key",
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

        for prompt in ["reset password help", "need password reset help"] {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {}", plaintext_key))
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(
                    serde_json::json!({
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": prompt}],
                        "trp_semantic_cache": {
                            "enabled": true,
                            "ttl_secs": 600,
                            "similarity_threshold": 0.7
                        }
                    })
                    .to_string(),
                )))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(resp.headers().get("x-trp-semantic-cache").is_none());
        }

        assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
        let provider_requests = provider_requests.lock().await;
        assert!(provider_requests
            .iter()
            .all(|request| request.get("trp_semantic_cache").is_none()));
        let status = api.semantic_cache_status().expect("semantic cache enabled");
        assert_eq!(status.hits, 0);
        assert_eq!(status.misses, 0);
        assert_eq!(status.skips, 2);
    }

    #[tokio::test]
    async fn semantic_cache_survives_restart_with_store_backing() {
        let temp_db = NamedTempFile::new().unwrap();
        let store_url = format!("sqlite://{}", temp_db.path().display());
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let provider_addr = start_upstream_async({
            let provider_calls = Arc::clone(&provider_calls);
            move |_req: Request<Incoming>| {
                let provider_calls = Arc::clone(&provider_calls);
                async move {
                    provider_calls.fetch_add(1, Ordering::SeqCst);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "id": "chatcmpl-semantic-restart",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Reset your password from the account settings page."
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 9,
                                    "total_tokens": 29
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
        let (plugins_first, api_first) =
            setup_gateway_with_store(semantic_cache_config(), &providers, &store_url).await;
        let (plaintext_key, _) = api_first
            .create_virtual_key(
                Some("project-semantic-restart"),
                "semantic-restart-key",
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
                    "messages": [{"role": "user", "content": "reset password help"}],
                    "trp_semantic_cache": {
                        "enabled": true,
                        "ttl_secs": 600,
                        "similarity_threshold": 0.7
                    }
                })
                .to_string(),
            )))
            .unwrap();
        let first_resp = send_request(&proxy_first, first_req).await;
        assert_eq!(
            first_resp
                .headers()
                .get("x-trp-semantic-cache")
                .and_then(|value| value.to_str().ok()),
            Some("miss")
        );
        let status_first = api_first
            .semantic_cache_status()
            .expect("semantic cache enabled");
        assert!(status_first.store_backed);
        assert_eq!(status_first.entry_count, 1);

        let (plugins_second, api_second) =
            setup_gateway_with_store(semantic_cache_config(), &providers, &store_url).await;
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
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "need password reset help"}],
                    "trp_semantic_cache": {
                        "enabled": true,
                        "ttl_secs": 600,
                        "similarity_threshold": 0.7
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
                .get("x-trp-semantic-cache")
                .and_then(|value| value.to_str().ok()),
            Some("hit")
        );
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        let status_second = api_second
            .semantic_cache_status()
            .expect("semantic cache enabled");
        assert!(status_second.store_backed);
        assert_eq!(status_second.entry_count, 1);
        assert_eq!(status_second.hits, 1);
    }
}
