#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::{Request, Response, StatusCode};
    use tokio::sync::Mutex;
    use tokio::time::sleep;

    use plugin_llm_gateway::store::{
        ProjectPromptRecord, ProjectPromptRolloutRecord, RequestLogEntry,
    };
    use plugin_llm_gateway::CreatePluginsOptions;
    use proxy_core::config::{
        PluginConfig, ProviderFamilyConfig, ProviderKeyConfig, ProviderSurfaceCatalog,
    };
    use proxy_core::plugin::PluginChain;

    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream_async,
        TestProxyConfig,
    };

    fn prompt_registry_config() -> PluginConfig {
        PluginConfig {
            name: "prompt_registry".into(),
            enabled: true,
            config: toml::Value::Table(toml::value::Map::new()),
        }
    }

    fn cost_tracker_config() -> PluginConfig {
        PluginConfig {
            name: "cost_tracker".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(100.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t.insert("default_cost_per_1k_input".into(), toml::Value::Float(0.01));
                t.insert(
                    "default_cost_per_1k_output".into(),
                    toml::Value::Float(0.02),
                );
                t
            }),
        }
    }

    fn plugin_configs() -> Vec<PluginConfig> {
        vec![prompt_registry_config(), cost_tracker_config()]
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

    async fn setup_gateway(
        configs: Vec<PluginConfig>,
        providers: &[ProviderKeyConfig],
    ) -> (Arc<PluginChain>, plugin_llm_gateway::api::LlmGatewayApi) {
        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            providers,
            &[],
            CreatePluginsOptions::default(),
            None,
        )
        .await
        .expect("create plugins");
        (Arc::new(PluginChain::new(plugins)), api)
    }

    async fn wait_for_project_logs(
        api: &plugin_llm_gateway::api::LlmGatewayApi,
        project_id: &str,
    ) -> Vec<RequestLogEntry> {
        for _ in 0..20 {
            let logs = api
                .get_request_logs(None, None, Some(project_id), 10)
                .await
                .expect("store enabled")
                .expect("request logs query");
            if !logs.is_empty() {
                return logs;
            }
            sleep(Duration::from_millis(50)).await;
        }
        api.get_request_logs(None, None, Some(project_id), 10)
            .await
            .expect("store enabled")
            .expect("request logs query")
    }

    #[tokio::test]
    async fn openai_prompt_ref_injects_system_message_and_persists_audit() {
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
                                "id": "chatcmpl-prompt-1",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Resolved prompt"
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 24,
                                    "completion_tokens": 6,
                                    "total_tokens": 30
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
        let (plugins, api) = setup_gateway(plugin_configs(), &providers).await;
        api.upsert_project_prompt(ProjectPromptRecord {
            project_id: "project-prompts".to_string(),
            prompt_name: "support".to_string(),
            version: "v1".to_string(),
            environment: "prod".to_string(),
            description: Some("Support system prompt".to_string()),
            target: "system".to_string(),
            template_text: "You are helping {{customer}} with {{topic}}.".to_string(),
            variables_schema_json: Some(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "customer": { "type": "string" },
                        "topic": { "type": "string" }
                    },
                    "required": ["customer", "topic"]
                })
                .to_string(),
            ),
            rollout_metadata_json: None,
            active: true,
            updated_at: "0".to_string(),
        })
        .await
        .expect("governance enabled")
        .expect("upsert prompt");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-prompts"),
                "prompt-key",
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
                    "messages": [{"role": "user", "content": "Need help"}],
                    "trp_prompt_ref": {
                        "name": "support",
                        "variables": {
                            "customer": "Acme",
                            "topic": "billing"
                        }
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let body_json: serde_json::Value = serde_json::from_slice(&body).expect("response json");
        assert_eq!(
            body_json["choices"][0]["message"]["content"].as_str(),
            Some("Resolved prompt")
        );

        let provider_requests = provider_requests.lock().await;
        assert_eq!(provider_requests.len(), 1);
        assert!(provider_requests[0].get("trp_prompt_ref").is_none());
        assert_eq!(
            provider_requests[0]["messages"][0]["role"].as_str(),
            Some("system")
        );
        assert_eq!(
            provider_requests[0]["messages"][0]["content"].as_str(),
            Some("You are helping Acme with billing.")
        );
        assert_eq!(
            provider_requests[0]["messages"][1]["role"].as_str(),
            Some("user")
        );
        drop(provider_requests);

        let logs = wait_for_project_logs(&api, "project-prompts").await;
        assert!(!logs.is_empty());
        assert_eq!(logs[0].project_id.as_deref(), Some("project-prompts"));
        assert_eq!(logs[0].prompt_name.as_deref(), Some("support"));
        assert_eq!(logs[0].prompt_version.as_deref(), Some("v1"));
        assert_eq!(logs[0].prompt_environment.as_deref(), Some("prod"));
    }

    #[tokio::test]
    async fn openai_prompt_ref_uses_applied_canary_rollout_candidate_version() {
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
                                "id": "chatcmpl-prompt-canary",
                                "object": "chat.completion",
                                "model": "gpt-4o",
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Resolved prompt"
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 18,
                                    "completion_tokens": 5,
                                    "total_tokens": 23
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
        let (plugins, api) = setup_gateway(plugin_configs(), &providers).await;
        for (version, template_text, active) in [
            ("v1", "Use baseline playbook.", true),
            ("v2", "Use canary playbook.", false),
        ] {
            api.upsert_project_prompt(ProjectPromptRecord {
                project_id: "project-prompts-canary".to_string(),
                prompt_name: "support".to_string(),
                version: version.to_string(),
                environment: "prod".to_string(),
                description: Some(format!("Prompt {version}")),
                target: "system".to_string(),
                template_text: template_text.to_string(),
                variables_schema_json: None,
                rollout_metadata_json: None,
                active,
                updated_at: "0".to_string(),
            })
            .await
            .expect("governance enabled")
            .expect("upsert prompt");
        }
        api.upsert_project_prompt_rollout(ProjectPromptRolloutRecord {
            project_id: "project-prompts-canary".to_string(),
            prompt_name: "support".to_string(),
            rollout_id: "rollout-canary-1".to_string(),
            policy_name: "prod-strict".to_string(),
            baseline_version: Some("v1".to_string()),
            candidate_version: "v2".to_string(),
            baseline_run_id: "run-base".to_string(),
            candidate_run_id: "run-candidate".to_string(),
            target_environment: Some("prod".to_string()),
            status: "applied_canary".to_string(),
            recommendation_action: Some("canary".to_string()),
            comparison_json: serde_json::json!({
                "runtime_rollout": {
                    "mode": "canary",
                    "traffic_percent": 100
                }
            })
            .to_string(),
            created_at: "1".to_string(),
            applied_at: Some("2".to_string()),
        })
        .await
        .expect("governance enabled")
        .expect("upsert rollout");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-prompts-canary"),
                "prompt-key",
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
                    "messages": [{"role": "user", "content": "Need help"}],
                    "trp_prompt_ref": {
                        "name": "support"
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let provider_requests = provider_requests.lock().await;
        assert_eq!(provider_requests.len(), 1);
        assert_eq!(
            provider_requests[0]["messages"][0]["content"].as_str(),
            Some("Use canary playbook.")
        );
        drop(provider_requests);

        let logs = wait_for_project_logs(&api, "project-prompts-canary").await;
        assert!(!logs.is_empty());
        assert_eq!(logs[0].prompt_version.as_deref(), Some("v2"));
    }

    #[tokio::test]
    async fn anthropic_prompt_ref_supports_explicit_version_and_merges_system() {
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
                                "id": "msg-prompt-1",
                                "type": "message",
                                "role": "assistant",
                                "model": "claude-sonnet-4-20250514",
                                "content": [{
                                    "type": "text",
                                    "text": "Anthropic prompt resolved"
                                }],
                                "usage": {
                                    "input_tokens": 11,
                                    "output_tokens": 4
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
        let (plugins, api) = setup_gateway(plugin_configs(), &providers).await;
        api.upsert_project_prompt(ProjectPromptRecord {
            project_id: "project-prompts-anthropic".to_string(),
            prompt_name: "policy".to_string(),
            version: "v1".to_string(),
            environment: "prod".to_string(),
            description: Some("Primary policy prompt".to_string()),
            target: "system".to_string(),
            template_text: "Use standard policy for {{customer}}.".to_string(),
            variables_schema_json: None,
            rollout_metadata_json: None,
            active: true,
            updated_at: "0".to_string(),
        })
        .await
        .expect("governance enabled")
        .expect("upsert prompt v1");
        api.upsert_project_prompt(ProjectPromptRecord {
            project_id: "project-prompts-anthropic".to_string(),
            prompt_name: "policy".to_string(),
            version: "v2".to_string(),
            environment: "prod".to_string(),
            description: Some("Premium policy prompt".to_string()),
            target: "system".to_string(),
            template_text: "Use premium policy for {{customer}}.".to_string(),
            variables_schema_json: None,
            rollout_metadata_json: None,
            active: false,
            updated_at: "1".to_string(),
        })
        .await
        .expect("governance enabled")
        .expect("upsert prompt v2");
        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-prompts-anthropic"),
                "prompt-key",
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
            .header("x-api-key", plaintext_key)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(
                serde_json::json!({
                    "model": "claude-sonnet-4-20250514",
                    "system": "Existing guardrails",
                    "messages": [{"role": "user", "content": "Need premium help"}],
                    "trp_prompt_ref": {
                        "name": "policy",
                        "version": "v2",
                        "variables": {
                            "customer": "Delta"
                        }
                    }
                })
                .to_string(),
            )))
            .unwrap();

        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let provider_requests = provider_requests.lock().await;
        assert_eq!(provider_requests.len(), 1);
        assert!(provider_requests[0].get("trp_prompt_ref").is_none());
        assert_eq!(
            provider_requests[0]["system"].as_str(),
            Some("Use premium policy for Delta.\n\nExisting guardrails")
        );
    }
}
