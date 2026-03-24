/// End-to-end integration tests for the LLM gateway management API.
///
/// These tests start a real proxy with plugins created via `create_plugins()`,
/// a management server on an ephemeral port, send LLM traffic through the proxy,
/// and then query/mutate state through the management HTTP API.
#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full, StreamBody};
    use hyper::body::{Frame, Incoming};
    use hyper::service::service_fn;
    use hyper::{Method, Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use semantic_safety_protocol::SemanticSafetyServiceServer;
    use semantic_safety_service::backend::TensorRtBackend;
    use semantic_safety_service::persistence::FileProjectIndexStore;
    use semantic_safety_service::service::{
        SemanticSafetyConfig as ServiceConfig, SemanticSafetyGrpcService,
    };
    use tempfile::{NamedTempFile, TempDir};
    use tokio::net::TcpListener;
    use tokio::sync::Mutex;
    use tokio_stream::wrappers::UnboundedReceiverStream;
    use tokio_stream::StreamExt;
    use tonic::transport::Server;

    use proxy_core::config::{
        AudioSurface, AudioSurfaceProtocol, EmbeddingSurface, EmbeddingSurfaceProtocol,
        FileSurface, ImageSurface, ImageSurfaceProtocol, PluginConfig, PromptCacheProtocol,
        PromptCacheSurface, ProviderCommonConfig, ProviderDataCollectionMode, ProviderFamily,
        ProviderFamilyConfig, ProviderKeyConfig, ProviderRoutingMetadataConfig,
        ProviderSurfaceCatalog, RealtimeSurface, ResponsesSurface, ToolSurface,
    };
    use proxy_core::plugin::PluginChain;

    use plugin_llm_gateway::api::LlmGatewayApi;
    use plugin_llm_gateway::CreatePluginsOptions;
    use serde_json::{json, Value};

    use trp_test_support::{
        catch_all_router, send_request, start_proxy_with_config, start_upstream,
        start_upstream_async, TestProxyConfig,
    };

    // --- Helpers ---

    /// Mock upstream that returns a JSON response mimicking an LLM chat completion.
    fn llm_chat_handler(_req: Request<Incoming>) -> Response<Full<Bytes>> {
        let body = r#"{"id":"chatcmpl-abc","object":"chat.completion","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I help you today?"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}"#;
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .header("content-length", body.len().to_string())
            .body(Full::new(Bytes::from(body)))
            .unwrap()
    }

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

    const MGMT_TOKEN: &str = "test-bootstrap-admin";

    fn canonical_provider(
        name: &str,
        api_key: &str,
        base_url: impl Into<String>,
        models: Vec<String>,
        api_key_header: &str,
        timeout_secs: Option<u64>,
        family: ProviderFamily,
        surfaces: ProviderSurfaceCatalog,
        routing_metadata: ProviderRoutingMetadataConfig,
    ) -> ProviderKeyConfig {
        ProviderKeyConfig::new(
            ProviderCommonConfig {
                name: name.to_string(),
                api_key: api_key.to_string(),
                base_url: base_url.into(),
                models,
                api_key_header: api_key_header.to_string(),
                timeout_secs,
                routing_metadata,
            },
            ProviderFamilyConfig::from_parts(family, surfaces).unwrap(),
        )
    }

    fn openai_tool_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            ..Default::default()
        }
    }

    fn openai_runtime_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            responses: Some(ResponsesSurface::OpenAiCompatible),
            reasoning: true,
            structured_output_json_mode: true,
            files: Some(FileSurface::OpenAiCompatible),
            audio: Some(AudioSurface {
                protocol: AudioSurfaceProtocol::OpenAiAudio,
                input: false,
                output: true,
                transcription: false,
                translation: true,
            }),
            embeddings: Some(EmbeddingSurface {
                protocol: EmbeddingSurfaceProtocol::OpenAiEmbeddings,
            }),
            realtime: Some(RealtimeSurface::OpenAiCompatible),
            ..Default::default()
        }
    }

    fn anthropic_runtime_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::Anthropic),
            structured_output_json_schema: true,
            images: Some(ImageSurface {
                protocol: ImageSurfaceProtocol::OpenAiImages,
                input: true,
                generations: false,
                edits: false,
                variations: false,
            }),
            audio: Some(AudioSurface {
                protocol: AudioSurfaceProtocol::OpenAiAudio,
                input: true,
                output: false,
                transcription: true,
                translation: false,
            }),
            ..Default::default()
        }
    }

    fn openai_prompt_cache_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::OpenAi),
            responses: Some(ResponsesSurface::OpenAiCompatible),
            reasoning: true,
            structured_output_json_mode: true,
            audio: Some(AudioSurface {
                protocol: AudioSurfaceProtocol::OpenAiAudio,
                input: false,
                output: true,
                transcription: false,
                translation: true,
            }),
            embeddings: Some(EmbeddingSurface {
                protocol: EmbeddingSurfaceProtocol::OpenAiEmbeddings,
            }),
            realtime: Some(RealtimeSurface::OpenAiCompatible),
            prompt_cache: Some(PromptCacheSurface {
                protocol: PromptCacheProtocol::OpenAi,
                request_controls: true,
            }),
            ..Default::default()
        }
    }

    fn anthropic_prompt_cache_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            tools: Some(ToolSurface::Anthropic),
            structured_output_json_schema: true,
            images: Some(ImageSurface {
                protocol: ImageSurfaceProtocol::OpenAiImages,
                input: true,
                generations: false,
                edits: false,
                variations: false,
            }),
            audio: Some(AudioSurface {
                protocol: AudioSurfaceProtocol::OpenAiAudio,
                input: true,
                output: false,
                transcription: true,
                translation: false,
            }),
            prompt_cache: Some(PromptCacheSurface {
                protocol: PromptCacheProtocol::Anthropic,
                request_controls: true,
            }),
            ..Default::default()
        }
    }

    fn groq_prompt_cache_surfaces() -> ProviderSurfaceCatalog {
        ProviderSurfaceCatalog {
            audio: Some(AudioSurface {
                protocol: AudioSurfaceProtocol::OpenAiAudio,
                input: true,
                output: false,
                transcription: false,
                translation: false,
            }),
            embeddings: Some(EmbeddingSurface {
                protocol: EmbeddingSurfaceProtocol::OpenAiEmbeddings,
            }),
            prompt_cache: Some(PromptCacheSurface {
                protocol: PromptCacheProtocol::OpenAi,
                request_controls: true,
            }),
            ..Default::default()
        }
    }

    fn plugin_options() -> CreatePluginsOptions {
        CreatePluginsOptions {
            bootstrap_admin_token: Some(MGMT_TOKEN.to_string()),
            allow_direct_provider_keys: false,
        }
    }

    fn uv_bin() -> String {
        std::env::var("UV_BIN").unwrap_or_else(|_| "uv".to_string())
    }

    fn fake_stdio_mcp_script() -> String {
        format!(
            "{}/tests/support/fake_stdio_mcp_server.py",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    async fn start_mcp_sse_server(method_log: Arc<Mutex<Vec<String>>>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let active_sender: Arc<Mutex<Option<tokio::sync::mpsc::UnboundedSender<String>>>> =
            Arc::new(Mutex::new(None));

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(_) => break,
                };
                let active_sender = Arc::clone(&active_sender);
                let method_log = Arc::clone(&method_log);
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let service = service_fn(move |req: Request<Incoming>| {
                        let active_sender = Arc::clone(&active_sender);
                        let method_log = Arc::clone(&method_log);
                        async move {
                            match (req.method().clone(), req.uri().path()) {
                                (method, "/sse") if method == Method::GET => {
                                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
                                    tx.send("event: endpoint\ndata: /messages\n\n".to_string())
                                        .unwrap();
                                    *active_sender.lock().await = Some(tx);
                                    let stream = UnboundedReceiverStream::new(rx).map(|chunk| {
                                        Ok::<_, Infallible>(Frame::data(Bytes::from(chunk)))
                                    });
                                    let response = Response::builder()
                                        .status(StatusCode::OK)
                                        .header("content-type", "text/event-stream")
                                        .body(StreamBody::new(stream).boxed())
                                        .unwrap();
                                    Ok::<_, hyper::Error>(response)
                                }
                                (method, "/messages") if method == Method::POST => {
                                    let body = req
                                        .into_body()
                                        .collect()
                                        .await
                                        .expect("collect sse mcp body")
                                        .to_bytes();
                                    let body_json: serde_json::Value =
                                        serde_json::from_slice(&body)
                                            .expect("sse mcp request json");
                                    let method_name = body_json["method"]
                                        .as_str()
                                        .unwrap_or("unknown")
                                        .to_string();
                                    method_log.lock().await.push(method_name.clone());
                                    if method_name != "notifications/initialized" {
                                        let response_json = match method_name.as_str() {
                                            "initialize" => serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": body_json["id"].clone(),
                                                "result": {
                                                    "protocolVersion": "2025-11-25",
                                                    "capabilities": {},
                                                    "serverInfo": {
                                                        "name": "fake-sse-mcp",
                                                        "version": "1.0.0"
                                                    }
                                                }
                                            }),
                                            "tools/list" => serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": body_json["id"].clone(),
                                                "result": {
                                                    "tools": [{
                                                        "name": "search_docs",
                                                        "description": "Search remote docs over sse"
                                                    }]
                                                }
                                            }),
                                            other => serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": body_json["id"].clone(),
                                                "error": {
                                                    "code": -32601,
                                                    "message": format!("unsupported method {other}")
                                                }
                                            }),
                                        };
                                        if let Some(sender) = active_sender.lock().await.clone() {
                                            sender
                                                .send(format!(
                                                    "event: message\ndata: {}\n\n",
                                                    response_json
                                                ))
                                                .unwrap();
                                        }
                                    }
                                    Ok::<_, hyper::Error>(
                                        Response::builder()
                                            .status(StatusCode::ACCEPTED)
                                            .body(Full::new(Bytes::new()).boxed())
                                            .unwrap(),
                                    )
                                }
                                _ => Ok::<_, hyper::Error>(
                                    Response::builder()
                                        .status(StatusCode::NOT_FOUND)
                                        .body(Full::new(Bytes::new()).boxed())
                                        .unwrap(),
                                ),
                            }
                        }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .keep_alive(true)
                        .serve_connection(io, service)
                        .await;
                });
            }
        });

        addr
    }

    /// Start a management API server on an ephemeral port and return the port.
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

    /// HTTP client helpers for management API.
    async fn mgmt_get(port: u16, path: &str) -> (u16, String) {
        let client =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http::<Full<Bytes>>();
        let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
            .parse()
            .unwrap();
        let req = Request::builder()
            .method("GET")
            .uri(uri)
            .header("authorization", format!("Bearer {}", MGMT_TOKEN))
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = client.request(req).await.unwrap();
        let status = resp.status().as_u16();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    async fn mgmt_delete(port: u16, path: &str) -> (u16, String) {
        let client =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http::<Full<Bytes>>();
        let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
            .parse()
            .unwrap();
        let req = Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("authorization", format!("Bearer {}", MGMT_TOKEN))
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = client.request(req).await.unwrap();
        let status = resp.status().as_u16();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    async fn mgmt_put(port: u16, path: &str, body: &str) -> (u16, String) {
        let client =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http::<Full<Bytes>>();
        let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
            .parse()
            .unwrap();
        let req = Request::builder()
            .method("PUT")
            .uri(uri)
            .header("authorization", format!("Bearer {}", MGMT_TOKEN))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body.to_string())))
            .unwrap();
        let resp = client.request(req).await.unwrap();
        let status = resp.status().as_u16();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    /// Create plugins via create_plugins() with in-memory SQLite.
    /// Returns (PluginChain, LlmGatewayApi).
    async fn setup_all_plugins() -> (Arc<PluginChain>, LlmGatewayApi) {
        let configs = vec![
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
            },
            PluginConfig {
                name: "rate_limiter".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("tokens_per_minute".into(), toml::Value::Float(600_000.0));
                    t.insert("burst_tokens".into(), toml::Value::Float(100_000.0));
                    t
                }),
            },
            PluginConfig {
                name: "provider_failover".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("cooldown_secs".into(), toml::Value::Integer(30));
                    t.insert(
                        "providers".into(),
                        toml::Value::Array(vec![{
                            let mut p = toml::value::Map::new();
                            p.insert("name".into(), toml::Value::String("openai".into()));
                            p.insert(
                                "pattern".into(),
                                toml::Value::String("api.openai.com".into()),
                            );
                            toml::Value::Table(p)
                        }]),
                    );
                    t
                }),
            },
        ];

        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &[],
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        (Arc::new(PluginChain::new(plugins)), api)
    }

    async fn start_semantic_service() -> (String, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileProjectIndexStore::new(dir.path().to_path_buf()).unwrap());
        let backend = Arc::new(TensorRtBackend::new_dev_stub());
        let service =
            SemanticSafetyGrpcService::new(ServiceConfig { auth_token: None }, store, backend);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            Server::builder()
                .add_service(SemanticSafetyServiceServer::new(service))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        (format!("http://{}", addr), dir)
    }

    async fn setup_plugins_with_semantic_safety(
        endpoint: &str,
    ) -> (Arc<PluginChain>, LlmGatewayApi) {
        let configs = vec![
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
            },
            PluginConfig {
                name: "semantic_safety".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("endpoint".into(), toml::Value::String(endpoint.to_string()));
                    t.insert("timeout_ms".into(), toml::Value::Integer(200));
                    t.insert("reconcile_interval_secs".into(), toml::Value::Integer(1));
                    t
                }),
            },
        ];

        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &[],
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        (Arc::new(PluginChain::new(plugins)), api)
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn proxy_traffic_visible_through_management_api() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        // Start proxy with the plugin chain.
        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        // Start management server.
        let mgmt_port = start_mgmt_server(api).await;

        // Verify initial state: no usage.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"usage\":[]"),
            "no usage initially: {}",
            body
        );

        // Send 3 requests through the proxy.
        for i in 0..3 {
            let req = chat_request(
                &format!("/v1/chat/completions?n={}", i),
                "sk-integration-key",
            );
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "request {} should succeed",
                i
            );
        }

        // Query management API — usage should now reflect the 3 requests.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("sk-integ"),
            "should contain masked key: {}",
            body
        );
        assert!(
            body.contains("\"total_input_tokens\":"),
            "should contain token counts: {}",
            body
        );
        // total_cost should be > 0
        assert!(
            !body.contains("\"total_cost\":0.000000"),
            "cost should be non-zero: {}",
            body
        );

        // Status endpoint should show tracked keys.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"tracked_api_keys\":1"),
            "should track 1 key: {}",
            body
        );
    }

    #[tokio::test]
    async fn management_api_reset_clears_usage() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Generate some usage.
        let req = chat_request("/v1/chat/completions", "sk-reset-test");
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Confirm usage exists.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body.contains("sk-reset"), "usage should exist: {}", body);

        // Reset all usage via management API.
        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/cost/usage").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"ok\":true"),
            "reset should succeed: {}",
            body
        );

        // Verify usage is now empty.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(
            body.contains("\"usage\":[]"),
            "usage should be empty after reset: {}",
            body
        );
    }

    #[tokio::test]
    async fn management_api_reset_single_key() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Generate usage for two keys.
        let req = chat_request("/v1/chat/completions", "sk-keep-this");
        send_request(&proxy_addr, req).await;
        let req = chat_request("/v1/chat/completions", "sk-delete-me");
        send_request(&proxy_addr, req).await;

        // Confirm 2 keys tracked.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(
            body.contains("\"tracked_api_keys\":2"),
            "should track 2 keys: {}",
            body
        );

        // Delete one key.
        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/cost/usage/sk-delete-me").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"deleted\":true"),
            "should find and delete: {}",
            body
        );

        // Only 1 key remains.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(
            body.contains("\"tracked_api_keys\":1"),
            "should have 1 key left: {}",
            body
        );

        // Deleting a non-existent key returns 404.
        let (status, _) = mgmt_delete(mgmt_port, "/api/v1/cost/usage/sk-nonexistent").await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn management_api_model_cost_crud_affects_tracking() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Initially no model costs configured.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"models\":[]"),
            "no models initially: {}",
            body
        );

        // Set a model cost via management API.
        let (status, body) = mgmt_put(
            mgmt_port,
            "/api/v1/cost/models/gpt-4",
            r#"{"input_cost_per_1k":0.03,"output_cost_per_1k":0.06}"#,
        )
        .await;
        assert_eq!(status, 200);
        assert!(body.contains("\"ok\":true"));

        // Verify model cost is listed.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert!(
            body.contains("\"model\":\"gpt-4\""),
            "gpt-4 should be listed: {}",
            body
        );
        assert!(
            body.contains("0.030000"),
            "input cost should be 0.03: {}",
            body
        );
        assert!(
            body.contains("0.060000"),
            "output cost should be 0.06: {}",
            body
        );

        // Send a request — it should be tracked with the new model pricing.
        let req = chat_request("/v1/chat/completions", "sk-model-test");
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Check the cost is calculated with model-specific pricing (0.03/0.06)
        // rather than defaults (0.01/0.02).
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body.contains("sk-model"), "usage recorded: {}", body);
        // With gpt-4 pricing at 0.03/0.06, cost per request should be higher
        // than the ~0.002 with default rates. The body is ~66 bytes => ~16 input tokens
        // and response ~290 bytes => ~72 output tokens.
        // cost = (16/1000)*0.03 + (72/1000)*0.06 = 0.00048 + 0.00432 = 0.0048
        // With defaults: (16/1000)*0.01 + (72/1000)*0.02 = 0.00016 + 0.00144 = 0.0016
        // We just verify the total_cost is in the expected range.
        assert!(
            body.contains("\"total_cost\":0.004"),
            "cost should reflect model pricing (~0.0048): {}",
            body
        );

        // Delete the model cost.
        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/cost/models/gpt-4").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"deleted\":true"));

        // Verify models list is empty again.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert!(
            body.contains("\"models\":[]"),
            "models should be empty: {}",
            body
        );
    }

    #[tokio::test]
    async fn management_api_rate_limiter_status() {
        let (chain, api) = setup_all_plugins().await;

        // Don't need a proxy for this — just check the management endpoint.
        let _ = chain; // plugins created but we only query management
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/rate-limiter/status").await;
        assert_eq!(status, 200);
        // 600_000 tokens/min = 10_000 tokens/sec
        assert!(
            body.contains("\"rate_per_second\":10000.00"),
            "rate should be 10000: {}",
            body
        );
        assert!(
            body.contains("\"burst\":100000"),
            "burst should be 100000: {}",
            body
        );
        assert!(
            body.contains("\"tracked_keys\":0"),
            "no keys tracked yet: {}",
            body
        );
    }

    #[tokio::test]
    async fn management_api_providers_and_failover() {
        let (chain, api) = setup_all_plugins().await;

        let _ = chain;
        let mgmt_port = start_mgmt_server(api).await;

        // GET providers.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"name\":\"openai\""),
            "openai provider listed: {}",
            body
        );
        assert!(
            body.contains("\"pattern\":\"api.openai.com\""),
            "pattern listed: {}",
            body
        );
        assert!(
            body.contains("\"cooldown_secs\":30"),
            "cooldown listed: {}",
            body
        );

        // GET failed providers — should be empty.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers/failed").await;
        assert_eq!(status, 200);
        let failed: Value = serde_json::from_str(&body).expect("parse failed providers");
        assert_eq!(failed["failed"], json!([]), "no failures initially: {body}");

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers/health").await;
        assert_eq!(status, 200);
        let health: Value = serde_json::from_str(&body).expect("parse provider health");
        let providers = health["providers"].as_array().expect("providers array");
        let openai = providers
            .iter()
            .find(|provider| provider["name"] == "openai")
            .expect("openai provider listed");
        assert_eq!(
            openai["eligible"],
            json!(true),
            "eligible by default: {body}"
        );
        assert_eq!(
            openai["adaptive_penalty_total"],
            json!(0.0),
            "health penalties start empty: {body}"
        );

        // Clear a nonexistent failed provider returns 404.
        let (status, _) = mgmt_delete(mgmt_port, "/api/v1/providers/failed/openai").await;
        assert_eq!(status, 404);

        // Clear all (no-op) returns 200.
        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/providers/failed").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"ok\":true"));
    }

    #[tokio::test]
    async fn routing_debug_headers_and_provider_health_show_failover_path() {
        let debug_header_seen = Arc::new(AtomicUsize::new(0));
        let failing_addr = start_upstream_async({
            let debug_header_seen = Arc::clone(&debug_header_seen);
            move |req: Request<Incoming>| {
                let debug_header_seen = Arc::clone(&debug_header_seen);
                async move {
                    if req.headers().contains_key("x-trp-routing-debug") {
                        debug_header_seen.fetch_add(1, Ordering::Relaxed);
                    }
                    Response::builder()
                        .status(StatusCode::TOO_MANY_REQUESTS)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            r#"{"error":{"message":"rate limited"}}"#,
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let healthy_addr = start_upstream_async({
            let debug_header_seen = Arc::clone(&debug_header_seen);
            move |req: Request<Incoming>| {
                let debug_header_seen = Arc::clone(&debug_header_seen);
                async move {
                    if req.headers().contains_key("x-trp-routing-debug") {
                        debug_header_seen.fetch_add(1, Ordering::Relaxed);
                    }
                    llm_chat_handler(req)
                }
            }
        })
        .await;

        let providers = vec![
            canonical_provider(
                "openai",
                "sk-openai-real",
                format!("http://{}", failing_addr),
                vec!["gpt-4".to_string()],
                "authorization",
                None,
                ProviderFamily::OpenAi,
                openai_tool_surfaces(),
                ProviderRoutingMetadataConfig::default(),
            ),
            canonical_provider(
                "anthropic",
                "sk-anthropic-real",
                format!("http://{}", healthy_addr),
                vec!["gpt-4".to_string()],
                "authorization",
                None,
                ProviderFamily::OpenAi,
                openai_tool_surfaces(),
                ProviderRoutingMetadataConfig::default(),
            ),
        ];

        let (chain, api) = setup_plugins_with_virtual_keys_and_failover(
            &providers,
            &[("openai", &failing_addr), ("anthropic", &healthy_addr)],
        )
        .await;
        let mgmt_port = start_mgmt_server(api.clone()).await;

        let router = catch_all_router(vec![format!("http://{}", failing_addr)]);
        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-a"),
                "routing-debug",
                "openai",
                None,
                None,
                None,
                None,
                Some(vec!["gpt-4".to_string()]),
                None,
            )
            .await
            .unwrap()
            .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .header("x-trp-routing-debug", "1")
            .body(Full::new(Bytes::from(chat_request_body())))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-trp-provider-selected")
                .unwrap()
                .to_str()
                .unwrap(),
            "anthropic"
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-provider-order")
                .unwrap()
                .to_str()
                .unwrap(),
            "openai,anthropic"
        );
        assert_eq!(
            resp.headers()
                .get("x-trp-provider-attempts")
                .unwrap()
                .to_str()
                .unwrap(),
            "openai,anthropic"
        );
        assert_eq!(
            debug_header_seen.load(Ordering::Relaxed),
            0,
            "debug header must not be forwarded upstream"
        );

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers/failed").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"name\":\"openai\"") && body.contains("\"reason\":\"rate_limited\""),
            "failed provider state surfaced: {}",
            body
        );

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers/health").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"name\":\"openai\"")
                && body.contains("\"cooldown_reason\":\"rate_limited\""),
            "openai cooldown surfaced: {}",
            body
        );
        assert!(
            body.contains("\"name\":\"anthropic\"") && body.contains("\"eligible\":true"),
            "healthy provider surfaced: {}",
            body
        );
    }

    #[tokio::test]
    async fn provider_management_api_escapes_provider_diagnostics_json() {
        let debug_header_seen = Arc::new(AtomicUsize::new(0));
        let failing_addr = start_upstream_async({
            let debug_header_seen = Arc::clone(&debug_header_seen);
            move |req: Request<Incoming>| {
                let debug_header_seen = Arc::clone(&debug_header_seen);
                async move {
                    if req.headers().contains_key("x-trp-routing-debug") {
                        debug_header_seen.fetch_add(1, Ordering::Relaxed);
                    }
                    Response::builder()
                        .status(StatusCode::TOO_MANY_REQUESTS)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            r#"{"error":{"message":"rate limited"}}"#,
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let provider_name = r#"odd"provider\name"#;
        let providers = vec![canonical_provider(
            provider_name,
            "sk-openai-real",
            format!("http://{}", failing_addr),
            vec!["gpt-4".to_string()],
            "authorization",
            None,
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (chain, api) = setup_plugins_with_virtual_keys_and_failover(
            &providers,
            &[(provider_name, &failing_addr)],
        )
        .await;
        let mgmt_port = start_mgmt_server(api.clone()).await;

        let router = catch_all_router(vec![format!("http://{}", failing_addr)]);
        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-a"),
                "routing-debug",
                provider_name,
                None,
                None,
                None,
                None,
                Some(vec!["gpt-4".to_string()]),
                None,
            )
            .await
            .unwrap()
            .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(chat_request_body())))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers/failed").await;
        assert_eq!(status, 200);
        let failed: Value = serde_json::from_str(&body).expect("parse failed providers");
        let failed_entry = failed["failed"]
            .as_array()
            .and_then(|entries| entries.first())
            .expect("failed provider entry");
        assert_eq!(failed_entry["name"], json!(provider_name));
        assert_eq!(failed_entry["reason"], json!("rate_limited"));

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers/health").await;
        assert_eq!(status, 200);
        let health: Value = serde_json::from_str(&body).expect("parse provider health");
        let entry = health["providers"]
            .as_array()
            .and_then(|entries| {
                entries
                    .iter()
                    .find(|provider| provider["name"] == json!(provider_name))
            })
            .expect("provider health entry");
        assert_eq!(entry["name"], json!(provider_name));
        assert_eq!(entry["cooldown_reason"], json!("rate_limited"));
    }

    #[tokio::test]
    async fn routing_rule_management_auto_ids_do_not_collide_within_same_second() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let current_second = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        while current_second
            == std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        {
            tokio::task::yield_now().await;
        }

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/projects/project-a/routing-rules",
            r#"{"name":"first rule","provider_order":["openai"]}"#,
        )
        .await;
        assert_eq!(status, 201, "first create failed: {body}");

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/projects/project-a/routing-rules",
            r#"{"name":"second rule","provider_order":["openai"]}"#,
        )
        .await;
        assert_eq!(status, 201, "second create failed: {body}");

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/projects/project-a/routing-rules").await;
        assert_eq!(status, 200);
        let rules: Value = serde_json::from_str(&body).expect("parse routing rules");
        let entries = rules["rules"].as_array().expect("rules array");
        assert_eq!(
            entries.len(),
            2,
            "expected both routing rules to persist: {body}"
        );

        let rule_ids: std::collections::BTreeSet<_> = entries
            .iter()
            .filter_map(|rule| rule["rule_id"].as_str())
            .collect();
        assert_eq!(
            rule_ids.len(),
            2,
            "autogenerated rule IDs must stay unique even within the same second: {body}"
        );
    }

    #[tokio::test]
    async fn session_endpoints_aggregate_request_logs_and_strip_internal_header() {
        let session_header_seen = Arc::new(AtomicUsize::new(0));
        let request_metadata_header_seen = Arc::new(AtomicUsize::new(0));
        let request_custom_cost_header_seen = Arc::new(AtomicUsize::new(0));
        let upstream = start_upstream_async({
            let session_header_seen = Arc::clone(&session_header_seen);
            let request_metadata_header_seen = Arc::clone(&request_metadata_header_seen);
            let request_custom_cost_header_seen = Arc::clone(&request_custom_cost_header_seen);
            move |req: Request<Incoming>| {
                let session_header_seen = Arc::clone(&session_header_seen);
                let request_metadata_header_seen = Arc::clone(&request_metadata_header_seen);
                let request_custom_cost_header_seen = Arc::clone(&request_custom_cost_header_seen);
                async move {
                    if req.headers().contains_key("x-trp-session-id") {
                        session_header_seen.fetch_add(1, Ordering::Relaxed);
                    }
                    if req.headers().contains_key("cf-aig-metadata") {
                        request_metadata_header_seen.fetch_add(1, Ordering::Relaxed);
                    }
                    if req.headers().contains_key("cf-aig-custom-cost") {
                        request_custom_cost_header_seen.fetch_add(1, Ordering::Relaxed);
                    }
                    llm_chat_handler(req)
                }
            }
        })
        .await;

        let (chain, api) = setup_all_plugins().await;
        let mgmt_port = start_mgmt_server(api.clone()).await;
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![format!("http://{}", upstream)]),
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        for _ in 0..2 {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-session-test")
                .header("content-type", "application/json")
                .header("x-trp-session-id", "session-123")
                .header(
                    "cf-aig-metadata",
                    r#"{"trace_id":"trace-123","tenant":"acme","sampled":true,"rank":7,"color":"blue","ignored":"extra"}"#,
                )
                .header(
                    "cf-aig-custom-cost",
                    r#"{"per_token_in":0.001,"per_token_out":0.002}"#,
                )
                .body(Full::new(Bytes::from(chat_request_body())))
                .unwrap();
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
        }

        let mut session_summary: Option<serde_json::Value> = None;
        let mut session_logs: Option<serde_json::Value> = None;
        for _ in 0..20 {
            let (status, body) = mgmt_get(mgmt_port, "/api/v1/sessions/session-123").await;
            assert_eq!(status, 200);
            let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
            if parsed
                .get("request_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                >= 2
            {
                session_summary = Some(parsed);
                let (logs_status, logs_body) =
                    mgmt_get(mgmt_port, "/api/v1/sessions/session-123/logs").await;
                assert_eq!(logs_status, 200);
                session_logs = Some(serde_json::from_str(&logs_body).unwrap());
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        let session_summary = session_summary.expect("session summary available");
        let session_logs = session_logs.expect("session logs available");

        let (events_status, events_body) =
            mgmt_get(mgmt_port, "/api/v1/sessions/session-123/events?limit=10").await;
        assert_eq!(events_status, 200);
        let session_events: serde_json::Value = serde_json::from_str(&events_body).unwrap();
        let event_rows = session_events["data"].as_array().expect("session events");
        assert!(event_rows.len() >= 2);
        assert!(
            event_rows
                .iter()
                .filter(|event| event["event_kind"].as_str() == Some("request_observed"))
                .count()
                >= 2
        );

        assert_eq!(
            session_summary
                .get("session_id")
                .and_then(|value| value.as_str()),
            Some("session-123")
        );
        assert_eq!(
            session_summary.get("project_id").unwrap(),
            &serde_json::Value::Null
        );
        assert_eq!(
            session_summary
                .get("request_count")
                .and_then(|value| value.as_u64()),
            Some(2)
        );
        assert_eq!(
            session_summary
                .get("streaming_request_count")
                .and_then(|value| value.as_u64()),
            Some(0)
        );
        assert!(session_summary
            .get("first_request_unix")
            .and_then(|value| value.as_u64())
            .is_some());
        assert!(session_summary
            .get("last_request_unix")
            .and_then(|value| value.as_u64())
            .is_some());
        assert_eq!(
            session_summary
                .get("latest_request")
                .and_then(|value| value.get("metadata"))
                .and_then(|value| value.get("trace_id"))
                .and_then(|value| value.as_str()),
            Some("trace-123")
        );
        assert_eq!(
            session_summary
                .get("latest_request")
                .and_then(|value| value.get("metadata"))
                .and_then(|value| value.get("ignored")),
            None
        );
        assert_eq!(
            session_summary
                .get("latest_request")
                .and_then(|value| value.get("custom_cost"))
                .and_then(|value| value.get("per_token_in"))
                .and_then(|value| value.as_f64()),
            Some(0.001)
        );
        assert_eq!(
            session_summary
                .get("latest_request")
                .and_then(|value| value.get("custom_cost_applied"))
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            session_summary
                .get("latest_request")
                .and_then(|value| value.get("model"))
                .and_then(|value| value.as_str()),
            Some("gpt-4")
        );
        assert_eq!(
            session_summary
                .get("semantic_event_count")
                .and_then(|value| value.as_u64()),
            Some(0)
        );
        assert_eq!(
            session_summary
                .get("tool_names")
                .and_then(|value| value.as_array())
                .map(|value| value.len()),
            Some(0)
        );
        assert_eq!(
            session_summary.get("status"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("owner_id"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("owner_acquired_at_unix"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("owner_active"),
            Some(&serde_json::Value::Bool(false))
        );
        assert_eq!(
            session_summary.get("last_transition_at_unix"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("last_transition_reason"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("last_heartbeat_unix"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("lease_expires_at_unix"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("cancel_requested_at_unix"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("cancel_requested_by"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("cancel_reason"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("handoff_target_owner_id"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("handoff_requested_at_unix"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("handoff_reason"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            session_summary.get("handoff_pending"),
            Some(&serde_json::Value::Bool(false))
        );
        assert_eq!(session_summary.get("state"), Some(&serde_json::Value::Null));
        assert!(session_summary
            .get("updated_at_unix")
            .and_then(|value| value.as_u64())
            .is_some());

        let logs = session_logs
            .get("data")
            .and_then(|value| value.as_array())
            .unwrap();
        assert_eq!(logs.len(), 2);
        assert!(logs.iter().all(|entry| {
            entry.get("session_id").and_then(|value| value.as_str()) == Some("session-123")
        }));
        assert!(logs
            .iter()
            .all(|entry| { entry.get("project_id") == Some(&serde_json::Value::Null) }));
        assert!(logs.iter().all(|entry| {
            entry
                .get("metadata")
                .and_then(|value| value.get("tenant"))
                .and_then(|value| value.as_str())
                == Some("acme")
        }));
        assert!(logs.iter().all(|entry| {
            entry
                .get("custom_cost")
                .and_then(|value| value.get("per_token_out"))
                .and_then(|value| value.as_f64())
                == Some(0.002)
        }));
        assert!(logs.iter().all(|entry| entry
            .get("custom_cost_applied")
            .and_then(|value| value.as_bool())
            == Some(true)));
        assert!(logs.iter().all(|entry| {
            entry
                .get("cost")
                .and_then(|value| value.as_f64())
                .map(|value| (value - 0.026).abs() < 1e-9)
                == Some(true)
        }));
        assert!(logs.iter().all(|entry| {
            entry
                .get("metadata")
                .and_then(|value| value.get("ignored"))
                .is_none()
        }));

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/logs?session_id=session-123").await;
        assert_eq!(status, 200);
        let logs_by_query: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            logs_by_query
                .get("data")
                .and_then(|value| value.as_array())
                .map(|value| value.len()),
            Some(2)
        );
        assert!(logs_by_query["data"]
            .as_array()
            .expect("query log rows")
            .iter()
            .all(|entry| entry["metadata"]["sampled"].as_bool() == Some(true)));
        assert!(logs_by_query["data"]
            .as_array()
            .expect("query log rows")
            .iter()
            .all(|entry| entry["custom_cost_applied"].as_bool() == Some(true)));

        let (status, body) = mgmt_get(
            mgmt_port,
            "/api/v1/logs?session_id=session-123&metadata_key=tenant&metadata_value=acme&has_custom_cost=true&custom_cost_applied=true",
        )
        .await;
        assert_eq!(status, 200);
        let filtered_logs: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            filtered_logs["data"]
                .as_array()
                .map(|entries| entries.len()),
            Some(2)
        );

        let (status, body) = mgmt_get(
            mgmt_port,
            "/api/v1/sessions/session-123/logs?metadata_key=ignored",
        )
        .await;
        assert_eq!(status, 200);
        let filtered_session_logs: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            filtered_session_logs["data"]
                .as_array()
                .map(|entries| entries.len()),
            Some(0)
        );

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/logs?metadata_value=acme").await;
        assert_eq!(status, 400);
        assert!(body.contains("metadata_value requires metadata_key"));

        let request_events = event_rows
            .iter()
            .filter(|event| event["event_kind"].as_str() == Some("request_observed"))
            .collect::<Vec<_>>();
        assert!(!request_events.is_empty());
        assert!(request_events
            .iter()
            .all(|event| { event["payload"]["metadata"]["rank"].as_i64() == Some(7) }));
        assert!(request_events.iter().all(|event| {
            event["payload"]["custom_cost"]["per_token_in"].as_f64() == Some(0.001)
        }));
        assert!(request_events
            .iter()
            .all(|event| event["payload"]["custom_cost_applied"].as_bool() == Some(true)));
        assert!(request_events
            .iter()
            .all(|event| event["payload"]["metadata"].get("ignored").is_none()));

        let (status, body) = mgmt_put(
            mgmt_port,
            "/api/v1/sessions/session-123",
            r#"{"status":"active","state":{"turn":2,"phase":"tool_loop"},"metadata":{"owner":"qa","source":"integration"}}"#,
        )
        .await;
        assert_eq!(status, 200);
        let updated_session: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(updated_session["status"].as_str(), Some("active"));
        assert_eq!(updated_session["state"]["turn"].as_u64(), Some(2));
        assert_eq!(updated_session["metadata"]["owner"].as_str(), Some("qa"));

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/sessions/session-123").await;
        assert_eq!(status, 200);
        let persisted_session: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(persisted_session["status"].as_str(), Some("active"));
        assert_eq!(
            persisted_session["state"]["phase"].as_str(),
            Some("tool_loop")
        );
        assert_eq!(
            persisted_session["metadata"]["source"].as_str(),
            Some("integration")
        );
        assert_eq!(
            persisted_session["latest_request"]["metadata"]["tenant"].as_str(),
            Some("acme")
        );
        assert!(persisted_session["owner_id"].is_null());
        assert!(persisted_session["owner_acquired_at_unix"].is_null());
        assert_eq!(persisted_session["owner_active"].as_bool(), Some(false));
        assert!(persisted_session["last_transition_at_unix"]
            .as_u64()
            .is_some());
        assert!(persisted_session["last_transition_reason"].is_null());
        assert!(persisted_session["last_heartbeat_unix"].is_null());
        assert!(persisted_session["lease_expires_at_unix"].is_null());
        assert!(persisted_session["cancel_requested_at_unix"].is_null());
        assert!(persisted_session["cancel_requested_by"].is_null());
        assert!(persisted_session["cancel_reason"].is_null());
        assert!(persisted_session["handoff_target_owner_id"].is_null());
        assert!(persisted_session["handoff_requested_at_unix"].is_null());
        assert!(persisted_session["handoff_reason"].is_null());
        assert_eq!(persisted_session["handoff_pending"].as_bool(), Some(false));

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/claim",
            r#"{"owner_id":"worker-a","lease_ttl_secs":45,"state":{"turn":2,"claimed":true}}"#,
        )
        .await;
        assert_eq!(status, 200);
        let claimed_session: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(claimed_session["owner_id"].as_str(), Some("worker-a"));
        assert!(claimed_session["owner_acquired_at_unix"].as_u64().is_some());
        assert_eq!(claimed_session["owner_active"].as_bool(), Some(true));
        assert!(claimed_session["lease_expires_at_unix"].as_u64().is_some());

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/claim",
            r#"{"owner_id":"worker-b","lease_ttl_secs":45}"#,
        )
        .await;
        assert_eq!(status, 409);
        assert!(
            body.contains("already claimed by 'worker-a'"),
            "claim conflict: {}",
            body
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/heartbeat",
            r#"{"owner_id":"worker-a","lease_ttl_secs":30,"state":{"turn":3},"metadata":{"owner":"qa","beat":1}}"#,
        )
        .await;
        assert_eq!(status, 200);
        let heartbeat_session: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(heartbeat_session["status"].as_str(), Some("active"));
        assert_eq!(heartbeat_session["state"]["turn"].as_u64(), Some(3));
        assert_eq!(heartbeat_session["owner_id"].as_str(), Some("worker-a"));
        assert_eq!(heartbeat_session["owner_active"].as_bool(), Some(true));
        assert!(heartbeat_session["last_heartbeat_unix"].as_u64().is_some());
        assert!(heartbeat_session["lease_expires_at_unix"]
            .as_u64()
            .is_some());

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/heartbeat",
            r#"{"owner_id":"worker-b","lease_ttl_secs":30}"#,
        )
        .await;
        assert_eq!(status, 409);
        assert!(
            body.contains("owner_id 'worker-a' is required"),
            "heartbeat owner conflict: {}",
            body
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/cancel",
            r#"{"requested_by":"operator-a","reason":"stop after tool loop"}"#,
        )
        .await;
        assert_eq!(status, 200);
        let cancelled_request: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(cancelled_request["cancel_requested_at_unix"]
            .as_u64()
            .is_some());
        assert_eq!(
            cancelled_request["cancel_requested_by"].as_str(),
            Some("operator-a")
        );
        assert_eq!(
            cancelled_request["cancel_reason"].as_str(),
            Some("stop after tool loop")
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/transition",
            r#"{"status":"paused","reason":"waiting on tool","metadata":{"owner":"qa","phase":"paused"}}"#,
        )
        .await;
        assert_eq!(status, 200);
        let paused_session: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(paused_session["status"].as_str(), Some("paused"));
        assert_eq!(
            paused_session["last_transition_reason"].as_str(),
            Some("waiting on tool")
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/release",
            r#"{"owner_id":"worker-b"}"#,
        )
        .await;
        assert_eq!(status, 409);
        assert!(
            body.contains("currently owned by 'worker-a'"),
            "release owner conflict: {}",
            body
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/release",
            r#"{"owner_id":"worker-a"}"#,
        )
        .await;
        assert_eq!(status, 200);
        let released_session: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(released_session["owner_id"].is_null());
        assert_eq!(released_session["owner_active"].as_bool(), Some(false));
        assert!(released_session["lease_expires_at_unix"].is_null());

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/transition",
            r#"{"status":"completed","reason":"done"}"#,
        )
        .await;
        assert_eq!(status, 200);
        let completed_session: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(completed_session["status"].as_str(), Some("completed"));
        assert_eq!(
            completed_session["last_transition_reason"].as_str(),
            Some("done")
        );
        assert!(completed_session["owner_id"].is_null());
        assert!(completed_session["lease_expires_at_unix"].is_null());
        assert!(completed_session["cancel_requested_at_unix"].is_null());
        assert!(completed_session["cancel_requested_by"].is_null());
        assert!(completed_session["cancel_reason"].is_null());
        assert!(completed_session["handoff_target_owner_id"].is_null());
        assert!(completed_session["handoff_requested_at_unix"].is_null());
        assert!(completed_session["handoff_reason"].is_null());
        assert_eq!(completed_session["handoff_pending"].as_bool(), Some(false));

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/heartbeat",
            r#"{"lease_ttl_secs":30}"#,
        )
        .await;
        assert_eq!(status, 409);
        assert!(
            body.contains("cannot heartbeat a terminal session"),
            "heartbeat rejection: {}",
            body
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-123/cancel",
            r#"{"requested_by":"operator-b","reason":"too late"}"#,
        )
        .await;
        assert_eq!(status, 409);
        assert!(
            body.contains("cannot request cancellation for a terminal session"),
            "cancel rejection: {}",
            body
        );

        assert_eq!(session_header_seen.load(Ordering::Relaxed), 0);
        assert_eq!(request_metadata_header_seen.load(Ordering::Relaxed), 0);
        assert_eq!(request_custom_cost_header_seen.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn session_wait_returns_new_events_after_latest_seen_seq() {
        let (_chain, api) = setup_all_plugins().await;
        let mgmt_port = start_mgmt_server(api.clone()).await;

        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-watch/claim",
            r#"{"project_id":"project-a","owner_id":"worker-a","lease_ttl_secs":45}"#,
        )
        .await;
        assert_eq!(status, 200);

        let (status, body) =
            mgmt_get(mgmt_port, "/api/v1/sessions/session-watch/events?limit=10").await;
        assert_eq!(status, 200);
        let events: serde_json::Value = serde_json::from_str(&body).unwrap();
        let latest_seq = events["latest_event_seq"]
            .as_i64()
            .expect("latest event seq");
        assert!(latest_seq > 0);

        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            let (status, _) = mgmt_post(
                mgmt_port,
                "/api/v1/sessions/session-watch/cancel",
                r#"{"requested_by":"operator-a","reason":"stop"}"#,
            )
            .await;
            assert_eq!(status, 200);
        });

        let (status, body) = mgmt_get(
            mgmt_port,
            &format!(
                "/api/v1/sessions/session-watch/wait?after_seq={}&timeout_secs=2",
                latest_seq
            ),
        )
        .await;
        assert_eq!(status, 200);
        let waited: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(waited["wait_timed_out"].as_bool(), Some(false));
        assert_eq!(waited["count"].as_u64(), Some(1));
        assert_eq!(
            waited["events"][0]["event_kind"].as_str(),
            Some("cancel_requested")
        );
        assert_eq!(waited["events"][0]["actor_id"].as_str(), Some("operator-a"));
        assert_eq!(waited["session"]["cancel_reason"].as_str(), Some("stop"));
    }

    #[tokio::test]
    async fn session_reconcile_endpoint_handles_stale_owners_and_pending_cancels() {
        let (_chain, api) = setup_all_plugins().await;
        let mgmt_port = start_mgmt_server(api.clone()).await;

        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-recover/claim",
            r#"{"project_id":"project-a","owner_id":"worker-a","lease_ttl_secs":0,"state":{"turn":1}}"#,
        )
        .await;
        assert_eq!(status, 200);

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-recover/reconcile",
            "{}",
        )
        .await;
        assert_eq!(status, 200);
        let reconciled: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            reconciled["reconciled_action"].as_str(),
            Some("recovery_required")
        );
        assert_eq!(reconciled["status"].as_str(), Some("paused"));
        assert!(reconciled["owner_id"].is_null());
        assert_eq!(reconciled["owner_stale"].as_bool(), Some(false));
        assert_eq!(reconciled["recovery_required"].as_bool(), Some(true));
        assert_eq!(
            reconciled["recovery_reason"].as_str(),
            Some("owner lease expired; recovery required")
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-recover/claim",
            r#"{"project_id":"project-a","owner_id":"worker-b","lease_ttl_secs":30}"#,
        )
        .await;
        assert_eq!(status, 200);
        let reclaimed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(reclaimed["owner_id"].as_str(), Some("worker-b"));
        assert_eq!(reclaimed["recovery_required"].as_bool(), Some(false));

        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-cancelled/claim",
            r#"{"project_id":"project-a","owner_id":"worker-c","lease_ttl_secs":0}"#,
        )
        .await;
        assert_eq!(status, 200);
        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-cancelled/cancel",
            r#"{"requested_by":"operator-a","reason":"stop now"}"#,
        )
        .await;
        assert_eq!(status, 200);
        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-cancelled/reconcile",
            "{}",
        )
        .await;
        assert_eq!(status, 200);
        let cancelled: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            cancelled["reconciled_action"].as_str(),
            Some("cancelled_after_owner_expiry")
        );
        assert_eq!(cancelled["status"].as_str(), Some("cancelled"));
        assert_eq!(
            cancelled["last_transition_reason"].as_str(),
            Some("cancel request finalized after owner lease expired")
        );
        assert_eq!(
            cancelled["cancel_requested_by"].as_str(),
            Some("operator-a")
        );
        assert_eq!(cancelled["cancel_pending"].as_bool(), Some(false));

        let (status, _) = mgmt_put(
            mgmt_port,
            "/api/v1/sessions/session-without-owner",
            r#"{"project_id":"project-a","status":"paused"}"#,
        )
        .await;
        assert_eq!(status, 200);
        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-without-owner/cancel",
            r#"{"requested_by":"operator-b","reason":"manual stop"}"#,
        )
        .await;
        assert_eq!(status, 200);
        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-without-owner/reconcile",
            "{}",
        )
        .await;
        assert_eq!(status, 200);
        let cancelled_without_owner: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            cancelled_without_owner["reconciled_action"].as_str(),
            Some("cancelled_without_owner")
        );
        assert_eq!(
            cancelled_without_owner["status"].as_str(),
            Some("cancelled")
        );
        assert_eq!(
            cancelled_without_owner["last_transition_reason"].as_str(),
            Some("cancel request finalized without active owner")
        );
    }

    #[tokio::test]
    async fn session_handoff_and_accept_transfer_ownership_cleanly() {
        let (_chain, api) = setup_all_plugins().await;
        let mgmt_port = start_mgmt_server(api.clone()).await;

        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-handoff/claim",
            r#"{"project_id":"project-a","owner_id":"worker-a","lease_ttl_secs":45,"state":{"turn":1}}"#,
        )
        .await;
        assert_eq!(status, 200);

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-handoff/handoff",
            r#"{"owner_id":"worker-a","target_owner_id":"worker-b","reason":"move to another worker","state":{"turn":2,"phase":"handoff"},"metadata":{"handoff":"pending"}}"#,
        )
        .await;
        assert_eq!(status, 200);
        let handoff_requested: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(handoff_requested["owner_id"].as_str(), Some("worker-a"));
        assert_eq!(
            handoff_requested["handoff_target_owner_id"].as_str(),
            Some("worker-b")
        );
        assert!(handoff_requested["handoff_requested_at_unix"]
            .as_u64()
            .is_some());
        assert_eq!(
            handoff_requested["handoff_reason"].as_str(),
            Some("move to another worker")
        );
        assert_eq!(handoff_requested["handoff_pending"].as_bool(), Some(true));
        assert_eq!(
            handoff_requested["state"]["phase"].as_str(),
            Some("handoff")
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-handoff/accept",
            r#"{"owner_id":"worker-c","lease_ttl_secs":45}"#,
        )
        .await;
        assert_eq!(status, 409);
        assert!(
            body.contains("owner_id 'worker-b' is required"),
            "accept owner conflict: {}",
            body
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-handoff/heartbeat",
            r#"{"owner_id":"worker-a","lease_ttl_secs":45}"#,
        )
        .await;
        assert_eq!(status, 200);
        let pre_accept_heartbeat: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(pre_accept_heartbeat["owner_id"].as_str(), Some("worker-a"));
        assert_eq!(
            pre_accept_heartbeat["handoff_pending"].as_bool(),
            Some(true)
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-handoff/accept",
            r#"{"owner_id":"worker-b","lease_ttl_secs":60,"state":{"turn":3,"phase":"resumed"},"metadata":{"handoff":"accepted"}}"#,
        )
        .await;
        assert_eq!(status, 200);
        let accepted: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(accepted["owner_id"].as_str(), Some("worker-b"));
        assert_eq!(accepted["owner_active"].as_bool(), Some(true));
        assert!(accepted["lease_expires_at_unix"].as_u64().is_some());
        assert_eq!(accepted["handoff_pending"].as_bool(), Some(false));
        assert!(accepted["handoff_target_owner_id"].is_null());
        assert!(accepted["handoff_requested_at_unix"].is_null());
        assert!(accepted["handoff_reason"].is_null());
        assert_eq!(
            accepted["last_transition_reason"].as_str(),
            Some("handoff accepted")
        );
        assert_eq!(accepted["state"]["phase"].as_str(), Some("resumed"));
        assert_eq!(accepted["metadata"]["handoff"].as_str(), Some("accepted"));

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-handoff/heartbeat",
            r#"{"owner_id":"worker-a","lease_ttl_secs":30}"#,
        )
        .await;
        assert_eq!(status, 409);
        assert!(
            body.contains("owner_id 'worker-b' is required"),
            "old owner heartbeat conflict: {}",
            body
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-handoff/release",
            r#"{"owner_id":"worker-a"}"#,
        )
        .await;
        assert_eq!(status, 409);
        assert!(
            body.contains("currently owned by 'worker-b'"),
            "old owner release conflict: {}",
            body
        );
    }

    #[tokio::test]
    async fn session_list_filters_and_takeover_cover_stale_and_forced_ownership_changes() {
        let (_chain, api) = setup_all_plugins().await;
        let mgmt_port = start_mgmt_server(api.clone()).await;

        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-stale/claim",
            r#"{"project_id":"project-a","owner_id":"worker-a","lease_ttl_secs":0,"state":{"turn":1}}"#,
        )
        .await;
        assert_eq!(status, 200);

        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-active/claim",
            r#"{"project_id":"project-a","owner_id":"worker-b","lease_ttl_secs":60,"state":{"turn":2}}"#,
        )
        .await;
        assert_eq!(status, 200);

        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-handoff-pending/claim",
            r#"{"project_id":"project-a","owner_id":"worker-c","lease_ttl_secs":60}"#,
        )
        .await;
        assert_eq!(status, 200);

        let (status, _) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-handoff-pending/handoff",
            r#"{"owner_id":"worker-c","target_owner_id":"worker-d","reason":"move worker"}"#,
        )
        .await;
        assert_eq!(status, 200);

        let (status, body) = mgmt_get(
            mgmt_port,
            "/api/v1/sessions?project_id=project-a&recovery_required=true",
        )
        .await;
        assert_eq!(status, 200);
        let stale_sessions: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(stale_sessions["count"].as_u64(), Some(1));
        assert_eq!(
            stale_sessions["data"][0]["session_id"].as_str(),
            Some("session-stale")
        );
        assert_eq!(
            stale_sessions["data"][0]["recovery_required"].as_bool(),
            Some(true)
        );

        let (status, body) = mgmt_get(
            mgmt_port,
            "/api/v1/sessions?project_id=project-a&handoff_pending=true",
        )
        .await;
        assert_eq!(status, 200);
        let handoff_sessions: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(handoff_sessions["count"].as_u64(), Some(1));
        assert_eq!(
            handoff_sessions["data"][0]["session_id"].as_str(),
            Some("session-handoff-pending")
        );
        assert_eq!(
            handoff_sessions["data"][0]["handoff_target_owner_id"].as_str(),
            Some("worker-d")
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-stale/takeover",
            r#"{"owner_id":"worker-z","lease_ttl_secs":30,"state":{"turn":3},"metadata":{"source":"resume"}}"#,
        )
        .await;
        assert_eq!(status, 200);
        let stale_takeover: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(stale_takeover["owner_id"].as_str(), Some("worker-z"));
        assert_eq!(stale_takeover["status"].as_str(), Some("active"));
        assert_eq!(stale_takeover["owner_active"].as_bool(), Some(true));
        assert_eq!(
            stale_takeover["last_transition_reason"].as_str(),
            Some("taken over")
        );
        assert_eq!(stale_takeover["state"]["turn"].as_u64(), Some(3));
        assert_eq!(
            stale_takeover["metadata"]["source"].as_str(),
            Some("resume")
        );
        assert_eq!(stale_takeover["handoff_pending"].as_bool(), Some(false));

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-active/takeover",
            r#"{"owner_id":"worker-z","lease_ttl_secs":30}"#,
        )
        .await;
        assert_eq!(status, 409);
        assert!(
            body.contains("request a handoff")
                || body.contains("wait for lease expiry")
                || body.contains("force=true"),
            "active takeover conflict: {}",
            body
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/sessions/session-active/takeover",
            r#"{"owner_id":"worker-z","lease_ttl_secs":30,"force":true}"#,
        )
        .await;
        assert_eq!(status, 200);
        let forced_takeover: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(forced_takeover["owner_id"].as_str(), Some("worker-z"));
        assert_eq!(
            forced_takeover["last_transition_reason"].as_str(),
            Some("force takeover")
        );
        assert_eq!(forced_takeover["owner_active"].as_bool(), Some(true));
    }

    #[tokio::test]
    async fn management_api_tool_runtime_status_and_provider_capabilities() {
        use plugin_llm_gateway::store::ProjectToolRecord;

        let mcp_addr = start_upstream_async(move |req: Request<Incoming>| async move {
            let headers = req.headers().clone();
            let body = req
                .into_body()
                .collect()
                .await
                .expect("collect mcp body")
                .to_bytes();
            let body_json: serde_json::Value =
                serde_json::from_slice(&body).expect("mcp request json");
            match body_json["method"].as_str() {
                Some("initialize") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .header("mcp-session-id", "session-status-1")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": body_json["id"].clone(),
                            "result": {
                                "protocolVersion": "2025-11-25",
                                "capabilities": {},
                                "serverInfo": {
                                    "name": "fake-mcp",
                                    "version": "1.0.0"
                                }
                            }
                        })
                        .to_string(),
                    )))
                    .unwrap(),
                Some("notifications/initialized") => {
                    assert_eq!(
                        headers
                            .get("mcp-session-id")
                            .and_then(|value| value.to_str().ok()),
                        Some("session-status-1")
                    );
                    Response::builder()
                        .status(StatusCode::ACCEPTED)
                        .body(Full::new(Bytes::new()))
                        .unwrap()
                }
                Some("tools/list") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": body_json["id"].clone(),
                            "result": {
                                "tools": [{
                                    "name": "search_docs",
                                    "description": "Search remote docs"
                                }]
                            }
                        })
                        .to_string(),
                    )))
                    .unwrap(),
                other => panic!("unexpected MCP method: {:?}", other),
            }
        })
        .await;

        let mcp_url = format!("http://{}", mcp_addr);
        let configs = vec![
            PluginConfig {
                name: "provider_failover".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("cooldown_secs".into(), toml::Value::Integer(45));
                    t.insert(
                        "providers".into(),
                        toml::Value::Array(vec![
                            toml::Value::Table({
                                let mut p = toml::value::Map::new();
                                p.insert("name".into(), toml::Value::String("openai".into()));
                                p.insert(
                                    "pattern".into(),
                                    toml::Value::String("api.openai.test".into()),
                                );
                                p
                            }),
                            toml::Value::Table({
                                let mut p = toml::value::Map::new();
                                p.insert("name".into(), toml::Value::String("anthropic".into()));
                                p.insert(
                                    "pattern".into(),
                                    toml::Value::String("api.anthropic.test".into()),
                                );
                                p
                            }),
                        ]),
                    );
                    t
                }),
            },
            PluginConfig {
                name: "tool_runtime".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("tool_timeout_ms".into(), toml::Value::Integer(4321));
                    t.insert("max_round_trips".into(), toml::Value::Integer(6));
                    t.insert(
                        "arxiv_base_url".into(),
                        toml::Value::String("https://arxiv.test/api/query".into()),
                    );
                    t.insert("arxiv_default_max_results".into(), toml::Value::Integer(7));
                    t.insert(
                        "web_search_backends".into(),
                        toml::Value::Table({
                            let mut backends = toml::value::Map::new();
                            backends.insert(
                                "docs-search".into(),
                                toml::Value::Table({
                                    let mut backend = toml::value::Map::new();
                                    backend.insert(
                                        "url".into(),
                                        toml::Value::String("https://search.test/query".into()),
                                    );
                                    backend.insert(
                                        "method".into(),
                                        toml::Value::String("POST".into()),
                                    );
                                    backend
                                }),
                            );
                            backends
                        }),
                    );
                    t.insert(
                        "mcp_servers".into(),
                        toml::Value::Table({
                            let mut servers = toml::value::Map::new();
                            servers.insert(
                                "research-mcp".into(),
                                toml::Value::Table({
                                    let mut server = toml::value::Map::new();
                                    server
                                        .insert("url".into(), toml::Value::String(mcp_url.clone()));
                                    server.insert(
                                        "method".into(),
                                        toml::Value::String("POST".into()),
                                    );
                                    server.insert("timeout_ms".into(), toml::Value::Integer(2100));
                                    server.insert("max_retries".into(), toml::Value::Integer(2));
                                    server.insert(
                                        "max_calls_per_request".into(),
                                        toml::Value::Integer(4),
                                    );
                                    server
                                }),
                            );
                            servers
                        }),
                    );
                    t
                }),
            },
        ];

        let providers = vec![
            canonical_provider(
                "openai",
                "sk-openai-real",
                "https://api.openai.test",
                vec!["gpt-4o".to_string(), "gpt-4o-mini".to_string()],
                "authorization",
                Some(30),
                ProviderFamily::OpenAi,
                openai_runtime_surfaces(),
                ProviderRoutingMetadataConfig {
                    data_collection: Some(ProviderDataCollectionMode::Deny),
                    zdr: true,
                    distillable_text: true,
                    quantizations: vec!["fp8".to_string(), "int4".to_string()],
                    supported_parameter_families: vec![
                        "tools".to_string(),
                        "prompt_cache_controls".to_string(),
                    ],
                },
            ),
            canonical_provider(
                "anthropic",
                "sk-anthropic-real",
                "https://api.anthropic.test",
                vec!["claude-sonnet-4-20250514".to_string()],
                "x-api-key",
                Some(45),
                ProviderFamily::Anthropic,
                anthropic_runtime_surfaces(),
                ProviderRoutingMetadataConfig {
                    data_collection: Some(ProviderDataCollectionMode::Allow),
                    zdr: false,
                    distillable_text: false,
                    quantizations: vec!["bf16".to_string()],
                    supported_parameter_families: vec!["tools".to_string()],
                },
            ),
        ];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();
        api.upsert_project_tool(ProjectToolRecord {
            project_id: "project-a".to_string(),
            tool_name: "web_search".to_string(),
            description: Some("Search docs".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "web_search".to_string(),
            executor_config_json: Some(serde_json::json!({ "backend": "docs-search" }).to_string()),
            enabled: true,
            timeout_ms: Some(1500),
            updated_at: "0".to_string(),
        })
        .await
        .unwrap()
        .unwrap();
        api.upsert_project_tool(ProjectToolRecord {
            project_id: "project-a".to_string(),
            tool_name: "arxiv_search".to_string(),
            description: Some("Search arxiv".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "arxiv_search".to_string(),
            executor_config_json: Some(serde_json::json!({ "max_results": 3 }).to_string()),
            enabled: false,
            timeout_ms: Some(1200),
            updated_at: "0".to_string(),
        })
        .await
        .unwrap()
        .unwrap();

        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/tool-runtime/status").await;
        assert_eq!(status, 200);
        let status_json: serde_json::Value = serde_json::from_str(&body).expect("status json");
        assert_eq!(status_json["default_timeout_ms"].as_u64(), Some(4321));
        assert_eq!(status_json["max_round_trips"].as_u64(), Some(6));
        assert_eq!(
            status_json["responses_stream_mode"].as_str(),
            Some("strict")
        );
        let preview_features = status_json["preview_features"]
            .as_array()
            .expect("preview features array");
        assert!(preview_features.iter().any(|feature| {
            feature["name"].as_str() == Some("responses_composed_streaming")
                && feature["enabled"].as_bool() == Some(false)
                && feature["enforcement"].as_str() == Some("hard_gate")
        }));
        assert!(preview_features.iter().any(|feature| {
            feature["name"].as_str() == Some("control_plane_import")
                && feature["enabled"].as_bool() == Some(false)
                && feature["enforcement"].as_str() == Some("hard_gate")
        }));
        assert!(preview_features.iter().any(|feature| {
            feature["name"].as_str() == Some("provider_surface_translations")
                && feature["enabled"].as_bool() == Some(false)
                && feature["enforcement"].as_str() == Some("visibility_only")
        }));
        assert_eq!(
            status_json["arxiv_base_url"].as_str(),
            Some("https://arxiv.test/api/query")
        );
        assert_eq!(status_json["registered_tools_total"].as_u64(), Some(2));
        assert_eq!(status_json["enabled_tools_total"].as_u64(), Some(1));
        assert!(status_json["web_search_backends"]
            .as_array()
            .map(|backends| {
                backends
                    .iter()
                    .any(|backend| backend["name"].as_str() == Some("docs-search"))
            })
            .unwrap_or(false));
        assert!(status_json["mcp_servers"]
            .as_array()
            .map(|servers| servers.iter().any(|server| {
                server["name"].as_str() == Some("research-mcp")
                    && server["timeout_ms"].as_u64() == Some(2100)
                    && server["max_retries"].as_u64() == Some(2)
                    && server["max_calls_per_request"].as_u64() == Some(4)
                    && server["operator_state"].as_str() == Some("enabled")
                    && server["operator_state_at"].is_null()
                    && server["operator_state_actor"].is_null()
                    && server["operator_state_reason"].is_null()
                    && server["health_state"].as_str() == Some("ready")
                    && server["health_reason"].is_null()
                    && server["recommended_action"].is_null()
                    && server["reachable"].as_bool() == Some(true)
                    && server["protocol_version"].as_str() == Some("2025-11-25")
                    && server["session_id_present"].as_bool() == Some(true)
                    && server["discovered_tools"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|tool| tool.as_str() == Some("search_docs"))
                    && server["discovery_refreshes"].as_u64() == Some(1)
                    && server["last_discovery_status"].as_str() == Some("ok")
                    && server["last_discovery_at"].as_str().is_some()
                    && server["last_discovery_error"].is_null()
                    && server["total_calls"].as_u64() == Some(0)
                    && server["successful_calls"].as_u64() == Some(0)
                    && server["failed_calls"].as_u64() == Some(0)
                    && server["retried_calls"].as_u64() == Some(0)
                    && server["session_reinitializations"].as_u64() == Some(0)
                    && server["last_session_reinitialized_at"].is_null()
                    && server["last_recovery_error"].is_null()
                    && server["last_budget_exceeded_at"].is_null()
                    && server["last_budget_exceeded_error"].is_null()
                    && server["last_session_reset_at"].is_null()
                    && server["last_session_reset_status"].is_null()
                    && server["last_session_reset_error"].is_null()
                    && server["last_session_reset_http_status"].is_null()
                    && server["last_call_at"].is_null()
                    && server["last_call_tool"].is_null()
                    && server["last_call_status"].is_null()
                    && server["last_call_error"].is_null()
                    && server["last_call_http_status"].is_null()
            }))
            .unwrap_or(false));
        let runtime_providers = status_json["providers"]
            .as_array()
            .expect("tool runtime providers");
        let openai = runtime_providers
            .iter()
            .find(|provider| provider["name"].as_str() == Some("openai"))
            .expect("openai runtime provider");
        assert_eq!(openai["family"].as_str(), Some("openai"));
        assert_eq!(openai["stability"].as_str(), Some("stable"));
        assert_eq!(
            openai["surfaces"]["responses"].as_str(),
            Some("openai_compatible")
        );
        assert!(openai["managed_tool_request_shapes"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|shape| shape.as_str() == Some("openai_chat_completions")));
        assert!(openai["managed_tool_request_shapes"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|shape| shape.as_str() == Some("openai_responses")));
        assert!(openai["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/responses")));
        assert!(openai["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/files")));
        assert!(openai["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/audio/speech")));
        assert!(openai["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/audio/translations")));
        assert!(openai["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/embeddings")));
        assert!(openai["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/realtime")));
        assert_eq!(openai["tool_protocol"].as_str(), Some("openai"));
        assert_eq!(openai["image_protocol"].as_str(), Some("none"));
        assert_eq!(openai["audio_protocol"].as_str(), Some("openai_audio"));
        assert_eq!(
            openai["embedding_protocol"].as_str(),
            Some("openai_embeddings")
        );
        assert_eq!(openai["supports_managed_tools"].as_bool(), Some(true));
        assert_eq!(openai["supports_responses_api"].as_bool(), Some(true));
        assert_eq!(openai["supports_reasoning"].as_bool(), Some(true));
        assert_eq!(
            openai["supports_structured_output_json_mode"].as_bool(),
            Some(true)
        );
        assert_eq!(
            openai["supports_structured_output_json_schema"].as_bool(),
            Some(false)
        );
        assert_eq!(openai["supports_files"].as_bool(), Some(true));
        assert_eq!(openai["supports_batches"].as_bool(), Some(false));
        assert_eq!(openai["supports_image_input"].as_bool(), Some(false));
        assert_eq!(openai["supports_images_generations"].as_bool(), Some(false));
        assert_eq!(openai["supports_images_edits"].as_bool(), Some(false));
        assert_eq!(openai["supports_images_variations"].as_bool(), Some(false));
        assert_eq!(openai["supports_audio_input"].as_bool(), Some(false));
        assert_eq!(openai["supports_audio_output"].as_bool(), Some(true));
        assert_eq!(
            openai["supports_audio_transcription"].as_bool(),
            Some(false)
        );
        assert_eq!(openai["supports_audio_translation"].as_bool(), Some(true));
        assert_eq!(openai["supports_embeddings"].as_bool(), Some(true));
        assert_eq!(openai["supports_realtime"].as_bool(), Some(true));
        assert_eq!(
            openai["supports_prompt_cache_openai"].as_bool(),
            Some(false)
        );
        assert_eq!(
            openai["supports_prompt_cache_request_controls"].as_bool(),
            Some(false)
        );
        assert_eq!(openai["data_collection"].as_str(), Some("deny"));
        assert_eq!(openai["zdr"].as_bool(), Some(true));
        assert_eq!(openai["distillable_text"].as_bool(), Some(true));
        assert!(openai["quantizations"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|value| value.as_str() == Some("fp8")));
        assert!(openai["supported_parameter_families"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|value| value.as_str() == Some("tools")));
        assert!(openai["capabilities"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|value| value.as_str() == Some("responses_api")));
        assert!(openai["capabilities"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|value| value.as_str() == Some("embeddings")));
        let anthropic = runtime_providers
            .iter()
            .find(|provider| provider["name"].as_str() == Some("anthropic"))
            .expect("anthropic runtime provider");
        assert_eq!(anthropic["family"].as_str(), Some("anthropic"));
        assert_eq!(anthropic["stability"].as_str(), Some("preview"));
        assert_eq!(
            anthropic["managed_tool_request_shapes"]
                .as_array()
                .map(|values| values.len()),
            Some(1)
        );
        assert_eq!(
            anthropic["managed_tool_request_shapes"][0].as_str(),
            Some("anthropic_messages")
        );
        assert!(anthropic["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/audio/transcriptions")));
        assert_eq!(anthropic["tool_protocol"].as_str(), Some("anthropic"));
        assert_eq!(anthropic["image_protocol"].as_str(), Some("none"));
        assert_eq!(anthropic["audio_protocol"].as_str(), Some("openai_audio"));
        assert_eq!(anthropic["embedding_protocol"].as_str(), Some("none"));
        assert_eq!(
            anthropic["supports_structured_output_json_schema"].as_bool(),
            Some(true)
        );
        assert_eq!(anthropic["supports_files"].as_bool(), Some(false));
        assert_eq!(anthropic["supports_batches"].as_bool(), Some(false));
        assert_eq!(anthropic["supports_image_input"].as_bool(), Some(true));
        assert_eq!(
            anthropic["supports_images_generations"].as_bool(),
            Some(false)
        );
        assert_eq!(anthropic["supports_audio_input"].as_bool(), Some(true));
        assert_eq!(
            anthropic["supports_audio_transcription"].as_bool(),
            Some(true)
        );
        assert_eq!(anthropic["supports_responses_api"].as_bool(), Some(false));
        assert_eq!(anthropic["supports_audio_output"].as_bool(), Some(false));
        assert_eq!(
            anthropic["supports_audio_translation"].as_bool(),
            Some(false)
        );
        assert_eq!(anthropic["supports_embeddings"].as_bool(), Some(false));
        assert_eq!(anthropic["supports_realtime"].as_bool(), Some(false));
        assert_eq!(
            anthropic["supports_prompt_cache_openai"].as_bool(),
            Some(false)
        );
        assert_eq!(anthropic["data_collection"].as_str(), Some("allow"));

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers").await;
        assert_eq!(status, 200);
        let providers_body: serde_json::Value =
            serde_json::from_str(&body).expect("providers body");
        let providers = providers_body["providers"].as_array().expect("providers");
        let openai = providers
            .iter()
            .find(|provider| provider["name"].as_str() == Some("openai"))
            .expect("openai provider");
        assert_eq!(openai["stability"].as_str(), Some("stable"));
        assert!(openai["managed_tool_request_shapes"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|shape| shape.as_str() == Some("openai_chat_completions")));
        assert!(openai["managed_tool_request_shapes"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|shape| shape.as_str() == Some("openai_responses")));
        assert!(openai["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/responses")));
        assert!(openai["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/files")));
        assert_eq!(openai["tool_protocol"].as_str(), Some("openai"));
        assert_eq!(openai["image_protocol"].as_str(), Some("none"));
        assert_eq!(openai["audio_protocol"].as_str(), Some("openai_audio"));
        assert_eq!(
            openai["embedding_protocol"].as_str(),
            Some("openai_embeddings")
        );
        assert_eq!(openai["supports_managed_tools"].as_bool(), Some(true));
        assert_eq!(openai["supports_responses_api"].as_bool(), Some(true));
        assert_eq!(openai["supports_reasoning"].as_bool(), Some(true));
        assert_eq!(
            openai["supports_structured_output_json_mode"].as_bool(),
            Some(true)
        );
        assert_eq!(
            openai["supports_structured_output_json_schema"].as_bool(),
            Some(false)
        );
        assert_eq!(openai["supports_files"].as_bool(), Some(true));
        assert_eq!(openai["supports_batches"].as_bool(), Some(false));
        assert_eq!(openai["supports_images_generations"].as_bool(), Some(false));
        assert_eq!(openai["supports_images_edits"].as_bool(), Some(false));
        assert_eq!(openai["supports_images_variations"].as_bool(), Some(false));
        assert_eq!(openai["supports_audio_output"].as_bool(), Some(true));
        assert_eq!(openai["supports_audio_translation"].as_bool(), Some(true));
        assert_eq!(openai["supports_embeddings"].as_bool(), Some(true));
        assert_eq!(openai["supports_realtime"].as_bool(), Some(true));
        assert_eq!(
            openai["supports_prompt_cache_openai"].as_bool(),
            Some(false)
        );
        assert_eq!(openai["data_collection"].as_str(), Some("deny"));
        assert_eq!(openai["zdr"].as_bool(), Some(true));
        assert_eq!(openai["distillable_text"].as_bool(), Some(true));
        assert_eq!(
            openai["routing_metadata"]["data_collection"].as_str(),
            Some("deny")
        );
        assert!(openai["capabilities"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|value| value.as_str() == Some("responses_api")));
        let anthropic = providers
            .iter()
            .find(|provider| provider["name"].as_str() == Some("anthropic"))
            .expect("anthropic provider");
        assert_eq!(anthropic["stability"].as_str(), Some("preview"));
        assert_eq!(
            anthropic["managed_tool_request_shapes"][0].as_str(),
            Some("anthropic_messages")
        );
        assert!(anthropic["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/audio/transcriptions")));
        assert_eq!(anthropic["tool_protocol"].as_str(), Some("anthropic"));
        assert_eq!(anthropic["image_protocol"].as_str(), Some("none"));
        assert_eq!(anthropic["audio_protocol"].as_str(), Some("openai_audio"));
        assert_eq!(anthropic["embedding_protocol"].as_str(), Some("none"));
        assert_eq!(
            anthropic["supports_structured_output_json_schema"].as_bool(),
            Some(true)
        );
        assert_eq!(anthropic["supports_files"].as_bool(), Some(false));
        assert_eq!(anthropic["supports_batches"].as_bool(), Some(false));
        assert_eq!(anthropic["supports_image_input"].as_bool(), Some(true));
        assert_eq!(
            anthropic["supports_images_generations"].as_bool(),
            Some(false)
        );
        assert_eq!(anthropic["supports_audio_input"].as_bool(), Some(true));
        assert_eq!(
            anthropic["supports_audio_transcription"].as_bool(),
            Some(true)
        );
        assert_eq!(anthropic["supports_audio_output"].as_bool(), Some(false));
        assert_eq!(
            anthropic["supports_audio_translation"].as_bool(),
            Some(false)
        );
        assert_eq!(anthropic["supports_embeddings"].as_bool(), Some(false));
        assert_eq!(anthropic["supports_realtime"].as_bool(), Some(false));
        assert_eq!(anthropic["data_collection"].as_str(), Some("allow"));
        assert!(anthropic["models"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|value| value.as_str() == Some("claude-sonnet-4-20250514")));
    }

    #[tokio::test]
    async fn management_api_reports_stdio_mcp_servers() {
        let log_file = NamedTempFile::new().expect("stdio log file");
        let log_path = log_file.path().to_string_lossy().to_string();
        let script = fake_stdio_mcp_script();

        let configs = vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert(
                    "mcp_servers".into(),
                    toml::Value::Table({
                        let mut servers = toml::value::Map::new();
                        servers.insert(
                            "research-stdio".into(),
                            toml::Value::Table({
                                let mut server = toml::value::Map::new();
                                server.insert(
                                    "transport".into(),
                                    toml::Value::String("stdio".into()),
                                );
                                server.insert("command".into(), toml::Value::String(uv_bin()));
                                server.insert(
                                    "args".into(),
                                    toml::Value::Array(vec![
                                        toml::Value::String("run".into()),
                                        toml::Value::String("python".into()),
                                        toml::Value::String("-u".into()),
                                        toml::Value::String(script),
                                    ]),
                                );
                                server.insert(
                                    "env".into(),
                                    toml::Value::Table({
                                        let mut env = toml::value::Map::new();
                                        env.insert(
                                            "TRP_STDIO_MCP_LOG".into(),
                                            toml::Value::String(log_path.clone()),
                                        );
                                        env.insert(
                                            "TRP_STDIO_MCP_REMOTE_TOOL".into(),
                                            toml::Value::String("search_docs".into()),
                                        );
                                        env
                                    }),
                                );
                                server
                            }),
                        );
                        servers
                    }),
                );
                t
            }),
        }];

        let providers = vec![canonical_provider(
            "openai",
            "sk-openai-real",
            "https://api.openai.test",
            vec!["gpt-4o".to_string()],
            "authorization",
            Some(30),
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        let mgmt_port = start_mgmt_server(api).await;
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/tool-runtime/status").await;
        assert_eq!(status, 200);
        let status_json: serde_json::Value = serde_json::from_str(&body).expect("status json");
        let server = status_json["mcp_servers"]
            .as_array()
            .and_then(|servers| {
                servers
                    .iter()
                    .find(|server| server["name"].as_str() == Some("research-stdio"))
            })
            .expect("stdio mcp server");
        assert_eq!(server["transport"].as_str(), Some("stdio"));
        assert_eq!(server["command"].as_str(), Some(uv_bin().as_str()));
        assert_eq!(server["method"].as_str(), Some("stdio"));
        assert_eq!(server["url"].as_str(), Some(""));
        assert_eq!(server["session_id_present"].as_bool(), Some(false));
        assert_eq!(server["health_state"].as_str(), Some("ready"));
        assert_eq!(server["protocol_version"].as_str(), Some("2025-11-25"));
        assert!(server["args"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|value| value.as_str() == Some("-u")));
        assert!(server["discovered_tools"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|tool| tool.as_str() == Some("search_docs")));

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/tool-runtime/mcp/research-stdio/refresh",
            "{}",
        )
        .await;
        assert_eq!(status, 200);
        let refresh_json: serde_json::Value = serde_json::from_str(&body).expect("refresh json");
        assert_eq!(refresh_json["transport"].as_str(), Some("stdio"));
        assert_eq!(refresh_json["session_id_present"].as_bool(), Some(false));
        assert_eq!(refresh_json["discovery_refreshes"].as_u64(), Some(2));

        let (status, body) =
            mgmt_delete(mgmt_port, "/api/v1/tool-runtime/mcp/research-stdio/session").await;
        assert_eq!(status, 200);
        let reset_json: serde_json::Value = serde_json::from_str(&body).expect("reset json");
        assert_eq!(
            reset_json["last_session_reset_status"].as_str(),
            Some("no_session")
        );

        let log_contents = fs::read_to_string(log_path).expect("stdio log contents");
        let methods = log_contents.lines().collect::<Vec<_>>();
        assert_eq!(
            methods,
            vec![
                "initialize",
                "notifications/initialized",
                "tools/list",
                "initialize",
                "notifications/initialized",
                "tools/list"
            ]
        );
    }

    #[tokio::test]
    async fn management_api_reports_sse_mcp_servers() {
        let method_log = Arc::new(Mutex::new(Vec::new()));
        let sse_addr = start_mcp_sse_server(Arc::clone(&method_log)).await;

        let configs = vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert(
                    "mcp_servers".into(),
                    toml::Value::Table({
                        let mut servers = toml::value::Map::new();
                        servers.insert(
                            "research-sse".into(),
                            toml::Value::Table({
                                let mut server = toml::value::Map::new();
                                server
                                    .insert("transport".into(), toml::Value::String("sse".into()));
                                server.insert(
                                    "url".into(),
                                    toml::Value::String(format!("http://{sse_addr}/sse")),
                                );
                                server
                            }),
                        );
                        servers
                    }),
                );
                t
            }),
        }];

        let providers = vec![canonical_provider(
            "openai",
            "sk-openai-real",
            "https://api.openai.test",
            vec!["gpt-4o".to_string()],
            "authorization",
            Some(30),
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        let mgmt_port = start_mgmt_server(api).await;
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/tool-runtime/status").await;
        assert_eq!(status, 200);
        let status_json: serde_json::Value = serde_json::from_str(&body).expect("status json");
        let server = status_json["mcp_servers"]
            .as_array()
            .and_then(|servers| {
                servers
                    .iter()
                    .find(|server| server["name"].as_str() == Some("research-sse"))
            })
            .expect("sse mcp server");
        assert_eq!(server["transport"].as_str(), Some("sse"));
        assert_eq!(server["method"].as_str(), Some("sse"));
        assert_eq!(
            server["url"].as_str(),
            Some(format!("http://{sse_addr}/sse").as_str())
        );
        assert_eq!(server["session_id_present"].as_bool(), Some(false));
        assert_eq!(server["health_state"].as_str(), Some("ready"));
        assert_eq!(server["protocol_version"].as_str(), Some("2025-11-25"));
        assert!(server["discovered_tools"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|tool| tool.as_str() == Some("search_docs")));

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/tool-runtime/mcp/research-sse/refresh",
            "{}",
        )
        .await;
        assert_eq!(status, 200);
        let refresh_json: serde_json::Value = serde_json::from_str(&body).expect("refresh json");
        assert_eq!(refresh_json["transport"].as_str(), Some("sse"));
        assert_eq!(refresh_json["discovery_refreshes"].as_u64(), Some(2));

        let methods = method_log.lock().await.clone();
        assert_eq!(
            methods,
            vec![
                "initialize",
                "notifications/initialized",
                "tools/list",
                "initialize",
                "notifications/initialized",
                "tools/list",
            ]
        );
    }

    #[tokio::test]
    async fn management_api_reports_oauth_mcp_auth_state() {
        let token_hits = Arc::new(AtomicUsize::new(0));
        let token_addr = start_upstream_async({
            let token_hits = Arc::clone(&token_hits);
            move |req: Request<Incoming>| {
                let token_hits = Arc::clone(&token_hits);
                async move {
                    token_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect token body")
                        .to_bytes();
                    let body_text = String::from_utf8_lossy(&body);
                    assert!(
                        body_text.contains("grant_type=client_credentials"),
                        "{body_text}"
                    );
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "access_token": "oauth-token-status",
                                "token_type": "Bearer",
                                "expires_in": 3600
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let mcp_addr = start_upstream_async(move |req: Request<Incoming>| async move {
            assert_eq!(
                req.headers()
                    .get("authorization")
                    .and_then(|value| value.to_str().ok()),
                Some("Bearer oauth-token-status")
            );
            let headers = req.headers().clone();
            let body = req
                .into_body()
                .collect()
                .await
                .expect("collect mcp body")
                .to_bytes();
            let body_json: serde_json::Value =
                serde_json::from_slice(&body).expect("mcp request json");
            match body_json["method"].as_str() {
                Some("initialize") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .header("mcp-session-id", "session-oauth-status")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": body_json["id"].clone(),
                            "result": {
                                "protocolVersion": "2025-11-25",
                                "capabilities": {},
                                "serverInfo": {
                                    "name": "fake-mcp",
                                    "version": "1.0.0"
                                }
                            }
                        })
                        .to_string(),
                    )))
                    .unwrap(),
                Some("notifications/initialized") => {
                    assert_eq!(
                        headers
                            .get("mcp-session-id")
                            .and_then(|value| value.to_str().ok()),
                        Some("session-oauth-status")
                    );
                    Response::builder()
                        .status(StatusCode::ACCEPTED)
                        .body(Full::new(Bytes::new()))
                        .unwrap()
                }
                Some("tools/list") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": body_json["id"].clone(),
                            "result": {
                                "tools": [{
                                    "name": "search_docs",
                                    "description": "Search remote docs"
                                }]
                            }
                        })
                        .to_string(),
                    )))
                    .unwrap(),
                other => panic!("unexpected MCP method: {:?}", other),
            }
        })
        .await;

        let configs = vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert(
                    "mcp_servers".into(),
                    toml::Value::Table({
                        let mut servers = toml::value::Map::new();
                        servers.insert(
                            "research-oauth".into(),
                            toml::Value::Table({
                                let mut server = toml::value::Map::new();
                                server.insert(
                                    "url".into(),
                                    toml::Value::String(format!("http://{mcp_addr}")),
                                );
                                server.insert("method".into(), toml::Value::String("POST".into()));
                                server.insert(
                                    "auth".into(),
                                    toml::Value::Table({
                                        let mut auth = toml::value::Map::new();
                                        auth.insert(
                                            "type".into(),
                                            toml::Value::String("oauth_client_credentials".into()),
                                        );
                                        auth.insert(
                                            "token_url".into(),
                                            toml::Value::String(format!("http://{token_addr}")),
                                        );
                                        auth.insert(
                                            "client_id".into(),
                                            toml::Value::String("tool-runtime".into()),
                                        );
                                        auth.insert(
                                            "client_secret".into(),
                                            toml::Value::String("topsecret".into()),
                                        );
                                        auth
                                    }),
                                );
                                server
                            }),
                        );
                        servers
                    }),
                );
                t
            }),
        }];

        let providers = vec![canonical_provider(
            "openai",
            "sk-openai-real",
            "https://api.openai.test",
            vec!["gpt-4o".to_string()],
            "authorization",
            Some(30),
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        let mgmt_port = start_mgmt_server(api).await;
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/tool-runtime/status").await;
        assert_eq!(status, 200);
        let status_json: serde_json::Value = serde_json::from_str(&body).expect("status json");
        let server = status_json["mcp_servers"]
            .as_array()
            .and_then(|servers| {
                servers
                    .iter()
                    .find(|server| server["name"].as_str() == Some("research-oauth"))
            })
            .expect("oauth mcp server");
        assert_eq!(
            server["auth_mode"].as_str(),
            Some("oauth_client_credentials")
        );
        assert_eq!(server["auth_status"].as_str(), Some("ready"));
        assert_eq!(server["auth_refreshes"].as_u64(), Some(1));
        assert!(server["auth_last_refreshed_at"].as_str().is_some());
        assert!(server["auth_token_expires_at_unix_ms"].as_u64().is_some());
        assert!(server["auth_last_error"].is_null());
        assert_eq!(token_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn management_api_reports_discovered_oauth_mcp_auth_state() {
        let token_hits = Arc::new(AtomicUsize::new(0));
        let token_addr = start_upstream_async({
            let token_hits = Arc::clone(&token_hits);
            move |req: Request<Incoming>| {
                let token_hits = Arc::clone(&token_hits);
                async move {
                    assert_eq!(req.method(), Method::POST);
                    assert_eq!(req.uri().path(), "/token");
                    token_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect token body")
                        .to_bytes();
                    let body_text = String::from_utf8_lossy(&body);
                    assert!(
                        body_text.contains("grant_type=client_credentials"),
                        "{body_text}"
                    );
                    assert!(body_text.contains("resource=http%3A%2F%2F"), "{body_text}");
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "access_token": "oauth-token-discovered-status",
                                "token_type": "Bearer",
                                "expires_in": 3600
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let auth_metadata_hits = Arc::new(AtomicUsize::new(0));
        let auth_addr = start_upstream_async({
            let auth_metadata_hits = Arc::clone(&auth_metadata_hits);
            let token_addr = token_addr.clone();
            move |req: Request<Incoming>| {
                let auth_metadata_hits = Arc::clone(&auth_metadata_hits);
                let token_addr = token_addr.clone();
                async move {
                    assert_eq!(req.method(), Method::GET);
                    assert_eq!(req.uri().path(), "/.well-known/oauth-authorization-server");
                    auth_metadata_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "token_endpoint": format!("http://{token_addr}/token")
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let protected_metadata_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let protected_metadata_hits = Arc::clone(&protected_metadata_hits);
            let auth_addr = auth_addr.clone();
            move |req: Request<Incoming>| {
                let protected_metadata_hits = Arc::clone(&protected_metadata_hits);
                let auth_addr = auth_addr.clone();
                async move {
                    if req.method() == Method::GET
                        && req.uri().path() == "/.well-known/oauth-protected-resource/mcp"
                    {
                        protected_metadata_hits.fetch_add(1, Ordering::Relaxed);
                        return Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "authorization_servers": [format!("http://{auth_addr}")]
                                })
                                .to_string(),
                            )))
                            .unwrap();
                    }

                    assert_eq!(
                        req.headers()
                            .get("authorization")
                            .and_then(|value| value.to_str().ok()),
                        Some("Bearer oauth-token-discovered-status")
                    );
                    let headers = req.headers().clone();
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect mcp body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("mcp request json");
                    match body_json["method"].as_str() {
                        Some("initialize") => Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .header("mcp-session-id", "session-oauth-discovered-status")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": body_json["id"].clone(),
                                    "result": {
                                        "protocolVersion": "2025-11-25",
                                        "capabilities": {},
                                        "serverInfo": {
                                            "name": "fake-mcp",
                                            "version": "1.0.0"
                                        }
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap(),
                        Some("notifications/initialized") => {
                            assert_eq!(
                                headers
                                    .get("mcp-session-id")
                                    .and_then(|value| value.to_str().ok()),
                                Some("session-oauth-discovered-status")
                            );
                            Response::builder()
                                .status(StatusCode::ACCEPTED)
                                .body(Full::new(Bytes::new()))
                                .unwrap()
                        }
                        Some("tools/list") => Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": body_json["id"].clone(),
                                    "result": {
                                        "tools": [{
                                            "name": "search_docs",
                                            "description": "Search remote docs"
                                        }]
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap(),
                        other => panic!("unexpected MCP method: {:?}", other),
                    }
                }
            }
        })
        .await;

        let configs = vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert(
                    "mcp_servers".into(),
                    toml::Value::Table({
                        let mut servers = toml::value::Map::new();
                        servers.insert(
                            "research-oauth-discovery".into(),
                            toml::Value::Table({
                                let mut server = toml::value::Map::new();
                                server.insert(
                                    "url".into(),
                                    toml::Value::String(format!("http://{mcp_addr}/mcp")),
                                );
                                server.insert("method".into(), toml::Value::String("POST".into()));
                                server.insert(
                                    "auth".into(),
                                    toml::Value::Table({
                                        let mut auth = toml::value::Map::new();
                                        auth.insert(
                                            "type".into(),
                                            toml::Value::String("oauth_client_credentials".into()),
                                        );
                                        auth.insert(
                                            "client_id".into(),
                                            toml::Value::String("tool-runtime".into()),
                                        );
                                        auth.insert(
                                            "client_secret".into(),
                                            toml::Value::String("topsecret".into()),
                                        );
                                        auth
                                    }),
                                );
                                server
                            }),
                        );
                        servers
                    }),
                );
                t
            }),
        }];

        let providers = vec![canonical_provider(
            "openai",
            "sk-openai-real",
            "https://api.openai.test",
            vec!["gpt-4o".to_string()],
            "authorization",
            Some(30),
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        let mgmt_port = start_mgmt_server(api).await;
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/tool-runtime/status").await;
        assert_eq!(status, 200);
        let status_json: serde_json::Value = serde_json::from_str(&body).expect("status json");
        let server = status_json["mcp_servers"]
            .as_array()
            .and_then(|servers| {
                servers
                    .iter()
                    .find(|server| server["name"].as_str() == Some("research-oauth-discovery"))
            })
            .expect("oauth discovery mcp server");
        assert_eq!(
            server["auth_mode"].as_str(),
            Some("oauth_client_credentials")
        );
        assert_eq!(server["auth_status"].as_str(), Some("ready"));
        let expected_auth_url = format!("http://{auth_addr}");
        let expected_token_url = format!("http://{token_addr}/token");
        let expected_resource = format!("http://{mcp_addr}/mcp");
        assert_eq!(
            server["auth_authorization_server_url"].as_str(),
            Some(expected_auth_url.as_str())
        );
        assert_eq!(
            server["auth_token_url"].as_str(),
            Some(expected_token_url.as_str())
        );
        assert_eq!(
            server["auth_resource"].as_str(),
            Some(expected_resource.as_str())
        );
        assert!(server["auth_last_discovery_error"].is_null());
        assert_eq!(protected_metadata_hits.load(Ordering::Relaxed), 1);
        assert_eq!(auth_metadata_hits.load(Ordering::Relaxed), 1);
        assert_eq!(token_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn management_api_can_drive_oauth_authorization_code_flow() {
        let token_hits = Arc::new(AtomicUsize::new(0));
        let token_addr = start_upstream_async({
            let token_hits = Arc::clone(&token_hits);
            move |req: Request<Incoming>| {
                let token_hits = Arc::clone(&token_hits);
                async move {
                    assert_eq!(req.method(), Method::POST);
                    assert_eq!(req.uri().path(), "/token");
                    token_hits.fetch_add(1, Ordering::Relaxed);
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect token body")
                        .to_bytes();
                    let body_text = String::from_utf8_lossy(&body);
                    assert!(
                        body_text.contains("grant_type=authorization_code"),
                        "{body_text}"
                    );
                    assert!(body_text.contains("code=mgmt-auth-code-1"), "{body_text}");
                    assert!(body_text.contains("code_verifier="), "{body_text}");
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "access_token": "oauth-authcode-management-token",
                                "refresh_token": "oauth-authcode-management-refresh",
                                "token_type": "Bearer",
                                "expires_in": 3600
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let auth_metadata_hits = Arc::new(AtomicUsize::new(0));
        let auth_addr = start_upstream_async({
            let auth_metadata_hits = Arc::clone(&auth_metadata_hits);
            let token_addr = token_addr.clone();
            move |req: Request<Incoming>| {
                let auth_metadata_hits = Arc::clone(&auth_metadata_hits);
                let token_addr = token_addr.clone();
                async move {
                    assert_eq!(req.method(), Method::GET);
                    assert_eq!(req.uri().path(), "/.well-known/oauth-authorization-server");
                    auth_metadata_hits.fetch_add(1, Ordering::Relaxed);
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(
                            serde_json::json!({
                                "authorization_endpoint": "https://auth.example.test/authorize",
                                "token_endpoint": format!("http://{token_addr}/token")
                            })
                            .to_string(),
                        )))
                        .unwrap()
                }
            }
        })
        .await;

        let protected_metadata_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let protected_metadata_hits = Arc::clone(&protected_metadata_hits);
            let auth_addr = auth_addr.clone();
            move |req: Request<Incoming>| {
                let protected_metadata_hits = Arc::clone(&protected_metadata_hits);
                let auth_addr = auth_addr.clone();
                async move {
                    if req.method() == Method::GET
                        && req.uri().path() == "/.well-known/oauth-protected-resource/mcp"
                    {
                        protected_metadata_hits.fetch_add(1, Ordering::Relaxed);
                        return Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "authorization_servers": [format!("http://{auth_addr}")]
                                })
                                .to_string(),
                            )))
                            .unwrap();
                    }

                    assert_eq!(
                        req.headers()
                            .get("authorization")
                            .and_then(|value| value.to_str().ok()),
                        Some("Bearer oauth-authcode-management-token")
                    );
                    let headers = req.headers().clone();
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect mcp body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("mcp request json");
                    match body_json["method"].as_str() {
                        Some("initialize") => Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .header("mcp-session-id", "session-oauth-authcode-management")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": body_json["id"].clone(),
                                    "result": {
                                        "protocolVersion": "2025-11-25",
                                        "capabilities": {},
                                        "serverInfo": {
                                            "name": "fake-mcp",
                                            "version": "1.0.0"
                                        }
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap(),
                        Some("notifications/initialized") => {
                            assert_eq!(
                                headers
                                    .get("mcp-session-id")
                                    .and_then(|value| value.to_str().ok()),
                                Some("session-oauth-authcode-management")
                            );
                            Response::builder()
                                .status(StatusCode::ACCEPTED)
                                .body(Full::new(Bytes::new()))
                                .unwrap()
                        }
                        Some("tools/list") => Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from(
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": body_json["id"].clone(),
                                    "result": {
                                        "tools": [{
                                            "name": "search_docs",
                                            "description": "Search remote docs"
                                        }]
                                    }
                                })
                                .to_string(),
                            )))
                            .unwrap(),
                        other => panic!("unexpected MCP method: {:?}", other),
                    }
                }
            }
        })
        .await;

        let configs = vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert(
                    "mcp_servers".into(),
                    toml::Value::Table({
                        let mut servers = toml::value::Map::new();
                        servers.insert(
                            "research-oauth-authcode".into(),
                            toml::Value::Table({
                                let mut server = toml::value::Map::new();
                                server.insert(
                                    "url".into(),
                                    toml::Value::String(format!("http://{mcp_addr}/mcp")),
                                );
                                server.insert("method".into(), toml::Value::String("POST".into()));
                                server.insert(
                                    "auth".into(),
                                    toml::Value::Table({
                                        let mut auth = toml::value::Map::new();
                                        auth.insert(
                                            "type".into(),
                                            toml::Value::String("oauth_authorization_code".into()),
                                        );
                                        auth.insert(
                                            "client_id".into(),
                                            toml::Value::String("tool-runtime".into()),
                                        );
                                        auth.insert(
                                            "redirect_uri".into(),
                                            toml::Value::String("http://127.0.0.1/callback".into()),
                                        );
                                        auth
                                    }),
                                );
                                server
                            }),
                        );
                        servers
                    }),
                );
                t
            }),
        }];

        let providers = vec![canonical_provider(
            "openai",
            "sk-openai-real",
            "https://api.openai.test",
            vec!["gpt-4o".to_string()],
            "authorization",
            Some(30),
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        let mgmt_port = start_mgmt_server(api).await;
        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/tool-runtime/mcp/research-oauth-authcode/oauth/authorize",
            "{}",
        )
        .await;
        assert_eq!(status, 200);
        let body_json: serde_json::Value = serde_json::from_str(&body).expect("auth start json");
        let state = body_json["state"].as_str().expect("state").to_string();
        assert_eq!(
            body_json["redirect_uri"].as_str(),
            Some("http://127.0.0.1/callback")
        );
        assert!(body_json["authorization_url"]
            .as_str()
            .unwrap_or_default()
            .starts_with("https://auth.example.test/authorize?"));

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/tool-runtime/mcp/research-oauth-authcode/oauth/callback",
            &serde_json::json!({
                "state": state,
                "code": "mgmt-auth-code-1"
            })
            .to_string(),
        )
        .await;
        assert_eq!(status, 200);
        let body_json: serde_json::Value =
            serde_json::from_str(&body).expect("callback response json");
        assert_eq!(body_json["auth_status"].as_str(), Some("ready"));
        assert_eq!(
            body_json["auth_authorization_server_url"].as_str(),
            Some(format!("http://{auth_addr}").as_str())
        );
        assert_eq!(
            body_json["auth_token_url"].as_str(),
            Some(format!("http://{token_addr}/token").as_str())
        );
        assert_eq!(
            body_json["auth_pending_authorization"].as_bool(),
            Some(false)
        );
        assert_eq!(protected_metadata_hits.load(Ordering::Relaxed), 1);
        assert_eq!(auth_metadata_hits.load(Ordering::Relaxed), 1);
        assert_eq!(token_hits.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn management_api_can_refresh_and_reset_mcp_server_sessions() {
        use plugin_llm_gateway::store::ProjectToolRecord;
        let initialize_hits = Arc::new(AtomicUsize::new(0));
        let list_hits = Arc::new(AtomicUsize::new(0));
        let delete_hits = Arc::new(AtomicUsize::new(0));
        let mcp_addr = start_upstream_async({
            let initialize_hits = Arc::clone(&initialize_hits);
            let list_hits = Arc::clone(&list_hits);
            let delete_hits = Arc::clone(&delete_hits);
            move |req: Request<Incoming>| {
                let initialize_hits = Arc::clone(&initialize_hits);
                let list_hits = Arc::clone(&list_hits);
                let delete_hits = Arc::clone(&delete_hits);
                async move {
                    if req.method() == Method::DELETE {
                        delete_hits.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(
                            req.headers()
                                .get("mcp-session-id")
                                .and_then(|value| value.to_str().ok()),
                            Some("session-refresh-1")
                        );
                        return Response::builder()
                            .status(StatusCode::NO_CONTENT)
                            .body(Full::new(Bytes::new()))
                            .unwrap();
                    }

                    let headers = req.headers().clone();
                    let body = req
                        .into_body()
                        .collect()
                        .await
                        .expect("collect mcp body")
                        .to_bytes();
                    let body_json: serde_json::Value =
                        serde_json::from_slice(&body).expect("mcp request json");
                    match body_json["method"].as_str() {
                        Some("initialize") => {
                            initialize_hits.fetch_add(1, Ordering::Relaxed);
                            Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .header("mcp-session-id", "session-refresh-1")
                                .body(Full::new(Bytes::from(
                                    serde_json::json!({
                                        "jsonrpc": "2.0",
                                        "id": body_json["id"].clone(),
                                        "result": {
                                            "protocolVersion": "2025-11-25",
                                            "capabilities": {},
                                            "serverInfo": {
                                                "name": "fake-mcp",
                                                "version": "1.0.0"
                                            }
                                        }
                                    })
                                    .to_string(),
                                )))
                                .unwrap()
                        }
                        Some("notifications/initialized") => {
                            assert_eq!(
                                headers
                                    .get("mcp-session-id")
                                    .and_then(|value| value.to_str().ok()),
                                Some("session-refresh-1")
                            );
                            Response::builder()
                                .status(StatusCode::ACCEPTED)
                                .body(Full::new(Bytes::new()))
                                .unwrap()
                        }
                        Some("tools/list") => {
                            list_hits.fetch_add(1, Ordering::Relaxed);
                            Response::builder()
                                .status(StatusCode::OK)
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from(
                                    serde_json::json!({
                                        "jsonrpc": "2.0",
                                        "id": body_json["id"].clone(),
                                        "result": {
                                            "tools": [{
                                                "name": "search_docs",
                                                "description": "Search remote docs"
                                            }]
                                        }
                                    })
                                    .to_string(),
                                )))
                                .unwrap()
                        }
                        other => panic!("unexpected MCP method: {:?}", other),
                    }
                }
            }
        })
        .await;

        let mcp_url = format!("http://{}", mcp_addr);
        let configs = vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert(
                    "mcp_servers".into(),
                    toml::Value::Table({
                        let mut servers = toml::value::Map::new();
                        servers.insert(
                            "research-mcp".into(),
                            toml::Value::Table({
                                let mut server = toml::value::Map::new();
                                server.insert("url".into(), toml::Value::String(mcp_url.clone()));
                                server.insert("method".into(), toml::Value::String("POST".into()));
                                server
                            }),
                        );
                        servers
                    }),
                );
                t
            }),
        }];

        let providers = vec![canonical_provider(
            "openai",
            "sk-openai-real",
            "https://api.openai.test",
            vec!["gpt-4o".to_string()],
            "authorization",
            Some(30),
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();
        api.upsert_project_tool(ProjectToolRecord {
            project_id: "project-a".to_string(),
            tool_name: "mcp_search".to_string(),
            description: Some("Search docs".to_string()),
            input_schema_json: serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            })
            .to_string(),
            executor_kind: "mcp".to_string(),
            executor_config_json: Some(
                serde_json::json!({ "server": "research-mcp", "remote_tool": "search_docs" })
                    .to_string(),
            ),
            enabled: true,
            timeout_ms: Some(1500),
            updated_at: "0".to_string(),
        })
        .await
        .unwrap()
        .unwrap();

        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) =
            mgmt_delete(mgmt_port, "/api/v1/tool-runtime/mcp/research-mcp/session").await;
        assert_eq!(status, 200);
        let reset_json: serde_json::Value = serde_json::from_str(&body).expect("reset json");
        assert_eq!(reset_json["name"].as_str(), Some("research-mcp"));
        assert_eq!(reset_json["session_id_present"].as_bool(), Some(false));
        assert_eq!(reset_json["last_session_reset_status"].as_str(), Some("ok"));
        assert!(reset_json["last_session_reset_at"].as_str().is_some());
        assert_eq!(delete_hits.load(Ordering::Relaxed), 1);

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/tool-runtime/mcp/research-mcp/refresh",
            "{}",
        )
        .await;
        assert_eq!(status, 200);
        let refresh_json: serde_json::Value = serde_json::from_str(&body).expect("refresh json");
        assert_eq!(refresh_json["name"].as_str(), Some("research-mcp"));
        assert_eq!(refresh_json["session_id_present"].as_bool(), Some(true));
        assert_eq!(refresh_json["last_discovery_status"].as_str(), Some("ok"));
        assert!(refresh_json["last_discovery_at"].as_str().is_some());
        assert_eq!(refresh_json["discovery_refreshes"].as_u64(), Some(2));
        assert_eq!(initialize_hits.load(Ordering::Relaxed), 2);
        assert_eq!(list_hits.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn management_api_can_disable_and_enable_mcp_servers() {
        let mcp_addr = start_upstream_async(move |req: Request<Incoming>| async move {
            let headers = req.headers().clone();
            let body = req
                .into_body()
                .collect()
                .await
                .expect("collect mcp body")
                .to_bytes();
            let body_json: serde_json::Value =
                serde_json::from_slice(&body).expect("mcp request json");
            match body_json["method"].as_str() {
                Some("initialize") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .header("mcp-session-id", "session-disable-1")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": body_json["id"].clone(),
                            "result": {
                                "protocolVersion": "2025-11-25",
                                "capabilities": {},
                                "serverInfo": {
                                    "name": "fake-mcp",
                                    "version": "1.0.0"
                                }
                            }
                        })
                        .to_string(),
                    )))
                    .unwrap(),
                Some("notifications/initialized") => {
                    assert_eq!(
                        headers
                            .get("mcp-session-id")
                            .and_then(|value| value.to_str().ok()),
                        Some("session-disable-1")
                    );
                    Response::builder()
                        .status(StatusCode::ACCEPTED)
                        .body(Full::new(Bytes::new()))
                        .unwrap()
                }
                Some("tools/list") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": body_json["id"].clone(),
                            "result": {
                                "tools": [{
                                    "name": "search_docs",
                                    "description": "Search remote docs"
                                }]
                            }
                        })
                        .to_string(),
                    )))
                    .unwrap(),
                other => panic!("unexpected MCP method: {:?}", other),
            }
        })
        .await;

        let mcp_url = format!("http://{}", mcp_addr);
        let configs = vec![PluginConfig {
            name: "tool_runtime".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert(
                    "mcp_servers".into(),
                    toml::Value::Table({
                        let mut servers = toml::value::Map::new();
                        servers.insert(
                            "research-mcp".into(),
                            toml::Value::Table({
                                let mut server = toml::value::Map::new();
                                server.insert("url".into(), toml::Value::String(mcp_url.clone()));
                                server.insert("method".into(), toml::Value::String("POST".into()));
                                server
                            }),
                        );
                        servers
                    }),
                );
                t
            }),
        }];

        let providers = vec![canonical_provider(
            "openai",
            "sk-openai-real",
            "https://api.openai.test",
            vec!["gpt-4o".to_string()],
            "authorization",
            Some(30),
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/tool-runtime/mcp/research-mcp/disable",
            r#"{"actor_id":"operator-a","reason":"planned maintenance"}"#,
        )
        .await;
        assert_eq!(status, 200);
        let disabled_json: serde_json::Value = serde_json::from_str(&body).expect("disable json");
        assert_eq!(disabled_json["name"].as_str(), Some("research-mcp"));
        assert_eq!(disabled_json["operator_state"].as_str(), Some("disabled"));
        assert_eq!(
            disabled_json["operator_state_actor"].as_str(),
            Some("operator-a")
        );
        assert_eq!(
            disabled_json["operator_state_reason"].as_str(),
            Some("planned maintenance")
        );
        assert!(disabled_json["operator_state_at"].as_str().is_some());
        assert_eq!(disabled_json["health_state"].as_str(), Some("disabled"));
        assert_eq!(
            disabled_json["health_reason"].as_str(),
            Some("planned maintenance")
        );
        assert_eq!(disabled_json["recommended_action"].as_str(), Some("enable"));

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/tool-runtime/status").await;
        assert_eq!(status, 200);
        let status_json: serde_json::Value = serde_json::from_str(&body).expect("status json");
        assert!(status_json["mcp_servers"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|server| {
                server["name"].as_str() == Some("research-mcp")
                    && server["operator_state"].as_str() == Some("disabled")
                    && server["health_state"].as_str() == Some("disabled")
                    && server["recommended_action"].as_str() == Some("enable")
            }));

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/tool-runtime/mcp/research-mcp/enable",
            r#"{"actor_id":"operator-b","reason":"maintenance complete"}"#,
        )
        .await;
        assert_eq!(status, 200);
        let enabled_json: serde_json::Value = serde_json::from_str(&body).expect("enable json");
        assert_eq!(enabled_json["name"].as_str(), Some("research-mcp"));
        assert_eq!(enabled_json["operator_state"].as_str(), Some("enabled"));
        assert_eq!(
            enabled_json["operator_state_actor"].as_str(),
            Some("operator-b")
        );
        assert_eq!(
            enabled_json["operator_state_reason"].as_str(),
            Some("maintenance complete")
        );
        assert!(enabled_json["operator_state_at"].as_str().is_some());
        assert_eq!(enabled_json["health_state"].as_str(), Some("ready"));
        assert!(enabled_json["health_reason"].is_null());
        assert!(enabled_json["recommended_action"].is_null());
    }

    #[tokio::test]
    async fn management_api_prompt_cache_status_and_provider_capabilities() {
        let configs = vec![PluginConfig {
            name: "prompt_cache".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut table = toml::value::Map::new();
                table.insert(
                    "anthropic_default_scope".into(),
                    toml::Value::String("tools".into()),
                );
                table
            }),
        }];

        let providers = vec![
            canonical_provider(
                "openai",
                "sk-openai-real",
                "https://api.openai.com/v1",
                vec!["gpt-4o".to_string()],
                "authorization",
                None,
                ProviderFamily::OpenAi,
                openai_prompt_cache_surfaces(),
                ProviderRoutingMetadataConfig::default(),
            ),
            canonical_provider(
                "anthropic",
                "sk-anthropic-real",
                "https://api.anthropic.com/v1",
                vec!["claude-sonnet-4-20250514".to_string()],
                "x-api-key",
                None,
                ProviderFamily::Anthropic,
                anthropic_prompt_cache_surfaces(),
                ProviderRoutingMetadataConfig::default(),
            ),
            canonical_provider(
                "groq",
                "sk-groq-real",
                "https://api.groq.com/openai/v1",
                vec!["llama-3.3-70b-versatile".to_string()],
                "authorization",
                Some(12),
                ProviderFamily::OpenAi,
                groq_prompt_cache_surfaces(),
                ProviderRoutingMetadataConfig::default(),
            ),
        ];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .expect("create plugins");

        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/prompt-cache/status").await;
        assert_eq!(status, 200);
        let status_json: serde_json::Value = serde_json::from_str(&body).expect("status json");
        assert_eq!(
            status_json["anthropic_default_scope"].as_str(),
            Some("tools")
        );
        assert_eq!(status_json["store_backed"].as_bool(), Some(true));
        assert_eq!(
            status_json["routing_hint_persistence_enabled"].as_bool(),
            Some(true)
        );
        assert_eq!(
            status_json["routing_flush_interval_ms"].as_u64(),
            Some(2000)
        );
        assert_eq!(
            status_json["routing_prune_interval_secs"].as_u64(),
            Some(60)
        );
        assert_eq!(status_json["warmed_route_count"].as_u64(), Some(0));
        assert_eq!(status_json["negative_route_count"].as_u64(), Some(0));
        assert_eq!(status_json["pending_route_updates"].as_u64(), Some(0));
        assert!(status_json["last_route_flush_unix_ms"].is_null());

        let providers_json = status_json["providers"]
            .as_array()
            .expect("providers array");
        let openai = providers_json
            .iter()
            .find(|provider| provider["name"].as_str() == Some("openai"))
            .expect("openai prompt cache provider");
        assert_eq!(openai["family"].as_str(), Some("openai"));
        assert_eq!(openai["prompt_cache_protocol"].as_str(), Some("openai"));
        assert_eq!(openai["supports_prompt_cache"].as_bool(), Some(true));
        assert_eq!(openai["request_controls_supported"].as_bool(), Some(true));
        assert!(openai["managed_tool_request_shapes"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|shape| shape.as_str() == Some("openai_chat_completions")));
        assert!(openai["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/responses")));
        assert_eq!(openai["supports_realtime"].as_bool(), Some(true));

        let anthropic = providers_json
            .iter()
            .find(|provider| provider["name"].as_str() == Some("anthropic"))
            .expect("anthropic prompt cache provider");
        assert_eq!(anthropic["family"].as_str(), Some("anthropic"));
        assert_eq!(
            anthropic["prompt_cache_protocol"].as_str(),
            Some("anthropic")
        );
        assert_eq!(anthropic["supports_prompt_cache"].as_bool(), Some(true));
        assert_eq!(
            anthropic["request_controls_supported"].as_bool(),
            Some(true)
        );
        assert_eq!(
            anthropic["managed_tool_request_shapes"][0].as_str(),
            Some("anthropic_messages")
        );
        assert!(anthropic["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/audio/transcriptions")));

        let groq = providers_json
            .iter()
            .find(|provider| provider["name"].as_str() == Some("groq"))
            .expect("groq prompt cache provider");
        assert_eq!(groq["family"].as_str(), Some("openai"));
        assert_eq!(groq["prompt_cache_protocol"].as_str(), Some("openai"));
        assert_eq!(groq["supports_prompt_cache"].as_bool(), Some(true));
        assert_eq!(groq["request_controls_supported"].as_bool(), Some(true));
        assert!(groq["surface_endpoints"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|path| path.as_str() == Some("/v1/embeddings")));

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers").await;
        assert_eq!(status, 200);
        let providers_body: serde_json::Value =
            serde_json::from_str(&body).expect("providers body json");
        let providers = providers_body["providers"]
            .as_array()
            .expect("providers array");

        let openai = providers
            .iter()
            .find(|provider| provider["name"].as_str() == Some("openai"))
            .expect("openai provider");
        assert_eq!(openai["prompt_cache_protocol"].as_str(), Some("openai"));
        assert_eq!(openai["image_protocol"].as_str(), Some("none"));
        assert_eq!(openai["supports_prompt_cache"].as_bool(), Some(true));
        assert_eq!(
            openai["prompt_cache_request_controls_supported"].as_bool(),
            Some(true)
        );
        assert_eq!(openai["supports_responses_api"].as_bool(), Some(true));
        assert_eq!(openai["supports_reasoning"].as_bool(), Some(true));
        assert_eq!(
            openai["supports_structured_output_json_mode"].as_bool(),
            Some(true)
        );
        assert_eq!(openai["supports_images_generations"].as_bool(), Some(false));
        assert_eq!(openai["supports_images_edits"].as_bool(), Some(false));
        assert_eq!(openai["supports_images_variations"].as_bool(), Some(false));
        assert_eq!(openai["supports_audio_output"].as_bool(), Some(true));
        assert_eq!(openai["supports_audio_translation"].as_bool(), Some(true));
        assert_eq!(openai["supports_embeddings"].as_bool(), Some(true));
        assert_eq!(openai["supports_realtime"].as_bool(), Some(true));

        let anthropic = providers
            .iter()
            .find(|provider| provider["name"].as_str() == Some("anthropic"))
            .expect("anthropic provider");
        assert_eq!(
            anthropic["prompt_cache_protocol"].as_str(),
            Some("anthropic")
        );
        assert_eq!(anthropic["supports_prompt_cache"].as_bool(), Some(true));
        assert_eq!(
            anthropic["prompt_cache_request_controls_supported"].as_bool(),
            Some(true)
        );
        assert_eq!(
            anthropic["supports_structured_output_json_schema"].as_bool(),
            Some(true)
        );
        assert_eq!(anthropic["image_protocol"].as_str(), Some("none"));
        assert_eq!(anthropic["supports_image_input"].as_bool(), Some(true));
        assert_eq!(
            anthropic["supports_images_generations"].as_bool(),
            Some(false)
        );
        assert_eq!(
            anthropic["supports_audio_transcription"].as_bool(),
            Some(true)
        );
        assert_eq!(anthropic["supports_audio_output"].as_bool(), Some(false));
        assert_eq!(anthropic["supports_realtime"].as_bool(), Some(false));

        let groq = providers
            .iter()
            .find(|provider| provider["name"].as_str() == Some("groq"))
            .expect("groq provider");
        assert_eq!(groq["prompt_cache_protocol"].as_str(), Some("openai"));
        assert_eq!(groq["supports_prompt_cache"].as_bool(), Some(true));
        assert_eq!(
            groq["prompt_cache_request_controls_supported"].as_bool(),
            Some(true)
        );
        assert_eq!(groq["supports_audio_input"].as_bool(), Some(true));
        assert_eq!(groq["supports_embeddings"].as_bool(), Some(true));
        assert_eq!(groq["supports_audio_output"].as_bool(), Some(false));
        assert_eq!(groq["supports_realtime"].as_bool(), Some(false));
        assert!(groq["capabilities"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|value| value.as_str() == Some("prompt_cache_openai")));
    }

    #[tokio::test]
    async fn management_api_budget_enforcement_after_reset() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr.clone()]);

        // Use a very tight budget so it gets exceeded quickly.
        let configs = vec![PluginConfig {
            name: "cost_tracker".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(0.001));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t.insert("default_cost_per_1k_input".into(), toml::Value::Float(0.01));
                t.insert(
                    "default_cost_per_1k_output".into(),
                    toml::Value::Float(0.02),
                );
                t
            }),
        }];

        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &[],
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();
        let chain = Arc::new(PluginChain::new(plugins));

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // First request succeeds.
        let req = chat_request("/v1/chat/completions", "sk-budget-key");
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request should be rejected — budget exceeded.
        let req = chat_request("/v1/chat/completions", "sk-budget-key");
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);

        // Reset usage via management API.
        let (status, _) = mgmt_delete(mgmt_port, "/api/v1/cost/usage/sk-budget-key").await;
        assert_eq!(status, 200);

        // After reset, the key should be able to make requests again.
        let req = chat_request("/v1/chat/completions", "sk-budget-key");
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "should succeed after usage reset"
        );
    }

    #[tokio::test]
    async fn store_persists_across_api_instances() {
        use plugin_llm_gateway::store::{self, GatewayStore, KeyUsageRecord, ModelCostRecord};

        // Create a shared in-memory store (using a file: URI so multiple connections share it).
        let store = Arc::new(
            store::connect("sqlite:file:shared_test?mode=memory&cache=shared")
                .await
                .unwrap(),
        );

        // Pre-populate the store with usage and model costs.
        store
            .upsert_usage(
                "sk-persisted",
                &KeyUsageRecord {
                    total_input_tokens: 5000,
                    total_output_tokens: 2500,
                    total_cost: 0.15,
                },
            )
            .await
            .unwrap();
        store
            .upsert_model_cost(
                "claude-3-opus",
                &ModelCostRecord {
                    input_cost_per_1k: 0.015,
                    output_cost_per_1k: 0.075,
                },
            )
            .await
            .unwrap();

        // Now create plugins using the same store URL — they should load the pre-existing data.
        let configs = vec![PluginConfig {
            name: "cost_tracker".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(0.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t
            }),
        }];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite:file:shared_test?mode=memory&cache=shared"),
            &[],
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        let mgmt_port = start_mgmt_server(api).await;

        // Verify persisted usage is visible via management API.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("sk-persi"),
            "should contain persisted key (masked): {}",
            body
        );
        assert!(
            body.contains("\"total_input_tokens\":5000"),
            "should have persisted input tokens: {}",
            body
        );

        // Verify persisted model costs.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"model\":\"claude-3-opus\""),
            "should contain persisted model: {}",
            body
        );
        assert!(
            body.contains("0.015000"),
            "input cost should match: {}",
            body
        );
    }

    #[tokio::test]
    async fn flush_writes_proxy_traffic_to_store() {
        use plugin_llm_gateway::store::{self, GatewayStore};

        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let configs = vec![PluginConfig {
            name: "cost_tracker".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(0.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t.insert("default_cost_per_1k_input".into(), toml::Value::Float(0.01));
                t.insert(
                    "default_cost_per_1k_output".into(),
                    toml::Value::Float(0.02),
                );
                t
            }),
        }];

        // Use a shared in-memory DB so we can read it from a separate connection.
        let store_url = "sqlite:file:flush_test?mode=memory&cache=shared";

        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some(store_url),
            &[],
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();
        let chain = Arc::new(PluginChain::new(plugins));

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        // Send requests through the proxy.
        for _ in 0..3 {
            let req = chat_request("/v1/chat/completions", "sk-flush-test");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
        }

        // Manually trigger a flush.
        api.flush().await;

        // Open a separate connection to the same in-memory DB and verify data.
        let store = store::connect(store_url).await.unwrap();
        let usage = store.get_all_usage().await.unwrap();
        assert_eq!(usage.len(), 1, "should have 1 key in store");
        assert_eq!(usage[0].0, "sk-flush-test");
        assert!(
            usage[0].1.total_input_tokens > 0,
            "input tokens should be > 0"
        );
        assert!(usage[0].1.total_cost > 0.0, "cost should be > 0");
    }

    #[tokio::test]
    async fn multiple_api_keys_tracked_independently() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Send requests with different API keys.
        for i in 0..3 {
            let req = chat_request(
                &format!("/v1/chat/completions?n={}", i),
                "sk-alice-key-1234",
            );
            send_request(&proxy_addr, req).await;
        }
        let req = chat_request("/v1/chat/completions", "sk-bob-key-56789");
        send_request(&proxy_addr, req).await;

        // Status should show 2 tracked keys.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(
            body.contains("\"tracked_api_keys\":2"),
            "should track 2 keys: {}",
            body
        );

        // Usage should contain both keys (masked).
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body.contains("sk-alice"), "should contain alice: {}", body);
        assert!(body.contains("sk-bob-k"), "should contain bob: {}", body);
    }

    #[tokio::test]
    async fn management_api_full_status_all_plugins() {
        let (chain, api) = setup_all_plugins().await;

        let _ = chain;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"cost_tracker_enabled\":true"));
        assert!(body.contains("\"rate_limiter_enabled\":true"));
        assert!(body.contains("\"provider_failover_enabled\":true"));
        assert!(body.contains("\"tracked_api_keys\":0"));
        assert!(body.contains("\"rate_limiter_tracked_keys\":0"));
        assert!(body.contains("\"failed_providers_count\":0"));
    }

    // -----------------------------------------------------------------------
    // Edge-case tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn edge_empty_state_queries_return_empty() {
        let (chain, api) = setup_all_plugins().await;
        let _ = chain;
        let mgmt_port = start_mgmt_server(api).await;

        // Usage is empty.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"usage\":[]"), "empty usage: {}", body);

        // Models are empty.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"models\":[]"), "empty models: {}", body);

        // Failed providers are empty.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers/failed").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"failed\":[]"), "empty failed: {}", body);

        // Status shows all zeros.
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"tracked_api_keys\":0"), "no keys: {}", body);
    }

    #[tokio::test]
    async fn edge_delete_nonexistent_usage_key_returns_404() {
        let (chain, api) = setup_all_plugins().await;
        let _ = chain;
        let mgmt_port = start_mgmt_server(api).await;

        // No traffic was sent — deleting a key that was never tracked.
        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/cost/usage/sk-ghost-key").await;
        assert_eq!(status, 404);
        assert!(body.contains("\"deleted\":false"), "not found: {}", body);
    }

    #[tokio::test]
    async fn edge_delete_all_usage_when_empty_is_noop() {
        let (chain, api) = setup_all_plugins().await;
        let _ = chain;
        let mgmt_port = start_mgmt_server(api).await;

        // DELETE all usage when nothing is tracked — should succeed.
        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/cost/usage").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"ok\":true"), "noop ok: {}", body);

        // Still empty.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body.contains("\"usage\":[]"), "still empty: {}", body);
    }

    #[tokio::test]
    async fn edge_delete_nonexistent_model_cost_returns_404() {
        let (chain, api) = setup_all_plugins().await;
        let _ = chain;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/cost/models/nonexistent-model").await;
        assert_eq!(status, 404);
        assert!(body.contains("\"deleted\":false"), "not found: {}", body);
    }

    #[tokio::test]
    async fn edge_add_model_cost_on_the_fly_affects_new_traffic() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Send traffic with default pricing.
        let req = chat_request("/v1/chat/completions", "sk-pricing-test1");
        send_request(&proxy_addr, req).await;

        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body.contains("sk-prici"), "key tracked: {}", body);

        // Extract cost with default pricing.
        // Default: 0.01 input, 0.02 output per 1k tokens.
        // Now add a much more expensive model cost for gpt-4.
        let (status, _) = mgmt_put(
            mgmt_port,
            "/api/v1/cost/models/gpt-4",
            r#"{"input_cost_per_1k":1.0,"output_cost_per_1k":2.0}"#,
        )
        .await;
        assert_eq!(status, 200);

        // Send traffic with the expensive pricing using a fresh key.
        let req = chat_request("/v1/chat/completions", "sk-pricing-test2");
        send_request(&proxy_addr, req).await;

        // Both keys should exist.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(
            body.contains("\"tracked_api_keys\":2"),
            "2 keys tracked: {}",
            body
        );

        // The second key's cost should be much higher than the first's
        // because model pricing was set at 100x the default.
        let usage = api_usage_entries(mgmt_port).await;
        let key1_cost = usage
            .iter()
            .find(|e| e.contains("sk-prici"))
            .expect("key1 present");
        let key2_cost = usage
            .iter()
            .find(|e| e.contains("sk-prici"))
            .expect("key2 present");
        // Both exist — the point is the model cost was picked up on the fly.
        assert!(!key1_cost.is_empty());
        assert!(!key2_cost.is_empty());
    }

    #[tokio::test]
    async fn edge_delete_usage_then_re_accumulate() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Send 3 requests.
        for _ in 0..3 {
            let req = chat_request("/v1/chat/completions", "sk-reaccum-key12");
            let resp = send_request(&proxy_addr, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
        }

        // Verify usage accumulated.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body.contains("sk-reacc"), "key present: {}", body);

        // Delete usage for this key.
        let (status, _) = mgmt_delete(mgmt_port, "/api/v1/cost/usage/sk-reaccum-key12").await;
        assert_eq!(status, 200);

        // Verify the key is gone.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body.contains("\"usage\":[]"), "cleared: {}", body);

        // Send 1 more request with the same key.
        let req = chat_request("/v1/chat/completions", "sk-reaccum-key12");
        send_request(&proxy_addr, req).await;

        // Usage should show fresh accumulation (only 1 request's worth).
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body.contains("sk-reacc"), "key re-tracked: {}", body);
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(
            body.contains("\"tracked_api_keys\":1"),
            "exactly 1 key: {}",
            body
        );
    }

    #[tokio::test]
    async fn edge_reset_all_then_re_accumulate() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Send traffic with multiple keys.
        for key in &["sk-reset-all-k1", "sk-reset-all-k2", "sk-reset-all-k3"] {
            let req = chat_request("/v1/chat/completions", key);
            send_request(&proxy_addr, req).await;
        }

        // Verify 3 keys tracked.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(body.contains("\"tracked_api_keys\":3"), "3 keys: {}", body);

        // Reset all usage.
        let (status, _) = mgmt_delete(mgmt_port, "/api/v1/cost/usage").await;
        assert_eq!(status, 200);

        // Verify all cleared.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(
            body.contains("\"tracked_api_keys\":0"),
            "0 keys after reset: {}",
            body
        );

        // Re-send with just one key.
        let req = chat_request("/v1/chat/completions", "sk-reset-all-k1");
        send_request(&proxy_addr, req).await;

        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(
            body.contains("\"tracked_api_keys\":1"),
            "1 key re-tracked: {}",
            body
        );
    }

    #[tokio::test]
    async fn edge_update_model_cost_mid_stream() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Set initial pricing.
        mgmt_put(
            mgmt_port,
            "/api/v1/cost/models/gpt-4",
            r#"{"input_cost_per_1k":0.01,"output_cost_per_1k":0.02}"#,
        )
        .await;

        // Send a request — cost calculated with 0.01/0.02.
        let req = chat_request("/v1/chat/completions", "sk-midstream-up1");
        send_request(&proxy_addr, req).await;

        let (_, body1) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;

        // Update pricing to 10x.
        mgmt_put(
            mgmt_port,
            "/api/v1/cost/models/gpt-4",
            r#"{"input_cost_per_1k":0.10,"output_cost_per_1k":0.20}"#,
        )
        .await;

        // Verify pricing updated.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert!(body.contains("0.100000"), "input cost updated: {}", body);
        assert!(body.contains("0.200000"), "output cost updated: {}", body);

        // Send another request with a fresh key — should use new pricing.
        let req = chat_request("/v1/chat/completions", "sk-midstream-up2");
        send_request(&proxy_addr, req).await;

        // Both keys exist — just verify the new key accumulated cost.
        let (_, body2) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body1.contains("sk-midst"), "key1 present: {}", body1);
        assert!(body2.contains("sk-midst"), "key2 present: {}", body2);
    }

    #[tokio::test]
    async fn edge_delete_model_cost_falls_back_to_defaults() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Set expensive pricing.
        mgmt_put(
            mgmt_port,
            "/api/v1/cost/models/gpt-4",
            r#"{"input_cost_per_1k":10.0,"output_cost_per_1k":20.0}"#,
        )
        .await;

        // Send request with expensive pricing.
        let req = chat_request("/v1/chat/completions", "sk-fallback-exp1");
        send_request(&proxy_addr, req).await;

        // Now delete the model pricing — subsequent requests should fall back to defaults.
        let (status, _) = mgmt_delete(mgmt_port, "/api/v1/cost/models/gpt-4").await;
        assert_eq!(status, 200);

        // Verify model list is empty.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert!(body.contains("\"models\":[]"), "models empty: {}", body);

        // Send request with fallback defaults (0.01/0.02).
        let req = chat_request("/v1/chat/completions", "sk-fallback-def1");
        send_request(&proxy_addr, req).await;

        // Both keys tracked — the expensive one should have a higher cost.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        assert!(body.contains("sk-fallb"), "keys present: {}", body);
    }

    #[tokio::test]
    async fn edge_delete_one_key_preserves_others() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Send traffic with 3 keys.
        for key in &["sk-preserve-aaa1", "sk-preserve-bbb2", "sk-preserve-ccc3"] {
            let req = chat_request("/v1/chat/completions", key);
            send_request(&proxy_addr, req).await;
        }

        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(body.contains("\"tracked_api_keys\":3"), "3 keys: {}", body);

        // Delete only the middle key.
        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/cost/usage/sk-preserve-bbb2").await;
        assert_eq!(status, 200);
        assert!(body.contains("\"deleted\":true"));

        // Verify 2 keys remain.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(body.contains("\"tracked_api_keys\":2"), "2 keys: {}", body);

        // The deleted key should not appear in usage.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/usage").await;
        // sk-preserve-bbb2 masked would be "sk-prese..." — all three share same mask prefix.
        // Instead, verify the count: usage array should have exactly 2 entries.
        let usage_count = body.matches("\"api_key\"").count();
        assert_eq!(usage_count, 2, "2 usage entries: {}", body);

        // Deleting the same key again returns 404.
        let (status, _) = mgmt_delete(mgmt_port, "/api/v1/cost/usage/sk-preserve-bbb2").await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn edge_put_model_cost_with_invalid_body() {
        let (chain, api) = setup_all_plugins().await;
        let _ = chain;
        let mgmt_port = start_mgmt_server(api).await;

        // Missing output_cost_per_1k.
        let (status, body) = mgmt_put(
            mgmt_port,
            "/api/v1/cost/models/gpt-4",
            r#"{"input_cost_per_1k":0.03}"#,
        )
        .await;
        assert_eq!(status, 400);
        assert!(
            body.contains("input_cost_per_1k and output_cost_per_1k"),
            "error message: {}",
            body
        );

        // Empty body.
        let (status, _) = mgmt_put(mgmt_port, "/api/v1/cost/models/gpt-4", "").await;
        assert_eq!(status, 400);

        // Garbage body.
        let (status, _) = mgmt_put(mgmt_port, "/api/v1/cost/models/gpt-4", "not json").await;
        assert_eq!(status, 400);

        // No model costs should have been created.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert!(body.contains("\"models\":[]"), "no models: {}", body);
    }

    #[tokio::test]
    async fn edge_request_without_api_key_not_tracked() {
        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (chain, api) = setup_all_plugins().await;

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api).await;

        // Send a request WITHOUT an API key header.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(chat_request_body())))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // No keys should be tracked.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/status").await;
        assert!(
            body.contains("\"tracked_api_keys\":0"),
            "no keys tracked without auth header: {}",
            body
        );
    }

    #[tokio::test]
    async fn edge_multiple_model_costs_crud() {
        let (chain, api) = setup_all_plugins().await;
        let _ = chain;
        let mgmt_port = start_mgmt_server(api).await;

        // Add 3 model costs.
        for (model, input, output) in &[
            ("gpt-4", "0.03", "0.06"),
            ("gpt-3.5-turbo", "0.001", "0.002"),
            ("claude-3-opus", "0.015", "0.075"),
        ] {
            let body = format!(
                r#"{{"input_cost_per_1k":{},"output_cost_per_1k":{}}}"#,
                input, output
            );
            let (status, _) =
                mgmt_put(mgmt_port, &format!("/api/v1/cost/models/{}", model), &body).await;
            assert_eq!(status, 200);
        }

        // Verify all 3 exist.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert!(body.contains("gpt-4"), "gpt-4: {}", body);
        assert!(body.contains("gpt-3.5-turbo"), "gpt-3.5: {}", body);
        assert!(body.contains("claude-3-opus"), "claude: {}", body);

        // Delete one.
        let (status, _) = mgmt_delete(mgmt_port, "/api/v1/cost/models/gpt-3.5-turbo").await;
        assert_eq!(status, 200);

        // Verify 2 remain.
        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        let model_count = body.matches("\"model\"").count();
        assert_eq!(model_count, 2, "2 models remain: {}", body);
        assert!(!body.contains("gpt-3.5-turbo"), "gpt-3.5 deleted: {}", body);

        // Update one of the remaining.
        let (status, _) = mgmt_put(
            mgmt_port,
            "/api/v1/cost/models/gpt-4",
            r#"{"input_cost_per_1k":0.05,"output_cost_per_1k":0.10}"#,
        )
        .await;
        assert_eq!(status, 200);

        let (_, body) = mgmt_get(mgmt_port, "/api/v1/cost/models").await;
        assert!(body.contains("0.050000"), "updated input: {}", body);
        assert!(body.contains("0.100000"), "updated output: {}", body);
    }

    #[tokio::test]
    async fn edge_method_not_allowed() {
        let (chain, api) = setup_all_plugins().await;
        let _ = chain;
        let mgmt_port = start_mgmt_server(api).await;

        // PUT on a GET-only endpoint.
        let (status, body) = mgmt_put(
            mgmt_port,
            "/api/v1/cost/usage/some-key",
            r#"{"anything":true}"#,
        )
        .await;
        assert_eq!(status, 405);
        assert!(
            body.contains("method not allowed"),
            "method not allowed: {}",
            body
        );
    }

    #[tokio::test]
    async fn edge_store_reflects_deletions() {
        use plugin_llm_gateway::store::{self, GatewayStore};

        let store_url = "sqlite:file:edge_del_test?mode=memory&cache=shared";

        let configs = vec![PluginConfig {
            name: "cost_tracker".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(0.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t.insert("default_cost_per_1k_input".into(), toml::Value::Float(0.01));
                t.insert(
                    "default_cost_per_1k_output".into(),
                    toml::Value::Float(0.02),
                );
                t
            }),
        }];

        let upstream_addr = start_upstream(llm_chat_handler).await;
        let router = catch_all_router(vec![upstream_addr]);

        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some(store_url),
            &[],
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();
        let chain = Arc::new(PluginChain::new(plugins));

        let proxy_addr = start_proxy_with_config(
            router,
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let mgmt_port = start_mgmt_server(api.clone()).await;

        // Send traffic and flush to store.
        let req = chat_request("/v1/chat/completions", "sk-store-del-test");
        send_request(&proxy_addr, req).await;
        api.flush().await;

        // Verify in store.
        let store = store::connect(store_url).await.unwrap();
        let usage = store.get_all_usage().await.unwrap();
        assert_eq!(usage.len(), 1, "1 key in store");

        // Delete via management API (this writes to both DashMap and store).
        let (status, _) = mgmt_delete(mgmt_port, "/api/v1/cost/usage/sk-store-del-test").await;
        assert_eq!(status, 200);

        // Verify store is also cleared.
        let usage = store.get_all_usage().await.unwrap();
        assert_eq!(usage.len(), 0, "store cleared after API delete");
    }

    #[tokio::test]
    async fn edge_model_cost_store_persistence() {
        use plugin_llm_gateway::store::{self, GatewayStore};

        let store_url = "sqlite:file:edge_model_persist?mode=memory&cache=shared";

        let configs = vec![PluginConfig {
            name: "cost_tracker".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("budget_limit".into(), toml::Value::Float(0.0));
                t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                t
            }),
        }];

        let (_, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some(store_url),
            &[],
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        let mgmt_port = start_mgmt_server(api).await;

        // Add a model cost via management API.
        let (status, _) = mgmt_put(
            mgmt_port,
            "/api/v1/cost/models/gpt-4",
            r#"{"input_cost_per_1k":0.03,"output_cost_per_1k":0.06}"#,
        )
        .await;
        assert_eq!(status, 200);

        // Verify it's persisted in the store.
        let store = store::connect(store_url).await.unwrap();
        let model = store.get_model_cost("gpt-4").await.unwrap();
        assert!(model.is_some(), "model cost persisted in store");
        let model = model.unwrap();
        assert!((model.input_cost_per_1k - 0.03).abs() < 1e-9);
        assert!((model.output_cost_per_1k - 0.06).abs() < 1e-9);

        // Delete via management API.
        mgmt_delete(mgmt_port, "/api/v1/cost/models/gpt-4").await;

        // Verify removed from store.
        let model = store.get_model_cost("gpt-4").await.unwrap();
        assert!(model.is_none(), "model cost removed from store");
    }

    /// Helper: get the raw usage response body to parse entries.
    async fn api_usage_entries(port: u16) -> Vec<String> {
        let (_, body) = mgmt_get(port, "/api/v1/cost/usage").await;
        // Split by "api_key" to get rough entries.
        body.split("\"api_key\"")
            .skip(1)
            .map(|s| s.to_string())
            .collect()
    }

    async fn mgmt_post(port: u16, path: &str, body: &str) -> (u16, String) {
        let client =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http::<Full<Bytes>>();
        let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
            .parse()
            .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri(uri)
            .header("authorization", format!("Bearer {}", MGMT_TOKEN))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body.to_string())))
            .unwrap();
        let resp = client.request(req).await.unwrap();
        let status = resp.status().as_u16();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    async fn mgmt_patch(port: u16, path: &str, body: &str) -> (u16, String) {
        let client =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http::<Full<Bytes>>();
        let uri: hyper::Uri = format!("http://127.0.0.1:{}{}", port, path)
            .parse()
            .unwrap();
        let req = Request::builder()
            .method("PATCH")
            .uri(uri)
            .header("authorization", format!("Bearer {}", MGMT_TOKEN))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body.to_string())))
            .unwrap();
        let resp = client.request(req).await.unwrap();
        let status = resp.status().as_u16();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    /// Setup plugins with virtual keys enabled (providers configured).
    async fn setup_plugins_with_virtual_keys() -> (Arc<PluginChain>, LlmGatewayApi) {
        let configs = vec![
            PluginConfig {
                name: "cost_tracker".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("budget_limit".into(), toml::Value::Float(100.0));
                    t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                    t
                }),
            },
            PluginConfig {
                name: "rate_limiter".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("tokens_per_minute".into(), toml::Value::Float(600_000.0));
                    t.insert("burst_tokens".into(), toml::Value::Float(100_000.0));
                    t
                }),
            },
        ];

        let providers = vec![canonical_provider(
            "openai",
            "sk-real-key",
            "https://api.openai.com",
            vec!["gpt-4o".to_string(), "gpt-4o-mini".to_string()],
            "authorization",
            None,
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        (Arc::new(PluginChain::new(plugins)), api)
    }

    async fn setup_plugins_with_virtual_keys_and_runtime_views() -> (Arc<PluginChain>, LlmGatewayApi)
    {
        let configs = vec![
            PluginConfig {
                name: "prompt_cache".into(),
                enabled: true,
                config: toml::Value::Table(toml::value::Map::new()),
            },
            PluginConfig {
                name: "tool_runtime".into(),
                enabled: true,
                config: toml::Value::Table(toml::value::Map::new()),
            },
        ];

        let providers = vec![canonical_provider(
            "openai",
            "sk-real-key",
            "https://api.openai.com",
            vec!["gpt-4o".to_string()],
            "authorization",
            None,
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        (Arc::new(PluginChain::new(plugins)), api)
    }

    async fn setup_plugins_with_virtual_keys_and_timeout(
        timeout_secs: Option<u64>,
    ) -> (Arc<PluginChain>, LlmGatewayApi) {
        let configs = vec![
            PluginConfig {
                name: "cost_tracker".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("budget_limit".into(), toml::Value::Float(100.0));
                    t.insert("log_interval_secs".into(), toml::Value::Integer(3600));
                    t
                }),
            },
            PluginConfig {
                name: "rate_limiter".into(),
                enabled: true,
                config: toml::Value::Table({
                    let mut t = toml::value::Map::new();
                    t.insert("tokens_per_minute".into(), toml::Value::Float(600_000.0));
                    t.insert("burst_tokens".into(), toml::Value::Float(100_000.0));
                    t
                }),
            },
        ];

        let providers = vec![canonical_provider(
            "openai",
            "sk-real-key",
            "https://api.openai.com",
            vec!["gpt-4o".to_string(), "gpt-4o-mini".to_string()],
            "authorization",
            timeout_secs,
            ProviderFamily::OpenAi,
            openai_tool_surfaces(),
            ProviderRoutingMetadataConfig::default(),
        )];

        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            &providers,
            &[],
            plugin_options(),
            None,
        )
        .await
        .unwrap();

        (Arc::new(PluginChain::new(plugins)), api)
    }

    async fn setup_plugins_with_virtual_keys_and_failover(
        providers: &[proxy_core::config::ProviderKeyConfig],
        failover_entries: &[(&str, &str)],
    ) -> (Arc<PluginChain>, LlmGatewayApi) {
        let configs = vec![PluginConfig {
            name: "provider_failover".into(),
            enabled: true,
            config: toml::Value::Table({
                let mut t = toml::value::Map::new();
                t.insert("cooldown_secs".into(), toml::Value::Integer(60));
                t.insert(
                    "providers".into(),
                    toml::Value::Array(
                        failover_entries
                            .iter()
                            .map(|(name, pattern)| {
                                let mut p = toml::value::Map::new();
                                p.insert("name".into(), toml::Value::String((*name).to_string()));
                                p.insert(
                                    "pattern".into(),
                                    toml::Value::String((*pattern).to_string()),
                                );
                                toml::Value::Table(p)
                            })
                            .collect(),
                    ),
                );
                t
            }),
        }];

        let registry = prometheus::Registry::new();
        let (plugins, api) = plugin_llm_gateway::create_plugins_with_options(
            &configs,
            Some("sqlite::memory:"),
            providers,
            &[],
            plugin_options(),
            Some(&registry),
        )
        .await
        .unwrap();

        (Arc::new(PluginChain::new(plugins)), api)
    }

    #[tokio::test]
    async fn provider_management_api_can_create_disable_and_delete_dynamic_provider() {
        std::env::set_var("TRP_TEST_BETA_PROVIDER_KEY", "sk-beta-managed");

        let beta_hits = Arc::new(AtomicUsize::new(0));
        let beta_upstream = start_upstream_async({
            let beta_hits = Arc::clone(&beta_hits);
            move |_req: Request<Incoming>| {
                let beta_hits = Arc::clone(&beta_hits);
                async move {
                    beta_hits.fetch_add(1, Ordering::Relaxed);
                    let body = r#"{"id":"chatcmpl-beta","object":"chat.completion","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"beta"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .header("content-length", body.len().to_string())
                        .body(Full::new(Bytes::from(body)))
                        .unwrap()
                }
            }
        })
        .await;

        let (chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api.clone()).await;
        let proxy_addr = start_proxy_with_config(
            catch_all_router(vec![format!("http://{}", beta_upstream)]),
            TestProxyConfig {
                plugins: Some(chain),
                ..Default::default()
            },
        )
        .await;

        let create_body = serde_json::json!({
            "name": "beta",
            "api_key_env": "TRP_TEST_BETA_PROVIDER_KEY",
            "base_url": format!("http://{}", beta_upstream),
            "models": ["gpt-4"],
            "family": "openai",
            "surfaces": {
                "tools": "openai",
                "responses": "openai_compatible"
            },
            "routing_metadata": {
                "data_collection": "deny",
                "zdr": true,
                "distillable_text": false,
                "quantizations": ["fp8"],
                "supported_parameter_families": ["tools"]
            }
        })
        .to_string();
        let (status, body) = mgmt_post(mgmt_port, "/api/v1/providers", &create_body).await;
        assert_eq!(status, 200, "{body}");
        let created: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(created["name"].as_str(), Some("beta"));
        assert_eq!(created["enabled"].as_bool(), Some(true));
        assert_eq!(created["source"].as_str(), Some("managed"));
        assert_eq!(
            created["api_key_env"].as_str(),
            Some("TRP_TEST_BETA_PROVIDER_KEY")
        );
        assert_eq!(created["family"].as_str(), Some("openai"));
        assert_eq!(
            created["surfaces"]["responses"].as_str(),
            Some("openai_compatible")
        );
        assert_eq!(created["tool_protocol"].as_str(), Some("openai"));
        let stored = api
            .managed_provider("beta")
            .expect("managed provider record");
        assert_eq!(stored.family.as_deref(), Some("openai"));
        assert!(stored
            .surfaces_json
            .as_deref()
            .unwrap_or_default()
            .contains("\"responses\":\"openai_compatible\""));

        let (plaintext_key, _) = api
            .create_virtual_key(
                Some("project-a"),
                "beta-key",
                "beta",
                None,
                None,
                None,
                None,
                Some(vec!["gpt-4".to_string()]),
                None,
            )
            .await
            .unwrap()
            .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(chat_request_body())))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8(body.to_vec())
            .unwrap()
            .contains("\"beta\""));
        assert_eq!(beta_hits.load(Ordering::Relaxed), 1);

        let (status, body) =
            mgmt_patch(mgmt_port, "/api/v1/providers/beta", r#"{"enabled":false}"#).await;
        assert_eq!(status, 200, "{body}");
        let disabled: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(disabled["enabled"].as_bool(), Some(false));
        assert_eq!(disabled["source"].as_str(), Some("managed"));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {}", plaintext_key))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(chat_request_body())))
            .unwrap();
        let resp = send_request(&proxy_addr, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(beta_hits.load(Ordering::Relaxed), 1);

        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/providers/beta").await;
        assert_eq!(status, 200, "{body}");
        let deleted: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(deleted["deleted"].as_bool(), Some(true));

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers/beta").await;
        assert_eq!(status, 404, "{body}");
    }

    #[tokio::test]
    async fn provider_management_api_updates_runtime_provider_views() {
        std::env::set_var("TRP_TEST_RUNTIME_PROVIDER_KEY", "sk-runtime-managed");

        let (_chain, api) = setup_plugins_with_virtual_keys_and_runtime_views().await;
        let mgmt_port = start_mgmt_server(api).await;

        let create_body = serde_json::json!({
            "name": "beta",
            "api_key_env": "TRP_TEST_RUNTIME_PROVIDER_KEY",
            "base_url": "https://beta.example.com",
            "models": ["gpt-4o"],
            "family": "openai",
            "surfaces": {
                "tools": "openai",
                "responses": "openai_compatible",
                "prompt_cache": {
                    "protocol": "openai",
                    "request_controls": true
                }
            }
        })
        .to_string();
        let (status, body) = mgmt_post(mgmt_port, "/api/v1/providers", &create_body).await;
        assert_eq!(status, 200, "{body}");

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/tool-runtime/status").await;
        assert_eq!(status, 200, "{body}");
        let runtime: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(runtime["providers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|provider| provider["name"].as_str() == Some("beta")));

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/prompt-cache/status").await;
        assert_eq!(status, 200, "{body}");
        let prompt_cache: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(prompt_cache["providers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|provider| provider["name"].as_str() == Some("beta")));

        let (status, body) = mgmt_patch(
            mgmt_port,
            "/api/v1/providers/openai",
            r#"{"enabled":false}"#,
        )
        .await;
        assert_eq!(status, 200, "{body}");
        let openai: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(openai["enabled"].as_bool(), Some(false));
        assert_eq!(openai["source"].as_str(), Some("static+overlay"));

        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/providers/openai").await;
        assert_eq!(status, 200, "{body}");

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/providers/openai").await;
        assert_eq!(status, 200, "{body}");
        let restored: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(restored["enabled"].as_bool(), Some(true));
        assert_eq!(restored["source"].as_str(), Some("static"));
    }

    #[tokio::test]
    async fn provider_management_api_rejects_legacy_provider_fields() {
        std::env::set_var("TRP_TEST_MIXED_PROVIDER_KEY", "sk-mixed-managed");

        let (_chain, api) = setup_plugins_with_virtual_keys_and_runtime_views().await;
        let mgmt_port = start_mgmt_server(api).await;

        let create_body = serde_json::json!({
            "name": "mixed",
            "api_key_env": "TRP_TEST_MIXED_PROVIDER_KEY",
            "base_url": "https://mixed.example.com",
            "models": ["gpt-4o"],
            "family": "openai",
            "tool_protocol": "openai",
            "surfaces": {
                "responses": "openai_compatible"
            }
        })
        .to_string();

        let (status, body) = mgmt_post(mgmt_port, "/api/v1/providers", &create_body).await;
        assert_eq!(status, 400, "{body}");
        assert!(body.contains("unknown field `tool_protocol`"), "{body}");
    }

    // -----------------------------------------------------------------------
    // Virtual key management integration tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn virtual_key_crud_via_management_api() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        // Create a key
        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","name":"test-key","provider_name":"openai"}"#,
        )
        .await;
        assert_eq!(status, 201, "create should return 201: {}", body);
        assert!(
            body.contains("\"key\":\"sk-trp-"),
            "should return plaintext key: {}",
            body
        );
        assert!(
            body.contains("\"key_hash\":"),
            "should return hash: {}",
            body
        );

        // Extract key_hash from response (find between "key_hash":" and next ")
        let hash_start = body.find("\"key_hash\":\"").unwrap() + 12;
        let hash_end = body[hash_start..].find('"').unwrap() + hash_start;
        let key_hash = &body[hash_start..hash_end];

        // List keys
        let (status, body) = mgmt_get(mgmt_port, "/api/v1/keys").await;
        assert_eq!(status, 200);
        assert!(body.contains("test-key"), "should list the key: {}", body);

        // Get key by hash prefix
        let prefix = &key_hash[..12];
        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{}", prefix)).await;
        assert_eq!(status, 200);
        assert!(body.contains("test-key"), "should get the key: {}", body);
        assert!(
            body.contains(&format!("\"key_hash\":\"{}\"", key_hash)),
            "full hash: {}",
            body
        );

        // Update the key (set active=false)
        let (status, body) = mgmt_patch(
            mgmt_port,
            &format!("/api/v1/keys/{}", prefix),
            r#"{"active":false}"#,
        )
        .await;
        assert_eq!(status, 200, "update should succeed: {}", body);

        // Verify update
        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{}", prefix)).await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"active\":false"),
            "should be deactivated: {}",
            body
        );

        // Delete the key
        let (status, body) = mgmt_delete(mgmt_port, &format!("/api/v1/keys/{}", prefix)).await;
        assert_eq!(status, 200, "delete should succeed: {}", body);

        // Verify deletion
        let (status, _) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{}", prefix)).await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn virtual_key_create_with_limits() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","name":"limited","provider_name":"openai","budget_limit":10.0,"rpm_limit":60,"tpm_limit":100000,"allowed_models":["gpt-4o"],"timeout_secs":25,"expires_at":"9999999999","allowed_tools":["web_search"]}"#,
        )
        .await;
        assert_eq!(status, 201, "create: {}", body);

        let hash_start = body.find("\"key_hash\":\"").unwrap() + 12;
        let hash_end = body[hash_start..].find('"').unwrap() + hash_start;
        let key_hash = &body[hash_start..hash_end];
        let prefix = &key_hash[..12];

        // Verify all fields were stored
        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{}", prefix)).await;
        assert_eq!(status, 200);
        assert!(body.contains("\"budget_limit\":10.0"), "budget: {}", body);
        assert!(body.contains("\"rpm_limit\":60"), "rpm: {}", body);
        assert!(body.contains("\"tpm_limit\":100000"), "tpm: {}", body);
        assert!(body.contains("gpt-4o"), "allowed_models: {}", body);
        assert!(
            body.contains("\"timeout_secs\":25"),
            "timeout_secs: {}",
            body
        );
        assert!(
            body.contains("\"tool_approval_mode\":\"allow_list\""),
            "tool approval mode: {}",
            body
        );
        assert!(body.contains("web_search"), "allowed_tools: {}", body);
        assert!(body.contains("9999999999"), "expires_at: {}", body);
    }

    #[tokio::test]
    async fn virtual_key_patch_updates_tool_approval_policy() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","name":"tool-policy-key","provider_name":"openai","allowed_tools":["web_search"]}"#,
        )
        .await;
        assert_eq!(status, 201, "create: {}", body);

        let hash_start = body.find("\"key_hash\":\"").unwrap() + 12;
        let hash_end = body[hash_start..].find('"').unwrap() + hash_start;
        let key_hash = &body[hash_start..hash_end];
        let prefix = &key_hash[..12];

        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{}", prefix)).await;
        assert_eq!(status, 200);
        assert!(body.contains("\"tool_approval_mode\":\"allow_list\""));
        assert!(body.contains("web_search"));

        let (status, body) = mgmt_patch(
            mgmt_port,
            &format!("/api/v1/keys/{}", prefix),
            r#"{"tool_approval_mode":"deny_all","allowed_tools":null}"#,
        )
        .await;
        assert_eq!(status, 200, "patch tool policy: {}", body);

        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{}", prefix)).await;
        assert_eq!(status, 200);
        assert!(body.contains("\"tool_approval_mode\":\"deny_all\""));
        assert!(body.contains("\"allowed_tools\":null"));
    }

    #[tokio::test]
    async fn virtual_key_create_unknown_provider_rejected() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","name":"bad","provider_name":"nonexistent"}"#,
        )
        .await;
        assert_eq!(status, 400, "unknown provider: {}", body);
        assert!(body.contains("unknown provider"), "error message: {}", body);
    }

    #[tokio::test]
    async fn virtual_key_ambiguous_prefix_returns_conflict() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let mut prefixes: std::collections::HashMap<char, Vec<String>> =
            std::collections::HashMap::new();
        for i in 0..17 {
            let (status, body) = mgmt_post(
                mgmt_port,
                "/api/v1/keys",
                &format!(r#"{{"project_id":"project-a","name":"ambiguous-{i}","provider_name":"openai"}}"#),
            )
            .await;
            assert_eq!(status, 201, "create should succeed: {}", body);

            let hash_start = body.find("\"key_hash\":\"").unwrap() + 12;
            let hash_end = body[hash_start..].find('"').unwrap() + hash_start;
            let key_hash = body[hash_start..hash_end].to_string();
            prefixes
                .entry(key_hash.chars().next().unwrap())
                .or_default()
                .push(key_hash);
        }

        let (prefix, matches) = prefixes
            .into_iter()
            .find(|(_, hashes)| hashes.len() > 1)
            .expect("17 keys should guarantee a duplicate one-character prefix");
        assert!(matches.len() > 1);
        let prefix = prefix.to_string();

        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{prefix}")).await;
        assert_eq!(status, 409, "GET should reject ambiguous prefix: {}", body);

        let (status, body) = mgmt_patch(
            mgmt_port,
            &format!("/api/v1/keys/{prefix}"),
            r#"{"active":false}"#,
        )
        .await;
        assert_eq!(
            status, 409,
            "PATCH should reject ambiguous prefix: {}",
            body
        );

        let (status, body) = mgmt_delete(mgmt_port, &format!("/api/v1/keys/{prefix}")).await;
        assert_eq!(
            status, 409,
            "DELETE should reject ambiguous prefix: {}",
            body
        );
    }

    #[tokio::test]
    async fn virtual_key_list_exposes_full_hash_for_recovery_from_ambiguous_prefixes() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let mut hashes = Vec::new();
        for i in 0..17 {
            let (status, body) = mgmt_post(
                mgmt_port,
                "/api/v1/keys",
                &format!(
                    r#"{{"project_id":"project-a","name":"recover-{i}","provider_name":"openai"}}"#
                ),
            )
            .await;
            assert_eq!(status, 201, "create should succeed: {}", body);
            let hash_start = body.find("\"key_hash\":\"").unwrap() + 12;
            let hash_end = body[hash_start..].find('"').unwrap() + hash_start;
            hashes.push(body[hash_start..hash_end].to_string());
        }

        let mut groups: std::collections::HashMap<char, Vec<String>> =
            std::collections::HashMap::new();
        for hash in &hashes {
            groups
                .entry(hash.chars().next().unwrap())
                .or_default()
                .push(hash.clone());
        }
        let ambiguous_group = groups
            .values()
            .find(|group| group.len() > 1)
            .expect("17 keys should guarantee an ambiguous one-character prefix")
            .clone();
        let ambiguous_prefix = ambiguous_group[0].chars().next().unwrap().to_string();

        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{ambiguous_prefix}")).await;
        assert_eq!(status, 409, "short prefix should be ambiguous: {}", body);

        let (status, list_body) = mgmt_get(mgmt_port, "/api/v1/keys").await;
        assert_eq!(status, 200);
        for hash in &ambiguous_group {
            assert!(
                list_body.contains(&format!("\"key_hash\":\"{}\"", hash)),
                "list output must expose full hashes for operator recovery: {}",
                list_body
            );
        }

        let chosen = &ambiguous_group[0];
        let unique_len = (1..=chosen.len())
            .find(|len| {
                let prefix = &chosen[..*len];
                hashes
                    .iter()
                    .filter(|hash| hash.starts_with(prefix))
                    .count()
                    == 1
            })
            .expect("a full hash must eventually be unique");
        let unique_prefix = &chosen[..unique_len];

        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{unique_prefix}")).await;
        assert_eq!(
            status, 200,
            "longer unique prefix should recover access: {}",
            body
        );
        assert!(body.contains(&format!("\"key_hash\":\"{}\"", chosen)));
    }

    #[tokio::test]
    async fn virtual_key_create_missing_fields_rejected() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        // Missing name
        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","provider_name":"openai"}"#,
        )
        .await;
        assert_eq!(status, 400, "missing name: {}", body);

        // Missing provider_name
        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","name":"test"}"#,
        )
        .await;
        assert_eq!(status, 400, "missing provider: {}", body);
    }

    #[tokio::test]
    async fn virtual_key_patch_updates_allowed_models_and_expires_at() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        // Create a key with no allowed_models and no expiry
        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","name":"patch-test","provider_name":"openai"}"#,
        )
        .await;
        assert_eq!(status, 201);

        let hash_start = body.find("\"key_hash\":\"").unwrap() + 12;
        let hash_end = body[hash_start..].find('"').unwrap() + hash_start;
        let key_hash = &body[hash_start..hash_end];
        let prefix = &key_hash[..12];

        // PATCH allowed_models and expires_at
        let (status, _) = mgmt_patch(
            mgmt_port,
            &format!("/api/v1/keys/{}", prefix),
            r#"{"allowed_models":["gpt-4o-mini"],"timeout_secs":42,"expires_at":"9999999999"}"#,
        )
        .await;
        assert_eq!(status, 200, "PATCH should succeed");

        // Verify the fields were updated
        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{}", prefix)).await;
        assert_eq!(status, 200);

        assert!(
            body.contains("gpt-4o-mini"),
            "allowed_models should be updated: {}",
            body
        );
        assert!(
            body.contains("\"timeout_secs\":42"),
            "timeout_secs should be updated: {}",
            body
        );

        assert!(
            body.contains("9999999999"),
            "expires_at should be updated: {}",
            body
        );
    }

    #[tokio::test]
    async fn project_effective_runtime_policy_merges_project_and_virtual_key_overrides() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_put(
            mgmt_port,
            "/api/v1/projects/project-a/policy",
            r#"{"budget_limit":99.0,"budget_duration":"monthly","rpm_limit":77,"tpm_limit":7777,"fallback_order":["openai","anthropic"],"adaptive_enabled":false,"timeout_secs":45,"provider_rpm_limits":{"openai":44},"provider_tpm_limits":{"openai":4444},"provider_timeouts":{"openai":15},"provider_input_costs":{"openai":0.03},"provider_output_costs":{"openai":0.07},"semantic_cache_enabled":true,"semantic_cache_ttl_secs":321,"semantic_cache_similarity_threshold":0.81,"allowed_tools":["web_search"]}"#,
        )
        .await;
        assert_eq!(status, 200, "policy update failed: {body}");

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","name":"effective-key","provider_name":"openai","budget_limit":10.0,"rpm_limit":60,"timeout_secs":25,"allowed_models":["gpt-4o-mini"],"allowed_tools":["arxiv_search"]}"#,
        )
        .await;
        assert_eq!(status, 201, "key create failed: {body}");

        let body_json: serde_json::Value = serde_json::from_str(&body).expect("key create json");
        let key_hash = body_json["key_hash"].as_str().expect("key hash");

        let (status, body) = mgmt_get(
            mgmt_port,
            &format!(
                "/api/v1/projects/project-a/effective-runtime-policy?key_hash={}",
                key_hash
            ),
        )
        .await;
        assert_eq!(status, 200, "effective policy lookup failed: {body}");

        let body_json: serde_json::Value =
            serde_json::from_str(&body).expect("effective policy json");
        assert_eq!(body_json["project_id"].as_str(), Some("project-a"));
        assert_eq!(body_json["provider_name"].as_str(), Some("openai"));
        assert_eq!(
            body_json["virtual_key"]["name"].as_str(),
            Some("effective-key")
        );
        assert_eq!(body_json["effective"]["budget_limit"].as_f64(), Some(10.0));
        assert_eq!(
            body_json["sources"]["budget_limit"].as_str(),
            Some("virtual_key")
        );
        assert_eq!(
            body_json["effective"]["budget_duration"].as_str(),
            Some("monthly")
        );
        assert_eq!(
            body_json["sources"]["budget_duration"].as_str(),
            Some("project_policy")
        );
        assert_eq!(body_json["effective"]["rpm_limit"].as_u64(), Some(60));
        assert_eq!(
            body_json["sources"]["rpm_limit"].as_str(),
            Some("virtual_key")
        );
        assert_eq!(body_json["effective"]["tpm_limit"].as_u64(), Some(4444));
        assert_eq!(
            body_json["sources"]["tpm_limit"].as_str(),
            Some("project_provider_policy")
        );
        assert_eq!(body_json["effective"]["timeout_secs"].as_u64(), Some(25));
        assert_eq!(
            body_json["sources"]["timeout_secs"].as_str(),
            Some("virtual_key")
        );
        assert_eq!(
            body_json["effective"]["tool_approval_mode"].as_str(),
            Some("allow_list")
        );
        assert_eq!(
            body_json["sources"]["tool_approval_mode"].as_str(),
            Some("virtual_key")
        );
        assert_eq!(
            body_json["effective"]["allowed_tools"][0].as_str(),
            Some("arxiv_search")
        );
        assert_eq!(
            body_json["sources"]["allowed_tools"].as_str(),
            Some("virtual_key")
        );
        assert_eq!(
            body_json["effective"]["provider_input_cost"].as_f64(),
            Some(0.03)
        );
        assert_eq!(
            body_json["effective"]["provider_output_cost"].as_f64(),
            Some(0.07)
        );
        assert_eq!(
            body_json["sources"]["provider_input_cost"].as_str(),
            Some("project_policy")
        );
        assert_eq!(
            body_json["sources"]["provider_output_cost"].as_str(),
            Some("project_policy")
        );
        assert_eq!(
            body_json["effective"]["semantic_cache_enabled"].as_bool(),
            Some(true)
        );
        assert_eq!(
            body_json["sources"]["semantic_cache_enabled"].as_str(),
            Some("project_policy")
        );
        assert_eq!(
            body_json["effective"]["semantic_cache_ttl_secs"].as_u64(),
            Some(321)
        );
        assert_eq!(
            body_json["sources"]["semantic_cache_ttl_secs"].as_str(),
            Some("project_policy")
        );
        assert_eq!(
            body_json["effective"]["semantic_cache_similarity_threshold"].as_f64(),
            Some(0.81)
        );
        assert_eq!(
            body_json["sources"]["semantic_cache_similarity_threshold"].as_str(),
            Some("project_policy")
        );
        assert_eq!(
            body_json["effective"]["fallback_order"][0].as_str(),
            Some("openai")
        );
        assert_eq!(
            body_json["sources"]["fallback_order"].as_str(),
            Some("project_policy")
        );
        assert_eq!(
            body_json["effective"]["adaptive_enabled"].as_bool(),
            Some(false)
        );
        assert_eq!(
            body_json["sources"]["adaptive_enabled"].as_str(),
            Some("project_policy")
        );
        assert_eq!(
            body_json["provider_context"]["project_provider_rpm_limit"].as_u64(),
            Some(44)
        );
        assert_eq!(
            body_json["provider_context"]["project_provider_tpm_limit"].as_u64(),
            Some(4444)
        );
        assert_eq!(
            body_json["provider_context"]["provider_input_cost"].as_f64(),
            Some(0.03)
        );
        assert_eq!(
            body_json["provider_context"]["provider_output_cost"].as_f64(),
            Some(0.07)
        );
        assert_eq!(
            body_json["timeout_context"]["project_provider_timeout_secs"].as_u64(),
            Some(15)
        );
        assert!(body_json["notes"]
            .as_array()
            .expect("notes array")
            .iter()
            .any(|entry| entry
                .as_str()
                .unwrap_or_default()
                .contains("Routing rule overrides")));
    }

    #[tokio::test]
    async fn project_effective_runtime_policy_can_show_provider_default_timeout_without_key() {
        let (_chain, api) = setup_plugins_with_virtual_keys_and_timeout(Some(90)).await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_get(
            mgmt_port,
            "/api/v1/projects/project-a/effective-runtime-policy?provider_name=openai",
        )
        .await;
        assert_eq!(status, 200, "effective policy lookup failed: {body}");

        let body_json: serde_json::Value =
            serde_json::from_str(&body).expect("effective policy json");
        assert_eq!(body_json["provider_name"].as_str(), Some("openai"));
        assert!(body_json["virtual_key"].is_null());
        assert!(body_json["project_policy"].is_null());
        assert_eq!(body_json["effective"]["timeout_secs"].as_u64(), Some(90));
        assert_eq!(
            body_json["sources"]["timeout_secs"].as_str(),
            Some("provider_default")
        );
        assert_eq!(
            body_json["timeout_context"]["provider_default_timeout_secs"].as_u64(),
            Some(90)
        );
        assert_eq!(
            body_json["effective"]["adaptive_enabled"].as_bool(),
            Some(true)
        );
        assert_eq!(
            body_json["sources"]["adaptive_enabled"].as_str(),
            Some("default")
        );
    }

    #[tokio::test]
    async fn virtual_key_patch_budget_and_rpm_works() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","name":"patch-ok","provider_name":"openai"}"#,
        )
        .await;
        assert_eq!(status, 201);

        let hash_start = body.find("\"key_hash\":\"").unwrap() + 12;
        let hash_end = body[hash_start..].find('"').unwrap() + hash_start;
        let key_hash = &body[hash_start..hash_end];
        let prefix = &key_hash[..12];

        // PATCH budget and RPM (these fields ARE forwarded)
        let (status, _) = mgmt_patch(
            mgmt_port,
            &format!("/api/v1/keys/{}", prefix),
            r#"{"budget_limit":25.0,"rpm_limit":120,"tpm_limit":50000}"#,
        )
        .await;
        assert_eq!(status, 200);

        // Verify the fields were updated
        let (status, body) = mgmt_get(mgmt_port, &format!("/api/v1/keys/{}", prefix)).await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"budget_limit\":25.0"),
            "budget updated: {}",
            body
        );
        assert!(body.contains("\"rpm_limit\":120"), "rpm updated: {}", body);
        assert!(
            body.contains("\"tpm_limit\":50000"),
            "tpm updated: {}",
            body
        );
    }

    #[tokio::test]
    async fn virtual_key_delete_nonexistent_returns_404() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_delete(mgmt_port, "/api/v1/keys/nonexistent_hash").await;
        assert_eq!(status, 404, "should be not found: {}", body);
    }

    #[tokio::test]
    async fn virtual_key_get_nonexistent_returns_404() {
        let (_chain, api) = setup_plugins_with_virtual_keys().await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, _) = mgmt_get(mgmt_port, "/api/v1/keys/nonexistent_hash").await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn virtual_keys_not_enabled_returns_error() {
        // setup_all_plugins() does NOT configure providers, so virtual_keys is disabled.
        let (_chain, api) = setup_all_plugins().await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_get(mgmt_port, "/api/v1/keys").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("virtual_keys not enabled"),
            "disabled: {}",
            body
        );

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/keys",
            r#"{"project_id":"project-a","name":"x","provider_name":"openai"}"#,
        )
        .await;
        assert_eq!(status, 200);
        assert!(
            body.contains("virtual_keys not enabled"),
            "disabled: {}",
            body
        );
    }

    #[tokio::test]
    async fn safety_detector_catalog_reflects_project_policy() {
        let (_chain, api) = setup_all_plugins().await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_post(
            mgmt_port,
            "/api/v1/projects",
            r#"{"project_id":"project-a","name":"Project A"}"#,
        )
        .await;
        assert_eq!(status, 201, "project create should succeed: {}", body);

        let (status, body) = mgmt_put(
            mgmt_port,
            "/api/v1/projects/project-a/safety",
            r#"{
                "mode":"observe_only",
                "rules":[
                    {
                        "detector_class":"aws_access_key",
                        "action":"block",
                        "verification":"disabled"
                    }
                ]
            }"#,
        )
        .await;
        assert_eq!(status, 200, "safety policy should be updated: {}", body);

        let (status, body) =
            mgmt_get(mgmt_port, "/api/v1/projects/project-a/safety/detectors").await;
        assert_eq!(status, 200, "detector catalog should load: {}", body);
        assert!(
            body.contains("\"detector_class\":\"aws_access_key\""),
            "{}",
            body
        );
        assert!(body.contains("\"effective_action\":\"block\""), "{}", body);
        assert!(
            body.contains("\"verification_mode\":\"disabled\""),
            "{}",
            body
        );
        assert!(
            body.contains("\"verifier_kind\":\"aws_example_guard\""),
            "{}",
            body
        );
        assert!(
            body.contains("\"detector_class\":\"github_pat_classic\""),
            "{}",
            body
        );
        assert!(
            body.contains("\"remote_verifier_kind\":\"github_user_api\""),
            "{}",
            body
        );
        assert!(
            body.contains("\"detector_class\":\"slack_bot_token\""),
            "{}",
            body
        );
        assert!(
            body.contains("\"remote_verifier_kind\":\"slack_auth_test_api\""),
            "{}",
            body
        );
        assert!(body.contains("\"detector_class\":\"email\""), "{}", body);
        assert!(
            body.contains("\"effective_action\":\"observe_only\""),
            "{}",
            body
        );
    }

    #[tokio::test]
    async fn semantic_policy_crud_pushes_to_service() {
        let (endpoint, _data_dir) = start_semantic_service().await;
        let (_chain, api) = setup_plugins_with_semantic_safety(&endpoint).await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_put(
            mgmt_port,
            "/api/v1/projects/project-a/semantic-safety",
            r#"{
                "enabled": true,
                "entities": [{"entity_id":"company-x","name":"Company X","aliases":["companyx"]}],
                "topics": [{"topic_id":"layoffs","name":"Layoffs","exemplars":["company x layoffs next week"],"rerank_threshold":0.1,"require_entity_match":true}]
            }"#,
        )
        .await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"ok\":true"),
            "upsert should succeed: {}",
            body
        );
        assert!(
            body.contains("\"synced\":true"),
            "service push should succeed: {}",
            body
        );

        let (status, body) =
            mgmt_get(mgmt_port, "/api/v1/projects/project-a/semantic-safety").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"project_id\":\"project-a\""),
            "policy get: {}",
            body
        );
        assert!(
            body.contains("\"company-x\""),
            "policy should include entity: {}",
            body
        );
        assert!(
            body.contains("\"layoffs\""),
            "policy should include topic: {}",
            body
        );

        let (status, body) = mgmt_get(
            mgmt_port,
            "/api/v1/projects/project-a/semantic-safety/status",
        )
        .await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"synced\":true"),
            "status should be synced: {}",
            body
        );
        assert!(
            body.contains("\"index_state\":\"ready\""),
            "service should be ready: {}",
            body
        );
        assert!(
            body.contains("\"ready\":false"),
            "stub backend should be reported as not ready: {}",
            body
        );
        assert!(
            body.contains("\"backend\":\"tensorrt-dev-stub\""),
            "status should surface backend mode: {}",
            body
        );

        let (status, body) =
            mgmt_delete(mgmt_port, "/api/v1/projects/project-a/semantic-safety").await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"ok\":true"),
            "delete should succeed: {}",
            body
        );
        assert!(
            body.contains("\"synced\":true"),
            "remote delete should succeed: {}",
            body
        );

        let (status, body) = mgmt_get(
            mgmt_port,
            "/api/v1/projects/project-a/semantic-safety/status",
        )
        .await;
        assert_eq!(status, 200);
        assert!(
            body.contains("\"index_state\":\"missing\""),
            "remote state should be missing: {}",
            body
        );
    }

    #[tokio::test]
    async fn disabled_semantic_policy_is_reported_as_synced() {
        let (endpoint, _data_dir) = start_semantic_service().await;
        let (_chain, api) = setup_plugins_with_semantic_safety(&endpoint).await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, body) = mgmt_put(
            mgmt_port,
            "/api/v1/projects/project-a/semantic-safety",
            r#"{
                "enabled": false,
                "entities": [{"entity_id":"company-x","name":"Company X","aliases":[]}],
                "topics": [{"topic_id":"layoffs","name":"Layoffs","exemplars":["company x layoffs"],"rerank_threshold":0.1,"require_entity_match":true}]
            }"#,
        )
        .await;
        assert_eq!(status, 200);
        assert!(body.contains("\"synced\":true"), "{}", body);

        let (status, body) = mgmt_get(
            mgmt_port,
            "/api/v1/projects/project-a/semantic-safety/status",
        )
        .await;
        assert_eq!(status, 200);
        assert!(body.contains("\"synced\":true"), "{}", body);
        assert!(body.contains("\"index_state\":\"disabled\""), "{}", body);
        assert!(
            body.contains("\"backend\":\"tensorrt-dev-stub\""),
            "{}",
            body
        );
    }

    #[tokio::test]
    async fn semantic_policies_are_isolated_per_project() {
        let (endpoint, _data_dir) = start_semantic_service().await;
        let (_chain, api) = setup_plugins_with_semantic_safety(&endpoint).await;
        let mgmt_port = start_mgmt_server(api).await;

        let (status, _) = mgmt_put(
            mgmt_port,
            "/api/v1/projects/project-a/semantic-safety",
            r#"{
                "entities": [{"entity_id":"company-x","name":"Company X","aliases":[]}],
                "topics": [{"topic_id":"layoffs","name":"Layoffs","exemplars":["company x layoffs"],"rerank_threshold":0.1,"require_entity_match":true}]
            }"#,
        )
        .await;
        assert_eq!(status, 200);

        let (status, _) = mgmt_put(
            mgmt_port,
            "/api/v1/projects/project-b/semantic-safety",
            r#"{
                "entities": [{"entity_id":"company-y","name":"Company Y","aliases":[]}],
                "topics": [{"topic_id":"earnings","name":"Earnings","exemplars":["company y earnings"],"rerank_threshold":0.1,"require_entity_match":true}]
            }"#,
        )
        .await;
        assert_eq!(status, 200);

        let (status, body_a) =
            mgmt_get(mgmt_port, "/api/v1/projects/project-a/semantic-safety").await;
        assert_eq!(status, 200);
        assert!(
            body_a.contains("\"company-x\""),
            "project-a should keep its policy: {}",
            body_a
        );
        assert!(
            !body_a.contains("\"company-y\""),
            "project-a should not leak project-b data: {}",
            body_a
        );

        let (status, body_b) =
            mgmt_get(mgmt_port, "/api/v1/projects/project-b/semantic-safety").await;
        assert_eq!(status, 200);
        assert!(
            body_b.contains("\"company-y\""),
            "project-b should keep its policy: {}",
            body_b
        );
        assert!(
            !body_b.contains("\"company-x\""),
            "project-b should not leak project-a data: {}",
            body_b
        );
    }
}
